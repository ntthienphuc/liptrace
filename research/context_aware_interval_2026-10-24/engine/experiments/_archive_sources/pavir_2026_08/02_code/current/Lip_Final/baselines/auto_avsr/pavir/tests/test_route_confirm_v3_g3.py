from __future__ import annotations

import copy
import json

import pytest
import torch

from datamodule.av_dataset import AVDataset
from datamodule.data_module import CustomBucketDataset
from pavir.route_confirm_v3_g3 import (
    FrozenCalibrationSchedule,
    load_frozen_schedule,
    validate_bucket_dataset,
    validate_emitted_batch,
)


def _payload() -> dict:
    selected = []
    for ordinal in range(64):
        sample_id = f"vicocktail:train:source-{ordinal}_clip"
        selected.append(
            {
                "selection_ordinal_zero_based": ordinal,
                "physical_batch_index": ordinal,
                "row_indices": [ordinal],
                "sample_ids": [sample_id],
                "source_ids": [f"source-{ordinal}"],
                "input_lengths": [10],
                "sample_count": 1,
                "source_count": 1,
                "padded_frames": 10,
                "E_eligible": True,
                "D_eligible": True,
                "D_eligible_ordinal_zero_based": ordinal,
                "D_eligible_rank_one_based": ordinal + 1,
                "D_quota_selected": ordinal % 2 == 1,
                "route_mass": {
                    route: {
                        "batch_count": 1,
                        "event_count": 1,
                        "sample_count": 1,
                        "source_count": 1,
                    }
                    for route in ("E", "D")
                },
            }
        )
    return {
        "schema_version": "pavir-route-confirm-v3-g2-calibration-schedule-v1",
        "protocol_name": "PAViR-RouteConfirm-v3",
        "status": "FROZEN_FOR_G3_CALIBRATION_ONLY",
        "method_contract": {"path": "/contract.json", "sha256": "a" * 64},
        "inputs": {
            "auto_avsr_train_labels": {
                "path": "/train.csv",
                "sha256": "b" * 64,
            },
            "canonical_id_map": {
                "path": "/canonical.jsonl",
                "sha256": "c" * 64,
            },
            "occurrences_sha256": "d" * 64,
            "routes_sha256": "e" * 64,
        },
        "schedule": {
            "physical_batch_builder": "AutoAVSR CustomBucketDataset",
            "num_buckets": 400,
            "max_padded_frames": 6000,
            "batch_size": None,
            "curriculum": False,
            "selection": (
                "torch.randperm(number_of_physical_batches, "
                "Generator().manual_seed(42))[:64]"
            ),
            "seed": 42,
            "selected_batch_count": 64,
            "D_context_fraction": 0.5,
            "D_quota_semantics": (
                "locked cumulative quota over D-eligible batch ordinals"
            ),
        },
        "physical_batch_count": 4000,
        "selected_physical_batch_indices": list(range(64)),
        "selected_batches": selected,
        "authorization": {
            "g3_wrapper_may_consume_after_g2_gate": True,
            "direct_calibration_authorized": False,
            "training_authorized": False,
        },
    }


def _write(tmp_path, payload: dict):
    path = tmp_path / "schedule.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_schedule_accepts_exact_frozen_inventory(tmp_path):
    schedule = load_frozen_schedule(_write(tmp_path, _payload()))
    assert isinstance(schedule, FrozenCalibrationSchedule)
    assert len(schedule.batches) == 64
    assert sum(row.d_quota_selected for row in schedule.batches) == 32
    assert schedule.batches[63].sample_ids == (
        "vicocktail:train:source-63_clip",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["selected_batches"][0].update(
            {"D_quota_selected": True}
        ),
        lambda value: value["selected_batches"][1].update(
            {"sample_ids": ["wrong-corpus:train:wrong"]}
        ),
        lambda value: value["selected_batches"][2].update(
            {"input_lengths": [11]}
        ),
        lambda value: value["selected_physical_batch_indices"].__setitem__(3, 4),
    ],
)
def test_schedule_rejects_quota_alignment_and_order_tamper(tmp_path, mutation):
    payload = _payload()
    mutation(payload)
    with pytest.raises(ValueError):
        load_frozen_schedule(_write(tmp_path, payload))


def test_schedule_hash_is_binding(tmp_path):
    path = _write(tmp_path, _payload())
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_schedule(path, expected_sha256="0" * 64)


def _fake_bucket_dataset(schedule: FrozenCalibrationSchedule):
    base = object.__new__(AVDataset)
    base.input_lengths = [10] * 64
    base.list = [
        ("fixture", f"source-{index}_clip", 10, torch.tensor([1]), None)
        for index in range(64)
    ]
    base.subset = "train"
    base.sample_id_map = None

    def sample_id(path):
        return "vicocktail:train:" + str(path)

    base.sample_id = sample_id
    bucket = object.__new__(CustomBucketDataset)
    bucket.dataset = base
    bucket.batches = [[index] for index in range(64)] + [
        [] for _ in range(4000 - 64)
    ]
    validate_bucket_dataset(bucket, schedule)
    return bucket


def test_reconstructed_bucket_rows_and_samples_are_binding(tmp_path):
    schedule = load_frozen_schedule(_write(tmp_path, _payload()))
    bucket = _fake_bucket_dataset(schedule)
    bucket.batches[7] = [8]
    with pytest.raises(ValueError, match="row order"):
        validate_bucket_dataset(bucket, schedule)


def test_runtime_batch_validator_rejects_sample_or_length_drift(tmp_path):
    schedule = load_frozen_schedule(_write(tmp_path, _payload()))
    spec = schedule.batches[0]
    batch = {
        "inputs": torch.zeros(1, 10, 1, 88, 88),
        "input_lengths": torch.tensor([10]),
        "targets": torch.ones(1, 3, 1, dtype=torch.long),
        "sample_ids": list(spec.sample_ids),
    }
    result = validate_emitted_batch(batch, spec)
    assert result["physical_batch_index"] == 0
    assert result["padded_frames"] == 10

    tampered = copy.copy(batch)
    tampered["input_lengths"] = torch.tensor([9])
    with pytest.raises(ValueError, match="frozen schedule"):
        validate_emitted_batch(tampered, spec)

    missing = copy.copy(batch)
    del missing["sample_ids"]
    missing["unrelated_extra_field"] = True
    with pytest.raises(ValueError, match="missing required"):
        validate_emitted_batch(missing, spec)
