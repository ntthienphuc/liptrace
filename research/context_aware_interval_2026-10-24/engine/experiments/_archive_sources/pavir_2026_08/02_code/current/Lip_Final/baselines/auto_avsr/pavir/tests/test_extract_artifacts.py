from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "extract_pavir_artifacts",
    ROOT / "tools" / "extract_pavir_artifacts.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ESTIMATOR_SPEC = importlib.util.spec_from_file_location(
    "estimate_pavir_routes_for_extract_test",
    ROOT / "tools" / "estimate_pavir_routes.py",
)
ESTIMATOR = importlib.util.module_from_spec(ESTIMATOR_SPEC)
assert ESTIMATOR_SPEC.loader is not None
sys.modules[ESTIMATOR_SPEC.name] = ESTIMATOR
ESTIMATOR_SPEC.loader.exec_module(ESTIMATOR)


SP_MODEL = ROOT / "benchmark_v1" / "vocabulary" / "vi_unigram_2000.model"
UNITS = (
    ROOT
    / "benchmark_v1"
    / "vocabulary"
    / "vi_unigram_2000.auto_avsr_units.txt"
)


def test_native_lightning_state_strip_is_exact():
    tensor = torch.tensor([1.0])
    assert MODULE.extract_native_state_dict(
        {"state_dict": {"model.frontend.weight": tensor}}
    ) == {"frontend.weight": tensor}
    with pytest.raises(ValueError, match="non-native"):
        MODULE.extract_native_state_dict(
            {"state_dict": {"model.frontend.weight": tensor, "other": tensor}}
        )
    with pytest.raises(ValueError, match="model-prefixed"):
        MODULE.extract_native_state_dict(
            {"state_dict": {"model.model.frontend.weight": tensor}}
        )


def test_endpoint_scan_freezes_every_word_occurrence(tmp_path):
    tokenizer = MODULE.ExactTokenizer(SP_MODEL, UNITS)
    endpoint_ids = tokenizer.encode("xin")
    endpoints = {
        "xin": [
            MODULE.Endpoint("edge-1", "target", "xin", endpoint_ids),
            MODULE.Endpoint("edge-2", "confusion", "xin", endpoint_ids),
        ]
    }
    cache = tmp_path / "train" / "clip.npy"
    cache.parent.mkdir()
    np.save(cache, np.zeros((12, 96, 96), dtype=np.uint8), allow_pickle=False)
    rows = [
        {
            "sample_id": "clip",
            "canonical_id": "vicocktail:train:clip",
            "source_id": "source-a",
            "partition": "VC-calib",
            "split": "train",
            "transcript": "xin chào xin",
            "actual_frames": 12,
        }
    ]
    samples, audit = MODULE.build_sample_specs(
        rows,
        corpus="vicocktail",
        graph_split="vc-calib",
        endpoints_by_word=endpoints,
        tokenizer=tokenizer,
        cache_root=tmp_path,
        forbidden=set(),
    )
    assert audit["endpoint_occurrences"] == 4
    assert len(samples) == 1
    assert [
        (event.edge_id, event.endpoint, event.word_index)
        for event in samples[0].occurrences
    ] == [
        ("edge-1", "target", 0),
        ("edge-2", "confusion", 0),
        ("edge-1", "target", 2),
        ("edge-2", "confusion", 2),
    ]


def test_promote_and_test_rows_fail_closed(tmp_path):
    tokenizer = MODULE.ExactTokenizer(SP_MODEL, UNITS)
    endpoint = MODULE.Endpoint(
        "edge-1", "target", "xin", tokenizer.encode("xin")
    )
    row = {
        "sample_id": "clip",
        "source_id": "source-a",
        "partition": "VC-promote",
        "split": "train",
        "transcript": "xin",
        "actual_frames": 4,
    }
    with pytest.raises(ValueError, match="promote/test"):
        MODULE.build_sample_specs(
            [row],
            corpus="vicocktail",
            graph_split="vc-calib",
            endpoints_by_word={"xin": [endpoint]},
            tokenizer=tokenizer,
            cache_root=tmp_path,
            forbidden=set(),
        )


def test_three_crop_views_are_locked_and_padding_is_native_zero():
    raw = torch.arange(96 * 96, dtype=torch.int64).remainder(256)
    raw = raw.to(torch.uint8).view(1, 1, 96, 96).repeat(1, 2, 1, 1)
    lengths = torch.tensor([1])
    center = MODULE.prepare_view(raw, lengths, "center")
    left = MODULE.prepare_view(raw, lengths, "left")
    right = MODULE.prepare_view(raw, lengths, "right")
    assert center.shape == (1, 2, 1, 88, 88)
    torch.testing.assert_close(
        center[0, 0, 0, 0, 0],
        (raw[0, 0, 4, 4].float() / 255.0 - 0.421) / 0.165,
    )
    assert left[0, 0, 0, 0, 0] != center[0, 0, 0, 0, 0]
    assert right[0, 0, 0, 0, 0] != center[0, 0, 0, 0, 0]
    assert torch.count_nonzero(center[0, 1]) == 0


def test_vc_firewall_rows_resolve_the_frozen_valid_cache(tmp_path):
    cache = tmp_path / "valid" / "clip.npy"
    cache.parent.mkdir()
    np.save(cache, np.zeros((4, 96, 96), dtype=np.uint8), allow_pickle=False)
    row = {
        "sample_id": "clip",
        "split": "train",
        "partition": "VC-calib",
        "cache_path": None,
    }
    assert MODULE._cache_path(row, tmp_path, "clip") == cache.resolve()


def test_extraction_batches_cap_padded_workload():
    samples = [
        MODULE.SampleSpec(
            sample_id=f"vicocktail:train:{index}",
            raw_sample_id=str(index),
            source_id="source",
            transcript="a",
            target_ids=(1,),
            cache_path=Path("/nonexistent"),
            estimated_frames=frames,
            expected_actual_frames=None,
            occurrences=(),
        )
        for index, frames in enumerate((60, 60, 60, 100, 100))
    ]
    batches = MODULE.length_aware_batches(
        samples,
        max_frames=220,
        batch_size=16,
    )
    assert [[sample.estimated_frames for sample in batch] for batch in batches] == [
        [60, 60, 60],
        [100, 100],
    ]
    assert all(
        len(batch) * max(sample.estimated_frames for sample in batch) <= 220
        for batch in batches
    )


def test_feature_view_cosine_is_not_a_decode_metric():
    cosines, minimum = MODULE.pairwise_cosines(
        (
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
        )
    )
    np.testing.assert_allclose(cosines, [1.0, 0.0, 0.0])
    assert minimum == 0.0


def test_deterministic_npz_is_pickle_free_and_byte_stable(tmp_path):
    arrays = {
        "schema_version": MODULE._scalar_text(MODULE.FEATURE_SCHEMA),
        **MODULE._empty_row_payload(3),
    }
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    MODULE.write_deterministic_npz(left, arrays)
    MODULE.write_deterministic_npz(right, arrays)
    assert left.read_bytes() == right.read_bytes()
    with np.load(left, allow_pickle=False) as payload:
        assert payload["vectors"].shape == (0, 3)
        assert not any(payload[name].dtype.hasobject for name in payload.files)


def _minimal_shard(
    path: Path,
    *,
    shard_index: int,
    sample_id: str,
    occurrence_id: str,
) -> None:
    cache_sha256 = str(shard_index) * 64
    rows = MODULE.records_to_arrays(
        [
            {
                "vector": np.asarray([1.0, 2.0, 3.0]),
                "edge_id": "edge-1",
                "endpoint": "target",
                "sample_id": sample_id,
                "source_id": f"source-{shard_index}",
                "ctc_log_likelihood": -1.0,
                "view_cosine_min": 0.99,
                "word_index": 0,
                "target_start": 0,
                "target_end": 1,
                "input_frames": 4,
                "target_length": 1,
                "ctc_feasible": True,
                "view_ctc_feasible": np.ones(3, dtype=np.bool_),
                "ctc_log_likelihood_views": np.full(3, -1.0, dtype=np.float32),
                "posterior_mass_views": np.ones(3, dtype=np.float32),
                "view_cosines": np.ones(3, dtype=np.float32),
                "cache_sha256": cache_sha256,
                "occurrence_id": occurrence_id,
            }
        ]
    )
    cache_inventory_sha256 = MODULE.cache_inventory_pairs(
        [(sample_id, cache_sha256)]
    )
    selected_sample_ids_sha256 = MODULE.sample_id_inventory_hash([sample_id])
    shard_contract = {
        "extraction_fingerprint": "common",
        "cache_inventory_sha256": cache_inventory_sha256,
        "shard": {"index": shard_index, "count": 2},
        "selected_sample_ids_sha256": selected_sample_ids_sha256,
    }
    metadata = {
        "schema_version": MODULE.FEATURE_SCHEMA,
        **shard_contract,
        "shard_fingerprint": MODULE._fingerprint(shard_contract),
        "selected_samples": 1,
        "selected_occurrences": 1,
        "inputs": {"manifest": {"sha256": "a" * 64}},
    }
    MODULE.write_deterministic_npz(
        path, MODULE.archive_payload(rows, metadata=metadata)
    )


def test_merge_requires_complete_shards_and_preserves_rows(tmp_path):
    shard0, shard1 = tmp_path / "s0.npz", tmp_path / "s1.npz"
    _minimal_shard(
        shard0, shard_index=0, sample_id="sample-a", occurrence_id="o-a"
    )
    _minimal_shard(
        shard1, shard_index=1, sample_id="sample-b", occurrence_id="o-b"
    )
    output = tmp_path / "merged.npz"
    summary = MODULE.merge_shards([shard1, shard0], output)
    assert summary["output_audit"]["rows"] == 2
    with np.load(output, allow_pickle=False) as payload:
        assert payload["sample_ids"].astype(str).tolist() == [
            "sample-a",
            "sample-b",
        ]
        assert int(payload["shard_index"]) == -1
    loaded = ESTIMATOR.load_features(output)
    assert loaded["vectors"].shape == (2, 3)
    with pytest.raises(ValueError, match="cover exactly"):
        MODULE.merge_shards([shard0], tmp_path / "bad.npz")


def test_tone_only_route_cannot_be_overridden_by_high_visual_auc():
    assert (
        ESTIMATOR.choose_identifiability_route(
            tone_only=True,
            auc=0.99,
            ci95=(0.90, 1.0),
            null_cutoff=0.51,
            encoder_ci_lower=0.60,
            decoder_ci_upper=0.55,
        )
        == "D"
    )
    assert (
        ESTIMATOR.choose_identifiability_route(
            tone_only=False,
            auc=0.99,
            ci95=(0.90, 1.0),
            null_cutoff=0.51,
            encoder_ci_lower=0.60,
            decoder_ci_upper=0.55,
        )
        == "E"
    )
