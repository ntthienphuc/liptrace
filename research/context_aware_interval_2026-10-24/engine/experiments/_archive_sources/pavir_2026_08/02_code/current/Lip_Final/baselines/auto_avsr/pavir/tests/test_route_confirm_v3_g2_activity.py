from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pavir.artifacts import CanonicalIDMap
from pavir.route_confirm_v3_g2_activity import (
    G2_MAX_PADDED_FRAMES,
    G2_SELECTED_BATCHES,
    SCHEMA_G2_ACTIVITY_AUDIT,
    audit_g2_activity,
    build_g2_physical_batches,
    select_g2_physical_batches,
)


def _write_inputs(
    root: Path,
    lengths: list[int],
    *,
    source_modulus: int = 4,
) -> tuple[Path, Path]:
    labels = root / "train.csv"
    canonical = root / "canonical.jsonl"
    label_lines = []
    canonical_lines = [
        json.dumps({"schema_version": "canonical-id-map-v1"})
    ]
    for index, length in enumerate(lengths):
        member = f"source-{index % source_modulus}/clip-{index:04d}.mp4"
        loader_path = f"/storage/train.tar#{member}"
        sample_id = f"vicocktail:train:{member[:-4]}"
        label_lines.append(
            f"vicocktail,{loader_path},{length},1 2,"
            f"/cache/clip-{index:04d}.npy"
        )
        canonical_lines.append(
            json.dumps(
                {
                    "split": "train",
                    "lookup_key": member[:-4],
                    "sample_id": sample_id,
                    "source_id": f"source-{index % source_modulus}",
                }
            )
        )
    labels.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    canonical.write_text(
        "\n".join(canonical_lines) + "\n", encoding="utf-8"
    )
    return labels, canonical


def _singleton_schedule(tmp_path: Path):
    labels, canonical = _write_inputs(
        tmp_path, [G2_MAX_PADDED_FRAMES] * 70
    )
    physical = build_g2_physical_batches(labels, canonical)
    assert len(physical) == 70
    assert all(len(batch.sample_ids) == 1 for batch in physical)
    return physical


def test_builds_exact_custom_bucket_batches_and_canonical_order(
    tmp_path: Path,
) -> None:
    labels, canonical_path = _write_inputs(
        tmp_path, [100, 60, 90, 70, 80]
    )
    canonical = CanonicalIDMap.load(canonical_path)
    physical = build_g2_physical_batches(labels, canonical)

    # With 400 buckets these distinct lengths occupy increasing buckets, so
    # CustomBucketDataset visits them shortest-to-longest.
    assert [batch.input_lengths for batch in physical] == [
        (60, 70, 80, 90, 100)
    ]
    assert physical[0].row_indices == (1, 3, 4, 2, 0)
    assert physical[0].padded_frames == 500
    assert physical[0].sample_ids[0].endswith("clip-0001")


def test_selection_is_the_explicit_seed42_randperm(tmp_path: Path) -> None:
    physical = _singleton_schedule(tmp_path)
    selected = select_g2_physical_batches(physical)
    expected = torch.randperm(
        len(physical), generator=torch.Generator().manual_seed(42)
    )[:G2_SELECTED_BATCHES].tolist()
    assert [batch.physical_batch_index for batch in selected] == expected
    assert len(set(expected)) == G2_SELECTED_BATCHES


def test_activity_audit_passes_exact_E_and_D_cumulative_gates(
    tmp_path: Path,
) -> None:
    physical = _singleton_schedule(tmp_path)
    selected = select_g2_physical_batches(physical)
    occurrences = {}
    # D activity is deliberately interleaved with D-ineligible selected
    # batches: the quota ordinal must advance only on eligible batches.
    for ordinal, batch in enumerate(selected[:32]):
        events = []
        if ordinal < 8:
            events.append(
                {"edge_id": "edge-E", "target_start": 0, "target_end": 1}
            )
        if ordinal % 2 == 0:
            events.append(
                {"edge_id": "edge-D", "target_start": 1, "target_end": 2}
            )
        if events:
            occurrences[batch.sample_ids[0]] = events

    audit = audit_g2_activity(
        physical,
        occurrences=occurrences,
        edge_routes={"edge-E": "E", "edge-D": "D", "edge-A": "A"},
        forbidden_ids=set(),
    )

    assert audit["schema_version"] == SCHEMA_G2_ACTIVITY_AUDIT
    assert audit["status"] == "PASS"
    assert audit["activity_gate"]["E_eligible_batch_count"] == 8
    assert audit["activity_gate"]["D_eligible_batch_count"] == 16
    assert audit["activity_gate"]["D_quota_selected_batch_count"] == 8
    d_records = [
        row for row in audit["selected_batches"] if row["D_eligible"]
    ]
    assert [
        row["D_eligible_ordinal_zero_based"] for row in d_records
    ] == list(range(16))
    assert [
        row["D_eligible_ordinal_zero_based"]
        for row in d_records
        if row["D_quota_selected"]
    ] == list(range(1, 16, 2))
    assert [
        row["selection_ordinal_zero_based"]
        for row in d_records
        if row["D_quota_selected"]
    ] == list(range(2, 32, 4))
    expected_e_sources = len(
        {batch.source_ids[0] for batch in selected[:8]}
    )
    assert audit["mass"]["full_train_occurrences"]["E"] == {
        "event_count": 8,
        "sample_count": 8,
        "source_count": expected_e_sources,
        "batch_count": 8,
    }
    assert audit["mass"]["selected_schedule"]["D"]["event_count"] == 16
    assert audit["mass"]["applied_schedule"]["D"]["event_count"] == 8


def test_activity_audit_blocks_leakage_integrity_and_low_mass(
    tmp_path: Path,
) -> None:
    physical = _singleton_schedule(tmp_path)
    selected = select_g2_physical_batches(physical)
    forbidden_sample = selected[0].sample_ids[0]
    occurrences = {
        forbidden_sample: [
            {"edge_id": "edge-A", "target_start": 0, "target_end": 1}
        ],
        "vicocktail:train:not-in-label": [
            {"edge_id": "unknown", "target_start": 0, "target_end": 1}
        ],
    }
    audit = audit_g2_activity(
        physical,
        occurrences=occurrences,
        edge_routes={"edge-A": "A"},
        forbidden_ids={forbidden_sample},
    )

    assert audit["status"] == "BLOCKED"
    assert set(audit["blocking_reasons"]) == {
        "insufficient_E_eligible_batches",
        "insufficient_D_quota_selected_batches",
        "forbidden_IDs_in_train_schedule",
        "forbidden_IDs_in_occurrences",
        "occurrence_samples_outside_train_schedule",
        "unknown_occurrence_edges",
        "inactive_route_A_occurrences",
    }
    assert audit["leakage"]["forbidden_train_sample_count"] == 1
    assert audit["integrity"]["unknown_occurrence_sample_count"] == 1
    assert audit["integrity"]["unknown_occurrence_edge_ids"] == ["unknown"]
    assert audit["integrity"]["inactive_route_A_edge_ids"] == ["edge-A"]


def test_compact_occurrence_counts_are_exactly_equivalent(
    tmp_path: Path,
) -> None:
    physical = _singleton_schedule(tmp_path)
    occurrences = {}
    compact = {}
    for batch in select_g2_physical_batches(physical):
        sample_id = batch.sample_ids[0]
        occurrences[sample_id] = [
            {"edge_id": "edge-E", "target_start": 0, "target_end": 1},
            {"edge_id": "edge-E", "target_start": 2, "target_end": 3},
            {"edge_id": "edge-D", "target_start": 4, "target_end": 5},
        ]
        compact[sample_id] = {"edge-E": 2, "edge-D": 1}
    arguments = {
        "edge_routes": {"edge-E": "E", "edge-D": "D"},
        "forbidden_ids": set(),
    }
    expanded_audit = audit_g2_activity(
        physical,
        occurrences=occurrences,
        **arguments,
    )
    compact_audit = audit_g2_activity(
        physical,
        occurrence_edge_counts=compact,
        **arguments,
    )

    assert compact_audit == expanded_audit
    assert compact_audit["mass"]["full_train_occurrences"]["E"][
        "event_count"
    ] == 128
    assert compact_audit["mass"]["full_train_occurrences"]["D"][
        "event_count"
    ] == 64


def test_label_parser_fails_closed_on_comma_ambiguity(tmp_path: Path) -> None:
    labels, canonical = _write_inputs(tmp_path, [100])
    labels.write_text(
        'vicocktail,/storage/train.tar#source-0/clip-0000.mp4,100,'
        '"1,2",/cache/clip.npy,unexpected\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="has 7 fields"):
        build_g2_physical_batches(labels, canonical)


def test_label_and_canonical_train_split_must_be_one_to_one(
    tmp_path: Path,
) -> None:
    labels, canonical = _write_inputs(tmp_path, [100, 100])
    labels.write_text(
        labels.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not one-to-one"):
        build_g2_physical_batches(labels, canonical)
