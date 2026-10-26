from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pavir.artifacts import canonical_json_bytes
from pavir.route_confirm_v3 import (
    SourceSplit,
    SupportThresholds,
    evaluate_edge,
)
from pavir.route_confirm_v3_io import (
    build_source_balanced_prototypes,
    compute_quality_filter,
    edge_evidence_from_record,
    evidence_record_sha256,
    load_locked_source_split,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE_SPLIT_LOCK = (
    ROOT / "pavir_campaign" / "route_confirm_v3" / "SOURCE_SPLIT_LOCK.json"
)


def test_real_source_split_lock_revalidates_full_lineage_and_inventory() -> None:
    locked = load_locked_source_split(SOURCE_SPLIT_LOCK)

    assert locked.split.source_count == 2_714
    assert len(locked.split.fit_sources) == 1_899
    assert len(locked.split.confirm_sources) == 815
    assert set(locked.split.fit_sources).isdisjoint(locked.split.confirm_sources)
    assert locked.manifest_row_count == 186_579
    assert locked.manifest_rows_by_partition == {
        "route_fit": 130_423,
        "route_confirm": 56_156,
    }
    assert locked.to_record()["lock"]["sha256"] == (
        "e9457645fb12c3c477652a5533ee2f63635a70874c3a551a6e0c1c9135b15b13"
    )


def test_source_split_lock_rejects_unsafe_scope_before_consumption(
    tmp_path: Path,
) -> None:
    payload = json.loads(SOURCE_SPLIT_LOCK.read_text(encoding="utf-8"))
    payload["scope"]["training_authorized"] = True
    path = tmp_path / "unsafe-lock.json"
    path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="unsafe authorization scope"):
        load_locked_source_split(path)


def test_quality_cutoff_uses_only_unique_finite_fit_positions() -> None:
    split = SourceSplit(
        fit_sources=("fit-a", "fit-b"),
        confirm_sources=("confirm-a", "confirm-b"),
    )
    result = compute_quality_filter(
        [0.0, 10.0, 20.0, 30.0, np.nan],
        [0.95, 0.95, 0.95, 0.89, 0.99],
        ["fit-a", "fit-b", "confirm-a", "confirm-b", "fit-a"],
        split,
    )

    assert result.cutoff == pytest.approx(1.0)
    np.testing.assert_array_equal(
        result.accepted,
        np.asarray([False, True, True, False, False]),
    )
    assert result.fit_position_count == 3
    assert result.finite_fit_position_count == 2
    assert result.accepted_fit_position_count == 1
    assert result.accepted_confirm_position_count == 1
    assert result.to_record()["cutoff_population"] == (
        "unique_route_fit_positions_unit_weight"
    )


def test_quality_cutoff_rejects_unknown_source() -> None:
    split = SourceSplit(
        fit_sources=("fit",),
        confirm_sources=("confirm",),
    )
    with pytest.raises(ValueError, match="unknown source"):
        compute_quality_filter(
            [0.0, 1.0],
            [0.95, 0.95],
            ["fit", "outside"],
            split,
        )


def test_source_balanced_prototype_equalizes_sources_not_occurrences() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    endpoints = np.asarray(
        [
            "target",
            "target",
            "target",
            "target",
            "confusion",
            "confusion",
            "confusion",
        ]
    )
    sources = np.asarray(["s1", "s1", "s1", "s2", "s1", "s1", "s2"])

    result = build_source_balanced_prototypes(vectors, endpoints, sources)

    expected = np.float32(1.0 / np.sqrt(2.0))
    np.testing.assert_allclose(
        result.vectors["target"],
        np.asarray([expected, expected], dtype=np.float32),
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.vectors["confusion"],
        np.asarray([-expected, -expected], dtype=np.float32),
        rtol=0,
        atol=1e-7,
    )
    assert result.occurrence_count_by_endpoint == {
        "target": 4,
        "confusion": 3,
    }
    assert result.source_count_by_endpoint == {
        "target": 2,
        "confusion": 2,
    }
    for endpoint, vector in result.vectors.items():
        expected_hash = hashlib.sha256(
            np.ascontiguousarray(vector.astype("<f4")).tobytes()
        ).hexdigest()
        assert result.fp32_prequantization_sha256[endpoint] == expected_hash


def test_source_balanced_prototype_requires_paired_nonzero_sources() -> None:
    with pytest.raises(ValueError, match="paired endpoint sources"):
        build_source_balanced_prototypes(
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            ["target", "confusion"],
            ["target-only-source", "confusion-only-source"],
        )
    with pytest.raises(ValueError, match="zero L2 norm"):
        build_source_balanced_prototypes(
            np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                dtype=np.float32,
            ),
            ["target", "confusion", "target", "confusion"],
            ["s1", "s1", "s2", "s2"],
        )


def _edge_fixture():
    fit_sources = ("fit-a", "fit-b")
    confirm_sources = ("confirm-a", "confirm-b")
    split = SourceSplit(
        fit_sources=fit_sources,
        confirm_sources=confirm_sources,
    )
    vectors: list[list[float]] = []
    endpoints: list[str] = []
    sources: list[str] = []
    for source in (*fit_sources, *confirm_sources):
        for endpoint, center in (("target", -1.0), ("confusion", 1.0)):
            for jitter in (-0.1, 0.1):
                vectors.append([center + jitter, jitter])
                endpoints.append(endpoint)
                sources.append(source)
    evidence = evaluate_edge(
        np.asarray(vectors, dtype=np.float32),
        endpoints,
        sources,
        split,
        edge_id="edge-roundtrip",
        tone_only=False,
        support_thresholds=SupportThresholds(
            fit_occurrences_per_endpoint=4,
            confirm_occurrences_per_endpoint=4,
            fit_paired_sources=2,
            confirm_paired_sources=2,
            maximum_source_endpoint_fraction=0.5,
        ),
        permutation_iterations=31,
        bootstrap_iterations=64,
    )
    return evidence


def test_edge_evidence_json_roundtrip_preserves_canonical_record() -> None:
    evidence = _edge_fixture()
    serialized = json.loads(
        canonical_json_bytes(evidence.to_record()).decode("utf-8")
    )
    restored = edge_evidence_from_record(serialized)

    assert canonical_json_bytes(restored.to_record()) == canonical_json_bytes(
        evidence.to_record()
    )
    assert evidence_record_sha256(restored) == evidence_record_sha256(evidence)


def test_edge_evidence_rejects_support_consistency_tampering() -> None:
    evidence = _edge_fixture()
    serialized = json.loads(
        canonical_json_bytes(evidence.to_record()).decode("utf-8")
    )
    serialized["support"]["passed"] = False

    with pytest.raises(ValueError, match="internally inconsistent"):
        edge_evidence_from_record(serialized)
