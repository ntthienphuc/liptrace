from __future__ import annotations

import copy
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from pavir.artifacts import canonical_json_bytes
from pavir.route_confirm_v3 import PartitionSupport, SourceSplit
from pavir.route_confirm_v3_g2_prototypes import (
    STATUS,
    _partition_audit,
    extract_g2_prototypes,
    validate_g2_prototype_batch,
)
from pavir.route_confirm_v3_runner import BundleContext


EDGE_IDS = ("edge-a", "edge-b", "edge-c", "edge-d", "edge-e")
ROUTE_ROWS = (
    {"edge_id": "edge-e", "route": "D"},
    {"edge_id": "edge-c", "route": "A"},
    {"edge_id": "edge-a", "route": "E"},
    {"edge_id": "edge-d", "route": "E"},
    {"edge_id": "edge-b", "route": "D"},
)
ACTIVE_EDGE_IDS = ("edge-a", "edge-b", "edge-d", "edge-e")


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _quality_sha256(accepted: np.ndarray) -> str:
    return hashlib.sha256(
        np.packbits(accepted, bitorder="little").tobytes()
    ).hexdigest()


def _bundle_context() -> BundleContext:
    fit_sources = tuple(f"fit-{index:02d}" for index in range(20))
    confirm_sources = tuple(f"confirm-{index:02d}" for index in range(20))
    vectors: list[list[float]] = []
    sources: list[str] = []
    accepted: list[bool] = []
    membership_positions: list[int] = []
    membership_edges: list[int] = []
    membership_endpoints: list[int] = []

    def add(
        *,
        edge_index: int,
        endpoint_code: int,
        source: str,
        vector: list[float],
        quality_accepted: bool = True,
    ) -> None:
        position = len(vectors)
        vectors.append(vector)
        sources.append(source)
        accepted.append(quality_accepted)
        membership_positions.append(position)
        membership_edges.append(edge_index)
        membership_endpoints.append(endpoint_code)

    for edge_index, _ in enumerate(EDGE_IDS):
        # Ten occurrence-rich and ten occurrence-light sources have equal
        # source weight in the prototype despite a 90:10 occurrence split.
        for source_index, source in enumerate(fit_sources):
            repeats = 9 if source_index < 10 else 1
            target = [1.0, 0.0] if source_index < 10 else [0.0, 1.0]
            confusion = (
                [-1.0, 0.0] if source_index < 10 else [0.0, -1.0]
            )
            for _ in range(repeats):
                add(
                    edge_index=edge_index,
                    endpoint_code=1,
                    source=source,
                    vector=target,
                )
                add(
                    edge_index=edge_index,
                    endpoint_code=0,
                    source=source,
                    vector=confusion,
                )
        # Confirmation vectors satisfy support but must never enter a
        # route-fit prototype.
        for source_index, source in enumerate(confirm_sources):
            for _ in range(2):
                add(
                    edge_index=edge_index,
                    endpoint_code=1,
                    source=source,
                    vector=[0.25 + source_index / 100.0, 1.0],
                )
                add(
                    edge_index=edge_index,
                    endpoint_code=0,
                    source=source,
                    vector=[-0.5, -1.0 - source_index / 100.0],
                )
        # Rejected route-fit positions are still graph memberships.  Zero
        # vectors would fail prototype normalization if the quality mask were
        # ignored.
        add(
            edge_index=edge_index,
            endpoint_code=1,
            source=fit_sources[0],
            vector=[0.0, 0.0],
            quality_accepted=False,
        )
        add(
            edge_index=edge_index,
            endpoint_code=0,
            source=fit_sources[0],
            vector=[0.0, 0.0],
            quality_accepted=False,
        )

    edge_dictionary = _readonly(np.asarray(EDGE_IDS, dtype=np.str_))
    membership_edge_array = _readonly(
        np.asarray(membership_edges, dtype=np.uint16)
    )
    starts = _readonly(
        np.searchsorted(
            membership_edge_array,
            np.arange(len(EDGE_IDS)),
            side="left",
        )
    )
    ends = _readonly(
        np.searchsorted(
            membership_edge_array,
            np.arange(len(EDGE_IDS)),
            side="right",
        )
    )
    quality = _readonly(np.asarray(accepted, dtype=np.bool_))
    return BundleContext(
        edge_dictionary=edge_dictionary,
        membership_positions=_readonly(
            np.asarray(membership_positions, dtype=np.uint32)
        ),
        membership_edges=membership_edge_array,
        membership_endpoints=_readonly(
            np.asarray(membership_endpoints, dtype=np.uint8)
        ),
        position_sources=_readonly(np.asarray(sources, dtype=np.str_)),
        position_vectors=_readonly(np.asarray(vectors, dtype=np.float16)),
        quality_accepted=quality,
        quality_mask_sha256=_quality_sha256(quality),
        starts=starts,
        ends=ends,
        split=SourceSplit(
            fit_sources=fit_sources,
            confirm_sources=confirm_sources,
        ),
    )


def _with_vectors(
    context: BundleContext,
    vectors: np.ndarray,
) -> BundleContext:
    return replace(
        context,
        position_vectors=_readonly(
            np.ascontiguousarray(vectors, dtype=np.float16)
        ),
    )


def _with_quality(
    context: BundleContext,
    accepted: np.ndarray,
) -> BundleContext:
    quality = _readonly(np.ascontiguousarray(accepted, dtype=np.bool_))
    return replace(
        context,
        quality_accepted=quality,
        quality_mask_sha256=_quality_sha256(quality),
    )


def test_active_e_and_d_are_included_a_is_excluded_and_four_workers_match() -> None:
    context = _bundle_context()
    serial = extract_g2_prototypes(context, ROUTE_ROWS, workers=1)
    parallel = extract_g2_prototypes(context, ROUTE_ROWS, workers=4)

    expected_keys = tuple(
        sorted(
            f"{edge_id}:{endpoint}"
            for edge_id in ACTIVE_EDGE_IDS
            for endpoint in ("confusion", "target")
        )
    )
    assert serial.keys == parallel.keys == expected_keys
    assert all(not key.startswith("edge-c:") for key in serial.keys)
    assert serial.vectors.dtype == np.dtype("<f2")
    assert serial.vectors.flags.writeable is False
    assert serial.vectors.tobytes() == parallel.vectors.tobytes()
    assert canonical_json_bytes(list(serial.per_edge_audit)) == (
        canonical_json_bytes(list(parallel.per_edge_audit))
    )
    assert serial.keys_sha256 == parallel.keys_sha256
    assert serial.vectors_sha256 == parallel.vectors_sha256
    assert serial.audit_sha256 == parallel.audit_sha256
    assert serial.numerical_thread_audit == parallel.numerical_thread_audit
    assert serial.numerical_thread_audit
    assert all(
        row["num_threads"] == 1
        for row in serial.numerical_thread_audit
    )
    assert [row["route"] for row in serial.per_edge_audit] == [
        "E",
        "D",
        "E",
        "D",
    ]

    record = validate_g2_prototype_batch(
        serial,
        ROUTE_ROWS,
        expected_quality_mask_sha256=context.quality_mask_sha256,
    )
    assert record["status"] == STATUS
    assert record["scope"]["training_authorized"] is False
    assert record["inventory"]["active_edges"] == 4


def test_confirm_and_quality_rejected_vectors_cannot_change_prototypes() -> None:
    context = _bundle_context()
    original = extract_g2_prototypes(context, ROUTE_ROWS)
    changed = np.asarray(context.position_vectors).copy()
    confirm = np.char.startswith(
        np.asarray(context.position_sources).astype(str),
        "confirm-",
    )
    rejected = ~np.asarray(context.quality_accepted)
    changed[confirm] = np.asarray([123.0, -321.0], dtype=np.float16)
    changed[rejected] = np.asarray([-777.0, 555.0], dtype=np.float16)
    perturbed = extract_g2_prototypes(
        _with_vectors(context, changed),
        ROUTE_ROWS,
    )

    assert perturbed.keys == original.keys
    assert perturbed.vectors.tobytes() == original.vectors.tobytes()
    assert canonical_json_bytes(list(perturbed.per_edge_audit)) == (
        canonical_json_bytes(list(original.per_edge_audit))
    )
    assert perturbed.vectors_sha256 == original.vectors_sha256
    assert perturbed.audit_sha256 == original.audit_sha256


def test_equal_source_arithmetic_and_fp32_prequantization_hash() -> None:
    batch = extract_g2_prototypes(_bundle_context(), ROUTE_ROWS)
    target = batch.vectors[batch.keys.index("edge-a:target")].astype(
        np.float32
    )
    confusion = batch.vectors[
        batch.keys.index("edge-a:confusion")
    ].astype(np.float32)
    expected = np.float32(1.0 / np.sqrt(2.0))
    np.testing.assert_allclose(
        target,
        np.asarray([expected, expected], dtype=np.float32),
        rtol=0,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        confusion,
        np.asarray([-expected, -expected], dtype=np.float32),
        rtol=0,
        atol=5e-4,
    )
    occurrence_weighted = np.asarray([0.9, 0.1], dtype=np.float32)
    occurrence_weighted /= np.linalg.norm(occurrence_weighted)
    assert not np.allclose(target, occurrence_weighted, atol=0.05)

    audit = batch.per_edge_audit[0]
    assert audit["support"]["fit"]["included_endpoint_occurrences"] == {
        "confusion": 100,
        "target": 100,
    }
    assert audit["support"]["fit"]["paired_source_count"] == 20
    assert audit["prototypes"]["target"]["occurrence_count"] == 100
    assert audit["prototypes"]["target"]["source_count"] == 20
    expected_fp32 = np.asarray([expected, expected], dtype="<f4")
    expected_hash = hashlib.sha256(
        np.ascontiguousarray(expected_fp32).tobytes()
    ).hexdigest()
    assert (
        audit["prototypes"]["target"]["fp32_prequantization_sha256"]
        == expected_hash
    )


def test_partition_audit_uses_core_target_zero_confusion_one_order() -> None:
    partition = PartitionSupport(
        partition="fit",
        observed_source_count=3,
        paired_sources=("a", "b"),
        one_class_sources=("c",),
        raw_endpoint_occurrences=(101, 202),
        included_endpoint_occurrences=(81, 162),
        maximum_source_endpoint_fractions=(0.11, 0.22),
        included_indices=(0, 1),
        passed=True,
        failures=(),
    )
    observed = _partition_audit(partition)
    core = partition.to_record()

    assert observed["raw_endpoint_occurrences"] == core[
        "raw_endpoint_occurrences"
    ] == {"target": 101, "confusion": 202}
    assert observed["included_endpoint_occurrences"] == core[
        "included_endpoint_occurrences"
    ] == {"target": 81, "confusion": 162}
    assert observed["maximum_source_endpoint_fractions"] == core[
        "maximum_source_endpoint_fractions"
    ] == {"target": 0.11, "confusion": 0.22}


def test_locked_support_is_recomputed_and_required_to_pass() -> None:
    context = _bundle_context()
    quality = np.asarray(context.quality_accepted).copy()
    sources = np.asarray(context.position_sources).astype(str)
    endpoints = np.asarray(context.membership_endpoints)
    edges = np.asarray(context.membership_edges)
    first_fit_target = np.flatnonzero(
        (sources == "fit-00")
        & (endpoints == 1)
        & (edges == 0)
        & quality
    )[0]
    quality[first_fit_target] = False

    with pytest.raises(ValueError, match="locked support failed for edge-a"):
        extract_g2_prototypes(
            _with_quality(context, quality),
            ROUTE_ROWS,
        )


def test_validator_rejects_key_dtype_norm_hash_and_audit_tampering() -> None:
    context = _bundle_context()
    batch = extract_g2_prototypes(context, ROUTE_ROWS)

    with pytest.raises(ValueError, match="keys/vectors"):
        validate_g2_prototype_batch(
            replace(batch, keys=tuple(reversed(batch.keys))),
            ROUTE_ROWS,
        )

    float32 = _readonly(batch.vectors.astype(np.float32))
    with pytest.raises(ValueError, match="keys/vectors"):
        validate_g2_prototype_batch(
            replace(batch, vectors=float32),
            ROUTE_ROWS,
        )

    scaled = batch.vectors.copy()
    scaled[0] *= np.float16(2.0)
    scaled = _readonly(scaled)
    with pytest.raises(ValueError, match="norm drifted"):
        validate_g2_prototype_batch(
            replace(batch, vectors=scaled),
            ROUTE_ROWS,
        )

    sign_flipped = batch.vectors.copy()
    sign_flipped[0] *= np.float16(-1.0)
    sign_flipped = _readonly(sign_flipped)
    with pytest.raises(ValueError, match="top-level hash mismatch"):
        validate_g2_prototype_batch(
            replace(batch, vectors=sign_flipped),
            ROUTE_ROWS,
        )

    audits = copy.deepcopy(list(batch.per_edge_audit))
    audits[0]["prototypes"]["target"]["fp32_prequantization_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audit fingerprint mismatch"):
        validate_g2_prototype_batch(
            replace(batch, per_edge_audit=tuple(audits)),
            ROUTE_ROWS,
        )

    unsafe_threads = tuple(
        {**row, "num_threads": 2}
        for row in batch.numerical_thread_audit
    )
    with pytest.raises(
        ValueError,
        match="numerical-thread audit",
    ):
        tampered = replace(
            batch,
            numerical_thread_audit=unsafe_threads,
            audit_sha256="",
        )
        tampered = replace(
            tampered,
            audit_sha256=hashlib.sha256(
                canonical_json_bytes(
                    {
                        "schema_version": "pavir-route-confirm-v3-g2-prototypes-v1",
                        "quality_mask_sha256": tampered.quality_mask_sha256,
                        "keys_sha256": tampered.keys_sha256,
                        "vectors_float16_sha256": tampered.vectors_sha256,
                        "numerical_thread_audit": [
                            dict(row)
                            for row in tampered.numerical_thread_audit
                        ],
                        "per_edge": [
                            dict(row) for row in tampered.per_edge_audit
                        ],
                    }
                )
            ).hexdigest(),
        )
        validate_g2_prototype_batch(tampered, ROUTE_ROWS)


@pytest.mark.parametrize("workers", [True, 0, 5])
def test_worker_topology_is_bounded(workers: object) -> None:
    with pytest.raises(ValueError, match=r"workers must be an integer in \[1, 4\]"):
        extract_g2_prototypes(  # type: ignore[arg-type]
            _bundle_context(),
            ROUTE_ROWS,
            workers=workers,
        )
