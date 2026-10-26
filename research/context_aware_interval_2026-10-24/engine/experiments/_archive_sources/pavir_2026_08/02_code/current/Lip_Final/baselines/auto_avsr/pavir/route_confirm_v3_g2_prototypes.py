"""Bounded G2 prototype extraction for PAViR-RouteConfirm-v3.

This module is deliberately narrower than an offline-artifact publisher.  It
accepts an already authenticated :class:`~pavir.route_confirm_v3_runner.BundleContext`,
recomputes the locked support audit for active E/D route rows, and builds the
precommitted route-fit-only source-balanced prototypes.  It neither writes
files nor authorizes calibration or training.

The parent process owns the full bundle.  Up to four ``fork`` workers inherit
the read-only arrays through copy-on-write and return only two small vectors
and a compact audit per edge.  Results are sorted and hashed in the parent, so
worker completion order cannot change the returned artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import multiprocessing
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .artifacts import canonical_json_bytes
from .route_confirm_v3 import (
    PROTOCOL,
    PartitionSupport,
    SourceSplit,
    SupportThresholds,
    audit_edge_support,
)
from .route_confirm_v3_io import build_source_balanced_prototypes
from .route_confirm_v3_runner import BundleContext


SCHEMA = "pavir-route-confirm-v3-g2-prototypes-v1"
STATUS = "COMPLETE_BOUNDED_G2_PROTOTYPES_NO_AUTHORIZATION"
ACTIVE_ROUTES = frozenset({"E", "D"})
VALID_ROUTES = frozenset({"E", "D", "A"})
MAXIMUM_WORKERS = 4
ENDPOINTS = ("confusion", "target")
# ``PartitionSupport`` tuples use the statistical core's class-label order:
# target=0, confusion=1.  Bundle membership codes use the opposite semantic
# order, so this must remain a separate constant.
SUPPORT_ENDPOINTS = ("target", "confusion")
_SHA256_HEX = frozenset("0123456789abcdef")
_FORK_STATE: tuple[BundleContext, Mapping[str, str]] | None = None
_WORKER_THREADPOOL_LIMITER: Any | None = None


@dataclass(frozen=True)
class G2PrototypeBatch:
    """In-memory, deterministic prototype artifact and compact edge audit."""

    keys: tuple[str, ...]
    vectors: np.ndarray
    per_edge_audit: tuple[Mapping[str, Any], ...]
    quality_mask_sha256: str
    keys_sha256: str
    vectors_sha256: str
    audit_sha256: str
    numerical_thread_audit: tuple[Mapping[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        """Return the compact JSON audit record for a future publisher."""

        return {
            "schema_version": SCHEMA,
            "protocol_name": PROTOCOL,
            "status": STATUS,
            "contract": {
                "active_routes": sorted(ACTIVE_ROUTES),
                "population": "quality-accepted paired route_fit memberships only",
                "arithmetic": (
                    "L2 occurrence; mean within source/endpoint; L2; "
                    "equal-source mean within edge/endpoint; L2"
                ),
                "storage_dtype": "float16",
                "maximum_fork_cow_workers": MAXIMUM_WORKERS,
            },
            "inventory": {
                "active_edges": len(self.per_edge_audit),
                "prototype_keys": len(self.keys),
                "vector_dimension": int(self.vectors.shape[1]),
                "keys_sha256": self.keys_sha256,
                "vectors_float16_sha256": self.vectors_sha256,
                "quality_mask_sha256": self.quality_mask_sha256,
            },
            "resource_topology": {
                "numerical_threads_per_worker": 1,
                "effective_numeric_libraries": [
                    dict(row) for row in self.numerical_thread_audit
                ],
            },
            "per_edge": [dict(row) for row in self.per_edge_audit],
            "audit_sha256": self.audit_sha256,
            "scope": {
                "bounded_prototype_primitive_only": True,
                "offline_artifacts_authorized": False,
                "calibration_authorized": False,
                "training_authorized": False,
            },
        }


@dataclass(frozen=True)
class _EdgeExtraction:
    edge_id: str
    route: str
    vectors: Mapping[str, np.ndarray]
    audit: Mapping[str, Any]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _float16_bytes(vectors: np.ndarray) -> bytes:
    values = np.ascontiguousarray(np.asarray(vectors).astype("<f2", copy=False))
    return values.tobytes()


def _membership_stream_sha256(
    positions: np.ndarray,
    endpoint_codes: np.ndarray,
) -> str:
    position_values = np.asarray(positions)
    endpoint_values = np.asarray(endpoint_codes)
    if (
        position_values.ndim != 1
        or endpoint_values.ndim != 1
        or len(position_values) != len(endpoint_values)
        or position_values.dtype.kind not in "iu"
        or endpoint_values.dtype.kind not in "iu"
        or np.any(position_values < 0)
        or np.any(position_values > np.iinfo(np.uint32).max)
        or np.any((endpoint_values != 0) & (endpoint_values != 1))
    ):
        raise ValueError("prototype membership stream is invalid")
    stream = np.empty(
        len(position_values),
        dtype=np.dtype([("position", "<u4"), ("endpoint", "u1")]),
    )
    stream["position"] = position_values
    stream["endpoint"] = endpoint_values
    return _sha256_bytes(stream.tobytes())


def _strict_route_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("route rows must be a sequence of mappings")
    routes: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"route row {index} is not a mapping")
        edge_id = row.get("edge_id")
        route = row.get("route")
        if not isinstance(edge_id, str) or not edge_id or edge_id in routes:
            raise ValueError("route rows contain an empty or duplicate edge_id")
        if not isinstance(route, str) or route not in VALID_ROUTES:
            raise ValueError(f"route row has an invalid route: {edge_id}")
        routes[edge_id] = route
    if not any(route in ACTIVE_ROUTES for route in routes.values()):
        raise ValueError("route rows contain no active E/D edge")
    return routes


def _validate_context(
    context: BundleContext,
    routes: Mapping[str, str],
) -> None:
    if not isinstance(context, BundleContext):
        raise TypeError("an authenticated BundleContext is required")
    edge_dictionary = np.asarray(context.edge_dictionary)
    membership_positions = np.asarray(context.membership_positions)
    membership_edges = np.asarray(context.membership_edges)
    membership_endpoints = np.asarray(context.membership_endpoints)
    sources = np.asarray(context.position_sources)
    vectors = np.asarray(context.position_vectors)
    accepted = np.asarray(context.quality_accepted)
    starts = np.asarray(context.starts)
    ends = np.asarray(context.ends)
    edge_ids = edge_dictionary.astype(str).tolist()
    if (
        edge_dictionary.ndim != 1
        or not edge_ids
        or edge_ids != sorted(edge_ids)
        or len(edge_ids) != len(set(edge_ids))
        or starts.shape != edge_dictionary.shape
        or ends.shape != edge_dictionary.shape
        or any(
            values.dtype.kind not in "iu"
            for values in (
                membership_positions,
                membership_edges,
                membership_endpoints,
                starts,
                ends,
            )
        )
        or any(
            values.ndim != 1
            for values in (
                membership_positions,
                membership_edges,
                membership_endpoints,
                sources,
                accepted,
            )
        )
        or not (
            len(membership_positions)
            == len(membership_edges)
            == len(membership_endpoints)
        )
        or vectors.ndim != 2
        or not len(vectors)
        or vectors.shape[1] <= 0
        or len(vectors) != len(sources)
        or len(accepted) != len(sources)
        or accepted.dtype.kind != "b"
        or not isinstance(context.split, SourceSplit)
    ):
        raise ValueError("BundleContext arrays are not aligned")
    missing = sorted(set(routes) - set(edge_ids))
    if missing:
        raise ValueError(f"route edge is absent from BundleContext: {missing[0]}")
    actual_quality_sha256 = _sha256_bytes(
        np.packbits(accepted, bitorder="little").tobytes()
    )
    if (
        not _is_sha256(context.quality_mask_sha256)
        or actual_quality_sha256 != context.quality_mask_sha256
    ):
        raise ValueError("BundleContext quality mask hash mismatch")


def _partition_audit(partition: PartitionSupport) -> dict[str, Any]:
    raw = {
        SUPPORT_ENDPOINTS[label]: int(
            partition.raw_endpoint_occurrences[label]
        )
        for label in range(2)
    }
    included = {
        SUPPORT_ENDPOINTS[label]: int(
            partition.included_endpoint_occurrences[label]
        )
        for label in range(2)
    }
    concentration = {
        SUPPORT_ENDPOINTS[label]: float(
            partition.maximum_source_endpoint_fractions[label]
        )
        for label in range(2)
    }
    return {
        "passed": bool(partition.passed),
        "failures": list(partition.failures),
        "observed_source_count": int(partition.observed_source_count),
        "paired_source_count": len(partition.paired_sources),
        "paired_source_ids_sha256": _sha256_lines(partition.paired_sources),
        "one_class_source_count": len(partition.one_class_sources),
        "one_class_source_ids_sha256": _sha256_lines(
            partition.one_class_sources
        ),
        "raw_endpoint_occurrences": raw,
        "included_endpoint_occurrences": included,
        "maximum_source_endpoint_fractions": concentration,
    }


def _edge_slice(
    context: BundleContext,
    edge_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    edge_index = int(np.searchsorted(context.edge_dictionary, edge_id))
    if (
        edge_index >= len(context.edge_dictionary)
        or str(context.edge_dictionary[edge_index]) != edge_id
    ):
        raise ValueError(f"active edge is absent from BundleContext: {edge_id}")
    start = int(context.starts[edge_index])
    end = int(context.ends[edge_index])
    expected_start = int(
        np.searchsorted(context.membership_edges, edge_index, side="left")
    )
    expected_end = int(
        np.searchsorted(context.membership_edges, edge_index, side="right")
    )
    if (
        start != expected_start
        or end != expected_end
        or start < 0
        or end <= start
        or end > len(context.membership_positions)
    ):
        raise ValueError(f"BundleContext membership bounds drifted: {edge_id}")
    positions = np.asarray(context.membership_positions[start:end])
    endpoint_codes = np.asarray(context.membership_endpoints[start:end])
    if (
        positions.dtype.kind not in "iu"
        or endpoint_codes.dtype.kind not in "iu"
        or np.any(context.membership_edges[start:end] != edge_index)
        or np.any(positions < 0)
        or np.any(positions >= len(context.position_sources))
        or np.any((endpoint_codes != 0) & (endpoint_codes != 1))
    ):
        raise ValueError(f"BundleContext edge membership is invalid: {edge_id}")
    accepted = np.asarray(context.quality_accepted[positions], dtype=np.bool_)
    positions = positions[accepted]
    endpoint_codes = endpoint_codes[accepted]
    if not len(positions):
        raise ValueError(f"active edge has no quality-accepted membership: {edge_id}")
    return positions, endpoint_codes


def _extract_edge(
    context: BundleContext,
    edge_id: str,
    route: str,
) -> _EdgeExtraction:
    positions, endpoint_codes = _edge_slice(context, edge_id)
    endpoints = np.where(endpoint_codes == 0, "confusion", "target")
    vectors = np.asarray(context.position_vectors[positions], dtype=np.float32)
    sources = np.asarray(context.position_sources[positions]).astype(str)
    support = audit_edge_support(
        vectors,
        endpoints,
        sources,
        context.split,
        thresholds=SupportThresholds(),
    )
    if not support.passed:
        first = support.failures[0] if support.failures else "unknown"
        raise ValueError(f"locked support failed for {edge_id}: {first}")
    fit_indices = np.asarray(support.fit.included_indices, dtype=np.int64)
    if not len(fit_indices):
        raise ValueError(f"active edge has no paired route_fit membership: {edge_id}")
    prototype = build_source_balanced_prototypes(
        vectors[fit_indices],
        endpoints[fit_indices],
        sources[fit_indices],
    )
    fit_positions = positions[fit_indices]
    fit_endpoint_codes = endpoint_codes[fit_indices]
    endpoint_records: dict[str, dict[str, Any]] = {}
    stored_vectors: dict[str, np.ndarray] = {}
    for endpoint in ENDPOINTS:
        fp32 = np.asarray(prototype.vectors[endpoint], dtype=np.float32)
        stored = np.ascontiguousarray(fp32.astype("<f2"))
        stored.setflags(write=False)
        stored_vectors[endpoint] = stored
        endpoint_records[endpoint] = {
            "occurrence_count": int(
                prototype.occurrence_count_by_endpoint[endpoint]
            ),
            "source_count": int(prototype.source_count_by_endpoint[endpoint]),
            "fp32_prequantization_sha256": (
                prototype.fp32_prequantization_sha256[endpoint]
            ),
            "float16_sha256": _sha256_bytes(_float16_bytes(stored)),
        }
    audit = {
        "edge_id": edge_id,
        "route": route,
        "accepted_memberships": len(positions),
        "accepted_membership_stream_sha256": _membership_stream_sha256(
            positions,
            endpoint_codes,
        ),
        "fit_prototype_memberships": len(fit_indices),
        "fit_prototype_membership_stream_sha256": _membership_stream_sha256(
            fit_positions,
            fit_endpoint_codes,
        ),
        "quality_mask_sha256": context.quality_mask_sha256,
        "support": {
            "passed": True,
            "failures": [],
            "fit": _partition_audit(support.fit),
            "confirm": _partition_audit(support.confirm),
        },
        "prototypes": endpoint_records,
    }
    return _EdgeExtraction(
        edge_id=edge_id,
        route=route,
        vectors=stored_vectors,
        audit=audit,
    )


def _fork_extract_edge(edge_id: str) -> _EdgeExtraction:
    if _FORK_STATE is None:
        raise RuntimeError("G2 fork worker state was not initialized")
    context, routes = _FORK_STATE
    return _extract_edge(context, edge_id, routes[edge_id])


def _numeric_thread_audit() -> tuple[Mapping[str, Any], ...]:
    records = tuple(
        sorted(
            (
                {
                    "user_api": str(record.get("user_api", "")),
                    "internal_api": str(record.get("internal_api", "")),
                    "prefix": str(record.get("prefix", "")),
                    "threading_layer": str(record.get("threading_layer", "")),
                    "num_threads": int(record.get("num_threads", -1)),
                }
                for record in threadpool_info()
                if record.get("user_api") in {"blas", "openmp"}
            ),
            key=lambda row: (
                row["user_api"],
                row["internal_api"],
                row["prefix"],
                row["threading_layer"],
            ),
        )
    )
    if not records:
        raise RuntimeError("no numeric threadpool was discoverable")
    if any(record["num_threads"] != 1 for record in records):
        raise RuntimeError("G2 numeric threadpool limit is not one")
    return records


def _initialize_fork_worker_threads() -> None:
    global _WORKER_THREADPOOL_LIMITER
    _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    _numeric_thread_audit()


def _audit_payload(batch: G2PrototypeBatch) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "quality_mask_sha256": batch.quality_mask_sha256,
        "keys_sha256": batch.keys_sha256,
        "vectors_float16_sha256": batch.vectors_sha256,
        "numerical_thread_audit": [
            dict(row) for row in batch.numerical_thread_audit
        ],
        "per_edge": [dict(row) for row in batch.per_edge_audit],
    }


def _validate_partition_record(
    value: object,
    *,
    name: str,
    minimum_occurrences: int,
    minimum_sources: int,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} support audit is missing")
    required = {
        "passed",
        "failures",
        "observed_source_count",
        "paired_source_count",
        "paired_source_ids_sha256",
        "one_class_source_count",
        "one_class_source_ids_sha256",
        "raw_endpoint_occurrences",
        "included_endpoint_occurrences",
        "maximum_source_endpoint_fractions",
    }
    if set(value) != required:
        raise ValueError(f"{name} support audit fields drifted")
    raw = value["raw_endpoint_occurrences"]
    included = value["included_endpoint_occurrences"]
    concentration = value["maximum_source_endpoint_fractions"]
    if not all(
        isinstance(record, Mapping) and set(record) == set(ENDPOINTS)
        for record in (raw, included, concentration)
    ):
        raise ValueError(f"{name} endpoint support audit is invalid")

    def strict_nonnegative_int(item: object) -> bool:
        return (
            isinstance(item, int)
            and not isinstance(item, bool)
            and item >= 0
        )

    if (
        value["passed"] is not True
        or value["failures"] != []
        or not strict_nonnegative_int(value["observed_source_count"])
        or not strict_nonnegative_int(value["paired_source_count"])
        or int(value["paired_source_count"]) < minimum_sources
        or not strict_nonnegative_int(value["one_class_source_count"])
        or not _is_sha256(value["paired_source_ids_sha256"])
        or not _is_sha256(value["one_class_source_ids_sha256"])
    ):
        raise ValueError(f"{name} support audit did not pass")
    if int(value["observed_source_count"]) != (
        int(value["paired_source_count"])
        + int(value["one_class_source_count"])
    ):
        raise ValueError(f"{name} source accounting drifted")
    for endpoint in ENDPOINTS:
        if (
            not strict_nonnegative_int(raw[endpoint])
            or not strict_nonnegative_int(included[endpoint])
            or int(included[endpoint]) < minimum_occurrences
            or int(included[endpoint]) < int(value["paired_source_count"])
            or int(raw[endpoint]) < int(included[endpoint])
        ):
            raise ValueError(f"{name} {endpoint} occurrence support is invalid")
        observed_fraction = concentration[endpoint]
        if (
            isinstance(observed_fraction, bool)
            or not isinstance(observed_fraction, (int, float))
            or not np.isfinite(float(observed_fraction))
            or not 0 <= float(observed_fraction)
            <= SupportThresholds().maximum_source_endpoint_fraction
        ):
            raise ValueError(f"{name} {endpoint} concentration is invalid")


def validate_g2_prototype_batch(
    batch: G2PrototypeBatch,
    route_rows: Sequence[Mapping[str, Any]],
    *,
    expected_quality_mask_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed on structural, arithmetic-audit, or hash drift."""

    if not isinstance(batch, G2PrototypeBatch):
        raise TypeError("G2PrototypeBatch is required")
    routes = _strict_route_rows(route_rows)
    active = {
        edge_id: route
        for edge_id, route in routes.items()
        if route in ACTIVE_ROUTES
    }
    expected_keys = tuple(
        sorted(
            f"{edge_id}:{endpoint}"
            for edge_id in active
            for endpoint in ENDPOINTS
        )
    )
    vectors = np.asarray(batch.vectors)
    if (
        not isinstance(batch.keys, tuple)
        or batch.keys != expected_keys
        or len(set(batch.keys)) != len(batch.keys)
        or vectors.dtype != np.dtype("<f2")
        or vectors.ndim != 2
        or vectors.shape[0] != len(expected_keys)
        or vectors.shape[1] <= 0
        or not vectors.flags.c_contiguous
        or vectors.flags.writeable
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("G2 prototype keys/vectors violate the storage contract")
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    if np.any(np.abs(norms - 1.0) > 5e-3):
        raise ValueError("G2 float16 prototype norm drifted")
    if (
        not _is_sha256(batch.quality_mask_sha256)
        or (
            expected_quality_mask_sha256 is not None
            and batch.quality_mask_sha256 != expected_quality_mask_sha256
        )
        or batch.keys_sha256 != _sha256_lines(batch.keys)
        or batch.vectors_sha256 != _sha256_bytes(_float16_bytes(vectors))
    ):
        raise ValueError("G2 prototype top-level hash mismatch")

    audits = batch.per_edge_audit
    if (
        not isinstance(audits, tuple)
        or [row.get("edge_id") for row in audits if isinstance(row, Mapping)]
        != sorted(active)
        or len(audits) != len(active)
    ):
        raise ValueError("G2 per-edge audit inventory drifted")
    index_by_key = {key: index for index, key in enumerate(batch.keys)}
    thresholds = SupportThresholds()
    for row in audits:
        if not isinstance(row, Mapping):
            raise ValueError("G2 per-edge audit row is invalid")
        required = {
            "edge_id",
            "route",
            "accepted_memberships",
            "accepted_membership_stream_sha256",
            "fit_prototype_memberships",
            "fit_prototype_membership_stream_sha256",
            "quality_mask_sha256",
            "support",
            "prototypes",
        }
        if set(row) != required:
            raise ValueError("G2 per-edge audit fields drifted")
        edge_id = row["edge_id"]
        if (
            not isinstance(edge_id, str)
            or edge_id not in active
            or row["route"] != active[edge_id]
            or row["quality_mask_sha256"] != batch.quality_mask_sha256
            or isinstance(row["accepted_memberships"], bool)
            or not isinstance(row["accepted_memberships"], int)
            or int(row["accepted_memberships"]) <= 0
            or isinstance(row["fit_prototype_memberships"], bool)
            or not isinstance(row["fit_prototype_memberships"], int)
            or int(row["fit_prototype_memberships"]) <= 0
            or int(row["fit_prototype_memberships"])
            > int(row["accepted_memberships"])
            or not _is_sha256(row["accepted_membership_stream_sha256"])
            or not _is_sha256(
                row["fit_prototype_membership_stream_sha256"]
            )
        ):
            raise ValueError(f"G2 edge audit identity/count mismatch: {edge_id}")
        support = row["support"]
        if (
            not isinstance(support, Mapping)
            or set(support) != {"passed", "failures", "fit", "confirm"}
            or support["passed"] is not True
            or support["failures"] != []
        ):
            raise ValueError(f"G2 edge support did not pass: {edge_id}")
        _validate_partition_record(
            support["fit"],
            name=f"{edge_id} fit",
            minimum_occurrences=thresholds.fit_occurrences_per_endpoint,
            minimum_sources=thresholds.fit_paired_sources,
        )
        _validate_partition_record(
            support["confirm"],
            name=f"{edge_id} confirm",
            minimum_occurrences=thresholds.confirm_occurrences_per_endpoint,
            minimum_sources=thresholds.confirm_paired_sources,
        )
        fit = support["fit"]
        confirm = support["confirm"]
        total_raw_memberships = sum(
            int(partition["raw_endpoint_occurrences"][endpoint])
            for partition in (fit, confirm)
            for endpoint in ENDPOINTS
        )
        if int(row["accepted_memberships"]) != total_raw_memberships:
            raise ValueError(f"G2 accepted membership count drifted: {edge_id}")
        prototypes = row["prototypes"]
        if not isinstance(prototypes, Mapping) or set(prototypes) != set(ENDPOINTS):
            raise ValueError(f"G2 prototype endpoint audit drifted: {edge_id}")
        fit_occurrences = fit["included_endpoint_occurrences"]
        expected_fit_memberships = sum(
            int(fit_occurrences[endpoint]) for endpoint in ENDPOINTS
        )
        if int(row["fit_prototype_memberships"]) != expected_fit_memberships:
            raise ValueError(f"G2 fit membership count drifted: {edge_id}")
        for endpoint in ENDPOINTS:
            record = prototypes[endpoint]
            if (
                not isinstance(record, Mapping)
                or set(record)
                != {
                    "occurrence_count",
                    "source_count",
                    "fp32_prequantization_sha256",
                    "float16_sha256",
                }
                or isinstance(record["occurrence_count"], bool)
                or not isinstance(record["occurrence_count"], int)
                or int(record["occurrence_count"])
                != int(fit_occurrences[endpoint])
                or isinstance(record["source_count"], bool)
                or not isinstance(record["source_count"], int)
                or int(record["source_count"])
                != int(fit["paired_source_count"])
                or not _is_sha256(record["fp32_prequantization_sha256"])
                or not _is_sha256(record["float16_sha256"])
            ):
                raise ValueError(
                    f"G2 prototype endpoint record is invalid: {edge_id}:{endpoint}"
                )
            key = f"{edge_id}:{endpoint}"
            stored_sha256 = _sha256_bytes(
                _float16_bytes(vectors[index_by_key[key]])
            )
            if record["float16_sha256"] != stored_sha256:
                raise ValueError(f"G2 float16 prototype hash mismatch: {key}")
    expected_audit_sha256 = _sha256_bytes(
        canonical_json_bytes(_audit_payload(batch))
    )
    if batch.audit_sha256 != expected_audit_sha256:
        raise ValueError("G2 prototype audit fingerprint mismatch")
    if (
        not batch.numerical_thread_audit
        or any(
            set(row)
            != {
                "user_api",
                "internal_api",
                "prefix",
                "threading_layer",
                "num_threads",
            }
            or row["num_threads"] != 1
            for row in batch.numerical_thread_audit
        )
    ):
        raise ValueError("G2 numerical-thread audit is invalid")
    return batch.to_record()


def extract_g2_prototypes(
    context: BundleContext,
    route_rows: Sequence[Mapping[str, Any]],
    *,
    workers: int = 1,
) -> G2PrototypeBatch:
    """Extract all active E/D prototypes with one parent bundle load.

    ``workers`` is intentionally bounded to one through four.  Values above
    four are rejected rather than silently changing the resource topology.
    """

    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= MAXIMUM_WORKERS
    ):
        raise ValueError(f"workers must be an integer in [1, {MAXIMUM_WORKERS}]")
    routes = _strict_route_rows(route_rows)
    _validate_context(context, routes)
    active_ids = sorted(
        edge_id
        for edge_id, route in routes.items()
        if route in ACTIVE_ROUTES
    )
    global _FORK_STATE
    if _FORK_STATE is not None:
        raise RuntimeError("a G2 fork-COW extraction is already active")
    with threadpool_limits(limits=1):
        numerical_thread_audit = _numeric_thread_audit()
        _FORK_STATE = (context, routes)
        try:
            if workers == 1 or len(active_ids) == 1:
                extracted = [
                    _fork_extract_edge(edge_id) for edge_id in active_ids
                ]
            else:
                try:
                    fork = multiprocessing.get_context("fork")
                except ValueError as error:
                    raise RuntimeError(
                        "G2 prototype workers require fork copy-on-write"
                    ) from error
                with fork.Pool(
                    processes=min(workers, len(active_ids)),
                    initializer=_initialize_fork_worker_threads,
                ) as pool:
                    extracted = list(
                        pool.imap_unordered(
                            _fork_extract_edge,
                            active_ids,
                            chunksize=1,
                        )
                    )
        finally:
            _FORK_STATE = None
    extracted.sort(key=lambda value: value.edge_id)
    vector_by_key = {
        f"{edge.edge_id}:{endpoint}": edge.vectors[endpoint]
        for edge in extracted
        for endpoint in ENDPOINTS
    }
    keys = tuple(sorted(vector_by_key))
    vectors = np.ascontiguousarray(
        np.stack([vector_by_key[key] for key in keys], axis=0).astype(
            "<f2",
            copy=False,
        )
    )
    vectors.setflags(write=False)
    per_edge = tuple(dict(edge.audit) for edge in extracted)
    provisional = G2PrototypeBatch(
        keys=keys,
        vectors=vectors,
        per_edge_audit=per_edge,
        quality_mask_sha256=context.quality_mask_sha256,
        keys_sha256=_sha256_lines(keys),
        vectors_sha256=_sha256_bytes(_float16_bytes(vectors)),
        audit_sha256="",
        numerical_thread_audit=numerical_thread_audit,
    )
    batch = replace(
        provisional,
        audit_sha256=_sha256_bytes(
            canonical_json_bytes(_audit_payload(provisional))
        ),
    )
    validate_g2_prototype_batch(
        batch,
        route_rows,
        expected_quality_mask_sha256=context.quality_mask_sha256,
    )
    return batch


__all__ = [
    "ACTIVE_ROUTES",
    "G2PrototypeBatch",
    "MAXIMUM_WORKERS",
    "SCHEMA",
    "STATUS",
    "SUPPORT_ENDPOINTS",
    "extract_g2_prototypes",
    "validate_g2_prototype_batch",
]
