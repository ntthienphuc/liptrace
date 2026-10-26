"""Fail-closed integration helpers for PAViR-RouteConfirm-v3.

The statistical routines in :mod:`pavir.route_confirm_v3` deliberately avoid
filesystem policy.  This module supplies the small, independently testable
integration layer needed by a future sharded runner:

* revalidate the immutable source-only split from its lock;
* compute the route-fit-only quality cutoff on unique positions;
* build source-balanced route-fit prototypes; and
* round-trip edge evidence through JSON without changing route semantics.

It does not authorize routing, offline artifacts, or training by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import canonical_json_bytes, sha256_file
from .route_confirm_v3 import (
    EXPECTED_SOURCE_COUNT,
    PROTOCOL,
    PROTOCOL_PRECOMMIT_SHA256,
    EdgeEvidence,
    EdgeSupport,
    PartitionSupport,
    SourceSplit,
    deterministic_source_partition,
)


SOURCE_SPLIT_LOCK_SCHEMA = "pavir-route-confirm-v3-source-split-lock-v1"
SOURCE_SPLIT_SUMMARY_SCHEMA = "pavir-route-confirm-v3-source-split-summary-v1"
SOURCE_ASSIGNMENT_SCHEMA = "pavir-route-confirm-v3-source-assignment-v1"
CORE_LOCK_SCHEMA = "pavir-route-confirm-v3-core-implementation-lock-v1"
SOURCE_LINEAGE_SCHEMA = "pavir-route-confirm-v3-source-lineage-v1"

QUALITY_PERCENTILE = 10.0
MINIMUM_VIEW_COSINE = 0.90


@dataclass(frozen=True)
class LockedSourceSplit:
    split: SourceSplit
    assignments_path: Path
    summary_path: Path
    lock_path: Path
    lock_sha256: str
    summary_sha256: str
    assignments_sha256: str
    manifest_sha256: str
    forbidden_ids_sha256: str
    manifest_row_count: int
    manifest_rows_by_partition: Mapping[str, int]

    def to_record(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "lock": {
                "path": str(self.lock_path),
                "sha256": self.lock_sha256,
            },
            "summary": {
                "path": str(self.summary_path),
                "sha256": self.summary_sha256,
            },
            "assignments": {
                "path": str(self.assignments_path),
                "sha256": self.assignments_sha256,
                "sources": self.split.source_count,
            },
            "manifest_sha256": self.manifest_sha256,
            "forbidden_ids_sha256": self.forbidden_ids_sha256,
            "manifest_row_count": self.manifest_row_count,
            "manifest_rows_by_partition": dict(self.manifest_rows_by_partition),
        }


@dataclass(frozen=True)
class QualityFilter:
    cutoff: float
    accepted: np.ndarray
    finite: np.ndarray
    fit_position_count: int
    finite_fit_position_count: int
    accepted_fit_position_count: int
    accepted_confirm_position_count: int

    def to_record(self) -> dict[str, Any]:
        return {
            "cutoff_population": "unique_route_fit_positions_unit_weight",
            "ctc_log_likelihood_percentile": QUALITY_PERCENTILE,
            "ctc_log_likelihood_cutoff": self.cutoff,
            "minimum_view_cosine": MINIMUM_VIEW_COSINE,
            "fit_position_count": self.fit_position_count,
            "finite_fit_position_count": self.finite_fit_position_count,
            "accepted_fit_position_count": self.accepted_fit_position_count,
            "accepted_confirm_position_count": self.accepted_confirm_position_count,
            "rejected_nonfinite_position_count": int(np.sum(~self.finite)),
            "accepted_position_count": int(np.sum(self.accepted)),
            "training_authorized": False,
        }


@dataclass(frozen=True)
class PrototypeResult:
    vectors: Mapping[str, np.ndarray]
    fp32_prequantization_sha256: Mapping[str, str]
    occurrence_count_by_endpoint: Mapping[str, int]
    source_count_by_endpoint: Mapping[str, int]

    def to_record(self) -> dict[str, Any]:
        return {
            "method": (
                "L2 occurrence; equal occurrence mean within source/endpoint; "
                "L2; equal source mean within edge/endpoint; L2"
            ),
            "storage_dtype": "float16",
            "fp32_prequantization_sha256": dict(
                self.fp32_prequantization_sha256
            ),
            "occurrence_count_by_endpoint": dict(
                self.occurrence_count_by_endpoint
            ),
            "source_count_by_endpoint": dict(self.source_count_by_endpoint),
        }


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _locked_file(record: object, *, context: str) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise ValueError(f"{context} file record is missing")
    raw_path = record.get("path")
    expected = str(record.get("sha256", "")).lower()
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{context} path is missing")
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(f"{context} SHA-256 is invalid")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{context} hash mismatch")
    return path, expected


def _sha256_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_locked_source_split(lock_path: str | Path) -> LockedSourceSplit:
    """Load and independently revalidate a source-only split lock."""

    resolved_lock = Path(lock_path).expanduser().resolve()
    if not resolved_lock.is_file():
        raise FileNotFoundError(resolved_lock)
    lock_sha256 = sha256_file(resolved_lock)
    lock = _load_object(resolved_lock)
    if (
        lock.get("schema_version") != SOURCE_SPLIT_LOCK_SCHEMA
        or lock.get("protocol_name") != PROTOCOL
        or not str(lock.get("status", "")).startswith("LOCKED_SOURCE_ONLY_")
    ):
        raise ValueError("unsupported or unlocked RouteConfirm-v3 source split")

    lineage = lock.get("lineage")
    implementation = lock.get("implementation")
    outputs = lock.get("outputs")
    scope = lock.get("scope")
    if not all(
        isinstance(value, dict)
        for value in (lineage, implementation, outputs, scope)
    ):
        raise ValueError("source split lock lacks lineage/implementation/output scope")
    if (
        scope.get("complete_route_authorized") is not False
        or scope.get("offline_training_artifacts_authorized") is not False
        or scope.get("training_authorized") is not False
    ):
        raise ValueError("source-only lock has an unsafe authorization scope")

    protocol_path, protocol_sha = _locked_file(
        lineage.get("protocol_precommit"),
        context="RouteConfirm-v3 protocol precommit",
    )
    if protocol_sha != PROTOCOL_PRECOMMIT_SHA256:
        raise ValueError("source split is bound to the wrong RouteConfirm-v3 protocol")
    protocol = _load_object(protocol_path)
    if (
        protocol.get("protocol_name") != PROTOCOL
        or not str(protocol.get("status", "")).startswith("LOCKED_")
    ):
        raise ValueError("source split protocol is not locked")

    core_path, core_sha = _locked_file(
        lineage.get("core_implementation_lock"),
        context="RouteConfirm-v3 core lock",
    )
    core = _load_object(core_path)
    if (
        core.get("schema_version") != CORE_LOCK_SCHEMA
        or core.get("protocol_name") != PROTOCOL
        or core.get("protocol_precommit", {}).get("sha256") != protocol_sha
    ):
        raise ValueError("source split core lock/protocol binding is invalid")

    manifest_path, manifest_sha = _locked_file(
        lineage.get("train_graph_v2_manifest"),
        context="TrainGraph-v2 manifest",
    )
    forbidden_path, forbidden_sha = _locked_file(
        lineage.get("forbidden_ids"),
        context="forbidden-ID registry",
    )
    _locked_file(
        lineage.get("prepare_complete"),
        context="TrainGraph-v2 preparation completion",
    )
    _locked_file(
        implementation.get("tool"),
        context="RouteConfirm-v3 source-split tool",
    )
    _locked_file(
        implementation.get("tests"),
        context="RouteConfirm-v3 source-split tests",
    )
    summary_path, summary_sha = _locked_file(
        outputs.get("summary"),
        context="RouteConfirm-v3 source-split summary",
    )
    assignments_path, assignments_sha = _locked_file(
        outputs.get("assignments"),
        context="RouteConfirm-v3 source assignments",
    )

    summary = _load_object(summary_path)
    summary_scope = summary.get("scope")
    if (
        summary.get("schema_version") != SOURCE_SPLIT_SUMMARY_SCHEMA
        or summary.get("protocol_name") != PROTOCOL
        or summary.get("status") != "LOCKED_SOURCE_ONLY"
        or not isinstance(summary_scope, dict)
        or summary_scope.get("complete_routing_authorized") is not False
        or summary_scope.get("offline_artifacts_authorized") is not False
        or summary_scope.get("training_authorized") is not False
    ):
        raise ValueError("source-split summary has unsafe or unsupported scope")
    summary_lineage = summary.get("lineage")
    if (
        not isinstance(summary_lineage, dict)
        or summary_lineage.get("schema_version") != SOURCE_LINEAGE_SCHEMA
        or summary_lineage.get("protocol_precommit_sha256") != protocol_sha
        or summary_lineage.get("core_lock_sha256") != core_sha
        or summary_lineage.get("manifest_sha256") != manifest_sha
        or summary_lineage.get("forbidden_ids_sha256") != forbidden_sha
    ):
        raise ValueError("source-split summary lineage mismatch")
    summary_assignment = summary.get("outputs", {}).get("assignments")
    if (
        not isinstance(summary_assignment, dict)
        or Path(str(summary_assignment.get("path", ""))).expanduser().resolve()
        != assignments_path
        or summary_assignment.get("sha256") != assignments_sha
        or int(summary_assignment.get("rows", -1)) != EXPECTED_SOURCE_COUNT
    ):
        raise ValueError("source-split summary does not authenticate assignments")

    rows: list[dict[str, Any]] = []
    with assignments_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"empty source-assignment line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"source-assignment line {line_number} is not an object")
            rows.append(row)
    if len(rows) != EXPECTED_SOURCE_COUNT:
        raise ValueError("source-assignment inventory is incomplete")

    sources: list[str] = []
    manifest_rows_by_partition = {"route_fit": 0, "route_confirm": 0}
    expected_lineage = str(summary_lineage.get("lineage_sha256", ""))
    for rank, row in enumerate(rows):
        source = row.get("source_id")
        partition = row.get("partition")
        count = row.get("manifest_row_count")
        if (
            row.get("schema_version") != SOURCE_ASSIGNMENT_SCHEMA
            or row.get("protocol_name") != PROTOCOL
            or row.get("lineage_sha256") != expected_lineage
            or row.get("rank") != rank
            or not isinstance(source, str)
            or not source
            or partition not in manifest_rows_by_partition
            or isinstance(count, bool)
            or int(count) <= 0
        ):
            raise ValueError(f"invalid source assignment at rank {rank}")
        sources.append(source)
        manifest_rows_by_partition[partition] += int(count)

    expected_rows = deterministic_source_partition(sources)
    for observed, expected in zip(rows, expected_rows):
        if (
            observed["source_id"] != expected.source_id
            or observed["source_digest_sha256"] != expected.source_digest_sha256
            or observed["rank"] != expected.rank
            or observed["partition"] != expected.partition
        ):
            raise ValueError("source assignment differs from deterministic protocol")

    partition = summary.get("partition")
    manifest_audit = summary.get("manifest_audit")
    if not isinstance(partition, dict) or not isinstance(manifest_audit, dict):
        raise ValueError("source-split summary lacks partition/manifest audit")
    if (
        int(partition.get("source_count", -1)) != EXPECTED_SOURCE_COUNT
        or int(partition.get("fit_source_count", -1)) != 1_899
        or int(partition.get("confirm_source_count", -1)) != 815
        or partition.get("manifest_rows_by_partition")
        != manifest_rows_by_partition
        or partition.get("protocol_order_source_ids_sha256")
        != _sha256_lines(sources)
        or int(manifest_audit.get("row_count", -1))
        != sum(manifest_rows_by_partition.values())
        or int(manifest_audit.get("source_count", -1)) != EXPECTED_SOURCE_COUNT
        or int(manifest_audit.get("forbidden_intersection_count", -1)) != 0
    ):
        raise ValueError("source-split summary counts/digests are inconsistent")

    lock_assignments = outputs.get("assignments", {})
    if (
        int(lock_assignments.get("rows", -1)) != EXPECTED_SOURCE_COUNT
        or int(lock_assignments.get("fit_sources", -1)) != 1_899
        or int(lock_assignments.get("confirm_sources", -1)) != 815
        or int(lock_assignments.get("fit_manifest_rows", -1))
        != manifest_rows_by_partition["route_fit"]
        or int(lock_assignments.get("confirm_manifest_rows", -1))
        != manifest_rows_by_partition["route_confirm"]
    ):
        raise ValueError("source-split lock/output counts are inconsistent")

    split = SourceSplit(
        fit_sources=tuple(
            row["source_id"] for row in rows if row["partition"] == "route_fit"
        ),
        confirm_sources=tuple(
            row["source_id"]
            for row in rows
            if row["partition"] == "route_confirm"
        ),
    )
    return LockedSourceSplit(
        split=split,
        assignments_path=assignments_path,
        summary_path=summary_path,
        lock_path=resolved_lock,
        lock_sha256=lock_sha256,
        summary_sha256=summary_sha,
        assignments_sha256=assignments_sha,
        manifest_sha256=manifest_sha,
        forbidden_ids_sha256=forbidden_sha,
        manifest_row_count=int(manifest_audit["row_count"]),
        manifest_rows_by_partition=manifest_rows_by_partition,
    )


def compute_quality_filter(
    ctc_log_likelihood: Sequence[float] | np.ndarray,
    view_cosine_min: Sequence[float] | np.ndarray,
    source_ids: Sequence[str] | np.ndarray,
    split: SourceSplit,
) -> QualityFilter:
    """Compute the frozen route-fit-only cutoff over unique positions."""

    likelihood = np.asarray(ctc_log_likelihood, dtype=np.float64)
    cosine = np.asarray(view_cosine_min, dtype=np.float64)
    raw_sources = np.asarray(source_ids)
    if (
        likelihood.ndim != 1
        or cosine.ndim != 1
        or raw_sources.ndim != 1
        or not len(likelihood)
        or not (len(likelihood) == len(cosine) == len(raw_sources))
    ):
        raise ValueError("quality-filter position arrays are empty or unaligned")
    sources = np.asarray([str(value) for value in raw_sources], dtype=str)
    if any(not value for value in sources):
        raise ValueError("quality-filter source IDs must be non-empty")
    fit_sources = frozenset(split.fit_sources)
    confirm_sources = frozenset(split.confirm_sources)
    if not fit_sources or not confirm_sources or fit_sources & confirm_sources:
        raise ValueError("quality filter requires disjoint non-empty source partitions")
    unknown = sorted(set(sources.tolist()) - fit_sources - confirm_sources)
    if unknown:
        raise ValueError(f"quality-filter position has unknown source: {unknown[0]}")
    fit_mask = np.fromiter(
        (source in fit_sources for source in sources),
        dtype=np.bool_,
        count=len(sources),
    )
    finite = np.isfinite(likelihood) & np.isfinite(cosine)
    finite_fit = finite & fit_mask
    if not np.any(finite_fit):
        raise ValueError("route-fit quality-cutoff population has no finite position")
    cutoff = float(
        np.percentile(
            likelihood[finite_fit],
            QUALITY_PERCENTILE,
            method="linear",
        )
    )
    accepted = (
        finite
        & (likelihood >= cutoff)
        & (cosine >= MINIMUM_VIEW_COSINE)
    )
    return QualityFilter(
        cutoff=cutoff,
        accepted=accepted,
        finite=finite,
        fit_position_count=int(np.sum(fit_mask)),
        finite_fit_position_count=int(np.sum(finite_fit)),
        accepted_fit_position_count=int(np.sum(accepted & fit_mask)),
        accepted_confirm_position_count=int(np.sum(accepted & ~fit_mask)),
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("prototype vectors must be finite non-empty rank-2 FP32")
    squared = np.sum(values * values, axis=1, dtype=np.float32)
    norms = np.sqrt(squared).astype(np.float32, copy=False)
    if np.any(norms <= 0):
        raise ValueError("prototype vector has zero L2 norm")
    return (values / norms[:, None]).astype(np.float32, copy=False)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("prototype mean must be a finite non-empty FP32 vector")
    norm = np.sqrt(np.sum(values * values, dtype=np.float32)).astype(np.float32)
    if float(norm) <= 0:
        raise ValueError("prototype mean has zero L2 norm")
    return (values / norm).astype(np.float32, copy=False)


def build_source_balanced_prototypes(
    vectors: np.ndarray,
    endpoints: Sequence[str] | np.ndarray,
    source_ids: Sequence[str] | np.ndarray,
) -> PrototypeResult:
    """Build both endpoint prototypes from paired route-fit occurrences."""

    matrix = np.asarray(vectors, dtype=np.float32)
    endpoint_values = np.asarray(endpoints).astype(str)
    sources = np.asarray(source_ids).astype(str)
    if (
        matrix.ndim != 2
        or endpoint_values.ndim != 1
        or sources.ndim != 1
        or not (len(matrix) == len(endpoint_values) == len(sources))
        or not len(matrix)
    ):
        raise ValueError("prototype vectors/endpoints/source IDs are unaligned")
    if set(endpoint_values.tolist()) != {"target", "confusion"}:
        raise ValueError("prototype population must contain both endpoints")
    if any(not value for value in sources):
        raise ValueError("prototype source IDs must be non-empty")
    endpoint_source_sets = {
        endpoint: set(sources[endpoint_values == endpoint].tolist())
        for endpoint in ("target", "confusion")
    }
    if endpoint_source_sets["target"] != endpoint_source_sets["confusion"]:
        raise ValueError("prototype population must contain paired endpoint sources")

    normalized_occurrences = _normalize_rows(matrix)
    result: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    occurrence_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for endpoint in ("target", "confusion"):
        endpoint_mask = endpoint_values == endpoint
        endpoint_sources = sorted(set(sources[endpoint_mask].tolist()))
        if not endpoint_sources:
            raise ValueError(f"prototype endpoint has no source: {endpoint}")
        source_prototypes: list[np.ndarray] = []
        for source in endpoint_sources:
            source_mask = endpoint_mask & (sources == source)
            source_mean = np.mean(
                normalized_occurrences[source_mask],
                axis=0,
                dtype=np.float32,
            )
            source_prototypes.append(_normalize_vector(source_mean))
        prototype = _normalize_vector(
            np.mean(
                np.stack(source_prototypes, axis=0),
                axis=0,
                dtype=np.float32,
            )
        )
        little_endian = np.ascontiguousarray(prototype.astype("<f4", copy=False))
        result[endpoint] = prototype
        digests[endpoint] = hashlib.sha256(little_endian.tobytes()).hexdigest()
        occurrence_counts[endpoint] = int(np.sum(endpoint_mask))
        source_counts[endpoint] = len(endpoint_sources)
    return PrototypeResult(
        vectors=result,
        fp32_prequantization_sha256=digests,
        occurrence_count_by_endpoint=occurrence_counts,
        source_count_by_endpoint=source_counts,
    )


def _pair_from_record(
    value: object,
    *,
    context: str,
    cast,
) -> tuple[Any, Any]:
    if not isinstance(value, dict) or set(value) != {"target", "confusion"}:
        raise ValueError(f"{context} must contain target/confusion exactly")
    return cast(value["target"]), cast(value["confusion"])


def _partition_support_from_record(record: object) -> PartitionSupport:
    if not isinstance(record, dict):
        raise ValueError("partition support record must be an object")
    return PartitionSupport(
        partition=str(record["partition"]),
        observed_source_count=int(record["observed_source_count"]),
        paired_sources=tuple(map(str, record["paired_sources"])),
        one_class_sources=tuple(map(str, record["one_class_sources"])),
        raw_endpoint_occurrences=_pair_from_record(
            record["raw_endpoint_occurrences"],
            context="raw endpoint occurrences",
            cast=int,
        ),
        included_endpoint_occurrences=_pair_from_record(
            record["included_endpoint_occurrences"],
            context="included endpoint occurrences",
            cast=int,
        ),
        maximum_source_endpoint_fractions=_pair_from_record(
            record["maximum_source_endpoint_fractions"],
            context="maximum source endpoint fractions",
            cast=float,
        ),
        included_indices=tuple(map(int, record["included_indices"])),
        passed=bool(record["passed"]),
        failures=tuple(map(str, record["failures"])),
    )


def edge_evidence_from_record(record: object) -> EdgeEvidence:
    """Strictly reconstruct one JSON-serialized :class:`EdgeEvidence`."""

    if not isinstance(record, dict):
        raise ValueError("edge evidence record must be an object")
    support_record = record.get("support")
    if not isinstance(support_record, dict):
        raise ValueError("edge evidence lacks support")
    support = EdgeSupport(
        fit=_partition_support_from_record(support_record.get("fit")),
        confirm=_partition_support_from_record(support_record.get("confirm")),
        passed=bool(support_record.get("passed")),
        failures=tuple(map(str, support_record.get("failures", []))),
    )
    ci_value = record.get("ci95")
    ci95 = (
        None
        if ci_value is None
        else (float(ci_value[0]), float(ci_value[1]))
    )
    theta = record.get("theta")
    evidence = EdgeEvidence(
        edge_id=str(record.get("edge_id", "")),
        tone_only=bool(record.get("tone_only")),
        support=support,
        converged=bool(record.get("converged")),
        seed=int(record.get("seed")),
        theta=None if theta is None else float(theta),
        source_aucs=tuple(map(float, record.get("source_aucs", []))),
        source_auc_ids=tuple(map(str, record.get("source_auc_ids", []))),
        ci95=ci95,
        p_e=float(record.get("p_e")),
        p_d=float(record.get("p_d")),
        permutation_exceedances=(
            None
            if record.get("permutation_exceedances") is None
            else int(record["permutation_exceedances"])
        ),
        warning_messages=tuple(map(str, record.get("warning_messages", []))),
    )
    if (
        not evidence.edge_id
        or evidence.support.fit.partition != "fit"
        or evidence.support.confirm.partition != "confirm"
        or evidence.support.passed
        != (evidence.support.fit.passed and evidence.support.confirm.passed)
        or evidence.support.failures
        != evidence.support.fit.failures + evidence.support.confirm.failures
        or len(evidence.source_aucs) != len(evidence.source_auc_ids)
        or not 0 <= evidence.p_e <= 1
        or not 0 <= evidence.p_d <= 1
        or any(not math.isfinite(value) for value in evidence.source_aucs)
        or (evidence.theta is not None and not math.isfinite(evidence.theta))
        or (
            evidence.ci95 is not None
            and (
                not all(math.isfinite(value) for value in evidence.ci95)
                or not 0 <= evidence.ci95[0] <= evidence.ci95[1] <= 1
            )
        )
    ):
        raise ValueError("edge evidence record is internally inconsistent")
    return evidence


def evidence_record_sha256(evidence: EdgeEvidence) -> str:
    """Return the canonical hash used by shard summaries."""

    return hashlib.sha256(canonical_json_bytes(evidence.to_record())).hexdigest()
