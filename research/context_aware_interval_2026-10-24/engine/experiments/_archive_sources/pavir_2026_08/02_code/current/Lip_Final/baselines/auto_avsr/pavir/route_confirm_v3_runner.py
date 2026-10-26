"""Authenticated, resumable execution scaffold for RouteConfirm-v3.

This module is intentionally separate from the locked statistical core and IO
primitives.  It binds those immutable implementations to the frozen graph,
full factorized bundle, bundle gate, source split, and full v3 preflight.  The
currently published preflight authorizes only bounded resource canaries, so
plans produced here cannot use both production Monte-Carlo defaults and none
of the outputs authorize offline artifacts or training.

The expensive unit of work is one support-eligible non-tone edge.  Each edge
is deterministic from its edge seed and is atomically checkpointed.  Shards
therefore affect scheduling only.  A complete merge includes every non-tone
graph edge in both multiplicity families and explicitly assigns p=1 to
support-skipped edges.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .artifacts import canonical_json_bytes, sha256_file
from .route_confirm_v3 import (
    PRODUCTION_BOOTSTRAP_ITERATIONS,
    PRODUCTION_PERMUTATIONS,
    PROTOCOL,
    PROTOCOL_PRECOMMIT_SHA256,
    EdgeEvidence,
    PermutationResult,
    RouteDecision,
    SourceSplit,
    SupportThresholds,
    audit_edge_support,
    benjamini_yekutieli,
    complete_non_tone_bh,
    decide_route,
    edge_seed,
    evaluate_edge,
    fit_weighted_probe,
    one_sample_auc_tost,
    source_bootstrap_ci,
    source_macro_auc,
    within_source_permutation_test,
    _paired_rank_specs,
)
from .route_confirm_v3_io import (
    compute_quality_filter,
    edge_evidence_from_record,
    evidence_record_sha256,
    load_locked_source_split,
)


PLAN_SCHEMA = "pavir-route-confirm-v3-edge-plan-v1"
CHECKPOINT_SCHEMA = "pavir-route-confirm-v3-edge-evidence-checkpoint-v1"
FAMILY_SCHEMA = "pavir-route-confirm-v3-family-merge-v1"
PREFLIGHT_SCHEMA = "pavir-route-confirm-v3-full-preflight-v1"
BUNDLE_GATE_SCHEMA = "pavir-factorized-full-bundle-gate-v1"
BUNDLE_SCHEMA = "pavir-factorized-feature-bundle-v1"
CORE_LOCK_SCHEMA = "pavir-route-confirm-v3-core-implementation-lock-v1"
IO_LOCK_SCHEMA = "pavir-route-confirm-v3-io-implementation-lock-v1"
GRAPH_SCHEMA = "pavir-confusion-graph-v1"

PLAN_STATUS = "READY_BOUNDED_RESOURCE_CANARY"
CHECKPOINT_STATUS = "COMPLETE_BOUNDED_EDGE_EVIDENCE"
_EDGE_ID = re.compile(r"e-[0-9a-f]{20}\Z")


@dataclass(frozen=True)
class BoundInputs:
    preflight_path: Path
    preflight_sha256: str
    preflight: Mapping[str, Any]
    graph_path: Path
    graph_sha256: str
    graph: Mapping[str, Any]
    features_path: Path
    features_sha256: str
    bundle_gate_path: Path
    bundle_gate_sha256: str
    source_split_lock_path: Path
    source_split_lock_sha256: str
    core_lock_path: Path
    core_lock_sha256: str
    io_lock_path: Path
    io_lock_sha256: str

    def records(self) -> dict[str, dict[str, Any]]:
        return {
            "graph": _file_record(self.graph_path, self.graph_sha256),
            "features": _file_record(self.features_path, self.features_sha256),
            "full_bundle_gate": _file_record(
                self.bundle_gate_path, self.bundle_gate_sha256
            ),
            "source_split_lock": _file_record(
                self.source_split_lock_path, self.source_split_lock_sha256
            ),
            "core_lock": _file_record(self.core_lock_path, self.core_lock_sha256),
            "io_lock": _file_record(self.io_lock_path, self.io_lock_sha256),
            "full_preflight": _file_record(
                self.preflight_path, self.preflight_sha256
            ),
        }


@dataclass(frozen=True)
class BundleContext:
    edge_dictionary: np.ndarray
    membership_positions: np.ndarray
    membership_edges: np.ndarray
    membership_endpoints: np.ndarray
    position_sources: np.ndarray
    position_vectors: np.ndarray
    quality_accepted: np.ndarray
    quality_mask_sha256: str
    starts: np.ndarray
    ends: np.ndarray
    split: SourceSplit


class PermutationBackend(Protocol):
    """Backend boundary for the one locked fixed-score permutation statistic."""

    name: str

    def run(
        self,
        endpoints: np.ndarray,
        scores: np.ndarray,
        sources: np.ndarray,
        *,
        edge_id: str,
        iterations: int,
        batch_size: int,
    ) -> PermutationResult:
        ...


@dataclass(frozen=True)
class LockedCPUBackend:
    name: str = "locked_cpu_numpy"

    def run(
        self,
        endpoints: np.ndarray,
        scores: np.ndarray,
        sources: np.ndarray,
        *,
        edge_id: str,
        iterations: int,
        batch_size: int,
    ) -> PermutationResult:
        return within_source_permutation_test(
            endpoints,
            scores,
            sources,
            edge_id=edge_id,
            iterations=iterations,
            batch_size=batch_size,
        )


@dataclass(frozen=True)
class SerializedCUDABackend:
    """NumPy-RNG/Torch-topk canary backend guarded by one cross-process lock.

    This backend is deliberately available only in bounded plans.  It preserves
    NumPy's edge-local random stream and moves the deterministic smallest-rank
    selection and rank summation to float64 CUDA.  A separate implementation
    lock/equivalence gate would still be required before confirmatory use.
    """

    lock_path: Path
    name: str = "cuda_numpy_rng_torch_topk_v1"

    def run(
        self,
        endpoints: np.ndarray,
        scores: np.ndarray,
        sources: np.ndarray,
        *,
        edge_id: str,
        iterations: int,
        batch_size: int,
    ) -> PermutationResult:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA permutation backend requested without CUDA")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            specs, observed_theta = _paired_rank_specs(endpoints, scores, sources)
            rng = np.random.default_rng(edge_seed(edge_id))
            null_sum = np.zeros(iterations, dtype=np.float64)
            device = torch.device("cuda:0")
            for _, ranks, positives, negatives in specs:
                count = len(ranks)
                choose_positive = positives <= negatives
                selected_count = positives if choose_positive else negatives
                rank_total = float(np.sum(ranks))
                offset = positives * (positives + 1) / 2
                denominator = positives * negatives
                rank_tensor = torch.as_tensor(
                    ranks, dtype=torch.float64, device=device
                )
                start = 0
                while start < iterations:
                    stop = min(start + batch_size, iterations)
                    priorities = rng.random((stop - start, count))
                    priority_tensor = torch.as_tensor(
                        priorities, dtype=torch.float64, device=device
                    )
                    chosen = torch.topk(
                        priority_tensor,
                        k=selected_count,
                        dim=1,
                        largest=False,
                        sorted=False,
                    ).indices
                    selected = torch.sum(rank_tensor[chosen], dim=1).cpu().numpy()
                    positive_rank_sum = (
                        selected if choose_positive else rank_total - selected
                    )
                    null_sum[start:stop] += (
                        positive_rank_sum - offset
                    ) / denominator
                    start = stop
            torch.cuda.synchronize()
            del rank_tensor
        null_thetas = null_sum / len(specs)
        exceedances = int(np.sum(null_thetas >= observed_theta))
        return PermutationResult(
            p_value=float((1 + exceedances) / (iterations + 1)),
            exceedances=exceedances,
            iterations=iterations,
            null_mean=float(np.mean(null_thetas)),
            null_std=float(np.std(null_thetas)),
            null_thetas=None,
        )


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _file_record(path: Path, digest: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": digest if digest is not None else sha256_file(resolved),
    }


def _resolve_locked_record(
    value: object,
    *,
    context: str,
    expected_path: Path | None = None,
    hash_file: bool = True,
) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} record is missing")
    raw_path = value.get("path")
    expected_sha = str(value.get("sha256", "")).lower()
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{context} path is missing")
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ValueError(f"{context} SHA-256 is invalid")
    path = Path(raw_path).expanduser().resolve()
    if expected_path is not None and path != expected_path.expanduser().resolve():
        raise ValueError(f"{context} path differs from the bound path")
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path}")
    if hash_file and sha256_file(path) != expected_sha:
        raise ValueError(f"{context} hash mismatch")
    return path, expected_sha


def _atomic_publish(path: Path, payload: Mapping[str, Any]) -> bool:
    """Publish canonical JSON once; an identical existing file is a resume hit."""

    encoded = canonical_json_bytes(payload)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace immutable artifact: {path}")
        return False
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _verify_lock_sources(lock: Mapping[str, Any], *, context: str) -> None:
    sources = lock.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{context} has no implementation source inventory")
    for index, record in enumerate(sources):
        _resolve_locked_record(
            record,
            context=f"{context} implementation source {index}",
        )


def authenticate_inputs(
    preflight_path: str | Path,
    *,
    core_lock_path: str | Path | None = None,
) -> BoundInputs:
    """Rehash and semantically authenticate every frozen runner input."""

    resolved_preflight = Path(preflight_path).expanduser().resolve()
    preflight_sha = sha256_file(resolved_preflight)
    preflight = _load_object(resolved_preflight)
    scope = preflight.get("scope")
    inputs = preflight.get("inputs")
    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA
        or preflight.get("protocol_name") != PROTOCOL
        or preflight.get("status") != "PASS_READY_FOR_V3_RESOURCE_CANARY"
        or not isinstance(scope, dict)
        or scope.get("support_and_resource_planning_only") is not True
        or scope.get("complete_routing_authorized") is not False
        or scope.get("offline_artifacts_authorized") is not False
        or scope.get("training_authorized") is not False
        or not isinstance(inputs, dict)
    ):
        raise ValueError("v3 full preflight has an unsafe or unsupported contract")

    graph_path, graph_sha = _resolve_locked_record(
        inputs.get("graph"), context="stable graph"
    )
    # Hashing the 5 GB bundle is intentional: the plan is an authenticated
    # execution boundary, not a path-only convenience wrapper.
    features_path, features_sha = _resolve_locked_record(
        inputs.get("features"), context="full factorized bundle"
    )
    gate_path, gate_sha = _resolve_locked_record(
        inputs.get("full_bundle_gate"), context="full bundle gate"
    )
    source_path, source_sha = _resolve_locked_record(
        inputs.get("source_split_lock"), context="source split lock"
    )
    io_path, io_sha = _resolve_locked_record(
        inputs.get("io_lock"), context="IO implementation lock"
    )

    graph = _load_object(graph_path)
    graph_edges = graph.get("edges")
    if (
        graph.get("schema_version") != GRAPH_SCHEMA
        or not isinstance(graph_edges, list)
        or not graph_edges
    ):
        raise ValueError("v3 runner requires the frozen stable graph")
    edge_ids = [str(row.get("edge_id", "")) for row in graph_edges]
    if (
        len(edge_ids) != len(set(edge_ids))
        or any(not _EDGE_ID.fullmatch(edge_id) for edge_id in edge_ids)
        or int(inputs["graph"].get("edge_count", -1)) != len(edge_ids)
    ):
        raise ValueError("stable graph edge inventory is invalid")

    gate = _load_object(gate_path)
    gate_bundle = gate.get("bundle")
    if (
        gate.get("schema_version") != BUNDLE_GATE_SCHEMA
        or gate.get("status") != "PASS"
        or gate.get("graph_sha256") != graph_sha
        or not isinstance(gate_bundle, dict)
        or Path(str(gate_bundle.get("path", ""))).expanduser().resolve()
        != features_path
        or gate_bundle.get("sha256") != features_sha
    ):
        raise ValueError("full bundle gate does not authenticate graph and bundle")

    io_lock = _load_object(io_path)
    io_scope = io_lock.get("scope")
    if (
        io_lock.get("schema_version") != IO_LOCK_SCHEMA
        or io_lock.get("protocol_name") != PROTOCOL
        or not str(io_lock.get("status", "")).startswith("LOCKED_")
        or not isinstance(io_scope, dict)
        or io_scope.get("complete_route_authorized") is not False
        or io_scope.get("offline_artifacts_authorized") is not False
        or io_scope.get("training_authorized") is not False
    ):
        raise ValueError("IO implementation lock has an unsafe contract")
    io_lineage = io_lock.get("lineage")
    if not isinstance(io_lineage, dict):
        raise ValueError("IO implementation lock lacks lineage")
    _resolve_locked_record(
        io_lineage.get("source_split_lock"),
        context="IO-bound source split lock",
        expected_path=source_path,
    )
    bound_core_path, bound_core_sha = _resolve_locked_record(
        io_lineage.get("core_implementation_lock"),
        context="IO-bound core implementation lock",
    )
    if core_lock_path is not None:
        requested_core = Path(core_lock_path).expanduser().resolve()
        if requested_core != bound_core_path:
            raise ValueError("requested core lock differs from IO-bound core lock")
    _verify_lock_sources(io_lock, context="IO implementation lock")

    core_lock = _load_object(bound_core_path)
    core_scope = core_lock.get("scope")
    if (
        core_lock.get("schema_version") != CORE_LOCK_SCHEMA
        or core_lock.get("protocol_name") != PROTOCOL
        or not str(core_lock.get("status", "")).startswith("LOCKED_")
        or core_lock.get("protocol_precommit", {}).get("sha256")
        != PROTOCOL_PRECOMMIT_SHA256
        or not isinstance(core_scope, dict)
        or core_scope.get("complete_route_authorized") is not False
        or core_scope.get("offline_artifacts_authorized") is not False
        or core_scope.get("training_authorized") is not False
    ):
        raise ValueError("core implementation lock has an unsafe contract")
    _resolve_locked_record(
        core_lock.get("protocol_precommit"),
        context="core-bound protocol precommit",
    )
    _verify_lock_sources(core_lock, context="core implementation lock")

    source_lock = _load_object(source_path)
    source_core_path, source_core_sha = _resolve_locked_record(
        source_lock.get("lineage", {}).get("core_implementation_lock"),
        context="source-split-bound core lock",
        expected_path=bound_core_path,
    )
    if source_core_sha != bound_core_sha or source_core_path != bound_core_path:
        raise ValueError("source split and IO lock bind different core locks")
    # This validates the complete source/manifest/forbidden/tool/test chain.
    load_locked_source_split(source_path)

    implementation = preflight.get("implementation", {})
    _resolve_locked_record(
        implementation.get("tool"),
        context="v3 full-preflight implementation",
    )
    rows = preflight.get("edges")
    if not isinstance(rows, list) or len(rows) != len(edge_ids):
        raise ValueError("v3 preflight edge inventory is incomplete")
    preflight_ids = [str(row.get("edge_id", "")) for row in rows]
    if preflight_ids != sorted(edge_ids) or len(preflight_ids) != len(set(preflight_ids)):
        raise ValueError("v3 preflight edge order/inventory differs from graph")
    graph_by_id = {str(row["edge_id"]): row for row in graph_edges}
    for row in rows:
        edge_id = str(row["edge_id"])
        if bool(row.get("tone_only")) != bool(graph_by_id[edge_id].get("tone_only")):
            raise ValueError(f"preflight tone flag differs from graph: {edge_id}")

    return BoundInputs(
        preflight_path=resolved_preflight,
        preflight_sha256=preflight_sha,
        preflight=preflight,
        graph_path=graph_path,
        graph_sha256=graph_sha,
        graph=graph,
        features_path=features_path,
        features_sha256=features_sha,
        bundle_gate_path=gate_path,
        bundle_gate_sha256=gate_sha,
        source_split_lock_path=source_path,
        source_split_lock_sha256=source_sha,
        core_lock_path=bound_core_path,
        core_lock_sha256=bound_core_sha,
        io_lock_path=io_path,
        io_lock_sha256=io_sha,
    )


def deterministic_lpt_shards(
    edge_work: Mapping[str, int],
    shard_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Deterministic longest-processing-time scheduling with stable ties."""

    if isinstance(shard_count, bool) or int(shard_count) <= 0:
        raise ValueError("shard_count must be positive")
    count = int(shard_count)
    validated: dict[str, int] = {}
    for edge_id, work in edge_work.items():
        if not isinstance(edge_id, str) or not edge_id:
            raise ValueError("edge work contains an invalid edge ID")
        if isinstance(work, bool) or int(work) <= 0:
            raise ValueError(f"edge work must be positive: {edge_id}")
        validated[edge_id] = int(work)
    if not validated:
        raise ValueError("cannot schedule an empty edge family")
    bins: list[list[str]] = [[] for _ in range(count)]
    loads = [0] * count
    for edge_id, work in sorted(
        validated.items(), key=lambda item: (-item[1], item[0])
    ):
        shard = min(range(count), key=lambda index: (loads[index], index))
        bins[shard].append(edge_id)
        loads[shard] += work
    return tuple(tuple(sorted(values)) for values in bins)


def _edge_work(row: Mapping[str, Any], *, permutations: int, bootstrap: int) -> int:
    fit = row["fit"]["included_endpoint_occurrences"]
    confirm = row["confirm"]["included_endpoint_occurrences"]
    fit_occurrences = int(fit["target"]) + int(fit["confusion"])
    confirm_occurrences = int(confirm["target"]) + int(confirm["confusion"])
    confirm_sources = int(row["confirm"]["paired_source_count"])
    if bool(row.get("tone_only")) or not bool(row.get("support_passed")):
        # These edges still receive exact support evidence, but evaluate_edge
        # short-circuits before probe/permutation/bootstrap.
        return max(1, int(row.get("accepted_memberships", 1)))
    # Permutations dominate and scale with held-out occurrences.  The smaller
    # terms keep the model useful for low-iteration canaries.
    return (
        confirm_occurrences * permutations
        + confirm_sources * bootstrap
        + fit_occurrences
    )


def _bounded_statistics(permutations: int, bootstrap: int) -> tuple[int, int]:
    values = (permutations, bootstrap)
    if any(isinstance(value, bool) or int(value) <= 0 for value in values):
        raise ValueError("Monte-Carlo counts must be positive integers")
    permutation_count, bootstrap_count = map(int, values)
    if (
        permutation_count > PRODUCTION_PERMUTATIONS
        or bootstrap_count > PRODUCTION_BOOTSTRAP_ITERATIONS
    ):
        raise ValueError("resource canary cannot exceed production defaults")
    if (
        permutation_count == PRODUCTION_PERMUTATIONS
        and bootstrap_count == PRODUCTION_BOOTSTRAP_ITERATIONS
    ):
        raise PermissionError(
            "current preflight does not authorize a production 199999/10000 run"
        )
    return permutation_count, bootstrap_count


def build_plan(
    bound: BoundInputs,
    *,
    shard_count: int,
    permutation_iterations: int,
    bootstrap_iterations: int,
    permutation_backend: str,
    permutation_batch_size: int,
    runner_path: str | Path,
) -> dict[str, Any]:
    """Create a deterministic, complete all-edge canary plan."""

    permutations, bootstrap = _bounded_statistics(
        permutation_iterations, bootstrap_iterations
    )
    rows = bound.preflight["edges"]
    eligible_non_tone_rows = [
        row
        for row in rows
        if bool(row.get("support_passed")) and not bool(row.get("tone_only"))
    ]
    eligible_ids = [str(row["edge_id"]) for row in eligible_non_tone_rows]
    if eligible_ids != list(bound.preflight.get("eligible_non_tone_edge_ids", [])):
        raise ValueError("preflight eligible non-tone inventory is inconsistent")
    if permutation_backend not in {
        "locked_cpu_numpy",
        "cuda_numpy_rng_torch_topk_v1",
    }:
        raise ValueError("unsupported bounded permutation backend")
    if (
        isinstance(permutation_batch_size, bool)
        or int(permutation_batch_size) <= 0
    ):
        raise ValueError("permutation batch size must be positive")
    work = {
        str(row["edge_id"]): _edge_work(
            row, permutations=permutations, bootstrap=bootstrap
        )
        for row in rows
    }
    assignments = deterministic_lpt_shards(work, shard_count)
    shards = [
        {
            "shard_index": index,
            "edge_count": len(edge_ids),
            "estimated_work_units": int(sum(work[edge_id] for edge_id in edge_ids)),
            "edge_ids": list(edge_ids),
        }
        for index, edge_ids in enumerate(assignments)
    ]
    non_tone = sorted(
        str(row["edge_id"]) for row in rows if not bool(row.get("tone_only"))
    )
    skipped_non_tone = sorted(set(non_tone) - set(eligible_ids))
    tone_ids = sorted(
        str(row["edge_id"]) for row in rows if bool(row.get("tone_only"))
    )
    runner = Path(runner_path).expanduser().resolve()
    runner_module = Path(__file__).resolve()
    runtime_packages = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "scipy", "scikit-learn")
    }
    if permutation_backend == "cuda_numpy_rng_torch_topk_v1":
        runtime_packages["torch"] = importlib.metadata.version("torch")
    return {
        "schema_version": PLAN_SCHEMA,
        "protocol_name": PROTOCOL,
        "status": PLAN_STATUS,
        "scope": {
            "bounded_resource_canary_only": True,
            "complete_confirmatory_route_authorized": False,
            "offline_artifacts_authorized": False,
            "training_authorized": False,
        },
        "inputs": bound.records(),
        "statistics": {
            "permutation_iterations": permutations,
            "bootstrap_iterations": bootstrap,
            "permutation_batch_size": int(permutation_batch_size),
            "permutation_backend": permutation_backend,
            "production_permutation_iterations": PRODUCTION_PERMUTATIONS,
            "production_bootstrap_iterations": PRODUCTION_BOOTSTRAP_ITERATIONS,
            "confirmatory_defaults_used": False,
        },
        "inventory": {
            "graph_edges": len(rows),
            "scheduled_all_graph_edges": len(rows),
            "non_tone_bh_family_edges": len(non_tone),
            "expensive_support_eligible_non_tone_edges": len(eligible_ids),
            "p1_support_skipped_non_tone_edges": len(skipped_non_tone),
            "tone_only_edges": len(tone_ids),
        },
        "family": {
            "non_tone_edge_ids": non_tone,
            "scheduled_edge_ids": sorted(work),
            "expensive_support_eligible_non_tone_edge_ids": eligible_ids,
            "p1_skipped_non_tone_edge_ids": skipped_non_tone,
            "tone_only_edge_ids": tone_ids,
        },
        "scheduling": {
            "algorithm": "deterministic-lpt-v1",
            "work_model": (
                "confirm_occurrences*permutations + "
                "confirm_paired_sources*bootstrap + fit_occurrences"
            ),
            "shard_count": int(shard_count),
            "shards": shards,
        },
        "implementation": {
            "runner": _file_record(runner),
            "runner_module": _file_record(runner_module),
            "locked_core_modified": False,
            "locked_io_modified": False,
        },
        "runtime_contract": {
            "packages": runtime_packages,
            "thread_environment": {
                name: "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "cuda_backend_confirmatory_authorized": False,
        },
    }


def publish_plan(
    preflight_path: str | Path,
    output_path: str | Path,
    *,
    shard_count: int,
    permutation_iterations: int,
    bootstrap_iterations: int,
    permutation_backend: str = "locked_cpu_numpy",
    permutation_batch_size: int = 2_048,
    runner_path: str | Path,
    core_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    bound = authenticate_inputs(preflight_path, core_lock_path=core_lock_path)
    plan = build_plan(
        bound,
        shard_count=shard_count,
        permutation_iterations=permutation_iterations,
        bootstrap_iterations=bootstrap_iterations,
        permutation_backend=permutation_backend,
        permutation_batch_size=permutation_batch_size,
        runner_path=runner_path,
    )
    _atomic_publish(Path(output_path), plan)
    return plan


def _verify_plan(
    plan_path: str | Path,
) -> tuple[Path, str, dict[str, Any], BoundInputs]:
    path = Path(plan_path).expanduser().resolve()
    digest = sha256_file(path)
    plan = _load_object(path)
    scope = plan.get("scope")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("protocol_name") != PROTOCOL
        or plan.get("status") != PLAN_STATUS
        or not isinstance(scope, dict)
        or scope.get("bounded_resource_canary_only") is not True
        or scope.get("complete_confirmatory_route_authorized") is not False
        or scope.get("offline_artifacts_authorized") is not False
        or scope.get("training_authorized") is not False
    ):
        raise ValueError("unsupported or unsafe v3 edge plan")
    statistics = plan.get("statistics", {})
    permutations, bootstrap = _bounded_statistics(
        statistics.get("permutation_iterations"),
        statistics.get("bootstrap_iterations"),
    )
    if (
        statistics.get("confirmatory_defaults_used") is not False
        or permutations != int(statistics["permutation_iterations"])
        or bootstrap != int(statistics["bootstrap_iterations"])
        or statistics.get("permutation_backend")
        not in {"locked_cpu_numpy", "cuda_numpy_rng_torch_topk_v1"}
        or isinstance(statistics.get("permutation_batch_size"), bool)
        or int(statistics.get("permutation_batch_size", 0)) <= 0
    ):
        raise ValueError("v3 plan statistical scope is inconsistent")

    plan_inputs = plan.get("inputs")
    if not isinstance(plan_inputs, dict):
        raise ValueError("v3 edge plan lacks input bindings")
    preflight_path, preflight_sha = _resolve_locked_record(
        plan_inputs.get("full_preflight"), context="plan-bound full preflight"
    )
    core_path, _ = _resolve_locked_record(
        plan_inputs.get("core_lock"), context="plan-bound core lock"
    )
    bound = authenticate_inputs(preflight_path, core_lock_path=core_path)
    if preflight_sha != bound.preflight_sha256 or plan_inputs != bound.records():
        raise ValueError("v3 edge plan input bindings are inconsistent")
    runner_path, runner_sha = _resolve_locked_record(
        plan.get("implementation", {}).get("runner"),
        context="plan-bound runner implementation",
    )
    if runner_sha != sha256_file(runner_path):
        raise ValueError("v3 runner implementation drift")
    _resolve_locked_record(
        plan.get("implementation", {}).get("runner_module"),
        context="plan-bound runner module",
        expected_path=Path(__file__).resolve(),
    )
    runtime = plan.get("runtime_contract")
    if not isinstance(runtime, dict):
        raise ValueError("v3 plan lacks a runtime contract")
    expected_packages = runtime.get("packages")
    if not isinstance(expected_packages, dict):
        raise ValueError("v3 plan lacks package versions")
    for package, expected_version in expected_packages.items():
        if importlib.metadata.version(str(package)) != str(expected_version):
            raise ValueError(f"runtime package drift: {package}")
    thread_environment = runtime.get("thread_environment")
    if not isinstance(thread_environment, dict) or any(
        os.environ.get(name) != str(value)
        for name, value in thread_environment.items()
    ):
        raise ValueError("BLAS/OpenMP thread environment differs from plan")
    if runtime.get("cuda_backend_confirmatory_authorized") is not False:
        raise ValueError("bounded plan has unsafe CUDA authorization")

    scheduling = plan.get("scheduling")
    family = plan.get("family")
    if not isinstance(scheduling, dict) or not isinstance(family, dict):
        raise ValueError("v3 edge plan lacks scheduling/family metadata")
    shards = scheduling.get("shards")
    shard_count = int(scheduling.get("shard_count", -1))
    if not isinstance(shards, list) or shard_count <= 0 or len(shards) != shard_count:
        raise ValueError("v3 edge plan shard inventory is invalid")
    scheduled: list[str] = []
    for index, shard in enumerate(shards):
        if (
            not isinstance(shard, dict)
            or shard.get("shard_index") != index
            or not isinstance(shard.get("edge_ids"), list)
            or shard["edge_ids"] != sorted(shard["edge_ids"])
            or int(shard.get("edge_count", -1)) != len(shard["edge_ids"])
        ):
            raise ValueError(f"invalid v3 plan shard {index}")
        scheduled.extend(map(str, shard["edge_ids"]))
    expected = list(map(str, family.get("scheduled_edge_ids", [])))
    if sorted(scheduled) != expected or len(scheduled) != len(set(scheduled)):
        raise ValueError("v3 edge plan shards do not partition scheduled edges")
    return path, digest, plan, bound


def _quality_record_equal(observed: Mapping[str, Any], expected: object) -> bool:
    if not isinstance(expected, dict) or set(observed) != set(expected):
        return False
    for key in observed:
        left, right = observed[key], expected[key]
        if isinstance(left, float):
            if not math.isclose(left, float(right), rel_tol=0.0, abs_tol=1e-12):
                return False
        elif left != right:
            return False
    return True


def load_bundle_context(bound: BoundInputs) -> BundleContext:
    """Load and validate the full bundle once for one shard process."""

    locked_split = load_locked_source_split(bound.source_split_lock_path)
    with np.load(bound.features_path, allow_pickle=False) as archive:
        required = {
            "schema_version",
            "edge_dictionary",
            "endpoint_dictionary",
            "membership_position_indices",
            "membership_edge_indices",
            "membership_endpoint_indices",
            "position_source_ids",
            "position_ctc_log_likelihood",
            "position_view_cosine_min",
            "position_vectors",
        }
        if required - set(archive.files):
            raise ValueError("full bundle lacks a RouteConfirm-v3 array")
        if str(archive["schema_version"].item()) != BUNDLE_SCHEMA:
            raise ValueError("v3 runner requires a direct full factorized bundle")
        edge_dictionary = np.asarray(archive["edge_dictionary"]).astype(str)
        endpoint_dictionary = np.asarray(archive["endpoint_dictionary"]).astype(str)
        # Preserve compact uint32/uint16/uint8 storage.  Global int64 casts
        # would add roughly 1.6 GiB and provide no benefit to searchsorted or
        # per-edge fancy indexing.
        membership_positions = np.asarray(
            archive["membership_position_indices"]
        )
        membership_edges = np.asarray(archive["membership_edge_indices"])
        membership_endpoints = np.asarray(
            archive["membership_endpoint_indices"]
        )
        position_sources = np.asarray(archive["position_source_ids"]).astype(str)
        likelihood = np.asarray(
            archive["position_ctc_log_likelihood"], dtype=np.float64
        )
        view_cosine = np.asarray(
            archive["position_view_cosine_min"], dtype=np.float64
        )
        position_vectors = np.asarray(archive["position_vectors"])

    graph_ids = sorted(str(row["edge_id"]) for row in bound.graph["edges"])
    if (
        edge_dictionary.tolist() != graph_ids
        or endpoint_dictionary.tolist() != ["confusion", "target"]
        or not (
            len(membership_positions)
            == len(membership_edges)
            == len(membership_endpoints)
        )
        or any(
            values.dtype.kind not in "iu"
            for values in (
                membership_positions,
                membership_edges,
                membership_endpoints,
            )
        )
        or len(position_vectors) != len(position_sources)
        or position_vectors.ndim != 2
        or not len(membership_positions)
        or membership_positions.min() < 0
        or membership_positions.max() >= len(position_sources)
        or membership_edges.min() < 0
        or membership_edges.max() >= len(edge_dictionary)
        or membership_endpoints.min() < 0
        or membership_endpoints.max() >= 2
        or np.any(membership_edges[1:] < membership_edges[:-1])
    ):
        raise ValueError("full bundle dictionaries or aligned arrays are invalid")
    quality = compute_quality_filter(
        likelihood,
        view_cosine,
        position_sources,
        locked_split.split,
    )
    if not _quality_record_equal(
        quality.to_record(), bound.preflight.get("quality_filter")
    ):
        raise ValueError("recomputed quality filter differs from authenticated preflight")
    starts = np.searchsorted(
        membership_edges, np.arange(len(edge_dictionary)), side="left"
    )
    ends = np.searchsorted(
        membership_edges, np.arange(len(edge_dictionary)), side="right"
    )
    packed_quality = np.packbits(quality.accepted, bitorder="little")
    quality_sha = hashlib.sha256(packed_quality.tobytes()).hexdigest()
    context = BundleContext(
        edge_dictionary=edge_dictionary,
        membership_positions=membership_positions,
        membership_edges=membership_edges,
        membership_endpoints=membership_endpoints,
        position_sources=position_sources,
        position_vectors=position_vectors,
        quality_accepted=quality.accepted,
        quality_mask_sha256=quality_sha,
        starts=starts,
        ends=ends,
        split=locked_split.split,
    )
    for value in (
        context.edge_dictionary,
        context.membership_positions,
        context.membership_edges,
        context.membership_endpoints,
        context.position_sources,
        context.position_vectors,
        context.quality_accepted,
        context.starts,
        context.ends,
    ):
        value.flags.writeable = False
    return context


def _support_matches_preflight(
    evidence: EdgeEvidence,
    preflight_row: Mapping[str, Any],
) -> bool:
    if evidence.support.passed is not bool(preflight_row.get("support_passed")):
        return False
    for name, support in (
        ("fit", evidence.support.fit),
        ("confirm", evidence.support.confirm),
    ):
        expected = preflight_row.get(name)
        if not isinstance(expected, dict):
            return False
        raw = {
            "target": support.raw_endpoint_occurrences[0],
            "confusion": support.raw_endpoint_occurrences[1],
        }
        included = {
            "target": support.included_endpoint_occurrences[0],
            "confusion": support.included_endpoint_occurrences[1],
        }
        fractions = {
            "target": support.maximum_source_endpoint_fractions[0],
            "confusion": support.maximum_source_endpoint_fractions[1],
        }
        failures = [
            failure.removeprefix(f"{name}_") for failure in support.failures
        ]
        if (
            raw != expected.get("raw_endpoint_occurrences")
            or included != expected.get("included_endpoint_occurrences")
            or len(support.paired_sources) != int(expected.get("paired_source_count", -1))
            or bool(support.passed) is not bool(expected.get("passed"))
            or failures != list(expected.get("failures", []))
        ):
            return False
        expected_fractions = expected.get("maximum_source_endpoint_fractions")
        if not isinstance(expected_fractions, dict) or any(
            not math.isclose(
                float(fractions[endpoint]),
                float(expected_fractions.get(endpoint)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for endpoint in ("target", "confusion")
        ):
            return False
    return list(evidence.support.failures) == list(preflight_row.get("failures", []))


def _checkpoint_payload(
    *,
    plan_path: Path,
    plan_sha256: str,
    plan: Mapping[str, Any],
    edge_id: str,
    evidence: EdgeEvidence,
    edge_input: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "protocol_name": PROTOCOL,
        "status": CHECKPOINT_STATUS,
        "scope": {
            "bounded_edge_evidence_only": True,
            "route_family_complete": False,
            "offline_artifacts_authorized": False,
            "training_authorized": False,
        },
        "plan": _file_record(plan_path, plan_sha256),
        "inputs": plan["inputs"],
        "statistics": plan["statistics"],
        "edge_id": edge_id,
        "edge_input": dict(edge_input or {}),
        "evidence": evidence.to_record(),
        "evidence_sha256": evidence_record_sha256(evidence),
    }


def validate_checkpoint(
    path: str | Path,
    *,
    plan_path: Path,
    plan_sha256: str,
    plan: Mapping[str, Any],
    edge_id: str,
) -> EdgeEvidence:
    checkpoint = _load_object(Path(path))
    scope = checkpoint.get("scope")
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("protocol_name") != PROTOCOL
        or checkpoint.get("status") != CHECKPOINT_STATUS
        or checkpoint.get("edge_id") != edge_id
        or checkpoint.get("plan") != _file_record(plan_path, plan_sha256)
        or checkpoint.get("inputs") != plan.get("inputs")
        or checkpoint.get("statistics") != plan.get("statistics")
        or not isinstance(scope, dict)
        or scope.get("bounded_edge_evidence_only") is not True
        or scope.get("route_family_complete") is not False
        or scope.get("offline_artifacts_authorized") is not False
        or scope.get("training_authorized") is not False
    ):
        raise ValueError(f"edge checkpoint contract mismatch: {path}")
    evidence = edge_evidence_from_record(checkpoint.get("evidence"))
    expected_tone = edge_id in set(plan["family"]["tone_only_edge_ids"])
    statistics = plan["statistics"]
    iterations = int(statistics["permutation_iterations"])
    expected_p = (
        None
        if evidence.permutation_exceedances is None
        else (1 + evidence.permutation_exceedances) / (iterations + 1)
    )
    edge_input = checkpoint.get("edge_input")
    if (
        not isinstance(edge_input, dict)
        or isinstance(edge_input.get("accepted_memberships"), bool)
        or int(edge_input.get("accepted_memberships", -1)) < 0
        or any(
            len(str(edge_input.get(key, ""))) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(edge_input.get(key, ""))
            )
            for key in (
                "accepted_membership_stream_sha256",
                "quality_mask_sha256",
            )
        )
    ):
        raise ValueError(f"edge checkpoint input binding mismatch: {path}")
    if (
        evidence.edge_id != edge_id
        or evidence.tone_only != expected_tone
        or evidence.seed != edge_seed(edge_id)
        or len(set(evidence.source_auc_ids)) != len(evidence.source_auc_ids)
        or (
            evidence.theta is not None
            and not math.isclose(
                evidence.theta,
                float(np.mean(evidence.source_aucs, dtype=np.float64)),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        or (
            expected_p is not None
            and evidence.p_e != expected_p
        )
        or (
            evidence.permutation_exceedances is not None
            and not 0 <= evidence.permutation_exceedances <= iterations
        )
        or (
            (expected_tone or not evidence.support.passed or not evidence.converged)
            and (
                evidence.p_e != 1.0
                or evidence.p_d != 1.0
                or evidence.theta is not None
                or evidence.ci95 is not None
            )
        )
        or checkpoint.get("evidence_sha256") != evidence_record_sha256(evidence)
    ):
        raise ValueError(f"edge checkpoint evidence mismatch: {path}")
    return evidence


def _backend_for_plan(
    plan: Mapping[str, Any], output_dir: Path
) -> PermutationBackend:
    name = str(plan["statistics"]["permutation_backend"])
    if name == "locked_cpu_numpy":
        return LockedCPUBackend()
    if name == "cuda_numpy_rng_torch_topk_v1":
        return SerializedCUDABackend(output_dir / "gpu-permutation.lock")
    raise ValueError(f"unsupported permutation backend: {name}")


def evaluate_edge_with_backend(
    vectors: np.ndarray,
    endpoints: np.ndarray,
    sources: np.ndarray,
    split: SourceSplit,
    *,
    edge_id: str,
    tone_only: bool,
    permutation_iterations: int,
    bootstrap_iterations: int,
    permutation_batch_size: int,
    backend: PermutationBackend,
) -> EdgeEvidence:
    """Use the locked core directly, replacing only its permutation primitive."""

    if backend.name == "locked_cpu_numpy":
        return evaluate_edge(
            vectors,
            endpoints,
            sources,
            split,
            edge_id=edge_id,
            tone_only=tone_only,
            support_thresholds=SupportThresholds(),
            permutation_iterations=permutation_iterations,
            bootstrap_iterations=bootstrap_iterations,
        )
    support = audit_edge_support(
        vectors,
        endpoints,
        sources,
        split,
        thresholds=SupportThresholds(),
    )
    seed = edge_seed(edge_id)
    if not support.passed or tone_only:
        return EdgeEvidence(
            edge_id=edge_id,
            tone_only=tone_only,
            support=support,
            converged=True,
            seed=seed,
            theta=None,
            source_aucs=(),
            source_auc_ids=(),
            ci95=None,
            p_e=1.0,
            p_d=1.0,
            permutation_exceedances=None,
            warning_messages=(),
        )
    fit_indices = np.asarray(support.fit.included_indices, dtype=np.int64)
    confirm_indices = np.asarray(support.confirm.included_indices, dtype=np.int64)
    probe = fit_weighted_probe(
        vectors[fit_indices],
        endpoints[fit_indices],
        sources[fit_indices],
        edge_id=edge_id,
    )
    if not probe.converged:
        return EdgeEvidence(
            edge_id=edge_id,
            tone_only=False,
            support=support,
            converged=False,
            seed=seed,
            theta=None,
            source_aucs=(),
            source_auc_ids=(),
            ci95=None,
            p_e=1.0,
            p_d=1.0,
            permutation_exceedances=None,
            warning_messages=probe.warning_messages,
        )
    scores = probe.confusion_scores(vectors[confirm_indices])
    confirm_endpoints = endpoints[confirm_indices]
    confirm_sources = sources[confirm_indices]
    macro = source_macro_auc(confirm_endpoints, scores, confirm_sources)
    if macro.excluded_one_class_sources:
        raise RuntimeError("paired confirmation filter admitted a one-class source")
    permutation = backend.run(
        confirm_endpoints,
        scores,
        confirm_sources,
        edge_id=edge_id,
        iterations=permutation_iterations,
        batch_size=permutation_batch_size,
    )
    bootstrap = source_bootstrap_ci(
        macro.source_aucs,
        edge_id=edge_id,
        iterations=bootstrap_iterations,
    )
    equivalence = one_sample_auc_tost(macro.source_aucs)
    return EdgeEvidence(
        edge_id=edge_id,
        tone_only=False,
        support=support,
        converged=True,
        seed=seed,
        theta=macro.theta,
        source_aucs=macro.source_aucs,
        source_auc_ids=macro.source_ids,
        ci95=(bootstrap.lower, bootstrap.upper),
        p_e=permutation.p_value,
        p_d=equivalence.p_equivalence,
        permutation_exceedances=permutation.exceedances,
        warning_messages=(),
    )


def _edge_evidence(
    edge_id: str,
    *,
    context: BundleContext,
    bound: BoundInputs,
    permutation_iterations: int,
    bootstrap_iterations: int,
    permutation_batch_size: int,
    backend: PermutationBackend,
) -> tuple[EdgeEvidence, dict[str, Any]]:
    edge_index = int(np.searchsorted(context.edge_dictionary, edge_id))
    if (
        edge_index >= len(context.edge_dictionary)
        or str(context.edge_dictionary[edge_index]) != edge_id
    ):
        raise ValueError(f"scheduled edge is absent from bundle: {edge_id}")
    start, end = int(context.starts[edge_index]), int(context.ends[edge_index])
    positions = context.membership_positions[start:end]
    endpoint_codes = context.membership_endpoints[start:end]
    accepted = context.quality_accepted[positions]
    positions = positions[accepted]
    endpoint_codes = endpoint_codes[accepted]
    if not len(positions):
        raise ValueError(f"scheduled edge has no quality-accepted memberships: {edge_id}")
    endpoints = np.where(endpoint_codes == 0, "confusion", "target")
    graph_by_id = {str(row["edge_id"]): row for row in bound.graph["edges"]}
    evidence = evaluate_edge_with_backend(
        np.asarray(context.position_vectors[positions], dtype=np.float32),
        endpoints,
        context.position_sources[positions],
        context.split,
        edge_id=edge_id,
        tone_only=bool(graph_by_id[edge_id]["tone_only"]),
        permutation_iterations=permutation_iterations,
        bootstrap_iterations=bootstrap_iterations,
        permutation_batch_size=permutation_batch_size,
        backend=backend,
    )
    row_by_id = {
        str(row["edge_id"]): row for row in bound.preflight["edges"]
    }
    if not _support_matches_preflight(evidence, row_by_id[edge_id]):
        raise ValueError(f"edge support differs from authenticated preflight: {edge_id}")
    stream = np.empty(
        len(positions),
        dtype=np.dtype([("position", "<u4"), ("endpoint", "u1")]),
    )
    stream["position"] = positions
    stream["endpoint"] = endpoint_codes
    edge_input = {
        "accepted_memberships": len(positions),
        "accepted_membership_stream_sha256": hashlib.sha256(
            stream.tobytes()
        ).hexdigest(),
        "quality_mask_sha256": context.quality_mask_sha256,
    }
    if edge_input["accepted_memberships"] != int(
        row_by_id[edge_id]["accepted_memberships"]
    ):
        raise ValueError(f"accepted membership count drift: {edge_id}")
    return evidence, edge_input


def run_shard(
    plan_path: str | Path,
    output_dir: str | Path,
    *,
    shard_index: int,
    max_new_edges: int | None = None,
) -> dict[str, Any]:
    """Run/resume one shard and atomically checkpoint every completed edge."""

    resolved_plan, plan_sha, plan, bound = _verify_plan(plan_path)
    shards = plan["scheduling"]["shards"]
    if (
        isinstance(shard_index, bool)
        or int(shard_index) < 0
        or int(shard_index) >= len(shards)
    ):
        raise ValueError("shard index is outside the plan")
    if max_new_edges is not None and (
        isinstance(max_new_edges, bool) or int(max_new_edges) <= 0
    ):
        raise ValueError("max_new_edges must be positive")
    shard_index = int(shard_index)
    edge_ids = list(map(str, shards[shard_index]["edge_ids"]))
    root = Path(output_dir).expanduser().resolve()
    checkpoint_dir = root / "edges"
    shard_dir = root / "shards" / f"shard-{shard_index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    lock_path = shard_dir / "runner.lock"
    with lock_path.open("a+b") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"shard {shard_index} already has an active runner") from error

        completed: list[str] = []
        pending: list[str] = []
        preflight_by_id = {
            str(row["edge_id"]): row for row in bound.preflight["edges"]
        }
        for edge_id in edge_ids:
            checkpoint_path = checkpoint_dir / f"{edge_id}.json"
            if checkpoint_path.is_file():
                resumed = validate_checkpoint(
                    checkpoint_path,
                    plan_path=resolved_plan,
                    plan_sha256=plan_sha,
                    plan=plan,
                    edge_id=edge_id,
                )
                checkpoint_record = _load_object(checkpoint_path)
                if (
                    int(checkpoint_record["edge_input"]["accepted_memberships"])
                    != int(preflight_by_id[edge_id]["accepted_memberships"])
                    or not _support_matches_preflight(
                        resumed, preflight_by_id[edge_id]
                    )
                ):
                    raise ValueError(
                        f"resumed checkpoint differs from preflight: {edge_id}"
                    )
                completed.append(edge_id)
            else:
                pending.append(edge_id)

        new_limit = len(pending) if max_new_edges is None else int(max_new_edges)
        context: BundleContext | None = None
        new_completed = 0
        statistics = plan["statistics"]
        backend = _backend_for_plan(plan, root)
        for edge_id in pending[:new_limit]:
            if context is None:
                context = load_bundle_context(bound)
            evidence, edge_input = _edge_evidence(
                edge_id,
                context=context,
                bound=bound,
                permutation_iterations=int(statistics["permutation_iterations"]),
                bootstrap_iterations=int(statistics["bootstrap_iterations"]),
                permutation_batch_size=int(statistics["permutation_batch_size"]),
                backend=backend,
            )
            checkpoint = _checkpoint_payload(
                plan_path=resolved_plan,
                plan_sha256=plan_sha,
                plan=plan,
                edge_id=edge_id,
                evidence=evidence,
                edge_input=edge_input,
            )
            _atomic_publish(checkpoint_dir / f"{edge_id}.json", checkpoint)
            completed.append(edge_id)
            new_completed += 1

        completed_set = set(completed)
        remaining = [edge_id for edge_id in edge_ids if edge_id not in completed_set]
        result = {
            "schema_version": "pavir-route-confirm-v3-shard-status-v1",
            "protocol_name": PROTOCOL,
            "status": (
                "COMPLETE_BOUNDED_SHARD_EVIDENCE"
                if not remaining
                else "PARTIAL_BOUNDED_SHARD_EVIDENCE"
            ),
            "scope": {
                "route_family_complete": False,
                "training_authorized": False,
            },
            "plan": _file_record(resolved_plan, plan_sha),
            "shard_index": shard_index,
            "edge_count": len(edge_ids),
            "completed_edges": len(completed_set),
            "newly_completed_edges": new_completed,
            "remaining_edges": len(remaining),
            "remaining_edge_ids": remaining,
        }
        status_path = shard_dir / "STATUS.json"
        if status_path.exists():
            status_path.unlink()
        _atomic_publish(status_path, result)
        return result


_FORK_STATE: dict[str, Any] | None = None
_FORK_BACKEND: PermutationBackend | None = None


def _fork_edge_worker(edge_id: str) -> str:
    global _FORK_BACKEND
    if _FORK_STATE is None:
        raise RuntimeError("fork worker state was not initialized")
    plan = _FORK_STATE["plan"]
    if _FORK_BACKEND is None:
        _FORK_BACKEND = _backend_for_plan(plan, _FORK_STATE["root"])
    statistics = plan["statistics"]
    evidence, edge_input = _edge_evidence(
        edge_id,
        context=_FORK_STATE["context"],
        bound=_FORK_STATE["bound"],
        permutation_iterations=int(statistics["permutation_iterations"]),
        bootstrap_iterations=int(statistics["bootstrap_iterations"]),
        permutation_batch_size=int(statistics["permutation_batch_size"]),
        backend=_FORK_BACKEND,
    )
    payload = _checkpoint_payload(
        plan_path=_FORK_STATE["plan_path"],
        plan_sha256=_FORK_STATE["plan_sha"],
        plan=plan,
        edge_id=edge_id,
        evidence=evidence,
        edge_input=edge_input,
    )
    _atomic_publish(_FORK_STATE["root"] / "edges" / f"{edge_id}.json", payload)
    return edge_id


def run_workers(
    plan_path: str | Path,
    output_dir: str | Path,
    *,
    workers: int,
    max_new_edges: int | None = None,
) -> dict[str, Any]:
    """Load the compressed bundle once, then fork shared read-only workers.

    CPU probe/bootstrap work runs concurrently.  If the bounded CUDA backend
    is selected, only its permutation critical section is serialized through
    ``gpu-permutation.lock`` while CPU preparation overlaps.
    """

    if isinstance(workers, bool) or int(workers) <= 0:
        raise ValueError("workers must be positive")
    if max_new_edges is not None and (
        isinstance(max_new_edges, bool) or int(max_new_edges) <= 0
    ):
        raise ValueError("max_new_edges must be positive")
    workers = int(workers)
    resolved_plan, plan_sha, plan, bound = _verify_plan(plan_path)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    preflight_by_id = {
        str(row["edge_id"]): row for row in bound.preflight["edges"]
    }
    with (root / "workers.lock").open("a+b") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another shared-worker runner is active") from error

        completed: list[str] = []
        pending: list[str] = []
        for edge_id in plan["family"]["scheduled_edge_ids"]:
            checkpoint_path = root / "edges" / f"{edge_id}.json"
            if not checkpoint_path.is_file():
                pending.append(edge_id)
                continue
            restored = validate_checkpoint(
                checkpoint_path,
                plan_path=resolved_plan,
                plan_sha256=plan_sha,
                plan=plan,
                edge_id=edge_id,
            )
            checkpoint = _load_object(checkpoint_path)
            if (
                int(checkpoint["edge_input"]["accepted_memberships"])
                != int(preflight_by_id[edge_id]["accepted_memberships"])
                or not _support_matches_preflight(
                    restored, preflight_by_id[edge_id]
                )
            ):
                raise ValueError(f"resumed checkpoint/preflight drift: {edge_id}")
            completed.append(edge_id)

        statistics = plan["statistics"]
        work = {
            edge_id: _edge_work(
                preflight_by_id[edge_id],
                permutations=int(statistics["permutation_iterations"]),
                bootstrap=int(statistics["bootstrap_iterations"]),
            )
            for edge_id in pending
        }
        pending = sorted(pending, key=lambda edge_id: (-work[edge_id], edge_id))
        if max_new_edges is not None:
            pending = pending[: int(max_new_edges)]
        newly_completed: list[str] = []
        if pending:
            context = load_bundle_context(bound)
            global _FORK_STATE, _FORK_BACKEND
            _FORK_STATE = {
                "plan_path": resolved_plan,
                "plan_sha": plan_sha,
                "plan": plan,
                "bound": bound,
                "context": context,
                "root": root,
            }
            _FORK_BACKEND = None
            process_count = min(workers, len(pending))
            if process_count == 1:
                newly_completed = [_fork_edge_worker(edge_id) for edge_id in pending]
            else:
                fork = multiprocessing.get_context("fork")
                with fork.Pool(processes=process_count) as pool:
                    newly_completed = list(
                        pool.imap_unordered(_fork_edge_worker, pending, chunksize=1)
                    )
            _FORK_STATE = None
            _FORK_BACKEND = None

        completed_set = set(completed) | set(newly_completed)
        all_edges = list(plan["family"]["scheduled_edge_ids"])
        remaining = [edge_id for edge_id in all_edges if edge_id not in completed_set]
        result = {
            "schema_version": "pavir-route-confirm-v3-worker-status-v1",
            "protocol_name": PROTOCOL,
            "status": (
                "COMPLETE_BOUNDED_ALL_EDGE_EVIDENCE"
                if not remaining
                else "PARTIAL_BOUNDED_ALL_EDGE_EVIDENCE"
            ),
            "scope": {
                "confirmatory_route_authorized": False,
                "training_authorized": False,
            },
            "plan": _file_record(resolved_plan, plan_sha),
            "physical_workers": workers,
            "bundle_loads_in_parent": 1 if pending else 0,
            "scheduled_edges": len(all_edges),
            "completed_edges": len(completed_set),
            "newly_completed_edges": len(newly_completed),
            "remaining_edges": len(remaining),
        }
        status_path = root / "STATUS.json"
        if status_path.exists():
            status_path.unlink()
        _atomic_publish(status_path, result)
        return result


def assemble_family(
    graph_rows: Sequence[Mapping[str, Any]],
    preflight_rows: Sequence[Mapping[str, Any]],
    evidence_by_edge: Mapping[str, EdgeEvidence],
) -> dict[str, Any]:
    """Assemble a schedule-independent diagnostic family result."""

    graph_by_id = {str(row["edge_id"]): row for row in graph_rows}
    preflight_by_id = {str(row["edge_id"]): row for row in preflight_rows}
    if (
        not graph_by_id
        or len(graph_by_id) != len(graph_rows)
        or set(preflight_by_id) != set(graph_by_id)
    ):
        raise ValueError("graph and preflight must cover the same unique edge family")
    eligible = {
        edge_id
        for edge_id, row in preflight_by_id.items()
        if bool(row.get("support_passed")) and not bool(row.get("tone_only"))
    }
    if set(evidence_by_edge) != set(graph_by_id):
        raise ValueError("evidence must cover every graph edge exactly once")
    for edge_id, evidence in evidence_by_edge.items():
        if (
            evidence.edge_id != edge_id
            or evidence.tone_only != bool(graph_by_id[edge_id].get("tone_only"))
            or evidence.support.passed
            != bool(preflight_by_id[edge_id].get("support_passed"))
        ):
            raise ValueError(f"invalid edge evidence: {edge_id}")

    non_tone = sorted(
        edge_id
        for edge_id, row in graph_by_id.items()
        if not bool(row.get("tone_only"))
    )
    observed_p_e = {
        edge_id: evidence_by_edge[edge_id].p_e for edge_id in non_tone
    }
    observed_p_d = {
        edge_id: evidence_by_edge[edge_id].p_d for edge_id in non_tone
    }
    skipped = set(non_tone) - eligible
    if any(
        observed_p_e[edge_id] != 1.0 or observed_p_d[edge_id] != 1.0
        for edge_id in skipped
    ):
        raise ValueError("support-skipped non-tone edge must enter both families at p=1")
    q_e = complete_non_tone_bh(observed_p_e, non_tone)
    q_d = complete_non_tone_bh(observed_p_d, non_tone)
    full_p_e = {
        edge_id: float(observed_p_e.get(edge_id, 1.0)) for edge_id in non_tone
    }
    full_p_d = {
        edge_id: float(observed_p_d.get(edge_id, 1.0)) for edge_id in non_tone
    }
    by_e = benjamini_yekutieli(full_p_e)
    by_d = benjamini_yekutieli(full_p_d)

    rows: list[dict[str, Any]] = []
    for edge_id in sorted(graph_by_id):
        graph_row = graph_by_id[edge_id]
        preflight_row = preflight_by_id[edge_id]
        tone_only = bool(graph_row.get("tone_only"))
        support_passed = bool(preflight_row.get("support_passed"))
        if tone_only:
            evidence = evidence_by_edge[edge_id]
            decision = RouteDecision(
                edge_id=edge_id,
                route="D" if support_passed else "A",
                reason="D_tone" if support_passed else "A_support",
                q_e=None,
                q_d=None,
            )
            p_e = p_d = q_e_value = q_d_value = by_e_value = by_d_value = None
            evidence_sha = evidence_record_sha256(evidence)
            family_status = "tone_prior" if support_passed else "support_skipped"
        elif support_passed:
            evidence = evidence_by_edge[edge_id]
            decision = decide_route(
                evidence, q_e=q_e[edge_id], q_d=q_d[edge_id]
            )
            p_e, p_d = evidence.p_e, evidence.p_d
            q_e_value, q_d_value = q_e[edge_id], q_d[edge_id]
            by_e_value, by_d_value = by_e[edge_id], by_d[edge_id]
            evidence_sha = evidence_record_sha256(evidence)
            family_status = "computed"
        else:
            evidence = evidence_by_edge[edge_id]
            decision = RouteDecision(
                edge_id=edge_id,
                route="A",
                reason="A_support",
                q_e=q_e[edge_id],
                q_d=q_d[edge_id],
            )
            p_e = p_d = 1.0
            q_e_value, q_d_value = q_e[edge_id], q_d[edge_id]
            by_e_value, by_d_value = by_e[edge_id], by_d[edge_id]
            evidence_sha = evidence_record_sha256(evidence)
            family_status = "support_skipped_p1"
        rows.append(
            {
                "edge_id": edge_id,
                "tone_only": tone_only,
                "support_passed": support_passed,
                "family_status": family_status,
                "p_e": p_e,
                "p_d": p_d,
                "q_e_bh": q_e_value,
                "q_d_bh": q_d_value,
                "q_e_by_sensitivity": by_e_value,
                "q_d_by_sensitivity": by_d_value,
                "route": decision.route,
                "reason": decision.reason,
                "evidence_sha256": evidence_sha,
            }
        )
    return {
        "non_tone_family_edge_count": len(non_tone),
        "computed_non_tone_edge_count": len(eligible),
        "p1_support_skipped_non_tone_edge_count": len(set(non_tone) - eligible),
        "tone_only_edge_count": len(graph_by_id) - len(non_tone),
        "edges": rows,
    }


def merge_plan(
    plan_path: str | Path,
    output_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate all checkpoints and publish one complete bounded family."""

    resolved_plan, plan_sha, plan, bound = _verify_plan(plan_path)
    root = Path(output_dir).expanduser().resolve()
    evidence: dict[str, EdgeEvidence] = {}
    preflight_by_id = {
        str(row["edge_id"]): row for row in bound.preflight["edges"]
    }
    quality_hashes: set[str] = set()
    for edge_id in plan["family"]["scheduled_edge_ids"]:
        checkpoint_path = root / "edges" / f"{edge_id}.json"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing edge checkpoint: {edge_id}")
        evidence[edge_id] = validate_checkpoint(
            checkpoint_path,
            plan_path=resolved_plan,
            plan_sha256=plan_sha,
            plan=plan,
            edge_id=edge_id,
        )
        checkpoint = _load_object(checkpoint_path)
        edge_input = checkpoint["edge_input"]
        quality_hashes.add(str(edge_input["quality_mask_sha256"]))
        if (
            int(edge_input["accepted_memberships"])
            != int(preflight_by_id[edge_id]["accepted_memberships"])
            or not _support_matches_preflight(
                evidence[edge_id], preflight_by_id[edge_id]
            )
        ):
            raise ValueError(f"checkpoint/preflight mismatch: {edge_id}")
    if len(quality_hashes) != 1:
        raise ValueError("edge checkpoints disagree on the global quality mask")
    family = assemble_family(
        bound.graph["edges"],
        bound.preflight["edges"],
        evidence,
    )
    output = {
        "schema_version": FAMILY_SCHEMA,
        "protocol_name": PROTOCOL,
        "status": "COMPLETE_BOUNDED_DIAGNOSTIC_FAMILY_NO_AUTHORIZATION",
        "scope": {
            "family_inventory_complete": True,
            "confirmatory_route_authorized": False,
            "offline_artifacts_authorized": False,
            "training_authorized": False,
        },
        "plan": _file_record(resolved_plan, plan_sha),
        "inputs": plan["inputs"],
        "statistics": plan["statistics"],
        "quality_mask_sha256": next(iter(quality_hashes)),
        "multiple_testing": {
            "primary": "BH independently over every non-tone graph edge",
            "sensitivity": "BY independently over every non-tone graph edge",
            "support_skipped_non_tone_p_value": 1.0,
            "tone_only_edges_excluded_from_both_families": True,
        },
        "inventory": {
            key: family[key]
            for key in (
                "non_tone_family_edge_count",
                "computed_non_tone_edge_count",
                "p1_support_skipped_non_tone_edge_count",
                "tone_only_edge_count",
            )
        },
        "edges": family["edges"],
    }
    _atomic_publish(Path(output_path), output)
    return output


def plan_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": plan["status"],
        "statistics": plan["statistics"],
        "inventory": plan["inventory"],
        "shards": [
            {
                key: shard[key]
                for key in ("shard_index", "edge_count", "estimated_work_units")
            }
            for shard in plan["scheduling"]["shards"]
        ],
    }


def family_summary(family: Mapping[str, Any]) -> dict[str, Any]:
    route_counts: dict[str, int] = {"E": 0, "D": 0, "A": 0}
    for row in family["edges"]:
        route_counts[str(row["route"])] += 1
    return {
        "status": family["status"],
        "inventory": family["inventory"],
        "diagnostic_route_counts": route_counts,
        "training_authorized": False,
    }
