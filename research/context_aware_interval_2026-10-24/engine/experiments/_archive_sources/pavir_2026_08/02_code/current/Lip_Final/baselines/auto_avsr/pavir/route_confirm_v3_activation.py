"""Fail-closed activation of exact full RouteConfirm-v3 evidence.

This module is deliberately separate from the locked statistical core,
``route_confirm_v3_io``, and the bounded resource-canary runner.  It consumes
an independently published resource/topology PASS and can then:

* publish one immutable all-2,236-edge evidence plan;
* run or resume the exact locked CPU plan; and
* publish complete-family p-values plus BH/BY corrections.

The resource gate does *not* authorize E/D/A decisions, prototypes, offline
training artifacts, or training.  Accordingly, this module never emits any of
those products.  A later family/route-mass gate must consume its complete
evidence merge.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import canonical_json_bytes, sha256_file
from .route_confirm_v3 import (
    PRODUCTION_BOOTSTRAP_ITERATIONS,
    PRODUCTION_PERMUTATIONS,
    PROTOCOL,
    EdgeEvidence,
    benjamini_yekutieli,
    complete_non_tone_bh,
    edge_seed,
)
from .route_confirm_v3_io import (
    edge_evidence_from_record,
    evidence_record_sha256,
)
from .route_confirm_v3_runner import (
    BoundInputs,
    BundleContext,
    LockedCPUBackend,
    _atomic_publish,
    _edge_evidence,
    _edge_work,
    _file_record,
    _support_matches_preflight,
    authenticate_inputs,
    deterministic_lpt_shards,
    load_bundle_context,
)


RESOURCE_GATE_SCHEMA = "pavir-route-confirm-v3-resource-topology-gate-v1"
RESOURCE_GATE_STATUS = "PASS_EXACT_FULL_EVIDENCE_PLAN_LAUNCH_ONLY"
PLAN_SCHEMA = "pavir-route-confirm-v3-exact-full-evidence-plan-v1"
CHECKPOINT_SCHEMA = "pavir-route-confirm-v3-exact-edge-evidence-v1"
MERGE_SCHEMA = "pavir-route-confirm-v3-exact-family-evidence-merge-v1"

PLAN_STATUS = "READY_EXACT_FULL_LOCKED_CPU_EVIDENCE_ONLY"
CHECKPOINT_STATUS = "COMPLETE_EXACT_FULL_EDGE_EVIDENCE"
MERGE_STATUS = "COMPLETE_EXACT_FULL_EVIDENCE_BH_BY_PENDING_ROUTE_GATE"

EXPECTED_RESOURCE_GATE = Path(
    "/root/workspace/ops/pavir_route_confirm_v3_resource_gate_20260731/"
    "RESOURCE_TOPOLOGY_GATE.json"
)
EXPECTED_RESOURCE_GATE_SHA256: str | None = (
    "25594ca8541dab115256fa3a119d81a27b9ef3fc3a5d3e1223855e7553ea4ac9"
)
EXPECTED_GRAPH_EDGES = 2_236
EXPECTED_NON_TONE_EDGES = 2_029
EXPECTED_ELIGIBLE_NON_TONE_EDGES = 1_823
EXPECTED_TONE_ONLY_EDGES = 207
EXPECTED_SKIPPED_NON_TONE_EDGES = 206
PERMUTATION_BATCH_SIZE = 2_048
MINIMUM_HOST_MEMORY_BYTES = 180 * 2**30
MINIMUM_AVAILABLE_MEMORY_BYTES = 96 * 2**30
MAXIMUM_SWAP_USED_BYTES = 1 * 2**30
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ProductionAuthorization:
    gate_path: Path
    gate_sha256: str
    gate: Mapping[str, Any]
    bound: BoundInputs
    workers: int
    selected_quality_mask_sha256: str
    canary_runner_path: Path
    canary_runner_sha256: str
    canary_runner_module_path: Path
    canary_runner_module_sha256: str

    def gate_record(self) -> dict[str, Any]:
        return _file_record(self.gate_path, self.gate_sha256)


def _load_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {resolved}")
    return value


def _locked_record(
    record: object,
    *,
    context: str,
    expected_path: Path | None = None,
) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise ValueError(f"{context} file record is missing")
    raw_path = record.get("path")
    digest = str(record.get("sha256", "")).lower()
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not _HEX64.fullmatch(digest)
    ):
        raise ValueError(f"{context} file record is invalid")
    path = Path(raw_path).expanduser().resolve()
    if expected_path is not None and path != expected_path.expanduser().resolve():
        raise ValueError(f"{context} path mismatch")
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{context} is absent or mutated: {path}")
    return path, digest


def _assert_safe_scope(scope: object, *, context: str) -> None:
    expected = {
        "exact_full_evidence_plan_creation_authorized": True,
        "exact_full_evidence_plan_launch_authorized": True,
        "complete_routing_authorized": False,
        "route_decisions_authorized": False,
        "offline_artifacts_authorized": False,
        "prototype_publication_authorized": False,
        "training_authorized": False,
    }
    if not isinstance(scope, dict) or scope != expected:
        raise ValueError(f"{context} authorization scope is unsafe or unsupported")


def validate_resource_gate_contract(gate: Mapping[str, Any]) -> tuple[int, dict[str, str]]:
    """Validate authorization fields without performing filesystem IO."""

    _assert_safe_scope(gate.get("scope"), context="resource topology gate")
    decision = gate.get("decision")
    checks = gate.get("checks")
    if (
        gate.get("schema_version") != RESOURCE_GATE_SCHEMA
        or gate.get("protocol_name") != PROTOCOL
        or gate.get("status") != RESOURCE_GATE_STATUS
        or not isinstance(decision, dict)
        or decision.get("selected_permutation_backend") != "locked_cpu_numpy"
        or isinstance(decision.get("selected_physical_workers"), bool)
        or int(decision.get("selected_physical_workers", 0)) not in {12, 16}
        or int(decision.get("permutation_iterations", -1))
        != PRODUCTION_PERMUTATIONS
        or int(decision.get("bootstrap_iterations", -1))
        != PRODUCTION_BOOTSTRAP_ITERATIONS
        or decision.get("thread_environment") != THREAD_ENVIRONMENT
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("resource topology gate contract is not an exact CPU PASS")
    return int(decision["selected_physical_workers"]), dict(
        decision["thread_environment"]
    )


def validate_live_resources(
    workers: int,
    *,
    meminfo_path: str | Path = "/proc/meminfo",
) -> dict[str, int]:
    """Require the live host to satisfy the topology's conservative envelope."""

    if isinstance(workers, bool) or int(workers) not in {12, 16}:
        raise ValueError("live resource check received an unsupported topology")
    values: dict[str, int] = {}
    for line in Path(meminfo_path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.strip().split()
        if not fields:
            continue
        multiplier = 1_024 if len(fields) > 1 and fields[1] == "kB" else 1
        try:
            values[key] = int(fields[0]) * multiplier
        except ValueError:
            continue
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    if required - set(values):
        raise ValueError("live meminfo lacks required memory fields")
    swap_used = values["SwapTotal"] - values["SwapFree"]
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - Linux is the production target
        available_cpus = os.cpu_count() or 0
    if (
        values["MemTotal"] < MINIMUM_HOST_MEMORY_BYTES
        or values["MemAvailable"] < MINIMUM_AVAILABLE_MEMORY_BYTES
        or swap_used < 0
        or swap_used > MAXIMUM_SWAP_USED_BYTES
        or available_cpus < int(workers)
    ):
        raise RuntimeError(
            "live CPU/memory resources no longer satisfy the exact-run gate"
        )
    return {
        "available_cpus": available_cpus,
        "mem_total_bytes": values["MemTotal"],
        "mem_available_bytes": values["MemAvailable"],
        "swap_used_bytes": swap_used,
    }


def _record_matches(
    record: object,
    *,
    path: Path,
    digest: str,
    context: str,
) -> None:
    observed_path, observed_sha = _locked_record(
        record,
        context=context,
        expected_path=path,
    )
    if observed_path != path.resolve() or observed_sha != digest:
        raise ValueError(f"{context} differs from authenticated preflight")


def _selected_candidate(
    gate: Mapping[str, Any],
    *,
    workers: int,
) -> Mapping[str, Any]:
    inputs = gate.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("resource topology gate lacks input lineage")
    candidates = inputs.get("cpu_topology_candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("resource gate lacks both CPU topology candidates")
    matching = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and not isinstance(candidate.get("workers"), bool)
        and int(candidate.get("workers", -1)) == workers
    ]
    if len(matching) != 1:
        raise ValueError("resource gate selected CPU topology is ambiguous")
    selected = matching[0]
    summary = inputs.get("selected_cpu_topology")
    if (
        not isinstance(summary, dict)
        or int(summary.get("workers", -1)) != workers
        or str(summary.get("root", "")) != str(selected.get("root", ""))
        or float(summary.get("wall_seconds", -1))
        != float(selected.get("runtime", {}).get("wall_seconds", -2))
        or selected.get("backend") != "locked_cpu_numpy"
        or int(selected.get("permutation_iterations", -1))
        != PRODUCTION_PERMUTATIONS
        or int(selected.get("bootstrap_iterations", -1)) != 1_000
        or int(selected.get("checkpoint_count", -1)) != workers
    ):
        raise ValueError("resource gate selected topology summary is inconsistent")
    quality_sha = str(selected.get("quality_mask_sha256", ""))
    if not _HEX64.fullmatch(quality_sha):
        raise ValueError("selected CPU topology lacks a quality-mask digest")
    return selected


def authenticate_resource_gate(
    gate_path: str | Path = EXPECTED_RESOURCE_GATE,
) -> ProductionAuthorization:
    """Authenticate the PASS gate and every production-relevant binding."""

    resolved_gate = Path(gate_path).expanduser().resolve()
    if resolved_gate != EXPECTED_RESOURCE_GATE.resolve():
        raise ValueError(
            "production activation requires the canonical resource gate path"
        )
    if not resolved_gate.is_file():
        raise FileNotFoundError(
            f"resource topology PASS is absent; production remains blocked: "
            f"{resolved_gate}"
        )
    gate_sha = sha256_file(resolved_gate)
    if EXPECTED_RESOURCE_GATE_SHA256 is None:
        raise PermissionError(
            "activation implementation is not yet bound to a reviewed "
            "resource-gate SHA-256"
        )
    if gate_sha != EXPECTED_RESOURCE_GATE_SHA256:
        raise ValueError("canonical resource topology gate hash mismatch")
    gate = _load_object(resolved_gate)
    workers, thread_environment = validate_resource_gate_contract(gate)
    if thread_environment != THREAD_ENVIRONMENT:
        raise ValueError("resource gate thread contract drift")
    _locked_record(
        gate.get("implementation", {}).get("publisher"),
        context="resource-gate publisher",
    )
    inputs = gate["inputs"]
    preflight_path, _ = _locked_record(
        inputs.get("v3_full_preflight"),
        context="resource-gate v3 full preflight",
    )
    bound = authenticate_inputs(preflight_path)
    _record_matches(
        inputs.get("stable_graph"),
        path=bound.graph_path,
        digest=bound.graph_sha256,
        context="resource-gate stable graph",
    )
    _record_matches(
        inputs.get("full_bundle"),
        path=bound.features_path,
        digest=bound.features_sha256,
        context="resource-gate full bundle",
    )
    _record_matches(
        inputs.get("full_bundle_gate"),
        path=bound.bundle_gate_path,
        digest=bound.bundle_gate_sha256,
        context="resource-gate full bundle gate",
    )
    _record_matches(
        inputs.get("source_split_lock"),
        path=bound.source_split_lock_path,
        digest=bound.source_split_lock_sha256,
        context="resource-gate source split lock",
    )
    _record_matches(
        inputs.get("core_implementation_lock"),
        path=bound.core_lock_path,
        digest=bound.core_lock_sha256,
        context="resource-gate core lock",
    )
    _record_matches(
        inputs.get("io_implementation_lock"),
        path=bound.io_lock_path,
        digest=bound.io_lock_sha256,
        context="resource-gate IO lock",
    )

    selected = _selected_candidate(gate, workers=workers)
    plan_path, _ = _locked_record(
        selected.get("plan"),
        context="selected CPU canary plan",
    )
    selected_plan = _load_object(plan_path)
    implementation = selected_plan.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("selected CPU canary plan lacks implementation binding")
    runner_path, runner_sha = _locked_record(
        implementation.get("runner"),
        context="selected canary runner CLI",
    )
    runner_module_path, runner_module_sha = _locked_record(
        implementation.get("runner_module"),
        context="selected canary runner module",
        expected_path=Path(__file__).with_name("route_confirm_v3_runner.py"),
    )
    runtime = selected_plan.get("runtime_contract")
    statistics = selected_plan.get("statistics")
    if (
        not isinstance(runtime, dict)
        or runtime.get("thread_environment") != THREAD_ENVIRONMENT
        or runtime.get("cuda_backend_confirmatory_authorized") is not False
        or not isinstance(statistics, dict)
        or statistics.get("permutation_backend") != "locked_cpu_numpy"
        or int(statistics.get("permutation_iterations", -1))
        != PRODUCTION_PERMUTATIONS
        or int(statistics.get("permutation_batch_size", -1))
        != PERMUTATION_BATCH_SIZE
        or int(selected_plan.get("scheduling", {}).get("shard_count", -1))
        != workers
        or selected_plan.get("inputs") != bound.records()
        or selected_plan.get("family", {}).get("scheduled_edge_ids")
        != [
            str(row["edge_id"])
            for row in bound.preflight["edges"]
        ]
    ):
        raise ValueError("selected CPU canary implementation/runtime drift")
    return ProductionAuthorization(
        gate_path=resolved_gate,
        gate_sha256=gate_sha,
        gate=gate,
        bound=bound,
        workers=workers,
        selected_quality_mask_sha256=str(selected["quality_mask_sha256"]),
        canary_runner_path=runner_path,
        canary_runner_sha256=runner_sha,
        canary_runner_module_path=runner_module_path,
        canary_runner_module_sha256=runner_module_sha,
    )


def _family_inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    edge_ids = [str(row["edge_id"]) for row in rows]
    non_tone = sorted(
        str(row["edge_id"]) for row in rows if not bool(row.get("tone_only"))
    )
    eligible = sorted(
        str(row["edge_id"])
        for row in rows
        if bool(row.get("support_passed")) and not bool(row.get("tone_only"))
    )
    tone = sorted(
        str(row["edge_id"]) for row in rows if bool(row.get("tone_only"))
    )
    skipped = sorted(set(non_tone) - set(eligible))
    if (
        edge_ids != sorted(edge_ids)
        or len(edge_ids) != EXPECTED_GRAPH_EDGES
        or len(non_tone) != EXPECTED_NON_TONE_EDGES
        or len(eligible) != EXPECTED_ELIGIBLE_NON_TONE_EDGES
        or len(tone) != EXPECTED_TONE_ONLY_EDGES
        or len(skipped) != EXPECTED_SKIPPED_NON_TONE_EDGES
    ):
        raise ValueError("preflight family inventory differs from frozen v3 counts")
    return {
        "scheduled_edge_ids": edge_ids,
        "non_tone_edge_ids": non_tone,
        "expensive_support_eligible_non_tone_edge_ids": eligible,
        "p1_skipped_non_tone_edge_ids": skipped,
        "tone_only_edge_ids": tone,
    }


def build_exact_plan(
    authorization: ProductionAuthorization,
    *,
    activation_cli_path: str | Path,
) -> dict[str, Any]:
    """Build the only plan shape accepted by the production executor."""

    rows = authorization.bound.preflight["edges"]
    family = _family_inventory(rows)
    work = {
        str(row["edge_id"]): _edge_work(
            row,
            permutations=PRODUCTION_PERMUTATIONS,
            bootstrap=PRODUCTION_BOOTSTRAP_ITERATIONS,
        )
        for row in rows
    }
    assignments = deterministic_lpt_shards(work, authorization.workers)
    shards = [
        {
            "shard_index": index,
            "edge_count": len(edge_ids),
            "estimated_work_units": int(
                sum(work[edge_id] for edge_id in edge_ids)
            ),
            "edge_ids": list(edge_ids),
        }
        for index, edge_ids in enumerate(assignments)
    ]
    activation_cli = Path(activation_cli_path).expanduser().resolve()
    activation_module = Path(__file__).resolve()
    return {
        "schema_version": PLAN_SCHEMA,
        "protocol_name": PROTOCOL,
        "status": PLAN_STATUS,
        "scope": {
            "exact_full_evidence_only": True,
            "complete_routing_authorized": False,
            "route_decisions_authorized": False,
            "offline_artifacts_authorized": False,
            "prototype_publication_authorized": False,
            "training_authorized": False,
        },
        "resource_topology_gate": authorization.gate_record(),
        "inputs": authorization.bound.records(),
        "statistics": {
            "permutation_backend": "locked_cpu_numpy",
            "permutation_iterations": PRODUCTION_PERMUTATIONS,
            "bootstrap_iterations": PRODUCTION_BOOTSTRAP_ITERATIONS,
            "permutation_batch_size": PERMUTATION_BATCH_SIZE,
            "confirmatory_defaults_used": True,
        },
        "inventory": {
            "graph_edges": EXPECTED_GRAPH_EDGES,
            "scheduled_all_graph_edges": EXPECTED_GRAPH_EDGES,
            "non_tone_bh_family_edges": EXPECTED_NON_TONE_EDGES,
            "expensive_support_eligible_non_tone_edges":
                EXPECTED_ELIGIBLE_NON_TONE_EDGES,
            "p1_support_skipped_non_tone_edges":
                EXPECTED_SKIPPED_NON_TONE_EDGES,
            "tone_only_edges": EXPECTED_TONE_ONLY_EDGES,
        },
        "family": family,
        "scheduling": {
            "algorithm": "deterministic-lpt-v1",
            "work_model": (
                "confirm_occurrences*permutations + "
                "confirm_paired_sources*bootstrap + fit_occurrences"
            ),
            "physical_workers": authorization.workers,
            "shard_count": authorization.workers,
            "parent_bundle_loads": 1,
            "shards": shards,
        },
        "implementation": {
            "activation_cli": _file_record(activation_cli),
            "activation_module": _file_record(activation_module),
            "gate_bound_canary_runner":
                _file_record(
                    authorization.canary_runner_path,
                    authorization.canary_runner_sha256,
                ),
            "gate_bound_canary_runner_module":
                _file_record(
                    authorization.canary_runner_module_path,
                    authorization.canary_runner_module_sha256,
                ),
            "locked_core_modified": False,
            "locked_io_modified": False,
        },
        "runtime_contract": {
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "scipy", "scikit-learn")
            },
            "thread_environment": dict(THREAD_ENVIRONMENT),
            "cuda_backend_confirmatory_authorized": False,
        },
        "expected_quality_mask_sha256":
            authorization.selected_quality_mask_sha256,
    }


def publish_exact_plan(
    gate_path: str | Path,
    output_path: str | Path,
    *,
    activation_cli_path: str | Path,
) -> dict[str, Any]:
    authorization = authenticate_resource_gate(gate_path)
    validate_live_resources(authorization.workers)
    plan = build_exact_plan(
        authorization,
        activation_cli_path=activation_cli_path,
    )
    _atomic_publish(Path(output_path), plan)
    return plan


def verify_exact_plan(
    plan_path: str | Path,
) -> tuple[Path, str, dict[str, Any], ProductionAuthorization]:
    path = Path(plan_path).expanduser().resolve()
    digest = sha256_file(path)
    plan = _load_object(path)
    gate_path, gate_sha = _locked_record(
        plan.get("resource_topology_gate"),
        context="plan-bound resource topology gate",
        expected_path=EXPECTED_RESOURCE_GATE,
    )
    authorization = authenticate_resource_gate(gate_path)
    if gate_sha != authorization.gate_sha256:
        raise ValueError("plan binds a different resource topology gate")
    implementation = plan.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("exact evidence plan lacks implementation lineage")
    activation_cli_path, _ = _locked_record(
        implementation.get("activation_cli"),
        context="plan-bound activation CLI",
    )
    _locked_record(
        implementation.get("activation_module"),
        context="plan-bound activation module",
        expected_path=Path(__file__).resolve(),
    )
    expected = build_exact_plan(
        authorization,
        activation_cli_path=activation_cli_path,
    )
    if canonical_json_bytes(plan) != canonical_json_bytes(expected):
        raise ValueError("exact evidence plan differs from reconstructed contract")
    if any(
        os.environ.get(name) != value
        for name, value in THREAD_ENVIRONMENT.items()
    ):
        raise ValueError("BLAS/OpenMP environment differs from exact plan")
    for package, version in plan["runtime_contract"]["packages"].items():
        if importlib.metadata.version(package) != str(version):
            raise ValueError(f"runtime package drift: {package}")
    return path, digest, plan, authorization


def _checkpoint_payload(
    *,
    plan_path: Path,
    plan_sha256: str,
    plan: Mapping[str, Any],
    edge_id: str,
    evidence: EdgeEvidence,
    edge_input: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "protocol_name": PROTOCOL,
        "status": CHECKPOINT_STATUS,
        "scope": {
            "exact_edge_evidence_only": True,
            "family_merge_complete": False,
            "complete_routing_authorized": False,
            "route_decisions_authorized": False,
            "offline_artifacts_authorized": False,
            "prototype_publication_authorized": False,
            "training_authorized": False,
        },
        "plan": _file_record(plan_path, plan_sha256),
        "resource_topology_gate": plan["resource_topology_gate"],
        "inputs": plan["inputs"],
        "statistics": plan["statistics"],
        "edge_id": edge_id,
        "edge_input": dict(edge_input),
        "evidence": evidence.to_record(),
        "evidence_sha256": evidence_record_sha256(evidence),
    }


def _hex_digest(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _strict_partition_invariants(
    partition,
    *,
    accepted_memberships: int,
) -> bool:
    included = tuple(partition.included_indices)
    paired = tuple(partition.paired_sources)
    one_class = tuple(partition.one_class_sources)
    return (
        included == tuple(sorted(included))
        and len(included) == len(set(included))
        and all(0 <= index < accepted_memberships for index in included)
        and sum(partition.included_endpoint_occurrences) == len(included)
        and all(
            0 <= included_count <= raw_count
            for included_count, raw_count in zip(
                partition.included_endpoint_occurrences,
                partition.raw_endpoint_occurrences,
                strict=True,
            )
        )
        and paired == tuple(sorted(paired))
        and one_class == tuple(sorted(one_class))
        and len(paired) == len(set(paired))
        and len(one_class) == len(set(one_class))
        and set(paired).isdisjoint(one_class)
    )


def validate_exact_checkpoint(
    path: str | Path,
    *,
    plan_path: Path,
    plan_sha256: str,
    plan: Mapping[str, Any],
    authorization: ProductionAuthorization,
    edge_id: str,
) -> tuple[EdgeEvidence, Mapping[str, Any]]:
    checkpoint = _load_object(path)
    expected_scope = {
        "exact_edge_evidence_only": True,
        "family_merge_complete": False,
        "complete_routing_authorized": False,
        "route_decisions_authorized": False,
        "offline_artifacts_authorized": False,
        "prototype_publication_authorized": False,
        "training_authorized": False,
    }
    expected_keys = {
        "schema_version",
        "protocol_name",
        "status",
        "scope",
        "plan",
        "resource_topology_gate",
        "inputs",
        "statistics",
        "edge_id",
        "edge_input",
        "evidence",
        "evidence_sha256",
    }
    if (
        set(checkpoint) != expected_keys
        or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("protocol_name") != PROTOCOL
        or checkpoint.get("status") != CHECKPOINT_STATUS
        or checkpoint.get("scope") != expected_scope
        or checkpoint.get("edge_id") != edge_id
        or checkpoint.get("plan") != _file_record(plan_path, plan_sha256)
        or checkpoint.get("resource_topology_gate")
        != plan.get("resource_topology_gate")
        or checkpoint.get("inputs") != plan.get("inputs")
        or checkpoint.get("statistics") != plan.get("statistics")
    ):
        raise ValueError(f"exact edge checkpoint contract mismatch: {path}")
    raw_evidence = checkpoint.get("evidence")
    evidence = edge_evidence_from_record(raw_evidence)
    if canonical_json_bytes(raw_evidence) != canonical_json_bytes(
        evidence.to_record()
    ):
        raise ValueError(f"edge evidence contains unknown/coerced fields: {path}")
    preflight_by_id = {
        str(row["edge_id"]): row
        for row in authorization.bound.preflight["edges"]
    }
    preflight_row = preflight_by_id.get(edge_id)
    if preflight_row is None:
        raise ValueError(f"checkpoint edge is outside preflight: {edge_id}")
    edge_input = checkpoint.get("edge_input")
    expected_tone = bool(preflight_row["tone_only"])
    iterations = PRODUCTION_PERMUTATIONS
    expected_p = (
        None
        if evidence.permutation_exceedances is None
        else (1 + evidence.permutation_exceedances) / (iterations + 1)
    )
    accepted_memberships = (
        -1
        if not isinstance(edge_input, dict)
        or isinstance(edge_input.get("accepted_memberships"), bool)
        else int(edge_input.get("accepted_memberships", -1))
    )
    active_statistic = (
        not expected_tone and evidence.support.passed and evidence.converged
    )
    if (
        not isinstance(edge_input, dict)
        or set(edge_input)
        != {
            "accepted_memberships",
            "accepted_membership_stream_sha256",
            "quality_mask_sha256",
        }
        or isinstance(edge_input.get("accepted_memberships"), bool)
        or int(edge_input.get("accepted_memberships", -1))
        != int(preflight_row["accepted_memberships"])
        or not _hex_digest(edge_input.get("accepted_membership_stream_sha256"))
        or edge_input.get("quality_mask_sha256")
        != authorization.selected_quality_mask_sha256
        or evidence.edge_id != edge_id
        or evidence.tone_only != expected_tone
        or evidence.seed != edge_seed(edge_id)
        or tuple(evidence.source_auc_ids) != tuple(sorted(evidence.source_auc_ids))
        or len(evidence.source_auc_ids) != len(set(evidence.source_auc_ids))
        or not _strict_partition_invariants(
            evidence.support.fit,
            accepted_memberships=accepted_memberships,
        )
        or not _strict_partition_invariants(
            evidence.support.confirm,
            accepted_memberships=accepted_memberships,
        )
        or (
            active_statistic
            and tuple(evidence.source_auc_ids)
            != tuple(evidence.support.confirm.paired_sources)
        )
        or (
            active_statistic
            and (
                evidence.theta is None
                or evidence.ci95 is None
                or evidence.permutation_exceedances is None
                or not evidence.source_aucs
            )
        )
        or (
            evidence.theta is not None
            and (
                not evidence.source_aucs
                or not math.isclose(
                    evidence.theta,
                    float(np.mean(evidence.source_aucs, dtype=np.float64)),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
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
                or evidence.source_aucs
                or evidence.source_auc_ids
                or evidence.ci95 is not None
                or evidence.permutation_exceedances is not None
            )
        )
        or not _support_matches_preflight(evidence, preflight_row)
        or checkpoint.get("evidence_sha256")
        != evidence_record_sha256(evidence)
    ):
        raise ValueError(f"exact edge checkpoint evidence mismatch: {path}")
    return evidence, edge_input


def _derive_edge_input(
    edge_id: str,
    *,
    context: BundleContext,
) -> dict[str, Any]:
    edge_index = int(np.searchsorted(context.edge_dictionary, edge_id))
    if (
        edge_index >= len(context.edge_dictionary)
        or str(context.edge_dictionary[edge_index]) != edge_id
    ):
        raise ValueError(f"edge is absent from authenticated bundle: {edge_id}")
    start, end = int(context.starts[edge_index]), int(context.ends[edge_index])
    positions = context.membership_positions[start:end]
    endpoint_codes = context.membership_endpoints[start:end]
    accepted = context.quality_accepted[positions]
    positions = positions[accepted]
    endpoint_codes = endpoint_codes[accepted]
    stream = np.empty(
        len(positions),
        dtype=np.dtype([("position", "<u4"), ("endpoint", "u1")]),
    )
    stream["position"] = positions
    stream["endpoint"] = endpoint_codes
    return {
        "accepted_memberships": len(positions),
        "accepted_membership_stream_sha256": hashlib.sha256(
            stream.tobytes()
        ).hexdigest(),
        "quality_mask_sha256": context.quality_mask_sha256,
    }


_FORK_STATE: dict[str, Any] | None = None


def _fork_worker(edge_id: str) -> str:
    if _FORK_STATE is None:
        raise RuntimeError("production fork state was not initialized")
    plan = _FORK_STATE["plan"]
    evidence, edge_input = _edge_evidence(
        edge_id,
        context=_FORK_STATE["context"],
        bound=_FORK_STATE["authorization"].bound,
        permutation_iterations=PRODUCTION_PERMUTATIONS,
        bootstrap_iterations=PRODUCTION_BOOTSTRAP_ITERATIONS,
        permutation_batch_size=PERMUTATION_BATCH_SIZE,
        backend=LockedCPUBackend(),
    )
    payload = _checkpoint_payload(
        plan_path=_FORK_STATE["plan_path"],
        plan_sha256=_FORK_STATE["plan_sha256"],
        plan=plan,
        edge_id=edge_id,
        evidence=evidence,
        edge_input=edge_input,
    )
    _atomic_publish(
        _FORK_STATE["output_root"] / "edges" / f"{edge_id}.json",
        payload,
    )
    return edge_id


def run_exact_plan(
    plan_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run every missing edge with the exact gate-selected CPU topology."""

    resolved_plan, plan_sha, plan, authorization = verify_exact_plan(plan_path)
    live_resources = validate_live_resources(authorization.workers)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    edge_root = root / "edges"
    scheduled = list(plan["family"]["scheduled_edge_ids"])
    expected_names = {f"{edge_id}.json" for edge_id in scheduled}
    if edge_root.exists():
        extras = sorted(
            path.name
            for path in edge_root.glob("*.json")
            if path.name not in expected_names
        )
        if extras:
            raise ValueError(f"production evidence directory has extra atoms: {extras[0]}")
    with (root / "exact-workers.lock").open("a+b") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another exact production runner is active") from error
        completed: list[str] = []
        pending: list[str] = []
        for edge_id in scheduled:
            checkpoint_path = edge_root / f"{edge_id}.json"
            if not checkpoint_path.is_file():
                pending.append(edge_id)
                continue
            validate_exact_checkpoint(
                checkpoint_path,
                plan_path=resolved_plan,
                plan_sha256=plan_sha,
                plan=plan,
                authorization=authorization,
                edge_id=edge_id,
            )
            completed.append(edge_id)

        preflight_by_id = {
            str(row["edge_id"]): row
            for row in authorization.bound.preflight["edges"]
        }
        pending.sort(
            key=lambda edge_id: (
                -_edge_work(
                    preflight_by_id[edge_id],
                    permutations=PRODUCTION_PERMUTATIONS,
                    bootstrap=PRODUCTION_BOOTSTRAP_ITERATIONS,
                ),
                edge_id,
            )
        )
        newly_completed: list[str] = []
        if pending:
            context = load_bundle_context(authorization.bound)
            if (
                context.quality_mask_sha256
                != authorization.selected_quality_mask_sha256
            ):
                raise ValueError(
                    "full bundle quality mask differs from selected resource canary"
                )
            global _FORK_STATE
            _FORK_STATE = {
                "plan_path": resolved_plan,
                "plan_sha256": plan_sha,
                "plan": plan,
                "authorization": authorization,
                "context": context,
                "output_root": root,
            }
            try:
                fork = multiprocessing.get_context("fork")
                with fork.Pool(processes=authorization.workers) as pool:
                    newly_completed = list(
                        pool.imap_unordered(_fork_worker, pending, chunksize=1)
                    )
            finally:
                _FORK_STATE = None

        completed_set = set(completed) | set(newly_completed)
        if completed_set != set(scheduled):
            raise RuntimeError("exact production runner returned incomplete evidence")
        status = {
            "schema_version": "pavir-route-confirm-v3-exact-run-status-v1",
            "protocol_name": PROTOCOL,
            "status": "COMPLETE_EXACT_FULL_EDGE_EVIDENCE_PENDING_FAMILY_MERGE",
            "scope": {
                "full_edge_evidence_complete": True,
                "complete_routing_authorized": False,
                "route_decisions_authorized": False,
                "offline_artifacts_authorized": False,
                "prototype_publication_authorized": False,
                "training_authorized": False,
            },
            "plan": _file_record(resolved_plan, plan_sha),
            "resource_topology_gate": authorization.gate_record(),
            "physical_workers": authorization.workers,
            "parent_bundle_loads": 1 if pending else 0,
            "scheduled_edges": len(scheduled),
            "completed_edges": len(completed_set),
            "resumed_edges": len(completed),
            "newly_completed_edges": len(newly_completed),
            "remaining_edges": 0,
            "launch_resource_check": live_resources,
        }
        complete_path = root / "COMPLETE.json"
        if complete_path.is_file():
            existing = _load_object(complete_path)
            stable_fields = (
                "schema_version",
                "protocol_name",
                "status",
                "scope",
                "plan",
                "resource_topology_gate",
                "physical_workers",
                "scheduled_edges",
                "completed_edges",
                "remaining_edges",
            )
            if any(existing.get(field) != status.get(field) for field in stable_fields):
                raise ValueError("existing exact completion record is inconsistent")
            return existing
        _atomic_publish(complete_path, status)
        return status


def assemble_exact_evidence_family(
    graph_rows: Sequence[Mapping[str, Any]],
    preflight_rows: Sequence[Mapping[str, Any]],
    evidence_by_edge: Mapping[str, EdgeEvidence],
) -> dict[str, Any]:
    """Compute complete p/BH/BY families without assigning E/D/A routes."""

    graph_by_id = {str(row["edge_id"]): row for row in graph_rows}
    preflight_by_id = {str(row["edge_id"]): row for row in preflight_rows}
    if (
        len(graph_by_id) != EXPECTED_GRAPH_EDGES
        or set(graph_by_id) != set(preflight_by_id)
        or set(evidence_by_edge) != set(graph_by_id)
    ):
        raise ValueError("exact family requires every unique graph edge")
    non_tone = sorted(
        edge_id
        for edge_id, row in graph_by_id.items()
        if not bool(row.get("tone_only"))
    )
    if len(non_tone) != EXPECTED_NON_TONE_EDGES:
        raise ValueError("exact non-tone family size drift")
    p_e = {edge_id: evidence_by_edge[edge_id].p_e for edge_id in non_tone}
    p_d = {edge_id: evidence_by_edge[edge_id].p_d for edge_id in non_tone}
    for edge_id in non_tone:
        expected_support = bool(preflight_by_id[edge_id]["support_passed"])
        evidence = evidence_by_edge[edge_id]
        if (
            evidence.edge_id != edge_id
            or evidence.tone_only
            or evidence.support.passed != expected_support
            or (
                not expected_support
                and (evidence.p_e != 1.0 or evidence.p_d != 1.0)
            )
        ):
            raise ValueError(f"invalid evidence family member: {edge_id}")
    q_e = complete_non_tone_bh(p_e, non_tone)
    q_d = complete_non_tone_bh(p_d, non_tone)
    by_e = benjamini_yekutieli(p_e)
    by_d = benjamini_yekutieli(p_d)
    rows: list[dict[str, Any]] = []
    for edge_id in sorted(graph_by_id):
        evidence = evidence_by_edge[edge_id]
        tone = bool(graph_by_id[edge_id]["tone_only"])
        support = bool(preflight_by_id[edge_id]["support_passed"])
        if evidence.tone_only != tone or evidence.support.passed != support:
            raise ValueError(f"tone/support family drift: {edge_id}")
        rows.append(
            {
                "edge_id": edge_id,
                "tone_only": tone,
                "support_passed": support,
                "family_status": (
                    "tone_prior_pending_route_gate"
                    if tone
                    else "computed"
                    if support
                    else "support_skipped_p1"
                ),
                "p_e": None if tone else p_e[edge_id],
                "p_d": None if tone else p_d[edge_id],
                "q_e_bh": None if tone else q_e[edge_id],
                "q_d_bh": None if tone else q_d[edge_id],
                "q_e_by_sensitivity": None if tone else by_e[edge_id],
                "q_d_by_sensitivity": None if tone else by_d[edge_id],
                "evidence_sha256": evidence_record_sha256(evidence),
            }
        )
    return {
        "non_tone_family_edge_count": len(non_tone),
        "computed_non_tone_edge_count": int(
            sum(
                bool(preflight_by_id[edge_id]["support_passed"])
                for edge_id in non_tone
            )
        ),
        "p1_support_skipped_non_tone_edge_count": int(
            sum(
                not bool(preflight_by_id[edge_id]["support_passed"])
                for edge_id in non_tone
            )
        ),
        "tone_only_edge_count": EXPECTED_TONE_ONLY_EDGES,
        "edges": rows,
    }


def merge_exact_evidence(
    plan_path: str | Path,
    output_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Publish complete evidence families; route decisions remain prohibited."""

    resolved_plan, plan_sha, plan, authorization = verify_exact_plan(plan_path)
    root = Path(output_dir).expanduser().resolve()
    complete_path = root / "COMPLETE.json"
    if not complete_path.is_file():
        raise FileNotFoundError("exact edge run has no immutable completion record")
    completion = _load_object(complete_path)
    if (
        completion.get("status")
        != "COMPLETE_EXACT_FULL_EDGE_EVIDENCE_PENDING_FAMILY_MERGE"
        or completion.get("plan") != _file_record(resolved_plan, plan_sha)
        or int(completion.get("completed_edges", -1)) != EXPECTED_GRAPH_EDGES
        or int(completion.get("remaining_edges", -1)) != 0
    ):
        raise ValueError("exact edge completion record is invalid")
    scheduled = list(plan["family"]["scheduled_edge_ids"])
    expected_paths = {
        (root / "edges" / f"{edge_id}.json").resolve() for edge_id in scheduled
    }
    observed_paths = {
        path.resolve() for path in (root / "edges").glob("e-*.json")
    }
    if observed_paths != expected_paths:
        raise ValueError("exact checkpoint directory has missing or extra edge atoms")

    context = load_bundle_context(authorization.bound)
    if context.quality_mask_sha256 != authorization.selected_quality_mask_sha256:
        raise ValueError("merge quality mask differs from resource gate")
    evidence: dict[str, EdgeEvidence] = {}
    checkpoint_inventory: list[dict[str, Any]] = []
    for edge_id in scheduled:
        checkpoint_path = root / "edges" / f"{edge_id}.json"
        restored, edge_input = validate_exact_checkpoint(
            checkpoint_path,
            plan_path=resolved_plan,
            plan_sha256=plan_sha,
            plan=plan,
            authorization=authorization,
            edge_id=edge_id,
        )
        if dict(edge_input) != _derive_edge_input(edge_id, context=context):
            raise ValueError(f"checkpoint membership stream drift: {edge_id}")
        evidence[edge_id] = restored
        checkpoint_inventory.append(
            {
                "edge_id": edge_id,
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
                "evidence_sha256": evidence_record_sha256(restored),
            }
        )
    if sha256_file(authorization.bound.features_path) != authorization.bound.features_sha256:
        raise ValueError("full bundle changed while exact family was being merged")
    family = assemble_exact_evidence_family(
        authorization.bound.graph["edges"],
        authorization.bound.preflight["edges"],
        evidence,
    )
    inventory_bytes = canonical_json_bytes(checkpoint_inventory)
    output = {
        "schema_version": MERGE_SCHEMA,
        "protocol_name": PROTOCOL,
        "status": MERGE_STATUS,
        "scope": {
            "full_edge_evidence_complete": True,
            "bh_by_family_statistics_complete": True,
            "complete_routing_authorized": False,
            "route_decisions_authorized": False,
            "offline_artifacts_authorized": False,
            "prototype_publication_authorized": False,
            "training_authorized": False,
        },
        "plan": _file_record(resolved_plan, plan_sha),
        "resource_topology_gate": authorization.gate_record(),
        "inputs": plan["inputs"],
        "statistics": plan["statistics"],
        "quality_mask_sha256": context.quality_mask_sha256,
        "checkpoint_inventory": {
            "count": len(checkpoint_inventory),
            "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        },
        "multiple_testing": {
            "primary": "BH independently over all 2029 non-tone graph edges",
            "sensitivity": "BY independently over all 2029 non-tone graph edges",
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
        "required_next_gate": (
            "independent complete-evidence/BH-BY audit and route-mass "
            "confirmation before E/D/A publication, prototypes, or training"
        ),
    }
    _atomic_publish(Path(output_path), output)
    return output


def plan_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": plan["status"],
        "scope": plan["scope"],
        "statistics": plan["statistics"],
        "workers": plan["scheduling"]["physical_workers"],
        "inventory": plan["inventory"],
        "production_launched": False,
        "training_authorized": False,
    }


def merge_summary(merge: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": merge["status"],
        "scope": merge["scope"],
        "inventory": merge["inventory"],
        "route_decisions_authorized": False,
        "training_authorized": False,
    }
