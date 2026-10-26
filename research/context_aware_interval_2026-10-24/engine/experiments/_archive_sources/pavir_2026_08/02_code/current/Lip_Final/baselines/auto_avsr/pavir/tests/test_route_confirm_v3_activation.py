from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from pavir.artifacts import canonical_json_bytes, sha256_file
from pavir.route_confirm_v3 import (
    EdgeEvidence,
    EdgeSupport,
    PartitionSupport,
    edge_seed,
)
from pavir.route_confirm_v3_activation import (
    CHECKPOINT_STATUS,
    EXPECTED_ELIGIBLE_NON_TONE_EDGES,
    EXPECTED_GRAPH_EDGES,
    EXPECTED_NON_TONE_EDGES,
    EXPECTED_SKIPPED_NON_TONE_EDGES,
    EXPECTED_TONE_ONLY_EDGES,
    PLAN_STATUS,
    ProductionAuthorization,
    _checkpoint_payload,
    assemble_exact_evidence_family,
    build_exact_plan,
    validate_exact_checkpoint,
    validate_live_resources,
    validate_resource_gate_contract,
)
from pavir.route_confirm_v3_runner import BoundInputs


def _resource_gate(*, workers: int = 16) -> dict:
    return {
        "schema_version": "pavir-route-confirm-v3-resource-topology-gate-v1",
        "protocol_name": "PAViR-RouteConfirm-v3",
        "status": "PASS_EXACT_FULL_EVIDENCE_PLAN_LAUNCH_ONLY",
        "decision": {
            "selected_permutation_backend": "locked_cpu_numpy",
            "selected_physical_workers": workers,
            "permutation_iterations": 199_999,
            "bootstrap_iterations": 10_000,
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
        },
        "checks": {
            "cpu16_canary_passed": True,
            "cross_topology_evidence_equal": True,
        },
        "scope": {
            "exact_full_evidence_plan_creation_authorized": True,
            "exact_full_evidence_plan_launch_authorized": True,
            "complete_routing_authorized": False,
            "route_decisions_authorized": False,
            "offline_artifacts_authorized": False,
            "prototype_publication_authorized": False,
            "training_authorized": False,
        },
    }


@pytest.mark.parametrize("workers", (12, 16))
def test_resource_gate_contract_accepts_only_locked_cpu_exact_defaults(
    workers: int,
) -> None:
    observed_workers, environment = validate_resource_gate_contract(
        _resource_gate(workers=workers)
    )
    assert observed_workers == workers
    assert set(environment.values()) == {"1"}


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("decision", "selected_permutation_backend"), "cuda_numpy_rng_torch_topk_v1"),
        (("decision", "selected_physical_workers"), 15),
        (("decision", "permutation_iterations"), 199_998),
        (("decision", "bootstrap_iterations"), 9_999),
        (("decision", "thread_environment", "OMP_NUM_THREADS"), "2"),
        (("scope", "route_decisions_authorized"), True),
        (("scope", "offline_artifacts_authorized"), True),
        (("scope", "training_authorized"), True),
        (("checks", "cpu16_canary_passed"), False),
    ),
)
def test_resource_gate_contract_fails_closed_on_tamper(path, value) -> None:
    gate = _resource_gate()
    cursor = gate
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError):
        validate_resource_gate_contract(gate)


def test_live_resource_gate_checks_cpu_memory_and_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "\n".join(
            (
                "MemTotal:       196608000 kB",
                "MemAvailable:   131072000 kB",
                "SwapTotal:       1048576 kB",
                "SwapFree:        1048576 kB",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "pavir.route_confirm_v3_activation.os.sched_getaffinity",
        lambda _pid: set(range(16)),
    )
    observed = validate_live_resources(16, meminfo_path=meminfo)
    assert observed["available_cpus"] == 16
    assert observed["swap_used_bytes"] == 0

    meminfo.write_text(
        "\n".join(
            (
                "MemTotal:       196608000 kB",
                "MemAvailable:    65536000 kB",
                "SwapTotal:       2097152 kB",
                "SwapFree:              0 kB",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="no longer satisfy"):
        validate_live_resources(16, meminfo_path=meminfo)


def _preflight_rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(EXPECTED_GRAPH_EDGES):
        edge_id = f"e-{index:020x}"
        tone = index >= EXPECTED_NON_TONE_EDGES
        support = (
            index < EXPECTED_ELIGIBLE_NON_TONE_EDGES
            or (
                tone
                and index
                < EXPECTED_NON_TONE_EDGES + 147
            )
        )
        rows.append(
            {
                "edge_id": edge_id,
                "tone_only": tone,
                "support_passed": support,
                "accepted_memberships": 8,
                "fit": {
                    "included_endpoint_occurrences": {
                        "target": 4,
                        "confusion": 4,
                    },
                    "paired_source_count": 2,
                },
                "confirm": {
                    "included_endpoint_occurrences": {
                        "target": 4,
                        "confusion": 4,
                    },
                    "paired_source_count": 2,
                },
            }
        )
    assert sum(not row["tone_only"] for row in rows) == EXPECTED_NON_TONE_EDGES
    assert sum(row["tone_only"] for row in rows) == EXPECTED_TONE_ONLY_EDGES
    assert (
        sum(row["support_passed"] and not row["tone_only"] for row in rows)
        == EXPECTED_ELIGIBLE_NON_TONE_EDGES
    )
    return rows


def _bound_inputs(tmp_path: Path, rows: list[dict]) -> BoundInputs:
    paths = {}
    for name in (
        "preflight",
        "graph",
        "features",
        "bundle_gate",
        "source",
        "core",
        "io",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"fixture":"{name}"}}\n', encoding="utf-8")
        paths[name] = path
    graph = {
        "edges": [
            {"edge_id": row["edge_id"], "tone_only": row["tone_only"]}
            for row in rows
        ]
    }
    preflight = {"edges": rows}
    return BoundInputs(
        preflight_path=paths["preflight"],
        preflight_sha256=sha256_file(paths["preflight"]),
        preflight=preflight,
        graph_path=paths["graph"],
        graph_sha256=sha256_file(paths["graph"]),
        graph=graph,
        features_path=paths["features"],
        features_sha256=sha256_file(paths["features"]),
        bundle_gate_path=paths["bundle_gate"],
        bundle_gate_sha256=sha256_file(paths["bundle_gate"]),
        source_split_lock_path=paths["source"],
        source_split_lock_sha256=sha256_file(paths["source"]),
        core_lock_path=paths["core"],
        core_lock_sha256=sha256_file(paths["core"]),
        io_lock_path=paths["io"],
        io_lock_sha256=sha256_file(paths["io"]),
    )


def _authorization(
    tmp_path: Path,
    rows: list[dict],
    *,
    workers: int = 16,
) -> ProductionAuthorization:
    gate_path = tmp_path / "gate.json"
    gate_path.write_bytes(canonical_json_bytes(_resource_gate(workers=workers)))
    canary_cli = tmp_path / "bounded-cli.py"
    canary_module = tmp_path / "bounded-module.py"
    canary_cli.write_text("# fixture\n", encoding="utf-8")
    canary_module.write_text("# fixture\n", encoding="utf-8")
    return ProductionAuthorization(
        gate_path=gate_path,
        gate_sha256=sha256_file(gate_path),
        gate=_resource_gate(workers=workers),
        bound=_bound_inputs(tmp_path, rows),
        workers=workers,
        selected_quality_mask_sha256="a" * 64,
        canary_runner_path=canary_cli,
        canary_runner_sha256=sha256_file(canary_cli),
        canary_runner_module_path=canary_module,
        canary_runner_module_sha256=sha256_file(canary_module),
    )


def test_exact_plan_has_no_runtime_override_or_downstream_authorization(
    tmp_path: Path,
) -> None:
    rows = _preflight_rows()
    authorization = _authorization(tmp_path, rows)
    cli = tmp_path / "activation-cli.py"
    cli.write_text("# fixture\n", encoding="utf-8")
    plan = build_exact_plan(authorization, activation_cli_path=cli)

    assert plan["status"] == PLAN_STATUS
    assert plan["statistics"] == {
        "permutation_backend": "locked_cpu_numpy",
        "permutation_iterations": 199_999,
        "bootstrap_iterations": 10_000,
        "permutation_batch_size": 2_048,
        "confirmatory_defaults_used": True,
    }
    assert plan["scheduling"]["physical_workers"] == 16
    assert plan["scheduling"]["parent_bundle_loads"] == 1
    assert len(plan["scheduling"]["shards"]) == 16
    scheduled = [
        edge_id
        for shard in plan["scheduling"]["shards"]
        for edge_id in shard["edge_ids"]
    ]
    assert len(scheduled) == EXPECTED_GRAPH_EDGES
    assert len(set(scheduled)) == EXPECTED_GRAPH_EDGES
    assert all(
        value is False
        for key, value in plan["scope"].items()
        if key != "exact_full_evidence_only"
    )


def _partition(name: str, *, passed: bool) -> PartitionSupport:
    failures = () if passed else (f"{name}_paired_sources",)
    return PartitionSupport(
        partition=name,
        observed_source_count=2,
        paired_sources=("s1", "s2") if passed else (),
        one_class_sources=(),
        raw_endpoint_occurrences=(4, 4),
        included_endpoint_occurrences=(4, 4) if passed else (0, 0),
        maximum_source_endpoint_fractions=(0.5, 0.5)
        if passed
        else (1.0, 1.0),
        included_indices=tuple(range(8)) if passed else (),
        passed=passed,
        failures=failures,
    )


def _evidence(
    edge_id: str,
    *,
    tone: bool,
    support: bool,
    theta: float = 0.5,
    p_e: float = 1.0,
    p_d: float = 1.0,
    exceedances: int | None = None,
) -> EdgeEvidence:
    fit = _partition("fit", passed=support)
    confirm = _partition("confirm", passed=support)
    active = support and not tone
    return EdgeEvidence(
        edge_id=edge_id,
        tone_only=tone,
        support=EdgeSupport(
            fit=fit,
            confirm=confirm,
            passed=support,
            failures=fit.failures + confirm.failures,
        ),
        converged=True,
        seed=edge_seed(edge_id),
        theta=theta if active else None,
        source_aucs=(theta - 0.01, theta + 0.01) if active else (),
        source_auc_ids=("s1", "s2") if active else (),
        ci95=(max(0.0, theta - 0.02), min(1.0, theta + 0.02))
        if active
        else None,
        p_e=p_e if active else 1.0,
        p_d=p_d if active else 1.0,
        permutation_exceedances=exceedances if active else None,
        warning_messages=(),
    )


def _support_preflight_row(edge_id: str) -> dict:
    return {
        "edge_id": edge_id,
        "tone_only": False,
        "support_passed": True,
        "accepted_memberships": 8,
        "failures": [],
        "fit": {
            "raw_endpoint_occurrences": {"target": 4, "confusion": 4},
            "included_endpoint_occurrences": {"target": 4, "confusion": 4},
            "paired_source_count": 2,
            "maximum_source_endpoint_fractions": {
                "target": 0.5,
                "confusion": 0.5,
            },
            "passed": True,
            "failures": [],
        },
        "confirm": {
            "raw_endpoint_occurrences": {"target": 4, "confusion": 4},
            "included_endpoint_occurrences": {"target": 4, "confusion": 4},
            "paired_source_count": 2,
            "maximum_source_endpoint_fractions": {
                "target": 0.5,
                "confusion": 0.5,
            },
            "passed": True,
            "failures": [],
        },
    }


def test_exact_checkpoint_rejects_p_grid_backend_and_quality_tamper(
    tmp_path: Path,
) -> None:
    edge_id = "e-00000000000000000000"
    row = _support_preflight_row(edge_id)
    authorization = _authorization(tmp_path, [row])
    # The checkpoint validator itself is edge-local; replace the fixture
    # preflight without invoking the full-family plan builder.
    authorization = replace(
        authorization,
        bound=replace(
            authorization.bound,
            preflight={"edges": [row]},
            graph={"edges": [{"edge_id": edge_id, "tone_only": False}]},
        ),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    plan = {
        "resource_topology_gate": authorization.gate_record(),
        "inputs": authorization.bound.records(),
        "statistics": {
            "permutation_backend": "locked_cpu_numpy",
            "permutation_iterations": 199_999,
            "bootstrap_iterations": 10_000,
            "permutation_batch_size": 2_048,
            "confirmatory_defaults_used": True,
        },
    }
    evidence = _evidence(
        edge_id,
        tone=False,
        support=True,
        theta=0.5,
        p_e=1 / 200_000,
        p_d=0.5,
        exceedances=0,
    )
    payload = _checkpoint_payload(
        plan_path=plan_path.resolve(),
        plan_sha256="b" * 64,
        plan=plan,
        edge_id=edge_id,
        evidence=evidence,
        edge_input={
            "accepted_memberships": 8,
            "accepted_membership_stream_sha256": "c" * 64,
            "quality_mask_sha256": "a" * 64,
        },
    )
    assert payload["status"] == CHECKPOINT_STATUS
    path = tmp_path / "edge.json"
    path.write_bytes(canonical_json_bytes(payload))
    restored, _ = validate_exact_checkpoint(
        path,
        plan_path=plan_path.resolve(),
        plan_sha256="b" * 64,
        plan=plan,
        authorization=authorization,
        edge_id=edge_id,
    )
    assert restored == evidence

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["evidence"]["p_e"] = 0.01
    tampered["evidence_sha256"] = "d" * 64
    path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="evidence mismatch"):
        validate_exact_checkpoint(
            path,
            plan_path=plan_path.resolve(),
            plan_sha256="b" * 64,
            plan=plan,
            authorization=authorization,
            edge_id=edge_id,
        )


def test_exact_family_has_complete_bh_by_but_no_route_labels() -> None:
    preflight = _preflight_rows()
    graph = [
        {"edge_id": row["edge_id"], "tone_only": row["tone_only"]}
        for row in preflight
    ]
    evidence = {}
    for row in preflight:
        edge_id = row["edge_id"]
        active = row["support_passed"] and not row["tone_only"]
        evidence[edge_id] = _evidence(
            edge_id,
            tone=row["tone_only"],
            support=row["support_passed"],
            theta=0.7 if active else 0.5,
            p_e=0.01 if active else 1.0,
            p_d=0.8 if active else 1.0,
            exceedances=1_999 if active else None,
        )
    first = assemble_exact_evidence_family(graph, preflight, evidence)
    second = assemble_exact_evidence_family(
        list(reversed(graph)),
        list(reversed(preflight)),
        dict(reversed(list(evidence.items()))),
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["non_tone_family_edge_count"] == EXPECTED_NON_TONE_EDGES
    assert (
        first["p1_support_skipped_non_tone_edge_count"]
        == EXPECTED_SKIPPED_NON_TONE_EDGES
    )
    assert first["tone_only_edge_count"] == EXPECTED_TONE_ONLY_EDGES
    assert all("route" not in row and "reason" not in row for row in first["edges"])
    skipped = [
        row for row in first["edges"] if row["family_status"] == "support_skipped_p1"
    ]
    assert len(skipped) == EXPECTED_SKIPPED_NON_TONE_EDGES
    assert all(row["p_e"] == row["p_d"] == 1.0 for row in skipped)
