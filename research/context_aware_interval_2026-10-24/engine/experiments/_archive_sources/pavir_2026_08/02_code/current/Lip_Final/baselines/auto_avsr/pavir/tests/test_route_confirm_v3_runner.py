from __future__ import annotations

import json

import numpy as np
import pytest

from pavir.artifacts import canonical_json_bytes
from pavir.route_confirm_v3 import (
    EdgeEvidence,
    EdgeSupport,
    PartitionSupport,
    edge_seed,
)
from pavir.route_confirm_v3_runner import (
    _atomic_publish,
    _bounded_statistics,
    _checkpoint_payload,
    assemble_family,
    deterministic_lpt_shards,
    validate_checkpoint,
)


def _partition(name: str, *, passed: bool = True) -> PartitionSupport:
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
    theta: float,
    ci95: tuple[float, float],
    p_e: float,
    p_d: float,
    tone_only: bool = False,
    support_passed: bool = True,
) -> EdgeEvidence:
    fit = _partition("fit", passed=support_passed)
    confirm = _partition("confirm", passed=support_passed)
    support_failures = fit.failures + confirm.failures
    return EdgeEvidence(
        edge_id=edge_id,
        tone_only=tone_only,
        support=EdgeSupport(
            fit=fit,
            confirm=confirm,
            passed=support_passed,
            failures=support_failures,
        ),
        converged=True,
        seed=edge_seed(edge_id),
        theta=theta if support_passed and not tone_only else None,
        source_aucs=(theta - 0.01, theta + 0.01)
        if support_passed and not tone_only
        else (),
        source_auc_ids=("s1", "s2") if support_passed and not tone_only else (),
        ci95=ci95 if support_passed and not tone_only else None,
        p_e=p_e if support_passed and not tone_only else 1.0,
        p_d=p_d if support_passed and not tone_only else 1.0,
        permutation_exceedances=0
        if support_passed and not tone_only
        else None,
        warning_messages=(),
    )


def _fixture_rows():
    graph = [
        {"edge_id": "e-00000000000000000001", "tone_only": False},
        {"edge_id": "e-00000000000000000002", "tone_only": False},
        {"edge_id": "e-00000000000000000003", "tone_only": False},
        {"edge_id": "e-00000000000000000004", "tone_only": True},
        {"edge_id": "e-00000000000000000005", "tone_only": True},
    ]
    preflight = [
        {**row, "support_passed": row["edge_id"] not in {
            "e-00000000000000000003",
            "e-00000000000000000005",
        }}
        for row in graph
    ]
    evidence = {
        graph[0]["edge_id"]: _evidence(
            graph[0]["edge_id"],
            theta=0.80,
            ci95=(0.70, 0.90),
            p_e=0.001,
            p_d=0.8,
        ),
        graph[1]["edge_id"]: _evidence(
            graph[1]["edge_id"],
            theta=0.50,
            ci95=(0.46, 0.54),
            p_e=0.8,
            p_d=0.001,
        ),
        graph[2]["edge_id"]: _evidence(
            graph[2]["edge_id"],
            theta=0.5,
            ci95=(0.4, 0.6),
            p_e=1.0,
            p_d=1.0,
            support_passed=False,
        ),
        graph[3]["edge_id"]: _evidence(
            graph[3]["edge_id"],
            theta=0.5,
            ci95=(0.4, 0.6),
            p_e=1.0,
            p_d=1.0,
            tone_only=True,
        ),
        graph[4]["edge_id"]: _evidence(
            graph[4]["edge_id"],
            theta=0.5,
            ci95=(0.4, 0.6),
            p_e=1.0,
            p_d=1.0,
            tone_only=True,
            support_passed=False,
        ),
    }
    return graph, preflight, evidence


def test_lpt_scheduling_is_deterministic_complete_and_disjoint() -> None:
    work = {
        "edge-c": 3,
        "edge-a": 9,
        "edge-d": 2,
        "edge-b": 7,
        "edge-e": 1,
    }
    observed = deterministic_lpt_shards(work, 3)
    assert observed == deterministic_lpt_shards(dict(reversed(list(work.items()))), 3)
    flattened = [edge_id for shard in observed for edge_id in shard]
    assert sorted(flattened) == sorted(work)
    assert len(flattened) == len(set(flattened))


def test_family_result_is_identical_across_one_two_three_worker_schedules() -> None:
    graph, preflight, evidence = _fixture_rows()
    expected = None
    work = {edge_id: index + 1 for index, edge_id in enumerate(evidence)}
    for worker_count in (1, 2, 3):
        scheduled = deterministic_lpt_shards(work, worker_count)
        gathered = {
            edge_id: evidence[edge_id]
            for shard in scheduled
            for edge_id in reversed(shard)
        }
        observed = assemble_family(graph, preflight, gathered)
        encoded = canonical_json_bytes(observed)
        expected = encoded if expected is None else expected
        assert encoded == expected

    by_id = {row["edge_id"]: row for row in observed["edges"]}
    skipped = by_id["e-00000000000000000003"]
    assert skipped["family_status"] == "support_skipped_p1"
    assert skipped["p_e"] == skipped["p_d"] == 1.0
    assert skipped["route"] == "A"
    assert by_id["e-00000000000000000004"]["reason"] == "D_tone"
    assert by_id["e-00000000000000000005"]["reason"] == "A_support"
    assert by_id["e-00000000000000000001"]["route"] == "E"
    assert by_id["e-00000000000000000002"]["route"] == "D"


def test_atomic_checkpoint_resume_and_tamper_detection(tmp_path) -> None:
    edge_id = "e-00000000000000000001"
    evidence = _evidence(
        edge_id,
        theta=0.8,
        ci95=(0.7, 0.9),
        p_e=0.1,
        p_d=0.9,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    plan = {
        "inputs": {"fixture": True},
        "statistics": {
            "permutation_iterations": 9,
            "bootstrap_iterations": 7,
        },
        "family": {"tone_only_edge_ids": []},
    }
    payload = _checkpoint_payload(
        plan_path=plan_path,
        plan_sha256="a" * 64,
        plan=plan,
        edge_id=edge_id,
        evidence=evidence,
        edge_input={
            "accepted_memberships": 8,
            "accepted_membership_stream_sha256": "b" * 64,
            "quality_mask_sha256": "c" * 64,
        },
    )
    checkpoint = tmp_path / "edges" / f"{edge_id}.json"
    assert _atomic_publish(checkpoint, payload) is True
    assert _atomic_publish(checkpoint, payload) is False
    restored = validate_checkpoint(
        checkpoint,
        plan_path=plan_path.resolve(),
        plan_sha256="a" * 64,
        plan=plan,
        edge_id=edge_id,
    )
    assert restored == evidence

    tampered = json.loads(checkpoint.read_text(encoding="utf-8"))
    tampered["evidence"]["p_e"] = 0.99
    checkpoint.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="evidence mismatch"):
        validate_checkpoint(
            checkpoint,
            plan_path=plan_path.resolve(),
            plan_sha256="a" * 64,
            plan=plan,
            edge_id=edge_id,
        )


def test_current_preflight_cannot_schedule_production_defaults() -> None:
    assert _bounded_statistics(999, 200) == (999, 200)
    with pytest.raises(PermissionError, match="does not authorize"):
        _bounded_statistics(199_999, 10_000)
    with pytest.raises(ValueError, match="cannot exceed"):
        _bounded_statistics(200_000, 10_000)
