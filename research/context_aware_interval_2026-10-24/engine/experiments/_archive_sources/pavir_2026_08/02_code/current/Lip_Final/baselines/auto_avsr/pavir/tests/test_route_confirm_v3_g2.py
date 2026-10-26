import copy
import hashlib
import random

import pytest

from pavir.artifacts import canonical_json_bytes
from pavir.route_confirm_v3_g2 import (
    ROUTE_SHUFFLE_SCOPE,
    adapt_g1_routes,
    audit_adapted_routes,
    audit_support_matched_route_shuffle,
    build_support_matched_route_shuffle,
)


_AUTHORITY_NAMES = (
    "g1_activation_receipt",
    "route_decisions",
    "route_mass_gate",
    "protocol_precommit",
    "core_lock",
    "io_lock",
    "source_split_lock",
    "stable_graph",
    "full_feature_bundle",
    "full_bundle_gate",
    "full_preflight",
    "exact_plan",
    "recovery_lineage_gate",
    "train_manifest",
    "auto_avsr_train_labels",
    "canonical_id_map",
    "tokenizer_model",
    "tokenizer_units",
    "forbidden_ids",
    "occurrence_builder",
)


def _sha(payload):
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _record(name, digest=None):
    return {
        "path": f"/frozen/{name}",
        "sha256": digest or hashlib.sha256(name.encode("utf-8")).hexdigest(),
    }


def _edge(index, route, support):
    reason = {
        "E": "E_confirmed",
        "D": "D_tone",
        "A": "A_support",
    }[route]
    return {
        "ci95": [0.75, 0.90] if route == "E" else None,
        "confusion_text": f"confusion-{index}",
        "confusion_token_ids": [1000 + index],
        "converged": route != "A",
        "edge_id": f"e-{index:020x}",
        "evidence_sha256": hashlib.sha256(
            f"evidence-{index}".encode("utf-8")
        ).hexdigest(),
        "p_d": 1.0 if route != "D" else 0.01,
        "p_e": 0.01 if route == "E" else 1.0,
        "q_d_bh": 1.0 if route != "D" else 0.02,
        "q_d_by_sensitivity": 1.0 if route != "D" else 0.04,
        "q_e_bh": 0.02 if route == "E" else 1.0,
        "q_e_by_sensitivity": 0.04 if route == "E" else 1.0,
        "reason": reason,
        "route": route,
        "stable_source_count": min(4, support),
        "stable_support": support,
        "support_passed": route != "A",
        "target_text": f"target-{index}",
        "target_token_ids": [100 + index],
        "theta": 0.82 if route == "E" else None,
        "tone_only": route == "D",
    }


def _fixtures(rows, *, maximum_attempts=10_000):
    counts = {
        route: sum(row["route"] == route for row in rows)
        for route in ("E", "D", "A")
    }
    authority = {name: _record(name) for name in _AUTHORITY_NAMES}
    bindings = {
        "stable_graph": authority["stable_graph"],
        "protocol_precommit": authority["protocol_precommit"],
        "core_implementation_lock": authority["core_lock"],
        "io_implementation_lock": authority["io_lock"],
        "source_split_lock": authority["source_split_lock"],
    }
    closed_scope = {
        "complete_routing_authorized": True,
        "route_decisions_authorized": True,
        "g2_guarded_offline_artifact_stage_authorized": False,
        "guarded_offline_artifacts_authorized": False,
        "offline_artifacts_authorized": False,
        "prototype_publication_authorized": False,
        "training_authorized": False,
    }
    reasons = {}
    for row in rows:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    decisions = {
        "schema_version": "pavir-route-confirm-v3-route-decisions-v1",
        "protocol_name": "PAViR-RouteConfirm-v3",
        "status": "COMPLETE_LOCKED_EDA_DECISIONS",
        "scope": copy.deepcopy(closed_scope),
        "authorization": copy.deepcopy(closed_scope),
        "bindings": copy.deepcopy(bindings),
        "inventory": {
            "edge_count": len(rows),
            "route_counts": counts,
            "reason_counts": dict(sorted(reasons.items())),
            "decision_rows_sha256": _sha(rows),
        },
        "edges": copy.deepcopy(rows),
    }
    authority["route_decisions"] = _record("route-decisions", _sha(decisions))
    open_g2_scope = {
        "complete_routing_authorized": True,
        "route_decisions_authorized": True,
        "g2_guarded_offline_artifact_stage_authorized": True,
        "guarded_offline_artifacts_authorized": True,
        "offline_artifacts_authorized": False,
        "prototype_publication_authorized": False,
        "training_authorized": False,
    }
    gate = {
        "schema_version": "pavir-route-confirm-v3-route-mass-gate-v1",
        "protocol_name": "PAViR-RouteConfirm-v3",
        "status": (
            "PASS_GUARDED_OFFLINE_ARTIFACTS_ONLY_"
            "NO_TRAINING_AUTHORIZATION"
        ),
        "decision": "PASS",
        "route_decisions": copy.deepcopy(authority["route_decisions"]),
        "bindings": copy.deepcopy(bindings),
        "scope": copy.deepcopy(open_g2_scope),
        "authorization": copy.deepcopy(open_g2_scope),
        "checks": {
            "locked_eda_decisions_applied": True,
            "all_locked_mass_checks_passed": True,
        },
        "locked_core_route_mass_audit": {
            "status": "PASS",
            "eligible_for_artifacts_and_training": True,
            "e_edge_count": counts["E"],
            "d_edge_count": counts["D"],
            "checks": {
                "minimum_e_edges": True,
                "minimum_d_edges": True,
            },
            "blockers": [],
        },
        "blockers": [],
    }
    authority["route_mass_gate"] = _record("route-mass-gate", _sha(gate))
    contract = {
        "schema_version": "pavir-route-confirm-v3-g2-method-contract-v1",
        "protocol_name": "PAViR-RouteConfirm-v3",
        "status": "FROZEN_PROSPECTIVELY_BEFORE_G2_RESULTS",
        "authority": authority,
        "fixed_counts": {
            "all_edges": len(rows),
            "route_E_edges": counts["E"],
            "route_D_edges": counts["D"],
            "route_A_edges": counts["A"],
            "active_edges": counts["E"] + counts["D"],
            "prototype_keys": 2 * (counts["E"] + counts["D"]),
            "train_rows": 19,
            "forbidden_ids": 3,
        },
        "route_shuffle": {
            "seed": 42,
            "population": "all and only frozen E/D edges sorted by edge_id",
            "method": (
                "first qualifying Python-random label permutation under "
                "deterministic rejection sampling"
            ),
            "maximum_attempts": maximum_attempts,
            "reject_identity": True,
            "preserve_exact_route_counts": True,
            "maximum_relative_stable_support_delta_per_route": 0.01,
            "leave_A_unchanged": True,
            "occurrence_sample_source_batch_mass": "report_only",
        },
        "authorization_on_pass": {
            "prototype_publication_authorized": True,
            "offline_artifact_bundle_authorized": True,
            "guarded_g3_real_w0_cuda_calibration_preflight_authorized": True,
            "gradient_calibration_authorized_without_g3_wrapper": False,
            "screen_training_authorized": False,
            "main_training_authorized": False,
            "training_authorized": False,
            "automatic_g3_launch": False,
        },
        "forbidden_actions": [
            "reroute_or_filter_any_edge",
            "change_any_graph_or_route_threshold",
            "use_route_confirm_sources_for_prototypes",
            "use_vc_calib_vc_promote_or_clean_test_rows",
            "run_legacy_route_estimation",
            "run_legacy_count_only_route_shuffle",
            "launch_calibration_or_training",
        ],
    }
    return decisions, gate, contract


def _balanced_fixture():
    rows = [
        _edge(0, "E", 100),
        _edge(1, "E", 100),
        _edge(2, "D", 100),
        _edge(3, "D", 100),
        _edge(4, "A", 25),
    ]
    return _fixtures(rows)


def test_adapter_preserves_every_g1_row_and_adds_only_exact_aliases():
    decisions, gate, contract = _balanced_fixture()
    frozen_inputs = (
        canonical_json_bytes(decisions),
        canonical_json_bytes(gate),
        canonical_json_bytes(contract),
    )

    result = adapt_g1_routes(decisions, gate, contract)

    assert result.routes["schema_version"] == "pavir-routes-v1"
    assert result.routes["route_counts"] == {"E": 2, "D": 2, "A": 1}
    assert [row["edge_id"] for row in result.routes["edges"]] == [
        row["edge_id"] for row in decisions["edges"]
    ]
    for source, adapted in zip(decisions["edges"], result.routes["edges"]):
        projected = {
            key: copy.deepcopy(value)
            for key, value in adapted.items()
            if key not in {"support", "source_count"}
        }
        assert projected == source
        assert adapted["support"] == source["stable_support"]
        assert adapted["source_count"] == source["stable_source_count"]
    assert audit_adapted_routes(
        decisions, gate, contract, result.routes
    ) == result.audit
    assert result.audit["authorization"]["training_authorized"] is False
    assert result.audit["authorization"]["prototype_publication_authorized"] is False
    assert frozen_inputs == (
        canonical_json_bytes(decisions),
        canonical_json_bytes(gate),
        canonical_json_bytes(contract),
    )


def test_adapter_fails_closed_on_authority_and_output_tamper():
    decisions, gate, contract = _balanced_fixture()
    tampered_decisions = copy.deepcopy(decisions)
    tampered_decisions["edges"][0]["stable_support"] += 1
    with pytest.raises(ValueError, match="payload hash differs"):
        adapt_g1_routes(tampered_decisions, gate, contract)

    adapted = adapt_g1_routes(decisions, gate, contract)
    tampered_routes = copy.deepcopy(adapted.routes)
    tampered_routes["edges"][0]["target_text"] = "tampered"
    with pytest.raises(ValueError, match="rows changed"):
        audit_adapted_routes(decisions, gate, contract, tampered_routes)


def test_adapter_rejects_gate_that_grants_training_even_with_matching_hash():
    decisions, gate, contract = _balanced_fixture()
    unsafe_gate = copy.deepcopy(gate)
    unsafe_gate["authorization"]["training_authorized"] = True
    unsafe_contract = copy.deepcopy(contract)
    unsafe_contract["authority"]["route_mass_gate"]["sha256"] = _sha(unsafe_gate)

    with pytest.raises(ValueError, match="authorization is unsafe"):
        adapt_g1_routes(decisions, unsafe_gate, unsafe_contract)


def test_support_matched_shuffle_is_seeded_first_pass_and_deterministic():
    decisions, gate, contract = _balanced_fixture()
    original = adapt_g1_routes(decisions, gate, contract).routes
    frozen_original = canonical_json_bytes(original)

    first = build_support_matched_route_shuffle(original, contract)
    second = build_support_matched_route_shuffle(original, contract)

    assert canonical_json_bytes(first.routes) == canonical_json_bytes(second.routes)
    assert first.audit == second.audit
    assert canonical_json_bytes(original) == frozen_original
    original_active = [
        row["route"] for row in original["edges"] if row["route"] in {"E", "D"}
    ]
    generator = random.Random(42)
    for expected_attempt in range(1, 10_001):
        labels = list(original_active)
        generator.shuffle(labels)
        if labels != original_active:
            break
    assert first.audit["qualifying_attempt"] == expected_attempt
    assert first.routes["route_counts"] == original["route_counts"]
    assert first.routes["scope"] == ROUTE_SHUFFLE_SCOPE
    assert "g2_route_adapter" not in first.routes
    assert [
        row["route"] for row in first.routes["edges"] if row["edge_id"].endswith("4")
    ] == ["A"]
    assert any(
        before["route"] != after["route"]
        for before, after in zip(original["edges"], first.routes["edges"])
    )
    assert all(
        delta <= 0.01
        for delta in first.audit["stable_support"]["relative_delta"].values()
    )
    assert audit_support_matched_route_shuffle(
        original,
        first.routes,
        contract,
        attempt=first.audit["qualifying_attempt"],
    ) == first.audit
    assert first.audit["authorization"]["training_authorized"] is False


def test_shuffle_audit_rejects_metadata_and_attempt_tamper():
    decisions, gate, contract = _balanced_fixture()
    original = adapt_g1_routes(decisions, gate, contract).routes
    shuffled = build_support_matched_route_shuffle(original, contract)

    metadata_tamper = copy.deepcopy(shuffled.routes)
    metadata_tamper["edges"][0]["target_text"] = "tampered"
    with pytest.raises(ValueError, match="non-route edge metadata"):
        audit_support_matched_route_shuffle(
            original,
            metadata_tamper,
            contract,
            attempt=shuffled.audit["qualifying_attempt"],
        )

    stale_scope = copy.deepcopy(shuffled.routes)
    stale_scope["scope"] = copy.deepcopy(original["scope"])
    stale_scope["g2_route_adapter"] = copy.deepcopy(
        original["g2_route_adapter"]
    )
    with pytest.raises(ValueError, match="stale adapter metadata"):
        audit_support_matched_route_shuffle(
            original,
            stale_scope,
            contract,
            attempt=shuffled.audit["qualifying_attempt"],
        )

    with pytest.raises(ValueError, match="first qualifying attempt"):
        audit_support_matched_route_shuffle(
            original,
            shuffled.routes,
            contract,
            attempt=shuffled.audit["qualifying_attempt"] + 1,
        )


def test_shuffle_fails_when_no_nonidentity_support_match_exists_in_bound():
    rows = [
        _edge(0, "E", 1000),
        _edge(1, "D", 1),
        _edge(2, "A", 20),
    ]
    decisions, gate, contract = _fixtures(rows, maximum_attempts=16)
    original = adapt_g1_routes(decisions, gate, contract).routes

    with pytest.raises(ValueError, match="no qualifying nonidentity"):
        build_support_matched_route_shuffle(original, contract)
