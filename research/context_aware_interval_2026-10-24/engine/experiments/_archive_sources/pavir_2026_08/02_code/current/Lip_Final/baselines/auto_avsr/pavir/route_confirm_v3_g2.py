"""Pure, fail-closed RouteConfirm-v3 G2 route-control primitives.

This module has no filesystem or training side effects.  It performs the two
route transformations frozen by the G2 method contract:

* adapt the authenticated G1 E/D/A decision rows to ``pavir-routes-v1``
  without dropping, filtering, or reinterpreting an edge; and
* construct the first deterministic, non-identity E/D label permutation that
  preserves exact route counts and stays within the frozen per-route stable
  support tolerance.

Prototype extraction, occurrence construction, publication, calibration, and
training authorization deliberately live outside this module.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import random
import re
from typing import Any, Mapping, Sequence

from .artifacts import canonical_json_bytes


PROTOCOL = "PAViR-RouteConfirm-v3"
METHOD_CONTRACT_SCHEMA = "pavir-route-confirm-v3-g2-method-contract-v1"
METHOD_CONTRACT_STATUS = "FROZEN_PROSPECTIVELY_BEFORE_G2_RESULTS"
G1_ROUTES_SCHEMA = "pavir-route-confirm-v3-route-decisions-v1"
G1_ROUTES_STATUS = "COMPLETE_LOCKED_EDA_DECISIONS"
G1_MASS_GATE_SCHEMA = "pavir-route-confirm-v3-route-mass-gate-v1"
G1_MASS_GATE_STATUS = (
    "PASS_GUARDED_OFFLINE_ARTIFACTS_ONLY_NO_TRAINING_AUTHORIZATION"
)
TRAINING_ROUTES_SCHEMA = "pavir-routes-v1"
ROUTE_ADAPTER_AUDIT_SCHEMA = (
    "pavir-route-confirm-v3-g2-route-adapter-audit-v1"
)
ROUTE_SHUFFLE_AUDIT_SCHEMA = (
    "pavir-route-confirm-v3-g2-support-matched-route-shuffle-audit-v1"
)
ROUTE_SHUFFLE_SCOPE = {
    "training_compatible_route_schema_only": True,
    "support_matched_route_shuffle_control": True,
    "route_labels_permuted": True,
    "edge_filtering_performed": False,
    "edge_filtering_or_rerouting_performed": True,
    "prototype_publication_authorized": False,
    "offline_artifact_bundle_authorized": False,
    "training_authorized": False,
}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_EDGE_ID = re.compile(r"e-[0-9a-f]{20}\Z")
_VALID_ROUTES = frozenset({"E", "D", "A"})
_REASON_ROUTE = {
    "E_confirmed": "E",
    "D_tone": "D",
    "D_equiv": "D",
    "A_support": "A",
    "A_reversed": "A",
    "A_uncertain": "A",
}
_G1_EDGE_FIELDS = frozenset(
    {
        "ci95",
        "confusion_text",
        "confusion_token_ids",
        "converged",
        "edge_id",
        "evidence_sha256",
        "p_d",
        "p_e",
        "q_d_bh",
        "q_d_by_sensitivity",
        "q_e_bh",
        "q_e_by_sensitivity",
        "reason",
        "route",
        "stable_source_count",
        "stable_support",
        "support_passed",
        "target_text",
        "target_token_ids",
        "theta",
        "tone_only",
    }
)
_TRAINING_EDGE_FIELDS = _G1_EDGE_FIELDS | {"support", "source_count"}
_AUTHORITY_RECORDS = frozenset(
    {
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
    }
)
_REQUIRED_FORBIDDEN_ACTIONS = frozenset(
    {
        "reroute_or_filter_any_edge",
        "change_any_graph_or_route_threshold",
        "use_route_confirm_sources_for_prototypes",
        "use_vc_calib_vc_promote_or_clean_test_rows",
        "run_legacy_route_estimation",
        "run_legacy_count_only_route_shuffle",
        "launch_calibration_or_training",
    }
)


@dataclass(frozen=True)
class RouteAdapterResult:
    """One training-schema route payload and its independently reproducible audit."""

    routes: dict[str, Any]
    audit: dict[str, Any]


@dataclass(frozen=True)
class RouteShuffleResult:
    """One support-matched shuffled route payload and its replay audit."""

    routes: dict[str, Any]
    audit: dict[str, Any]


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_object(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _require_int(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _require_number(
    value: object,
    *,
    context: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{context} must be <= {maximum}")
    return number


def _require_digest(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"{context} is not a lowercase SHA-256")
    return value


def _validate_file_record(
    value: object,
    *,
    context: str,
) -> Mapping[str, Any]:
    record = _require_object(value, context=context)
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{context} must contain path/sha256 exactly")
    if not isinstance(record["path"], str) or not record["path"]:
        raise ValueError(f"{context} path is empty")
    _require_digest(record["sha256"], context=f"{context} SHA-256")
    return record


def _route_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"E": 0, "D": 0, "A": 0}
    for row in rows:
        route = str(row.get("route", ""))
        if route not in counts:
            raise ValueError(f"invalid route: {route!r}")
        counts[route] += 1
    return counts


def _expected_route_counts(contract: Mapping[str, Any]) -> dict[str, int]:
    fixed = _require_object(contract.get("fixed_counts"), context="fixed counts")
    required = {
        "all_edges",
        "route_E_edges",
        "route_D_edges",
        "route_A_edges",
        "active_edges",
        "prototype_keys",
        "train_rows",
        "forbidden_ids",
    }
    if set(fixed) != required:
        raise ValueError("G2 fixed-count contract is incomplete or has extra fields")
    values = {
        name: _require_int(fixed[name], context=f"fixed_counts.{name}")
        for name in required
    }
    expected = {
        "E": values["route_E_edges"],
        "D": values["route_D_edges"],
        "A": values["route_A_edges"],
    }
    if (
        values["all_edges"] != sum(expected.values())
        or values["active_edges"] != expected["E"] + expected["D"]
        or values["prototype_keys"] != 2 * values["active_edges"]
        or expected["E"] <= 0
        or expected["D"] <= 0
    ):
        raise ValueError("G2 fixed counts are internally inconsistent")
    return expected


def _validate_method_contract(
    method_contract: object,
) -> tuple[Mapping[str, Any], dict[str, int]]:
    contract = _require_object(method_contract, context="G2 method contract")
    if (
        contract.get("schema_version") != METHOD_CONTRACT_SCHEMA
        or contract.get("protocol_name") != PROTOCOL
        or contract.get("status") != METHOD_CONTRACT_STATUS
    ):
        raise ValueError("unsupported or unfrozen G2 method contract")

    authority = _require_object(
        contract.get("authority"), context="G2 method authority"
    )
    missing = _AUTHORITY_RECORDS - set(authority)
    if missing:
        raise ValueError(f"G2 method authority is incomplete: {sorted(missing)}")
    for name in _AUTHORITY_RECORDS:
        _validate_file_record(authority[name], context=f"authority.{name}")

    expected_counts = _expected_route_counts(contract)
    shuffle = _require_object(
        contract.get("route_shuffle"), context="route-shuffle contract"
    )
    if (
        _require_int(shuffle.get("seed"), context="route shuffle seed") != 42
        or shuffle.get("population")
        != "all and only frozen E/D edges sorted by edge_id"
        or shuffle.get("method")
        != (
            "first qualifying Python-random label permutation under "
            "deterministic rejection sampling"
        )
        or _require_int(
            shuffle.get("maximum_attempts"),
            context="route shuffle maximum attempts",
            minimum=1,
        )
        <= 0
        or shuffle.get("reject_identity") is not True
        or shuffle.get("preserve_exact_route_counts") is not True
        or shuffle.get("leave_A_unchanged") is not True
        or shuffle.get("occurrence_sample_source_batch_mass") != "report_only"
    ):
        raise ValueError("route-shuffle contract differs from the frozen method")
    tolerance = _require_number(
        shuffle.get("maximum_relative_stable_support_delta_per_route"),
        context="route shuffle support tolerance",
        minimum=0.0,
        maximum=1.0,
    )
    if tolerance <= 0:
        raise ValueError("route shuffle support tolerance must be positive")

    authorization = _require_object(
        contract.get("authorization_on_pass"),
        context="G2 authorization contract",
    )
    expected_authorization = {
        "prototype_publication_authorized": True,
        "offline_artifact_bundle_authorized": True,
        "guarded_g3_real_w0_cuda_calibration_preflight_authorized": True,
        "gradient_calibration_authorized_without_g3_wrapper": False,
        "screen_training_authorized": False,
        "main_training_authorized": False,
        "training_authorized": False,
        "automatic_g3_launch": False,
    }
    if dict(authorization) != expected_authorization:
        raise ValueError("G2 authorization contract is unsafe or incomplete")
    forbidden = contract.get("forbidden_actions")
    if (
        not isinstance(forbidden, list)
        or len(forbidden) != len(set(map(str, forbidden)))
        or not _REQUIRED_FORBIDDEN_ACTIONS.issubset(map(str, forbidden))
    ):
        raise ValueError("G2 forbidden-action contract is incomplete")
    return contract, expected_counts


def _validate_probability(value: object, *, context: str) -> None:
    if value is None:
        return
    _require_number(value, context=context, minimum=0.0, maximum=1.0)


def _validate_token_ids(value: object, *, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty integer list")
    result = []
    for index, item in enumerate(value):
        result.append(
            _require_int(
                item,
                context=f"{context}[{index}]",
                minimum=1,
            )
        )
    return tuple(result)


def _validate_g1_edge(
    row: object,
    *,
    index: int,
    require_route_reason_match: bool = True,
) -> Mapping[str, Any]:
    edge = _require_object(row, context=f"G1 edge {index}")
    if set(edge) != _G1_EDGE_FIELDS:
        raise ValueError(f"G1 edge {index} fields differ from the locked schema")
    edge_id = str(edge["edge_id"])
    if not _EDGE_ID.fullmatch(edge_id):
        raise ValueError(f"G1 edge {index} has invalid edge_id")
    route = str(edge["route"])
    reason = str(edge["reason"])
    if (
        route not in _VALID_ROUTES
        or reason not in _REASON_ROUTE
        or (
            require_route_reason_match
            and _REASON_ROUTE.get(reason) != route
        )
    ):
        raise ValueError(f"G1 edge {edge_id} has inconsistent route/reason")
    if not isinstance(edge["target_text"], str) or not edge["target_text"]:
        raise ValueError(f"G1 edge {edge_id} has empty target text")
    if not isinstance(edge["confusion_text"], str) or not edge["confusion_text"]:
        raise ValueError(f"G1 edge {edge_id} has empty confusion text")
    target_ids = _validate_token_ids(
        edge["target_token_ids"], context=f"{edge_id}.target_token_ids"
    )
    confusion_ids = _validate_token_ids(
        edge["confusion_token_ids"],
        context=f"{edge_id}.confusion_token_ids",
    )
    if target_ids == confusion_ids:
        raise ValueError(f"G1 edge {edge_id} has identical token endpoints")
    support = _require_int(
        edge["stable_support"], context=f"{edge_id}.stable_support", minimum=1
    )
    sources = _require_int(
        edge["stable_source_count"],
        context=f"{edge_id}.stable_source_count",
        minimum=1,
    )
    if sources > support:
        raise ValueError(f"G1 edge {edge_id} has source_count > support")
    for name in ("tone_only", "support_passed", "converged"):
        if not isinstance(edge[name], bool):
            raise ValueError(f"G1 edge {edge_id} has non-boolean {name}")
    if route in {"E", "D"} and edge["support_passed"] is not True:
        raise ValueError(f"active G1 edge {edge_id} did not pass support")
    if route == "E" and edge["converged"] is not True:
        raise ValueError(f"route-E edge {edge_id} did not converge")
    if reason == "D_tone" and edge["tone_only"] is not True:
        raise ValueError(f"D_tone edge {edge_id} is not tone-only")
    if reason == "D_equiv" and edge["tone_only"] is not False:
        raise ValueError(f"D_equiv edge {edge_id} is tone-only")
    _require_digest(
        edge["evidence_sha256"], context=f"{edge_id}.evidence_sha256"
    )
    for name in (
        "theta",
        "p_e",
        "p_d",
        "q_e_bh",
        "q_d_bh",
        "q_e_by_sensitivity",
        "q_d_by_sensitivity",
    ):
        _validate_probability(edge[name], context=f"{edge_id}.{name}")
    ci95 = edge["ci95"]
    if ci95 is not None:
        if not isinstance(ci95, list) or len(ci95) != 2:
            raise ValueError(f"{edge_id}.ci95 must be null or a pair")
        lower = _require_number(
            ci95[0], context=f"{edge_id}.ci95[0]", minimum=0.0, maximum=1.0
        )
        upper = _require_number(
            ci95[1], context=f"{edge_id}.ci95[1]", minimum=0.0, maximum=1.0
        )
        if lower > upper:
            raise ValueError(f"{edge_id}.ci95 is reversed")
    return edge


def _validate_g1_authority(
    route_decisions: object,
    route_mass_gate: object,
    method_contract: object,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    list[Mapping[str, Any]],
    dict[str, int],
]:
    contract, expected_counts = _validate_method_contract(method_contract)
    authority = contract["authority"]
    decisions = _require_object(route_decisions, context="G1 route decisions")
    mass_gate = _require_object(route_mass_gate, context="G1 route-mass gate")
    route_digest = _payload_sha256(decisions)
    gate_digest = _payload_sha256(mass_gate)
    if route_digest != authority["route_decisions"]["sha256"]:
        raise ValueError("G1 route-decision payload hash differs from G2 authority")
    if gate_digest != authority["route_mass_gate"]["sha256"]:
        raise ValueError("G1 route-mass payload hash differs from G2 authority")
    if (
        decisions.get("schema_version") != G1_ROUTES_SCHEMA
        or decisions.get("protocol_name") != PROTOCOL
        or decisions.get("status") != G1_ROUTES_STATUS
    ):
        raise ValueError("unsupported G1 route-decision artifact")
    decision_scope = _require_object(
        decisions.get("scope"), context="G1 route-decision scope"
    )
    decision_authorization = _require_object(
        decisions.get("authorization"),
        context="G1 route-decision authorization",
    )
    for record in (decision_scope, decision_authorization):
        if (
            record.get("complete_routing_authorized") is not True
            or record.get("route_decisions_authorized") is not True
            or record.get("g2_guarded_offline_artifact_stage_authorized")
            is not False
            or record.get("guarded_offline_artifacts_authorized") is not False
            or record.get("offline_artifacts_authorized") is not False
            or record.get("prototype_publication_authorized") is not False
            or record.get("training_authorized") is not False
        ):
            raise ValueError("G1 route-decision scope is unsafe or inconsistent")

    binding_names = {
        "stable_graph": "stable_graph",
        "protocol_precommit": "protocol_precommit",
        "core_implementation_lock": "core_lock",
        "io_implementation_lock": "io_lock",
        "source_split_lock": "source_split_lock",
    }
    bindings = _require_object(decisions.get("bindings"), context="G1 bindings")
    for g1_name, authority_name in binding_names.items():
        if bindings.get(g1_name) != authority[authority_name]:
            raise ValueError(f"G1 binding differs from G2 authority: {g1_name}")

    raw_rows = decisions.get("edges")
    if not isinstance(raw_rows, list):
        raise ValueError("G1 route decisions lack an edge list")
    rows = [
        _validate_g1_edge(row, index=index)
        for index, row in enumerate(raw_rows)
    ]
    edge_ids = [str(row["edge_id"]) for row in rows]
    if edge_ids != sorted(edge_ids) or len(edge_ids) != len(set(edge_ids)):
        raise ValueError("G1 edges are not unique and sorted by edge_id")
    counts = _route_counts(rows)
    inventory = _require_object(
        decisions.get("inventory"), context="G1 route inventory"
    )
    reason_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if (
        counts != expected_counts
        or inventory.get("route_counts") != expected_counts
        or _require_int(
            inventory.get("edge_count"), context="G1 inventory edge count"
        )
        != len(rows)
        or len(rows) != sum(expected_counts.values())
        or inventory.get("reason_counts") != dict(sorted(reason_counts.items()))
        or inventory.get("decision_rows_sha256")
        != _payload_sha256(rows)
    ):
        raise ValueError("G1 route inventory or decision-row digest is inconsistent")

    if (
        mass_gate.get("schema_version") != G1_MASS_GATE_SCHEMA
        or mass_gate.get("protocol_name") != PROTOCOL
        or mass_gate.get("status") != G1_MASS_GATE_STATUS
        or mass_gate.get("decision") != "PASS"
        or mass_gate.get("route_decisions") != authority["route_decisions"]
        or mass_gate.get("bindings") != decisions.get("bindings")
        or mass_gate.get("blockers") != []
    ):
        raise ValueError("G1 route-mass gate is not the frozen PASS authority")
    gate_scope = _require_object(
        mass_gate.get("scope"), context="G1 route-mass scope"
    )
    gate_authorization = _require_object(
        mass_gate.get("authorization"),
        context="G1 route-mass authorization",
    )
    for record in (gate_scope, gate_authorization):
        if (
            record.get("complete_routing_authorized") is not True
            or record.get("route_decisions_authorized") is not True
            or record.get("g2_guarded_offline_artifact_stage_authorized")
            is not True
            or record.get("guarded_offline_artifacts_authorized") is not True
            or record.get("offline_artifacts_authorized") is not False
            or record.get("prototype_publication_authorized") is not False
            or record.get("training_authorized") is not False
        ):
            raise ValueError("G1 route-mass authorization is unsafe or closed")
    checks = _require_object(mass_gate.get("checks"), context="G1 mass checks")
    if not checks or any(value is not True for value in checks.values()):
        raise ValueError("G1 route-mass checks did not all pass")
    mass = _require_object(
        mass_gate.get("locked_core_route_mass_audit"),
        context="locked route-mass audit",
    )
    if (
        mass.get("status") != "PASS"
        or mass.get("eligible_for_artifacts_and_training") is not True
        or _require_int(mass.get("e_edge_count"), context="mass E edge count")
        != expected_counts["E"]
        or _require_int(mass.get("d_edge_count"), context="mass D edge count")
        != expected_counts["D"]
        or mass.get("blockers") != []
        or not isinstance(mass.get("checks"), Mapping)
        or any(value is not True for value in mass["checks"].values())
    ):
        raise ValueError("locked G1 route-mass audit is inconsistent")
    return decisions, mass_gate, contract, rows, expected_counts


def _project_g1_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: copy.deepcopy(value)
            for key, value in row.items()
            if key in _G1_EDGE_FIELDS
        }
        for row in rows
    ]


def _validate_training_rows(
    routes: Mapping[str, Any],
    expected_counts: Mapping[str, int],
    *,
    require_route_reason_match: bool = True,
) -> list[Mapping[str, Any]]:
    if routes.get("schema_version") != TRAINING_ROUTES_SCHEMA:
        raise ValueError("routes are not pavir-routes-v1")
    raw_rows = routes.get("edges")
    if not isinstance(raw_rows, list):
        raise ValueError("training routes lack an edge list")
    rows: list[Mapping[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row = _require_object(raw, context=f"training edge {index}")
        if set(row) != _TRAINING_EDGE_FIELDS:
            raise ValueError(
                f"training edge {index} does not preserve the exact G1 fields"
            )
        _validate_g1_edge(
            {key: row[key] for key in _G1_EDGE_FIELDS},
            index=index,
            require_route_reason_match=require_route_reason_match,
        )
        support = _require_int(
            row["support"], context=f"{row['edge_id']}.support", minimum=1
        )
        source_count = _require_int(
            row["source_count"],
            context=f"{row['edge_id']}.source_count",
            minimum=1,
        )
        if (
            support != row["stable_support"]
            or source_count != row["stable_source_count"]
        ):
            raise ValueError(f"training aliases drifted for {row['edge_id']}")
        rows.append(row)
    edge_ids = [str(row["edge_id"]) for row in rows]
    counts = _route_counts(rows)
    if (
        edge_ids != sorted(edge_ids)
        or len(edge_ids) != len(set(edge_ids))
        or counts != dict(expected_counts)
        or routes.get("route_counts") != dict(expected_counts)
    ):
        raise ValueError("training route inventory is inconsistent")
    return rows


def audit_adapted_routes(
    route_decisions: object,
    route_mass_gate: object,
    method_contract: object,
    routes: object,
    *,
    corpus: str = "vicocktail",
    split: str = "train",
) -> dict[str, Any]:
    """Independently prove that an adapter changed no G1 decision semantics."""

    decisions, gate, contract, g1_rows, expected_counts = _validate_g1_authority(
        route_decisions,
        route_mass_gate,
        method_contract,
    )
    if corpus not in {"vicocktail", "vlr"} or not split:
        raise ValueError("invalid route corpus/split")
    adapted = _require_object(routes, context="adapted training routes")
    rows = _validate_training_rows(adapted, expected_counts)
    route_sha = _payload_sha256(decisions)
    gate_sha = _payload_sha256(gate)
    contract_sha = _payload_sha256(contract)
    expected_inputs = {
        "graph_sha256": contract["authority"]["stable_graph"]["sha256"],
        "g1_route_decisions_sha256": route_sha,
        "g1_route_mass_gate_sha256": gate_sha,
        "g2_method_contract_sha256": contract_sha,
    }
    scope = _require_object(adapted.get("scope"), context="adapted route scope")
    if (
        adapted.get("decision_protocol") != PROTOCOL
        or adapted.get("corpus") != corpus
        or adapted.get("split") != split
        or adapted.get("inputs") != expected_inputs
        or scope.get("training_compatible_route_schema_only") is not True
        or scope.get("edge_filtering_or_rerouting_performed") is not False
        or scope.get("prototype_publication_authorized") is not False
        or scope.get("offline_artifact_bundle_authorized") is not False
        or scope.get("training_authorized") is not False
    ):
        raise ValueError("adapted route header/scope is unsafe or inconsistent")

    projected = _project_g1_rows(rows)
    if projected != list(g1_rows):
        raise ValueError("G1 edge rows changed during training-schema adaptation")
    alias_checks = all(
        row["support"] == row["stable_support"]
        and row["source_count"] == row["stable_source_count"]
        for row in rows
    )
    checks = {
        "g1_authority_hashes_match_method_contract": True,
        "g1_route_mass_gate_passed": True,
        "all_edges_preserved_in_exact_order": True,
        "all_g1_row_fields_preserved_exactly": True,
        "route_labels_unchanged": True,
        "stable_support_aliases_exact": alias_checks,
        "route_counts_exact": _route_counts(rows) == expected_counts,
        "no_training_authorization": True,
    }
    if not all(checks.values()):
        raise ValueError("adapted route audit failed")
    audit = {
        "schema_version": ROUTE_ADAPTER_AUDIT_SCHEMA,
        "protocol_name": PROTOCOL,
        "status": "PASS_EXACT_G1_TO_PAVIR_ROUTES_V1_NO_TRAINING",
        "inputs": expected_inputs,
        "inventory": {
            "edge_count": len(rows),
            "route_counts": dict(expected_counts),
            "active_edges": expected_counts["E"] + expected_counts["D"],
        },
        "row_preservation": {
            "g1_decision_rows_sha256": _payload_sha256(g1_rows),
            "adapted_g1_projection_sha256": _payload_sha256(projected),
            "adapted_training_rows_sha256": _payload_sha256(rows),
            "added_fields": {
                "support": "exact alias of stable_support",
                "source_count": "exact alias of stable_source_count",
            },
        },
        "checks": checks,
        "authorization": {
            "route_adapter_complete": True,
            "prototype_publication_authorized": False,
            "offline_artifact_bundle_authorized": False,
            "gradient_calibration_authorized": False,
            "screen_training_authorized": False,
            "main_training_authorized": False,
            "training_authorized": False,
        },
    }
    embedded = adapted.get("g2_route_adapter")
    if embedded is not None and embedded != audit:
        raise ValueError("embedded G2 route-adapter audit is inconsistent")
    return audit


def adapt_g1_routes(
    route_decisions: object,
    route_mass_gate: object,
    method_contract: object,
    *,
    corpus: str = "vicocktail",
    split: str = "train",
) -> RouteAdapterResult:
    """Adapt the exact G1 rows without filtering or changing any route label."""

    decisions, gate, contract, rows, expected_counts = _validate_g1_authority(
        route_decisions,
        route_mass_gate,
        method_contract,
    )
    adapted_rows = []
    for source in rows:
        row = copy.deepcopy(dict(source))
        if "support" in row or "source_count" in row:
            raise ValueError("G1 row already contains a training alias")
        row["support"] = row["stable_support"]
        row["source_count"] = row["stable_source_count"]
        adapted_rows.append(row)
    routes: dict[str, Any] = {
        "schema_version": TRAINING_ROUTES_SCHEMA,
        "decision_protocol": PROTOCOL,
        "corpus": corpus,
        "split": split,
        "inputs": {
            "graph_sha256": contract["authority"]["stable_graph"]["sha256"],
            "g1_route_decisions_sha256": _payload_sha256(decisions),
            "g1_route_mass_gate_sha256": _payload_sha256(gate),
            "g2_method_contract_sha256": _payload_sha256(contract),
        },
        "route_counts": dict(expected_counts),
        "scope": {
            "training_compatible_route_schema_only": True,
            "edge_filtering_or_rerouting_performed": False,
            "prototype_publication_authorized": False,
            "offline_artifact_bundle_authorized": False,
            "training_authorized": False,
        },
        "edges": adapted_rows,
    }
    audit = audit_adapted_routes(
        decisions,
        gate,
        contract,
        routes,
        corpus=corpus,
        split=split,
    )
    routes["g2_route_adapter"] = copy.deepcopy(audit)
    return RouteAdapterResult(routes=routes, audit=audit)


def _stable_support_by_route(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    result = {"E": 0, "D": 0}
    for row in rows:
        route = str(row["route"])
        if route in result:
            result[route] += _require_int(
                row["stable_support"],
                context=f"{row['edge_id']}.stable_support",
                minimum=1,
            )
    if result["E"] <= 0 or result["D"] <= 0:
        raise ValueError("support-matched shuffle requires positive E and D mass")
    return result


def _candidate_support(
    active_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> dict[str, int]:
    result = {"E": 0, "D": 0}
    if len(active_rows) != len(labels):
        raise ValueError("shuffle labels and active rows are unaligned")
    for row, route in zip(active_rows, labels):
        if route not in result:
            raise ValueError("shuffle candidate contains a non-E/D label")
        result[route] += int(row["stable_support"])
    return result


def _relative_support_delta(
    original: Mapping[str, int],
    candidate: Mapping[str, int],
) -> dict[str, float]:
    return {
        route: abs(int(candidate[route]) - int(original[route]))
        / int(original[route])
        for route in ("E", "D")
    }


def _candidate_qualifies(
    original_labels: Sequence[str],
    candidate_labels: Sequence[str],
    active_rows: Sequence[Mapping[str, Any]],
    original_support: Mapping[str, int],
    *,
    tolerance: float,
) -> tuple[bool, dict[str, int], dict[str, float]]:
    support = _candidate_support(active_rows, candidate_labels)
    delta = _relative_support_delta(original_support, support)
    qualifies = (
        list(candidate_labels) != list(original_labels)
        and {
            route: list(candidate_labels).count(route)
            for route in ("E", "D")
        }
        == {
            route: list(original_labels).count(route)
            for route in ("E", "D")
        }
        and all(delta[route] <= tolerance + 1e-15 for route in ("E", "D"))
    )
    return qualifies, support, delta


def _replay_first_qualifying_shuffle(
    active_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    maximum_attempts: int,
    tolerance: float,
    stop_at: int | None = None,
) -> tuple[list[str], int, dict[str, int], dict[str, float]]:
    original_labels = [str(row["route"]) for row in active_rows]
    original_support = _stable_support_by_route(active_rows)
    generator = random.Random(seed)
    limit = maximum_attempts if stop_at is None else stop_at
    if limit <= 0 or limit > maximum_attempts:
        raise ValueError("shuffle replay attempt is outside the frozen bound")
    for attempt in range(1, limit + 1):
        candidate = list(original_labels)
        generator.shuffle(candidate)
        qualifies, support, delta = _candidate_qualifies(
            original_labels,
            candidate,
            active_rows,
            original_support,
            tolerance=tolerance,
        )
        if qualifies:
            return candidate, attempt, support, delta
    raise ValueError(
        "no qualifying nonidentity support-matched route shuffle "
        f"within {limit} attempts"
    )


def audit_support_matched_route_shuffle(
    original_routes: object,
    shuffled_routes: object,
    method_contract: object,
    *,
    attempt: int,
) -> dict[str, Any]:
    """Replay rejection sampling and audit that the supplied map is the first pass."""

    contract, expected_counts = _validate_method_contract(method_contract)
    original = _require_object(original_routes, context="original routes")
    shuffled = _require_object(shuffled_routes, context="shuffled routes")
    original_rows = _validate_training_rows(original, expected_counts)
    shuffled_rows = _validate_training_rows(
        shuffled,
        expected_counts,
        require_route_reason_match=False,
    )
    if (
        original.get("decision_protocol") != PROTOCOL
        or shuffled.get("decision_protocol") != PROTOCOL
        or original.get("corpus") != shuffled.get("corpus")
        or original.get("split") != shuffled.get("split")
        or original.get("inputs") != shuffled.get("inputs")
    ):
        raise ValueError("route shuffle changed the training route header")
    if (
        shuffled.get("scope") != ROUTE_SHUFFLE_SCOPE
        or "g2_route_adapter" in shuffled
    ):
        raise ValueError(
            "route shuffle retained stale adapter metadata or an unsafe scope"
        )
    original_ids = [str(row["edge_id"]) for row in original_rows]
    shuffled_ids = [str(row["edge_id"]) for row in shuffled_rows]
    if original_ids != shuffled_ids:
        raise ValueError("route shuffle changed edge identity/order")
    original_projection = [
        {key: copy.deepcopy(value) for key, value in row.items() if key != "route"}
        for row in original_rows
    ]
    shuffled_projection = [
        {key: copy.deepcopy(value) for key, value in row.items() if key != "route"}
        for row in shuffled_rows
    ]
    if original_projection != shuffled_projection:
        raise ValueError("route shuffle changed non-route edge metadata")
    for before, after in zip(original_rows, shuffled_rows):
        if before["route"] == "A" and after["route"] != "A":
            raise ValueError("route shuffle changed an A edge")
        if before["route"] in {"E", "D"} and after["route"] not in {"E", "D"}:
            raise ValueError("route shuffle moved an active edge out of E/D")

    shuffle = contract["route_shuffle"]
    seed = int(shuffle["seed"])
    maximum_attempts = int(shuffle["maximum_attempts"])
    tolerance = float(
        shuffle["maximum_relative_stable_support_delta_per_route"]
    )
    requested_attempt = _require_int(
        attempt, context="shuffle qualifying attempt", minimum=1
    )
    active_original = [
        row for row in original_rows if str(row["route"]) in {"E", "D"}
    ]
    active_shuffled = {
        str(row["edge_id"]): str(row["route"])
        for row in shuffled_rows
        if str(row["route"]) in {"E", "D"}
    }
    expected_labels, observed_attempt, shuffled_support, delta = (
        _replay_first_qualifying_shuffle(
            active_original,
            seed=seed,
            maximum_attempts=maximum_attempts,
            tolerance=tolerance,
            stop_at=requested_attempt,
        )
    )
    if observed_attempt != requested_attempt:
        raise ValueError(
            "supplied shuffle attempt is not the first qualifying attempt"
        )
    expected_assignment = {
        str(row["edge_id"]): label
        for row, label in zip(active_original, expected_labels)
    }
    if active_shuffled != expected_assignment:
        raise ValueError("shuffled labels differ from deterministic replay")

    original_support = _stable_support_by_route(active_original)
    changed = sorted(
        str(before["edge_id"])
        for before, after in zip(original_rows, shuffled_rows)
        if before["route"] != after["route"]
    )
    if not changed:
        raise ValueError("route shuffle is an identity")
    checks = {
        "population_is_all_and_only_sorted_active_edges": (
            [str(row["edge_id"]) for row in active_original]
            == sorted(str(row["edge_id"]) for row in active_original)
        ),
        "first_qualifying_rejection_sample_replayed": True,
        "nonidentity": True,
        "exact_route_counts_preserved": (
            _route_counts(original_rows) == _route_counts(shuffled_rows)
            == expected_counts
        ),
        "route_A_unchanged": all(
            before["route"] != "A" or after["route"] == "A"
            for before, after in zip(original_rows, shuffled_rows)
        ),
        "non_route_edge_metadata_byte_semantics_preserved": (
            original_projection == shuffled_projection
        ),
        "E_support_delta_within_limit": delta["E"] <= tolerance + 1e-15,
        "D_support_delta_within_limit": delta["D"] <= tolerance + 1e-15,
        "occurrence_sample_source_batch_mass_not_used_for_selection": True,
        "control_scope_truthful_and_stale_adapter_removed": True,
        "no_training_authorization": True,
    }
    if not all(checks.values()):
        raise ValueError("support-matched route-shuffle audit failed")
    audit = {
        "schema_version": ROUTE_SHUFFLE_AUDIT_SCHEMA,
        "protocol_name": PROTOCOL,
        "status": "PASS_FIRST_SUPPORT_MATCHED_NONIDENTITY_SHUFFLE_NO_TRAINING",
        "method_contract_sha256": _payload_sha256(contract),
        "seed": seed,
        "qualifying_attempt": requested_attempt,
        "maximum_attempts": maximum_attempts,
        "population": shuffle["population"],
        "method": shuffle["method"],
        "route_counts": dict(expected_counts),
        "stable_support": {
            "original": original_support,
            "shuffled": shuffled_support,
            "relative_delta": delta,
            "maximum_relative_delta_per_route": tolerance,
        },
        "changed_edges": {
            "count": len(changed),
            "edge_ids_sha256": _payload_sha256(changed),
        },
        "row_preservation": {
            "original_non_route_projection_sha256": _payload_sha256(
                original_projection
            ),
            "shuffled_non_route_projection_sha256": _payload_sha256(
                shuffled_projection
            ),
        },
        "selection_exclusions": {
            "occurrence_sample_source_batch_mass": "report_only",
        },
        "checks": checks,
        "authorization": {
            "route_shuffle_complete": True,
            "prototype_publication_authorized": False,
            "offline_artifact_bundle_authorized": False,
            "gradient_calibration_authorized": False,
            "screen_training_authorized": False,
            "main_training_authorized": False,
            "training_authorized": False,
        },
    }
    embedded = shuffled.get("route_shuffle_control")
    if embedded is not None and embedded != audit:
        raise ValueError("embedded route-shuffle audit is inconsistent")
    return audit


def build_support_matched_route_shuffle(
    routes: object,
    method_contract: object,
) -> RouteShuffleResult:
    """Return the frozen first qualifying support-matched E/D permutation."""

    contract, expected_counts = _validate_method_contract(method_contract)
    original = _require_object(routes, context="training routes")
    rows = _validate_training_rows(original, expected_counts)
    active = [row for row in rows if str(row["route"]) in {"E", "D"}]
    shuffle = contract["route_shuffle"]
    labels, attempt, _, _ = _replay_first_qualifying_shuffle(
        active,
        seed=int(shuffle["seed"]),
        maximum_attempts=int(shuffle["maximum_attempts"]),
        tolerance=float(
            shuffle["maximum_relative_stable_support_delta_per_route"]
        ),
    )
    assignment = {
        str(row["edge_id"]): label for row, label in zip(active, labels)
    }
    output = copy.deepcopy(dict(original))
    output.pop("route_shuffle_control", None)
    output.pop("g2_route_adapter", None)
    output["scope"] = copy.deepcopy(ROUTE_SHUFFLE_SCOPE)
    for row in output["edges"]:
        edge_id = str(row["edge_id"])
        if edge_id in assignment:
            row["route"] = assignment[edge_id]
    output["route_counts"] = _route_counts(output["edges"])
    audit = audit_support_matched_route_shuffle(
        original,
        output,
        contract,
        attempt=attempt,
    )
    output["route_shuffle_control"] = copy.deepcopy(audit)
    return RouteShuffleResult(routes=output, audit=audit)


__all__ = [
    "G1_MASS_GATE_SCHEMA",
    "G1_ROUTES_SCHEMA",
    "METHOD_CONTRACT_SCHEMA",
    "PROTOCOL",
    "ROUTE_ADAPTER_AUDIT_SCHEMA",
    "ROUTE_SHUFFLE_AUDIT_SCHEMA",
    "ROUTE_SHUFFLE_SCOPE",
    "RouteAdapterResult",
    "RouteShuffleResult",
    "adapt_g1_routes",
    "audit_adapted_routes",
    "audit_support_matched_route_shuffle",
    "build_support_matched_route_shuffle",
]
