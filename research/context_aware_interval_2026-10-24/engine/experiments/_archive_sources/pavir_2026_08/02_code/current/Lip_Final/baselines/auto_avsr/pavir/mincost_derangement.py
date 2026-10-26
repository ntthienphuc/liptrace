"""Exact min-cost transportation for the PAViR event-negative control.

The feasibility/merge layer lives in :mod:`pavir.stratified_derangement`.
This module takes one feasible final component and selects the mass-exact
positive-edge to negative-edge transportation plan that best matches the
original endpoint difficulty.  It is an offline artifact primitive only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from .stratified_derangement import DerangementComponent


SCHEMA_STRATIFIED_MINCOST_DERANGEMENT = (
    "pavir-stratified-event-negative-mincost-derangement-v1"
)
ALGORITHM_STRATIFIED_MINCOST_DERANGEMENT = (
    "stratified-constrained-network-simplex-cross-pair-l1-v1"
)
COST_QUANTIZATION_SCALE = 1_000_000_000
COST_DEFINITION = (
    "round-half-to-even(1e9 * abs(dot(target_unit[p], confusion_unit[q]) "
    "- dot(target_unit[p], confusion_unit[p])))"
)
TIE_BREAK_DEFINITION = (
    "primary quantized total cost is lexicographically dominant; remaining "
    "ties minimize the total rank of arcs ordered by "
    "SHA256(seed|component_id|positive_edge_id|negative_edge_id), then IDs; "
    "nodes and arcs are inserted in canonical sorted order"
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _domain_hash(seed: int, component_id: str, *values: str) -> bytes:
    return hashlib.sha256(
        "\x1f".join(
            ("pavir-v4-mincost-v1", str(seed), component_id, *values)
        ).encode("utf-8")
    ).digest()


@dataclass(frozen=True)
class MinCostComponentAssignment:
    component_id: str
    flows_by_positive_edge: Mapping[str, tuple[tuple[str, int], ...]]
    primary_quantized_total: int
    primary_float_total: float
    combined_network_objective: int
    primary_multiplier: int
    arc_count: int
    flow_sha256: str
    networkx_version: str

    def negative_counts(self) -> Counter[str]:
        output: Counter[str] = Counter()
        for flows in self.flows_by_positive_edge.values():
            for edge_id, count in flows:
                output[edge_id] += int(count)
        return output

    def audit_record(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "arc_count": self.arc_count,
            "primary_quantized_total": self.primary_quantized_total,
            "primary_float_total": self.primary_float_total,
            "combined_network_objective": self.combined_network_objective,
            "primary_multiplier": self.primary_multiplier,
            "flow_sha256": self.flow_sha256,
            "networkx_version": self.networkx_version,
            "optimality": "exact network_simplex optimum for integer combined cost",
        }


def _validate_vectors(
    component: DerangementComponent,
    target_units: Mapping[str, np.ndarray],
    confusion_units: Mapping[str, np.ndarray],
    endpoint_cosines: Mapping[str, float],
    forbidden_negative_edges: Mapping[str, frozenset[str] | set[str]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    edge_ids = set(component.edge_ids)
    for name, values in (
        ("target units", target_units),
        ("confusion units", confusion_units),
        ("endpoint cosines", endpoint_cosines),
        ("forbidden negative edges", forbidden_negative_edges),
    ):
        if set(map(str, values)) != edge_ids:
            raise ValueError(f"component {name} edge population mismatch")
    targets: dict[str, np.ndarray] = {}
    confusions: dict[str, np.ndarray] = {}
    cosines: dict[str, float] = {}
    dimension: int | None = None
    for edge_id in component.edge_ids:
        target = np.asarray(target_units[edge_id], dtype=np.float64)
        confusion = np.asarray(confusion_units[edge_id], dtype=np.float64)
        cosine = float(endpoint_cosines[edge_id])
        if (
            target.ndim != 1
            or confusion.ndim != 1
            or target.shape != confusion.shape
            or not target.size
            or not np.isfinite(target).all()
            or not np.isfinite(confusion).all()
            or not math.isfinite(cosine)
        ):
            raise ValueError(f"invalid min-cost geometry for {edge_id}")
        if dimension is None:
            dimension = int(target.size)
        if target.size != dimension:
            raise ValueError("min-cost prototype dimensions differ")
        if abs(float(np.linalg.norm(target)) - 1.0) > 1e-8:
            raise ValueError(f"target vector is not unit normalized: {edge_id}")
        if abs(float(np.linalg.norm(confusion)) - 1.0) > 1e-8:
            raise ValueError(f"confusion vector is not unit normalized: {edge_id}")
        observed = float(np.dot(target, confusion))
        if abs(observed - cosine) > 1e-10:
            raise ValueError(f"endpoint cosine disagrees with vectors: {edge_id}")
        targets[edge_id] = target
        confusions[edge_id] = confusion
        cosines[edge_id] = cosine
    return targets, confusions, cosines


def solve_component_mincost(
    component: DerangementComponent,
    *,
    target_units: Mapping[str, np.ndarray],
    confusion_units: Mapping[str, np.ndarray],
    endpoint_cosines: Mapping[str, float],
    forbidden_negative_edges: Mapping[str, frozenset[str] | set[str]],
    seed: int,
) -> MinCostComponentAssignment:
    """Return the exact deterministic integer min-cost transportation plan."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("min-cost seed must be an integer")
    targets, confusions, cosines = _validate_vectors(
        component,
        target_units,
        confusion_units,
        endpoint_cosines,
        forbidden_negative_edges,
    )
    counts = component.counts_by_edge
    allowed_records: list[tuple[str, str, int, float]] = []
    for positive in sorted(component.edge_ids):
        forbidden = {str(value) for value in forbidden_negative_edges[positive]}
        for negative in sorted(component.edge_ids):
            if negative in forbidden:
                continue
            difference = abs(
                float(np.dot(targets[positive], confusions[negative]))
                - cosines[positive]
            )
            quantized = int(np.rint(difference * COST_QUANTIZATION_SCALE))
            if quantized < 0:
                raise AssertionError("quantized min-cost difference is negative")
            allowed_records.append((positive, negative, quantized, difference))
    if not allowed_records:
        raise ValueError("min-cost component has no allowed arcs")

    ranked = sorted(
        ((positive, negative) for positive, negative, _, _ in allowed_records),
        key=lambda pair: (
            _domain_hash(seed, component.component_id, pair[0], pair[1]),
            pair,
        ),
    )
    arc_rank = {pair: rank for rank, pair in enumerate(ranked)}
    # Any one-unit increase in primary cost must dominate the maximum possible
    # total tie-rank difference over the whole component.
    primary_multiplier = component.event_count * max(len(ranked) - 1, 0) + 1

    graph = nx.DiGraph()
    for edge_id in sorted(component.edge_ids):
        graph.add_node(f"p:{edge_id}", demand=-counts[edge_id])
    for edge_id in sorted(component.edge_ids):
        graph.add_node(f"q:{edge_id}", demand=counts[edge_id])
    cost_lookup: dict[tuple[str, str], tuple[int, float]] = {}
    for positive, negative, quantized, difference in sorted(
        allowed_records, key=lambda row: (row[0], row[1])
    ):
        rank = arc_rank[(positive, negative)]
        graph.add_edge(
            f"p:{positive}",
            f"q:{negative}",
            capacity=component.event_count,
            weight=quantized * primary_multiplier + rank,
        )
        cost_lookup[(positive, negative)] = (quantized, difference)

    try:
        combined_objective, raw_flow = nx.network_simplex(graph)
    except (nx.NetworkXError, nx.NetworkXUnfeasible) as error:
        raise ValueError(
            f"component is infeasible for exact min-cost transport: "
            f"{component.component_id}"
        ) from error

    flows: dict[str, tuple[tuple[str, int], ...]] = {}
    assigned: Counter[str] = Counter()
    primary_quantized_total = 0
    primary_float_total = 0.0
    records = []
    for positive in sorted(component.edge_ids):
        values = []
        positive_total = 0
        for negative in sorted(component.edge_ids):
            amount = int(
                raw_flow.get(f"p:{positive}", {}).get(f"q:{negative}", 0)
            )
            if amount <= 0:
                continue
            quantized, difference = cost_lookup[(positive, negative)]
            values.append((negative, amount))
            positive_total += amount
            assigned[negative] += amount
            primary_quantized_total += amount * quantized
            primary_float_total += amount * difference
            records.append(
                {
                    "positive_edge_id": positive,
                    "negative_edge_id": negative,
                    "events": amount,
                    "quantized_cost_per_event": quantized,
                }
            )
        if positive_total != counts[positive]:
            raise AssertionError("min-cost positive mass mismatch")
        flows[positive] = tuple(values)
    if assigned != Counter(counts):
        raise AssertionError("min-cost negative mass mismatch")
    expected_combined = sum(
        int(record["events"])
        * (
            int(record["quantized_cost_per_event"]) * primary_multiplier
            + arc_rank[
                (
                    str(record["positive_edge_id"]),
                    str(record["negative_edge_id"]),
                )
            ]
        )
        for record in records
    )
    if int(combined_objective) != expected_combined:
        raise AssertionError("network-simplex objective accounting mismatch")
    return MinCostComponentAssignment(
        component_id=component.component_id,
        flows_by_positive_edge=flows,
        primary_quantized_total=primary_quantized_total,
        primary_float_total=primary_float_total,
        combined_network_objective=int(combined_objective),
        primary_multiplier=primary_multiplier,
        arc_count=len(allowed_records),
        flow_sha256=_sha(records),
        networkx_version=str(nx.__version__),
    )


def assign_event_keys(
    event_keys_by_edge: Mapping[str, Sequence[tuple[Any, ...]]],
    assignments: Mapping[str, MinCostComponentAssignment],
    edge_components: Mapping[str, str],
    *,
    seed: int,
) -> dict[tuple[Any, ...], str]:
    """Map exact event identities to flow quotas without row-order bias."""

    output: dict[tuple[Any, ...], str] = {}
    for positive in sorted(event_keys_by_edge):
        try:
            component_id = edge_components[positive]
            flows = assignments[component_id].flows_by_positive_edge[positive]
        except KeyError as error:
            raise ValueError(f"event edge is absent from min-cost plan: {positive}") from error
        keys = list(event_keys_by_edge[positive])
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate exact event identity for {positive}")
        ordered_keys = sorted(
            keys,
            key=lambda key: (
                _domain_hash(
                    seed,
                    component_id,
                    "event",
                    positive,
                    json.dumps(key, ensure_ascii=False, separators=(",", ":")),
                ),
                key,
            ),
        )
        negative_runs = sorted(
            flows,
            key=lambda row: (
                _domain_hash(seed, component_id, "quota", positive, row[0]),
                row[0],
            ),
        )
        cursor = 0
        for negative, count in negative_runs:
            for key in ordered_keys[cursor : cursor + int(count)]:
                output[key] = negative
            cursor += int(count)
        if cursor != len(ordered_keys):
            raise AssertionError("event identities do not exhaust flow quotas")
    if len(output) != sum(len(values) for values in event_keys_by_edge.values()):
        raise AssertionError("min-cost event assignment is incomplete")
    return output
