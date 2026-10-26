"""Stratified event-level derangements for the PAViR exact-error control."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import statistics
from typing import Any, Mapping

SCHEMA_STRATIFIED_EVENT_DERANGEMENT = (
    "pavir-stratified-event-negative-derangement-v1"
)
ALGORITHM_STRATIFIED_EVENT_DERANGEMENT = (
    "unweighted-edge-rank-decile-nearest-component-constrained-maxflow-v1"
)
DECILE_DEFINITION = (
    "sort route-E edges by (target/confusion prototype cosine, edge_id); "
    "decile=min(9,floor(zero_based_rank*10/edge_count))"
)
MERGE_PRIORITY_DEFINITION = (
    "for the lexicographically first infeasible component, minimize "
    "(different_exact_token_length_pair, absolute_unweighted_cosine_center_"
    "distance, minimum_token_length_L1_distance, candidate_initial_strata, "
    "candidate_component_id)"
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


def _content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def unweighted_rank_deciles(
    endpoint_cosines: Mapping[str, float],
    *,
    bins: int = 10,
) -> dict[str, int]:
    """Assign deterministic equal-rank bins with one vote per route-E edge."""

    if bins <= 1:
        raise ValueError("decile bin count must exceed one")
    normalized: dict[str, float] = {}
    for raw_edge_id, raw_cosine in endpoint_cosines.items():
        edge_id = str(raw_edge_id)
        cosine = float(raw_cosine)
        if not edge_id or edge_id in normalized:
            raise ValueError(f"empty or duplicate cosine edge ID: {edge_id!r}")
        if not math.isfinite(cosine) or cosine < -1.00001 or cosine > 1.00001:
            raise ValueError(f"invalid endpoint cosine for {edge_id}: {cosine}")
        normalized[edge_id] = cosine
    if not normalized:
        raise ValueError("cannot assign deciles to an empty edge population")
    ordered = sorted(normalized, key=lambda edge_id: (normalized[edge_id], edge_id))
    size = len(ordered)
    return {
        edge_id: min(bins - 1, index * bins // size)
        for index, edge_id in enumerate(ordered)
    }


@dataclass(frozen=True)
class DerangementComponent:
    component_id: str
    initial_strata: tuple[tuple[int, int, int], ...]
    edge_ids: tuple[str, ...]
    event_counts: tuple[int, ...]
    length_pairs: tuple[tuple[int, int], ...]
    deciles: tuple[int, ...]
    cosine_center: float
    event_count: int
    maximum_multiplicity: int

    @property
    def counts_by_edge(self) -> dict[str, int]:
        return dict(zip(self.edge_ids, self.event_counts))

    @property
    def derangeable(self) -> bool:
        return (
            len(self.edge_ids) >= 2
            and self.maximum_multiplicity * 2 <= self.event_count
        )

    def audit_record(
        self,
        *,
        constrained_derangeable: bool | None = None,
    ) -> dict[str, Any]:
        record = {
            "component_id": self.component_id,
            "initial_strata": [list(key) for key in self.initial_strata],
            "edge_count": len(self.edge_ids),
            "event_count": self.event_count,
            "maximum_multiplicity": self.maximum_multiplicity,
            "length_pairs": [list(value) for value in self.length_pairs],
            "deciles": list(self.deciles),
            "unweighted_cosine_center": self.cosine_center,
            "edge_ids_sha256": _content_hash(list(self.edge_ids)),
            "event_multiset_sha256": _content_hash(
                [
                    {"edge_id": edge_id, "events": count}
                    for edge_id, count in zip(self.edge_ids, self.event_counts)
                ]
            ),
            "derangeable": self.derangeable,
        }
        if constrained_derangeable is not None:
            record["constrained_derangeable"] = constrained_derangeable
        return record


def _make_component(
    *,
    initial_strata: tuple[tuple[int, int, int], ...],
    edge_ids: tuple[str, ...],
    event_counts: Mapping[str, int],
    target_token_lengths: Mapping[str, int],
    confusion_token_lengths: Mapping[str, int],
    edge_deciles: Mapping[str, int],
    endpoint_cosines: Mapping[str, float],
) -> DerangementComponent:
    strata = tuple(sorted(set(initial_strata)))
    edges = tuple(sorted(set(edge_ids)))
    if not strata or not edges:
        raise ValueError("derangement component cannot be empty")
    counts = tuple(int(event_counts[edge_id]) for edge_id in edges)
    if any(count <= 0 for count in counts):
        raise ValueError("derangement component event counts must be positive")
    length_pairs = tuple(
        sorted(
            {
                (
                    int(target_token_lengths[edge_id]),
                    int(confusion_token_lengths[edge_id]),
                )
                for edge_id in edges
            }
        )
    )
    deciles = tuple(sorted({int(edge_deciles[edge_id]) for edge_id in edges}))
    cosine_center = statistics.fmean(endpoint_cosines[edge_id] for edge_id in edges)
    identity = {
        "initial_strata": [list(key) for key in strata],
        "edge_event_counts": [
            {"edge_id": edge_id, "events": count}
            for edge_id, count in zip(edges, counts)
        ],
    }
    return DerangementComponent(
        component_id=_content_hash(identity),
        initial_strata=strata,
        edge_ids=edges,
        event_counts=counts,
        length_pairs=length_pairs,
        deciles=deciles,
        cosine_center=cosine_center,
        event_count=sum(counts),
        maximum_multiplicity=max(counts),
    )


def _minimum_length_distance(
    left: DerangementComponent,
    right: DerangementComponent,
) -> int:
    return min(
        abs(left_target - right_target) + abs(left_confusion - right_confusion)
        for left_target, left_confusion in left.length_pairs
        for right_target, right_confusion in right.length_pairs
    )


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int


class _Dinic:
    def __init__(self, nodes: int):
        self.graph: list[list[_FlowEdge]] = [[] for _ in range(nodes)]

    def add_edge(self, source: int, target: int, capacity: int) -> _FlowEdge:
        forward = _FlowEdge(target, len(self.graph[target]), int(capacity))
        reverse = _FlowEdge(source, len(self.graph[source]), 0)
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return forward

    def max_flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            levels = [-1] * len(self.graph)
            levels[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge.capacity > 0 and levels[edge.to] < 0:
                        levels[edge.to] = levels[node] + 1
                        queue.append(edge.to)
            if levels[sink] < 0:
                return total
            offsets = [0] * len(self.graph)

            def push(node: int, available: int) -> int:
                if node == sink:
                    return available
                while offsets[node] < len(self.graph[node]):
                    edge = self.graph[node][offsets[node]]
                    if edge.capacity > 0 and levels[edge.to] == levels[node] + 1:
                        sent = push(edge.to, min(available, edge.capacity))
                        if sent:
                            edge.capacity -= sent
                            self.graph[edge.to][edge.reverse].capacity += sent
                            return sent
                    offsets[node] += 1
                return 0

            while sent := push(source, 1 << 60):
                total += sent


@dataclass(frozen=True)
class ConstrainedComponentAssignment:
    component_id: str
    runs_by_positive_edge: Mapping[str, tuple[tuple[str, int], ...]]
    flow_sha256: str

    def negative_edge_id(self, edge_id: str, occurrence_ordinal: int) -> str:
        try:
            runs = self.runs_by_positive_edge[str(edge_id)]
        except KeyError as error:
            raise KeyError(
                f"edge absent from constrained component assignment: {edge_id!r}"
            ) from error
        if isinstance(occurrence_ordinal, bool) or not isinstance(
            occurrence_ordinal, int
        ):
            raise IndexError("occurrence ordinal must be an integer")
        if occurrence_ordinal < 0 or not runs or occurrence_ordinal >= runs[-1][1]:
            raise IndexError(
                f"occurrence ordinal outside component assignment for {edge_id!r}: "
                f"{occurrence_ordinal!r}"
            )
        for negative_edge_id, cumulative_end in runs:
            if occurrence_ordinal < cumulative_end:
                return negative_edge_id
        raise AssertionError("component assignment run lookup failed")


def _seeded_hash(seed: int, *values: str) -> bytes:
    return hashlib.sha256(
        (str(seed) + "\x1f" + "\x1f".join(values)).encode("utf-8")
    ).digest()


def _solve_component_assignment(
    component: DerangementComponent,
    *,
    forbidden_negative_edges: Mapping[str, frozenset[str]],
    target_token_lengths: Mapping[str, int],
    confusion_token_lengths: Mapping[str, int],
    edge_deciles: Mapping[str, int],
    endpoint_cosines: Mapping[str, float],
    seed: int,
) -> ConstrainedComponentAssignment | None:
    if not component.derangeable:
        return None
    counts = component.counts_by_edge
    positives = sorted(
        component.edge_ids,
        key=lambda edge_id: (_seeded_hash(seed, "positive", edge_id), edge_id),
    )
    negatives = tuple(component.edge_ids)
    source = 0
    positive_offset = 1
    negative_offset = positive_offset + len(positives)
    sink = negative_offset + len(negatives)
    network = _Dinic(sink + 1)
    positive_nodes = {
        edge_id: positive_offset + index for index, edge_id in enumerate(positives)
    }
    negative_nodes = {
        edge_id: negative_offset + index for index, edge_id in enumerate(negatives)
    }
    for edge_id in positives:
        network.add_edge(source, positive_nodes[edge_id], counts[edge_id])
    for edge_id in negatives:
        network.add_edge(negative_nodes[edge_id], sink, counts[edge_id])

    flow_edges: dict[tuple[str, str], _FlowEdge] = {}
    negative_orders: dict[str, tuple[str, ...]] = {}
    total_events = component.event_count
    for positive in positives:
        allowed = [
            negative
            for negative in negatives
            if negative not in forbidden_negative_edges[positive]
        ]
        if not allowed:
            return None
        allowed.sort(
            key=lambda negative: (
                0
                if (
                    target_token_lengths[positive] == target_token_lengths[negative]
                    and confusion_token_lengths[positive]
                    == confusion_token_lengths[negative]
                )
                else 1,
                0 if edge_deciles[positive] == edge_deciles[negative] else 1,
                abs(endpoint_cosines[positive] - endpoint_cosines[negative]),
                _seeded_hash(seed, "negative", positive, negative),
                negative,
            )
        )
        negative_orders[positive] = tuple(allowed)
        for negative in allowed:
            flow_edges[(positive, negative)] = network.add_edge(
                positive_nodes[positive],
                negative_nodes[negative],
                total_events,
            )
    if network.max_flow(source, sink) != total_events:
        return None

    runs: dict[str, tuple[tuple[str, int], ...]] = {}
    flow_records = []
    assigned_counts: dict[str, int] = defaultdict(int)
    for positive in sorted(component.edge_ids):
        cumulative = 0
        edge_runs = []
        for negative in negative_orders[positive]:
            forward = flow_edges[(positive, negative)]
            flow = network.graph[forward.to][forward.reverse].capacity
            if flow <= 0:
                continue
            cumulative += flow
            assigned_counts[negative] += flow
            edge_runs.append((negative, cumulative))
            flow_records.append(
                {
                    "positive_edge_id": positive,
                    "negative_edge_id": negative,
                    "events": flow,
                }
            )
        if cumulative != counts[positive]:
            raise AssertionError("constrained assignment positive mass mismatch")
        runs[positive] = tuple(edge_runs)
    if dict(assigned_counts) != counts:
        raise AssertionError("constrained assignment negative mass mismatch")
    return ConstrainedComponentAssignment(
        component_id=component.component_id,
        runs_by_positive_edge=runs,
        flow_sha256=_content_hash(flow_records),
    )


@dataclass(frozen=True)
class StratifiedEventNegativePlan:
    seed: int
    edge_deciles: Mapping[str, int]
    edge_components: Mapping[str, str]
    component_plans: Mapping[str, ConstrainedComponentAssignment]
    initial_strata: tuple[dict[str, Any], ...]
    merge_trace: tuple[dict[str, Any], ...]
    final_components: tuple[DerangementComponent, ...]
    initial_strata_sha256: str
    merge_trace_sha256: str
    final_components_sha256: str
    decile_assignment_sha256: str
    component_seeds: Mapping[str, int] = field(repr=False)

    def negative_edge_id(self, edge_id: str, occurrence_ordinal: int) -> str:
        try:
            component_id = self.edge_components[str(edge_id)]
        except KeyError as error:
            raise KeyError(f"edge absent from stratified plan: {edge_id!r}") from error
        return self.component_plans[component_id].negative_edge_id(
            str(edge_id), occurrence_ordinal
        )


def build_stratified_event_negative_plan(
    *,
    event_counts: Mapping[str, int],
    target_token_lengths: Mapping[str, int],
    confusion_token_lengths: Mapping[str, int],
    endpoint_cosines: Mapping[str, float],
    forbidden_negative_edges: Mapping[str, set[str] | frozenset[str]] | None = None,
    seed: int,
) -> StratifiedEventNegativePlan:
    """Build the deterministic nearest-component merge and per-component plans."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("stratified derangement seed must be an integer")
    edge_ids = set(map(str, event_counts))
    if not edge_ids:
        raise ValueError("stratified derangement requires route-E events")
    for name, values in (
        ("target lengths", target_token_lengths),
        ("confusion lengths", confusion_token_lengths),
        ("endpoint cosines", endpoint_cosines),
    ):
        if set(map(str, values)) != edge_ids:
            raise ValueError(f"{name} edge population does not match event counts")
    normalized_counts: dict[str, int] = {}
    normalized_target_lengths: dict[str, int] = {}
    normalized_confusion_lengths: dict[str, int] = {}
    normalized_cosines: dict[str, float] = {}
    for edge_id in sorted(edge_ids):
        count = event_counts[edge_id]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"invalid event count for {edge_id}: {count!r}")
        target_length = target_token_lengths[edge_id]
        confusion_length = confusion_token_lengths[edge_id]
        if (
            isinstance(target_length, bool)
            or not isinstance(target_length, int)
            or target_length <= 0
            or isinstance(confusion_length, bool)
            or not isinstance(confusion_length, int)
            or confusion_length <= 0
        ):
            raise ValueError(f"invalid token lengths for {edge_id}")
        cosine = float(endpoint_cosines[edge_id])
        if not math.isfinite(cosine):
            raise ValueError(f"non-finite endpoint cosine for {edge_id}")
        normalized_counts[edge_id] = int(count)
        normalized_target_lengths[edge_id] = int(target_length)
        normalized_confusion_lengths[edge_id] = int(confusion_length)
        normalized_cosines[edge_id] = cosine

    raw_forbidden = forbidden_negative_edges or {}
    unknown_positive_edges = set(map(str, raw_forbidden)) - edge_ids
    if unknown_positive_edges:
        raise ValueError(
            "forbidden-negative map references unknown positive edges: "
            f"{sorted(unknown_positive_edges)[:5]}"
        )
    normalized_forbidden: dict[str, frozenset[str]] = {}
    for edge_id in sorted(edge_ids):
        values = raw_forbidden.get(edge_id, ())
        if isinstance(values, (str, bytes)):
            raise ValueError(
                f"forbidden negatives for {edge_id} must be an edge-ID collection"
            )
        forbidden = {str(value) for value in values}
        unknown_negative_edges = forbidden - edge_ids
        if unknown_negative_edges:
            raise ValueError(
                f"forbidden negatives for {edge_id} reference unknown edges: "
                f"{sorted(unknown_negative_edges)[:5]}"
            )
        # Edge identity is never a valid negative, even if the caller omits it.
        forbidden.add(edge_id)
        normalized_forbidden[edge_id] = frozenset(forbidden)

    component_assignments: dict[str, ConstrainedComponentAssignment | None] = {}
    component_seeds: dict[str, int] = {}

    def component_seed(component: DerangementComponent) -> int:
        value = component_seeds.get(component.component_id)
        if value is None:
            value = int.from_bytes(
                hashlib.sha256(
                    f"{seed}\x1f{component.component_id}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            component_seeds[component.component_id] = value
        return value

    def solve(
        component: DerangementComponent,
    ) -> ConstrainedComponentAssignment | None:
        if component.component_id not in component_assignments:
            component_assignments[component.component_id] = (
                _solve_component_assignment(
                    component,
                    forbidden_negative_edges=normalized_forbidden,
                    target_token_lengths=normalized_target_lengths,
                    confusion_token_lengths=normalized_confusion_lengths,
                    edge_deciles=edge_deciles,
                    endpoint_cosines=normalized_cosines,
                    seed=component_seed(component),
                )
            )
        return component_assignments[component.component_id]

    edge_deciles = unweighted_rank_deciles(normalized_cosines)
    grouped: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for edge_id in sorted(edge_ids):
        key = (
            normalized_target_lengths[edge_id],
            normalized_confusion_lengths[edge_id],
            edge_deciles[edge_id],
        )
        grouped[key].append(edge_id)

    components = [
        _make_component(
            initial_strata=(key,),
            edge_ids=tuple(grouped[key]),
            event_counts=normalized_counts,
            target_token_lengths=normalized_target_lengths,
            confusion_token_lengths=normalized_confusion_lengths,
            edge_deciles=edge_deciles,
            endpoint_cosines=normalized_cosines,
        )
        for key in sorted(grouped)
    ]
    initial_records = tuple(
        component.audit_record(constrained_derangeable=solve(component) is not None)
        for component in components
    )
    merge_trace = []
    while True:
        infeasible = sorted(
            (component for component in components if solve(component) is None),
            key=lambda component: (
                component.initial_strata,
                component.component_id,
            ),
        )
        if not infeasible:
            break
        if len(components) == 1:
            component = components[0]
            raise ValueError(
                "global route-E event multiset cannot satisfy the constrained "
                "derangement: "
                f"maximum_multiplicity={component.maximum_multiplicity}, "
                f"total_events={component.event_count}"
            )
        left = infeasible[0]
        left_lengths = set(left.length_pairs)

        def merge_priority(right: DerangementComponent) -> tuple[Any, ...]:
            same_lengths = bool(left_lengths.intersection(right.length_pairs))
            return (
                0 if same_lengths else 1,
                abs(left.cosine_center - right.cosine_center),
                _minimum_length_distance(left, right),
                right.initial_strata,
                right.component_id,
            )

        right = min(
            (component for component in components if component is not left),
            key=merge_priority,
        )
        priority = merge_priority(right)
        merged = _make_component(
            initial_strata=left.initial_strata + right.initial_strata,
            edge_ids=left.edge_ids + right.edge_ids,
            event_counts=normalized_counts,
            target_token_lengths=normalized_target_lengths,
            confusion_token_lengths=normalized_confusion_lengths,
            edge_deciles=edge_deciles,
            endpoint_cosines=normalized_cosines,
        )
        merge_trace.append(
            {
                "step": len(merge_trace),
                "left_component_id": left.component_id,
                "right_component_id": right.component_id,
                "result_component_id": merged.component_id,
                "same_exact_token_length_pair_priority": priority[0] == 0,
                "absolute_unweighted_cosine_center_distance": priority[1],
                "minimum_token_length_L1_distance": priority[2],
                "result_derangeable": merged.derangeable,
                "result_constrained_derangeable": solve(merged) is not None,
                "result_event_count": merged.event_count,
                "result_maximum_multiplicity": merged.maximum_multiplicity,
            }
        )
        components = [
            component
            for component in components
            if component is not left and component is not right
        ] + [merged]

    final_components = tuple(
        sorted(
            components,
            key=lambda component: (
                component.initial_strata,
                component.component_id,
            ),
        )
    )
    edge_components: dict[str, str] = {}
    component_plans: dict[str, ConstrainedComponentAssignment] = {}
    for component in final_components:
        assignment = solve(component)
        if assignment is None:
            raise AssertionError("final constrained component is infeasible")
        component_plans[component.component_id] = assignment
        for edge_id in component.edge_ids:
            if edge_id in edge_components:
                raise AssertionError(f"edge assigned to multiple components: {edge_id}")
            edge_components[edge_id] = component.component_id
    if set(edge_components) != edge_ids:
        raise AssertionError("final components do not cover the edge population")

    final_records = tuple(
        {
            **component.audit_record(constrained_derangeable=True),
            "assignment_flow_sha256": component_plans[
                component.component_id
            ].flow_sha256,
        }
        for component in final_components
    )
    decile_records = [
        {
            "edge_id": edge_id,
            "endpoint_cosine": normalized_cosines[edge_id],
            "decile": edge_deciles[edge_id],
        }
        for edge_id in sorted(edge_ids)
    ]
    return StratifiedEventNegativePlan(
        seed=seed,
        edge_deciles=edge_deciles,
        edge_components=edge_components,
        component_plans=component_plans,
        initial_strata=initial_records,
        merge_trace=tuple(merge_trace),
        final_components=final_components,
        initial_strata_sha256=_content_hash(initial_records),
        merge_trace_sha256=_content_hash(merge_trace),
        final_components_sha256=_content_hash(final_records),
        decile_assignment_sha256=_content_hash(decile_records),
        component_seeds=component_seeds,
    )
