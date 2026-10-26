from collections import Counter
import itertools
import math

import numpy as np
import pytest

from pavir.mincost_derangement import (
    COST_QUANTIZATION_SCALE,
    assign_event_keys,
    solve_component_mincost,
)
from pavir.stratified_derangement import DerangementComponent


def _unit(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    return np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)


def _fixture():
    edges = ("a", "b", "c", "d")
    targets = dict(zip(edges, map(_unit, (0.0, 70.0, 180.0, 250.0))))
    confusions = dict(zip(edges, map(_unit, (20.0, 95.0, 205.0, 275.0))))
    cosines = {
        edge: float(np.dot(targets[edge], confusions[edge])) for edge in edges
    }
    component = DerangementComponent(
        component_id="component-test",
        initial_strata=((1, 1, 0),),
        edge_ids=edges,
        event_counts=(1, 1, 1, 1),
        length_pairs=((1, 1),),
        deciles=(0,),
        cosine_center=float(np.mean(list(cosines.values()))),
        event_count=4,
        maximum_multiplicity=1,
    )
    forbidden = {edge: frozenset({edge}) for edge in edges}
    return component, targets, confusions, cosines, forbidden


def test_network_simplex_reaches_bruteforce_primary_optimum() -> None:
    component, targets, confusions, cosines, forbidden = _fixture()
    result = solve_component_mincost(
        component,
        target_units=targets,
        confusion_units=confusions,
        endpoint_cosines=cosines,
        forbidden_negative_edges=forbidden,
        seed=42,
    )
    brute = []
    for permutation in itertools.permutations(component.edge_ids):
        if any(
            negative in forbidden[positive]
            for positive, negative in zip(component.edge_ids, permutation)
        ):
            continue
        brute.append(
            sum(
                int(
                    np.rint(
                        abs(
                            float(np.dot(targets[positive], confusions[negative]))
                            - cosines[positive]
                        )
                        * COST_QUANTIZATION_SCALE
                    )
                )
                for positive, negative in zip(component.edge_ids, permutation)
            )
        )
    assert result.primary_quantized_total == min(brute)
    assert result.negative_counts() == Counter(component.event_counts and {e: 1 for e in component.edge_ids})


def test_mincost_and_event_assignment_are_byte_deterministic_in_semantics() -> None:
    component, targets, confusions, cosines, forbidden = _fixture()
    kwargs = dict(
        target_units=targets,
        confusion_units=confusions,
        endpoint_cosines=cosines,
        forbidden_negative_edges=forbidden,
        seed=42,
    )
    first = solve_component_mincost(component, **kwargs)
    second = solve_component_mincost(component, **kwargs)
    assert first == second
    keys = {edge: [(edge, 0), (edge, 1)] for edge in component.edge_ids}
    # Double each flow quota for this event-assignment-only check.
    doubled = type(first)(
        **{
            **first.__dict__,
            "flows_by_positive_edge": {
                edge: tuple((negative, count * 2) for negative, count in flows)
                for edge, flows in first.flows_by_positive_edge.items()
            },
        }
    )
    mapping_a = assign_event_keys(
        keys,
        {component.component_id: doubled},
        {edge: component.component_id for edge in component.edge_ids},
        seed=42,
    )
    mapping_b = assign_event_keys(
        {edge: list(reversed(value)) for edge, value in keys.items()},
        {component.component_id: doubled},
        {edge: component.component_id for edge in component.edge_ids},
        seed=42,
    )
    assert mapping_a == mapping_b


def test_mincost_fails_closed_when_semantic_arcs_are_infeasible() -> None:
    component, targets, confusions, cosines, _ = _fixture()
    forbidden = {
        edge: frozenset(component.edge_ids) for edge in component.edge_ids
    }
    with pytest.raises(ValueError, match="no allowed arcs|infeasible"):
        solve_component_mincost(
            component,
            target_units=targets,
            confusion_units=confusions,
            endpoint_cosines=cosines,
            forbidden_negative_edges=forbidden,
            seed=42,
        )
