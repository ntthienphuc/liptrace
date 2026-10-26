"""Deterministic multiset derangements for event-level negative controls.

The control keeps every route-E event in place and changes only the edge that
supplies its confusion prototype.  A plan is represented in grouped edge-ID
space, so applying it to a large occurrence file needs only one ordinal counter
per edge instead of one Python object per event.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
import hashlib
from typing import Mapping, Sequence


SCHEMA_EVENT_NEGATIVE_DERANGEMENT = "pavir-event-negative-derangement-v1"
ALGORITHM_EVENT_NEGATIVE_DERANGEMENT = (
    "sha256-seeded-edge-block-order-max-multiplicity-cyclic-shift-v1"
)


@dataclass(frozen=True)
class EventNegativeDerangementPlan:
    """A bijective, zero-identity assignment over a multiset of edge IDs."""

    seed: int
    edge_order: tuple[str, ...]
    edge_counts: tuple[int, ...]
    block_starts: tuple[int, ...]
    maximum_multiplicity: int
    total_events: int
    edge_indices: Mapping[str, int] = field(repr=False, compare=False)

    @classmethod
    def from_counts(
        cls,
        counts: Mapping[str, int],
        *,
        seed: int,
    ) -> "EventNegativeDerangementPlan":
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("derangement seed must be an integer")
        normalized: dict[str, int] = {}
        for raw_edge_id, raw_count in counts.items():
            edge_id = str(raw_edge_id)
            if not edge_id:
                raise ValueError("derangement edge IDs must be non-empty")
            if isinstance(raw_count, bool) or not isinstance(raw_count, int):
                raise ValueError(f"derangement count for {edge_id!r} must be an integer")
            if raw_count <= 0:
                raise ValueError(f"derangement count for {edge_id!r} must be positive")
            if edge_id in normalized:
                raise ValueError(f"duplicate derangement edge ID: {edge_id!r}")
            normalized[edge_id] = int(raw_count)
        if len(normalized) < 2:
            raise ValueError("event-negative derangement needs at least two edge IDs")

        total = sum(normalized.values())
        maximum = max(normalized.values())
        # This is the necessary and sufficient condition for deranging a
        # multiset when equality is defined by edge ID.
        if maximum * 2 > total:
            raise ValueError(
                "event-negative multiset cannot be deranged without identity: "
                f"maximum_multiplicity={maximum}, total_events={total}"
            )

        edge_order = sorted(
            normalized,
            key=lambda edge_id: (
                hashlib.sha256(
                    f"{seed}\x1f{edge_id}".encode("utf-8")
                ).digest(),
                edge_id,
            ),
        )
        edge_counts = tuple(normalized[edge_id] for edge_id in edge_order)
        starts = []
        cursor = 0
        for count in edge_counts:
            starts.append(cursor)
            cursor += count
        if cursor != total:
            raise AssertionError("derangement block accounting mismatch")
        return cls(
            seed=seed,
            edge_order=tuple(edge_order),
            edge_counts=edge_counts,
            block_starts=tuple(starts),
            maximum_multiplicity=maximum,
            total_events=total,
            edge_indices={edge_id: index for index, edge_id in enumerate(edge_order)},
        )

    @property
    def counts_by_edge(self) -> dict[str, int]:
        return dict(zip(self.edge_order, self.edge_counts))

    def negative_edge_id(self, edge_id: str, occurrence_ordinal: int) -> str:
        """Return the negative edge assigned to one occurrence of ``edge_id``.

        ``occurrence_ordinal`` is zero-based among events with the same positive
        edge ID, in any deterministic traversal of the occurrence artifact.
        """

        try:
            edge_index = self.edge_indices[str(edge_id)]
        except KeyError as error:
            raise KeyError(f"edge absent from derangement plan: {edge_id!r}") from error
        count = self.edge_counts[edge_index]
        if (
            isinstance(occurrence_ordinal, bool)
            or not isinstance(occurrence_ordinal, int)
            or occurrence_ordinal < 0
            or occurrence_ordinal >= count
        ):
            raise IndexError(
                f"occurrence ordinal outside [0,{count}) for edge {edge_id!r}: "
                f"{occurrence_ordinal!r}"
            )

        grouped_slot = self.block_starts[edge_index] + occurrence_ordinal
        negative_slot = (
            grouped_slot + self.maximum_multiplicity
        ) % self.total_events
        negative_index = bisect_right(self.block_starts, negative_slot) - 1
        negative = self.edge_order[negative_index]
        if negative == edge_id:
            raise AssertionError("derangement plan emitted an identity assignment")
        return negative


def derange_event_edge_ids(
    edge_ids: Sequence[str],
    *,
    seed: int,
) -> tuple[str, ...]:
    """Convenience wrapper used by tests and small in-memory callers."""

    values = tuple(str(edge_id) for edge_id in edge_ids)
    plan = EventNegativeDerangementPlan.from_counts(Counter(values), seed=seed)
    ordinals: Counter[str] = Counter()
    output = []
    for edge_id in values:
        output.append(plan.negative_edge_id(edge_id, ordinals[edge_id]))
        ordinals[edge_id] += 1
    return tuple(output)
