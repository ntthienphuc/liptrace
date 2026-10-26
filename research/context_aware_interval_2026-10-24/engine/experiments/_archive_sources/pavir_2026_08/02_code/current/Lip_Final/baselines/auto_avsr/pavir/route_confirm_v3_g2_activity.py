"""Static G2 activity schedule and integrity audit for RouteConfirm-v3.

This module intentionally does not load video, extract features, publish
artifacts, or launch calibration/training.  It only:

* maps the frozen AutoAVSR train label to canonical sample/source IDs;
* constructs the exact physical batches produced by ``CustomBucketDataset``;
* selects the prospectively frozen 64-batch schedule; and
* audits route-E / route-D occurrence activity and leakage.

The selected physical indices and the complete sample order within each batch
are emitted in the audit record so a later G3 wrapper can consume the exact
schedule instead of sampling it again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from datamodule.data_module import CustomBucketDataset
from pavir.artifacts import CanonicalIDMap, TrainingEvent, VALID_ROUTES
from pavir.routing import context_batch_selected


SCHEMA_G2_ACTIVITY_AUDIT = "pavir-route-confirm-v3-g2-activity-audit-v1"

G2_CORPUS = "vicocktail"
G2_SPLIT = "train"
G2_SEED = 42
G2_NUM_BUCKETS = 400
G2_MAX_PADDED_FRAMES = 6000
G2_BATCH_SIZE = None
G2_SELECTED_BATCHES = 64
G2_MINIMUM_E_ELIGIBLE_BATCHES = 8
G2_MINIMUM_D_QUOTA_SELECTED_BATCHES = 8
G2_D_CONTEXT_FRACTION = 0.5
G2_SELECTION_DESCRIPTION = (
    "torch.randperm(number_of_physical_batches, "
    "Generator().manual_seed(42))[:64]"
)
G2_D_QUOTA_DESCRIPTION = (
    "locked cumulative quota over D-eligible batch ordinals"
)

_AUTOAVSR_FIELD_COUNTS = frozenset({4, 5, 6, 8})
_ACTIVE_ROUTES = ("E", "D")


@dataclass(frozen=True)
class G2LabelSample:
    """One train-label row after strict canonical-ID resolution."""

    row_index: int
    dataset_name: str
    loader_path: str
    input_length: int
    sample_id: str
    source_id: str


@dataclass(frozen=True)
class PhysicalBatch:
    """An immutable AutoAVSR physical batch in loader sample order."""

    physical_batch_index: int
    row_indices: tuple[int, ...]
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    input_lengths: tuple[int, ...]
    padded_frames: int

    def __post_init__(self) -> None:
        cardinalities = {
            len(self.row_indices),
            len(self.sample_ids),
            len(self.source_ids),
            len(self.input_lengths),
        }
        if len(cardinalities) != 1 or not self.sample_ids:
            raise ValueError("physical batch fields have incompatible cardinalities")
        if self.physical_batch_index < 0:
            raise ValueError("physical batch index must be non-negative")
        if any(length <= 0 for length in self.input_lengths):
            raise ValueError("physical batch input lengths must be positive")
        expected_padded = len(self.input_lengths) * max(self.input_lengths)
        if self.padded_frames != expected_padded:
            raise ValueError(
                "physical batch padded_frames disagrees with its input lengths"
            )


def _parse_integer_sequence(value: str, *, context: str) -> tuple[int, ...]:
    """Mirror AutoAVSR's whitespace-separated integer field parser."""

    pieces = value.split()
    if not pieces:
        raise ValueError(f"{context} must be a non-empty integer sequence")
    try:
        return tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise ValueError(f"{context} contains a non-integer token") from error


def _parse_autoavsr_train_label(
    label_path: Path,
    canonical_ids: CanonicalIDMap,
) -> tuple[G2LabelSample, ...]:
    """Parse exactly the formats accepted by ``AVDataset.load_list``.

    AutoAVSR uses literal comma splitting rather than an RFC-CSV parser.  Using
    that same rule here is important: quoted commas must not silently produce
    a schedule that differs from training.
    """

    lines = label_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("AutoAVSR train label is empty")

    samples: list[G2LabelSample] = []
    seen_sample_ids: set[str] = set()
    for row_index, line in enumerate(lines):
        fields = line.split(",")
        if len(fields) not in _AUTOAVSR_FIELD_COUNTS:
            raise ValueError(
                "AutoAVSR train label line "
                f"{row_index + 1} has {len(fields)} fields; expected one of "
                f"{sorted(_AUTOAVSR_FIELD_COUNTS)}"
            )
        dataset_name, loader_path, raw_input_length, raw_token_ids = fields[:4]
        if dataset_name.strip().lower() != G2_CORPUS:
            raise ValueError(
                f"AutoAVSR train label line {row_index + 1} has unexpected "
                f"dataset {dataset_name!r}"
            )
        if not loader_path:
            raise ValueError(
                f"AutoAVSR train label line {row_index + 1} has an empty path"
            )
        try:
            input_length = int(raw_input_length)
        except ValueError as error:
            raise ValueError(
                f"AutoAVSR train label line {row_index + 1} has an invalid "
                "input length"
            ) from error
        if input_length <= 0:
            raise ValueError(
                f"AutoAVSR train label line {row_index + 1} has a non-positive "
                "input length"
            )
        _parse_integer_sequence(
            raw_token_ids,
            context=f"AutoAVSR train label line {row_index + 1} token IDs",
        )
        if len(fields) == 6:
            _parse_integer_sequence(
                fields[5],
                context=f"AutoAVSR train label line {row_index + 1} syllable IDs",
            )
        elif len(fields) == 8:
            for offset, name in enumerate(("onset", "rime", "tone"), start=5):
                _parse_integer_sequence(
                    fields[offset],
                    context=(
                        f"AutoAVSR train label line {row_index + 1} {name} IDs"
                    ),
                )

        try:
            sample_id = canonical_ids.lookup(G2_SPLIT, loader_path)
            source_id = canonical_ids.source_for(sample_id)
        except KeyError as error:
            raise ValueError(
                f"AutoAVSR train label line {row_index + 1} cannot be resolved "
                "by the canonical-ID map"
            ) from error
        if not sample_id.startswith(f"{G2_CORPUS}:{G2_SPLIT}:"):
            raise ValueError(
                f"canonical sample {sample_id!r} is outside the G2 train corpus"
            )
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate train-label canonical sample: {sample_id}")
        seen_sample_ids.add(sample_id)
        samples.append(
            G2LabelSample(
                row_index=row_index,
                dataset_name=dataset_name,
                loader_path=loader_path,
                input_length=input_length,
                sample_id=sample_id,
                source_id=source_id,
            )
        )

    canonical_train_ids = {
        sample_id
        for sample_id in canonical_ids.sample_ids
        if canonical_ids.split_for(sample_id) == G2_SPLIT
    }
    label_ids = set(seen_sample_ids)
    if label_ids != canonical_train_ids:
        missing = sorted(canonical_train_ids - label_ids)
        extra = sorted(label_ids - canonical_train_ids)
        raise ValueError(
            "AutoAVSR train label and canonical train split are not one-to-one: "
            f"missing={len(missing)}"
            + (f" first_missing={missing[0]!r}" if missing else "")
            + f", extra={len(extra)}"
            + (f" first_extra={extra[0]!r}" if extra else "")
        )
    return tuple(samples)


def build_g2_physical_batches(
    label_path: str | Path,
    canonical_ids: CanonicalIDMap | str | Path,
) -> tuple[PhysicalBatch, ...]:
    """Build exact G2 physical batches from the frozen AutoAVSR train label.

    ``CustomBucketDataset`` is called directly with its training defaults:
    internal shuffle disabled, 400 buckets, padded-frame cap 6000, and no
    fixed sample-count cap.
    """

    path = Path(label_path)
    if not path.is_file():
        raise FileNotFoundError(f"AutoAVSR train label is missing: {path}")
    canonical_map = (
        canonical_ids
        if isinstance(canonical_ids, CanonicalIDMap)
        else CanonicalIDMap.load(canonical_ids)
    )
    samples = _parse_autoavsr_train_label(path, canonical_map)
    lengths = [sample.input_length for sample in samples]
    if max(lengths) > G2_MAX_PADDED_FRAMES:
        raise ValueError(
            "AutoAVSR train label contains a clip longer than the locked "
            f"{G2_MAX_PADDED_FRAMES}-frame cap"
        )

    bucketed = CustomBucketDataset(
        dataset=samples,
        lengths=lengths,
        max_frames=G2_MAX_PADDED_FRAMES,
        num_buckets=G2_NUM_BUCKETS,
        shuffle=False,
        batch_size=G2_BATCH_SIZE,
    )
    batches: list[PhysicalBatch] = []
    flattened_rows: list[int] = []
    for physical_index, row_indices_list in enumerate(bucketed.batches):
        row_indices = tuple(int(index) for index in row_indices_list)
        flattened_rows.extend(row_indices)
        batch_samples = tuple(samples[index] for index in row_indices)
        input_lengths = tuple(sample.input_length for sample in batch_samples)
        padded_frames = len(input_lengths) * max(input_lengths)
        if padded_frames > G2_MAX_PADDED_FRAMES:
            raise RuntimeError(
                f"physical batch {physical_index} exceeds the locked padded "
                "frame cap"
            )
        batches.append(
            PhysicalBatch(
                physical_batch_index=physical_index,
                row_indices=tuple(sample.row_index for sample in batch_samples),
                sample_ids=tuple(sample.sample_id for sample in batch_samples),
                source_ids=tuple(sample.source_id for sample in batch_samples),
                input_lengths=input_lengths,
                padded_frames=padded_frames,
            )
        )

    expected_rows = list(range(len(samples)))
    if sorted(flattened_rows) != expected_rows or len(flattened_rows) != len(
        expected_rows
    ):
        raise RuntimeError(
            "CustomBucketDataset did not partition every train row exactly once"
        )
    if not batches:
        raise RuntimeError("CustomBucketDataset produced no physical batches")
    return tuple(batches)


def _validate_physical_batches(
    physical_batches: Sequence[PhysicalBatch],
) -> tuple[PhysicalBatch, ...]:
    batches = tuple(physical_batches)
    if len(batches) < G2_SELECTED_BATCHES:
        raise ValueError(
            f"G2 needs at least {G2_SELECTED_BATCHES} physical batches; "
            f"found {len(batches)}"
        )
    expected_indices = tuple(range(len(batches)))
    actual_indices = tuple(batch.physical_batch_index for batch in batches)
    if actual_indices != expected_indices:
        raise ValueError(
            "physical batches must be complete and ordered by contiguous "
            "physical_batch_index"
        )
    sample_ids = [
        sample_id for batch in batches for sample_id in batch.sample_ids
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("physical batches contain duplicate canonical sample IDs")
    if any(
        batch.padded_frames > G2_MAX_PADDED_FRAMES for batch in physical_batches
    ):
        raise ValueError("physical batch exceeds the locked padded-frame cap")
    return batches


def select_g2_physical_batches(
    physical_batches: Sequence[PhysicalBatch],
) -> tuple[PhysicalBatch, ...]:
    """Select the exact prospectively frozen G2 64-batch schedule."""

    batches = _validate_physical_batches(physical_batches)
    generator = torch.Generator().manual_seed(G2_SEED)
    selected_indices = torch.randperm(
        len(batches), generator=generator
    )[:G2_SELECTED_BATCHES].tolist()
    return tuple(batches[int(index)] for index in selected_indices)


def _route_value(value: Any, *, edge_id: str) -> str:
    if isinstance(value, str):
        route = value
    elif isinstance(value, Mapping):
        if "route" not in value:
            raise ValueError(f"edge {edge_id!r} has no route")
        route = str(value["route"])
    elif hasattr(value, "route"):
        route = str(value.route)
    else:
        raise TypeError(f"edge {edge_id!r} has an unsupported route record")
    if route not in VALID_ROUTES:
        raise ValueError(f"edge {edge_id!r} has invalid route {route!r}")
    return route


def _normalize_edge_routes(edge_routes: Mapping[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_edge_id, value in edge_routes.items():
        edge_id = str(raw_edge_id)
        if not edge_id:
            raise ValueError("edge route mapping contains an empty edge ID")
        normalized[edge_id] = _route_value(value, edge_id=edge_id)
    if not normalized:
        raise ValueError("edge route mapping is empty")
    return normalized


def _event_fields(event: Any, *, sample_id: str) -> tuple[str, int, int]:
    if isinstance(event, TrainingEvent):
        edge_id = event.edge_id
        start = event.target_start
        end = event.target_end
    elif isinstance(event, Mapping):
        missing = {"edge_id", "target_start", "target_end"} - set(event)
        if missing:
            raise ValueError(
                f"occurrence for {sample_id!r} is missing fields {sorted(missing)}"
            )
        edge_id = str(event["edge_id"])
        start = event["target_start"]
        end = event["target_end"]
    elif all(
        hasattr(event, field)
        for field in ("edge_id", "target_start", "target_end")
    ):
        edge_id = str(event.edge_id)
        start = event.target_start
        end = event.target_end
    else:
        raise TypeError(f"occurrence for {sample_id!r} has an unsupported record")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise ValueError(f"occurrence for {sample_id!r} has a non-integer span")
    if not edge_id or start < 0 or end <= start:
        raise ValueError(f"occurrence for {sample_id!r} has an invalid edge/span")
    return edge_id, int(start), int(end)


def _normalize_occurrence_edge_counts(
    *,
    occurrences: Mapping[str, Sequence[Any]] | None,
    occurrence_edge_counts: Mapping[str, Mapping[str, int]] | None,
) -> dict[str, dict[str, int]]:
    if (occurrences is None) == (occurrence_edge_counts is None):
        raise ValueError(
            "supply exactly one of occurrences or occurrence_edge_counts"
        )
    normalized: dict[str, dict[str, int]] = {}
    if occurrence_edge_counts is not None:
        for raw_sample_id, raw_counts in occurrence_edge_counts.items():
            sample_id = str(raw_sample_id)
            if not sample_id or not isinstance(raw_counts, Mapping):
                raise TypeError("compact occurrence counts are malformed")
            counts: dict[str, int] = {}
            for raw_edge_id, raw_count in raw_counts.items():
                edge_id = str(raw_edge_id)
                if (
                    not edge_id
                    or isinstance(raw_count, bool)
                    or not isinstance(raw_count, int)
                    or raw_count <= 0
                ):
                    raise ValueError(
                        f"compact occurrence count is invalid for {sample_id!r}"
                    )
                counts[edge_id] = int(raw_count)
            if not counts:
                raise ValueError(
                    f"compact occurrence row for {sample_id!r} is empty"
                )
            normalized[sample_id] = counts
        return normalized

    assert occurrences is not None
    for raw_sample_id, raw_events in occurrences.items():
        sample_id = str(raw_sample_id)
        if not sample_id:
            raise ValueError("occurrence mapping contains an empty sample ID")
        if not isinstance(raw_events, Sequence) or isinstance(
            raw_events, (str, bytes)
        ):
            raise TypeError(
                f"occurrences for {sample_id!r} must be a sequence"
            )
        events = tuple(_event_fields(event, sample_id=sample_id) for event in raw_events)
        if not events:
            raise ValueError(f"occurrence row for {sample_id!r} is empty")
        if len(events) != len(set(events)):
            raise ValueError(
                f"occurrence row for {sample_id!r} contains a duplicate event"
            )
        counts: dict[str, int] = {}
        for edge_id, _start, _end in events:
            counts[edge_id] = counts.get(edge_id, 0) + 1
        normalized[sample_id] = counts
    return normalized


def _empty_mass_sets() -> dict[str, dict[str, set[Any] | int]]:
    return {
        route: {
            "events": 0,
            "samples": set(),
            "sources": set(),
            "batches": set(),
        }
        for route in _ACTIVE_ROUTES
    }


def _finalize_mass(
    values: Mapping[str, Mapping[str, set[Any] | int]],
) -> dict[str, dict[str, int]]:
    return {
        route: {
            "event_count": int(route_values["events"]),
            "sample_count": len(route_values["samples"]),
            "source_count": len(route_values["sources"]),
            "batch_count": len(route_values["batches"]),
        }
        for route, route_values in values.items()
    }


def audit_g2_activity(
    physical_batches: Sequence[PhysicalBatch],
    *,
    occurrences: Mapping[str, Sequence[Any]] | None = None,
    occurrence_edge_counts: Mapping[str, Mapping[str, int]] | None = None,
    edge_routes: Mapping[str, Any],
    forbidden_ids: Sequence[str] | set[str] | frozenset[str],
) -> dict[str, Any]:
    """Audit the locked G2 schedule without performing model computation.

    Structural ambiguity in labels, batches, routes, or event rows raises an
    exception.  Empirical gate failures (insufficient activity, leakage,
    occurrence IDs outside the train schedule, unknown edges, or route-A
    occurrences) return a complete ``BLOCKED`` audit record.
    """

    batches = _validate_physical_batches(physical_batches)
    selected = select_g2_physical_batches(batches)
    routes = _normalize_edge_routes(edge_routes)
    event_counts = _normalize_occurrence_edge_counts(
        occurrences=occurrences,
        occurrence_edge_counts=occurrence_edge_counts,
    )
    forbidden = frozenset(str(sample_id) for sample_id in forbidden_ids)

    sample_locations: dict[str, tuple[int, str]] = {}
    train_sample_ids: set[str] = set()
    for batch in batches:
        for sample_id, source_id in zip(batch.sample_ids, batch.source_ids):
            sample_locations[sample_id] = (
                batch.physical_batch_index,
                source_id,
            )
            train_sample_ids.add(sample_id)

    unknown_occurrence_sample_ids = sorted(
        set(event_counts) - train_sample_ids
    )
    unknown_occurrence_edge_ids: set[str] = set()
    inactive_route_A_edge_ids: set[str] = set()
    invalid_event_count = 0
    valid_routed_counts: dict[str, tuple[int, int]] = {}
    for sample_id, sample_edge_counts in event_counts.items():
        routed = {"E": 0, "D": 0}
        for edge_id, count in sample_edge_counts.items():
            route = routes.get(edge_id)
            if route is None:
                unknown_occurrence_edge_ids.add(edge_id)
                invalid_event_count += count
            elif route == "A":
                inactive_route_A_edge_ids.add(edge_id)
                invalid_event_count += count
            else:
                routed[route] += count
        if sample_id in sample_locations and any(routed.values()):
            valid_routed_counts[sample_id] = (
                routed["E"],
                routed["D"],
            )

    def mass_for(
        route_batch_allowlist: Mapping[str, set[int]],
    ) -> dict[str, dict[str, int]]:
        mass = _empty_mass_sets()
        for sample_id, sample_counts in valid_routed_counts.items():
            batch_index, source_id = sample_locations[sample_id]
            for route, count in zip(_ACTIVE_ROUTES, sample_counts):
                if count <= 0:
                    continue
                if batch_index not in route_batch_allowlist[route]:
                    continue
                route_mass = mass[route]
                route_mass["events"] = int(route_mass["events"]) + count
                route_mass["samples"].add(sample_id)
                route_mass["sources"].add(source_id)
                route_mass["batches"].add(batch_index)
        return _finalize_mass(mass)

    all_batch_indices = set(range(len(batches)))
    selected_batch_indices = {
        batch.physical_batch_index for batch in selected
    }
    d_eligible_ordinal = 0
    d_quota_batch_indices: set[int] = set()
    selected_records: list[dict[str, Any]] = []
    e_eligible_count = 0
    d_eligible_count = 0
    for selection_ordinal, batch in enumerate(selected):
        per_route = _empty_mass_sets()
        for sample_id, source_id in zip(batch.sample_ids, batch.source_ids):
            for route, count in zip(
                _ACTIVE_ROUTES,
                valid_routed_counts.get(sample_id, (0, 0)),
            ):
                if count <= 0:
                    continue
                route_mass = per_route[route]
                route_mass["events"] = int(route_mass["events"]) + count
                route_mass["samples"].add(sample_id)
                route_mass["sources"].add(source_id)
                route_mass["batches"].add(batch.physical_batch_index)
        route_mass = _finalize_mass(per_route)
        e_eligible = route_mass["E"]["event_count"] > 0
        d_eligible = route_mass["D"]["event_count"] > 0
        if e_eligible:
            e_eligible_count += 1

        d_ordinal_zero_based: int | None = None
        d_quota_selected = False
        if d_eligible:
            d_ordinal_zero_based = d_eligible_ordinal
            d_quota_selected = context_batch_selected(
                batch.sample_ids,
                seed=G2_SEED,
                update=d_eligible_ordinal,
                fraction=G2_D_CONTEXT_FRACTION,
            )
            if d_quota_selected:
                d_quota_batch_indices.add(batch.physical_batch_index)
            d_eligible_ordinal += 1
            d_eligible_count += 1

        selected_records.append(
            {
                "selection_ordinal_zero_based": selection_ordinal,
                "physical_batch_index": batch.physical_batch_index,
                "row_indices": list(batch.row_indices),
                "sample_ids": list(batch.sample_ids),
                "source_ids": list(batch.source_ids),
                "input_lengths": list(batch.input_lengths),
                "sample_count": len(batch.sample_ids),
                "source_count": len(set(batch.source_ids)),
                "padded_frames": batch.padded_frames,
                "route_mass": route_mass,
                "E_eligible": e_eligible,
                "D_eligible": d_eligible,
                "D_eligible_ordinal_zero_based": d_ordinal_zero_based,
                "D_eligible_rank_one_based": (
                    None
                    if d_ordinal_zero_based is None
                    else d_ordinal_zero_based + 1
                ),
                "D_quota_selected": d_quota_selected,
            }
        )

    d_quota_selected_count = len(d_quota_batch_indices)
    activity_checks = {
        "E_eligible_batches_at_least_8": (
            e_eligible_count >= G2_MINIMUM_E_ELIGIBLE_BATCHES
        ),
        "D_quota_selected_batches_at_least_8": (
            d_quota_selected_count
            >= G2_MINIMUM_D_QUOTA_SELECTED_BATCHES
        ),
    }
    activity_passed = all(activity_checks.values())

    forbidden_train_ids = sorted(train_sample_ids & forbidden)
    forbidden_occurrence_ids = sorted(set(event_counts) & forbidden)
    leakage_passed = not forbidden_train_ids and not forbidden_occurrence_ids
    integrity_checks = {
        "physical_batches_cover_unique_samples": True,
        "selected_batches_are_unique": (
            len(selected_batch_indices) == G2_SELECTED_BATCHES
        ),
        "occurrence_samples_are_in_train_schedule": (
            not unknown_occurrence_sample_ids
        ),
        "occurrence_edges_are_known": not unknown_occurrence_edge_ids,
        "occurrences_reference_only_active_E_D_routes": (
            not inactive_route_A_edge_ids
        ),
    }
    integrity_passed = all(integrity_checks.values())

    blocking_reasons: list[str] = []
    if not activity_checks["E_eligible_batches_at_least_8"]:
        blocking_reasons.append("insufficient_E_eligible_batches")
    if not activity_checks["D_quota_selected_batches_at_least_8"]:
        blocking_reasons.append("insufficient_D_quota_selected_batches")
    if forbidden_train_ids:
        blocking_reasons.append("forbidden_IDs_in_train_schedule")
    if forbidden_occurrence_ids:
        blocking_reasons.append("forbidden_IDs_in_occurrences")
    if unknown_occurrence_sample_ids:
        blocking_reasons.append("occurrence_samples_outside_train_schedule")
    if unknown_occurrence_edge_ids:
        blocking_reasons.append("unknown_occurrence_edges")
    if inactive_route_A_edge_ids:
        blocking_reasons.append("inactive_route_A_occurrences")

    overall_passed = activity_passed and leakage_passed and integrity_passed
    full_mass = mass_for(
        {"E": all_batch_indices, "D": all_batch_indices}
    )
    selected_mass = mass_for(
        {"E": selected_batch_indices, "D": selected_batch_indices}
    )
    applied_mass = mass_for(
        {"E": selected_batch_indices, "D": d_quota_batch_indices}
    )
    return {
        "schema_version": SCHEMA_G2_ACTIVITY_AUDIT,
        "status": "PASS" if overall_passed else "BLOCKED",
        "passed": overall_passed,
        "blocking_reasons": blocking_reasons,
        "schedule": {
            "seed": G2_SEED,
            "physical_batch_builder": "AutoAVSR CustomBucketDataset",
            "num_buckets": G2_NUM_BUCKETS,
            "max_padded_frames": G2_MAX_PADDED_FRAMES,
            "batch_size": G2_BATCH_SIZE,
            "curriculum": False,
            "selection": G2_SELECTION_DESCRIPTION,
            "selected_batch_count": G2_SELECTED_BATCHES,
            "D_context_fraction": G2_D_CONTEXT_FRACTION,
            "D_quota_semantics": G2_D_QUOTA_DESCRIPTION,
        },
        "train_row_count": len(train_sample_ids),
        "physical_batch_count": len(batches),
        "selected_physical_batch_indices": [
            batch.physical_batch_index for batch in selected
        ],
        "selected_batches": selected_records,
        "mass": {
            "full_train_occurrences": full_mass,
            "selected_schedule": selected_mass,
            "applied_schedule": applied_mass,
        },
        "activity_gate": {
            "E_eligible_batch_count": e_eligible_count,
            "minimum_E_eligible_batches": G2_MINIMUM_E_ELIGIBLE_BATCHES,
            "D_eligible_batch_count": d_eligible_count,
            "D_quota_selected_batch_count": d_quota_selected_count,
            "minimum_D_quota_selected_batches": (
                G2_MINIMUM_D_QUOTA_SELECTED_BATCHES
            ),
            "checks": activity_checks,
            "passed": activity_passed,
        },
        "leakage": {
            "forbidden_registry_count": len(forbidden),
            "forbidden_train_sample_count": len(forbidden_train_ids),
            "forbidden_train_sample_ids": forbidden_train_ids,
            "forbidden_occurrence_sample_count": len(
                forbidden_occurrence_ids
            ),
            "forbidden_occurrence_sample_ids": forbidden_occurrence_ids,
            "passed": leakage_passed,
        },
        "integrity": {
            "occurrence_sample_count": len(event_counts),
            "valid_active_occurrence_sample_count": len(valid_routed_counts),
            "invalid_event_count": invalid_event_count,
            "unknown_occurrence_sample_count": len(
                unknown_occurrence_sample_ids
            ),
            "unknown_occurrence_sample_ids": unknown_occurrence_sample_ids,
            "unknown_occurrence_edge_count": len(
                unknown_occurrence_edge_ids
            ),
            "unknown_occurrence_edge_ids": sorted(
                unknown_occurrence_edge_ids
            ),
            "inactive_route_A_edge_count": len(
                inactive_route_A_edge_ids
            ),
            "inactive_route_A_edge_ids": sorted(
                inactive_route_A_edge_ids
            ),
            "checks": integrity_checks,
            "passed": integrity_passed,
        },
    }
