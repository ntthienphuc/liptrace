"""Exact G2-schedule bridge for the guarded RouteConfirm-v3 G3 run.

This module contains no model forward, gradient, optimizer, or publication
logic.  It authenticates the frozen 64-batch schedule, reconstructs the
unshuffled Auto-AVSR physical bucket dataset, and exposes those physical
batches in exactly the frozen order.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from datamodule.av_dataset import AVDataset
from datamodule.data_module import CustomBucketDataset, collate_pad
from datamodule.transforms import VideoTransform
from pavir.artifacts import CanonicalIDMap


SCHEMA = "pavir-route-confirm-v3-g2-calibration-schedule-v1"
PROTOCOL = "PAViR-RouteConfirm-v3"
STATUS = "FROZEN_FOR_G3_CALIBRATION_ONLY"
PHYSICAL_BATCH_COUNT = 4000
SELECTED_BATCH_COUNT = 64
NUM_BUCKETS = 400
MAX_PADDED_FRAMES = 6000
SEED = 42
CONTEXT_FRACTION = 0.5
NUM_WORKERS = 8

_TOP_KEYS = {
    "schema_version",
    "protocol_name",
    "status",
    "method_contract",
    "inputs",
    "schedule",
    "physical_batch_count",
    "selected_physical_batch_indices",
    "selected_batches",
    "authorization",
}
_ENTRY_KEYS = {
    "selection_ordinal_zero_based",
    "physical_batch_index",
    "row_indices",
    "sample_ids",
    "source_ids",
    "input_lengths",
    "sample_count",
    "source_count",
    "padded_frames",
    "E_eligible",
    "D_eligible",
    "D_eligible_ordinal_zero_based",
    "D_eligible_rank_one_based",
    "D_quota_selected",
    "route_mass",
}
_MASS_KEYS = {"batch_count", "event_count", "sample_count", "source_count"}
_SCHEDULE = {
    "physical_batch_builder": "AutoAVSR CustomBucketDataset",
    "num_buckets": NUM_BUCKETS,
    "max_padded_frames": MAX_PADDED_FRAMES,
    "batch_size": None,
    "curriculum": False,
    "selection": (
        "torch.randperm(number_of_physical_batches, "
        "Generator().manual_seed(42))[:64]"
    ),
    "seed": SEED,
    "selected_batch_count": SELECTED_BATCH_COUNT,
    "D_context_fraction": CONTEXT_FRACTION,
    "D_quota_semantics": (
        "locked cumulative quota over D-eligible batch ordinals"
    ),
}
_AUTHORIZATION = {
    "g3_wrapper_may_consume_after_g2_gate": True,
    "direct_calibration_authorized": False,
    "training_authorized": False,
}


def sha256_file(path: str | Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{context} inventory mismatch")


def _strict_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return int(value)


def _strict_int_tuple(
    value: Any,
    context: str,
    *,
    minimum: int = 0,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON list")
    return tuple(
        _strict_int(item, f"{context}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _quota_selected(eligible_ordinal: int) -> bool:
    before = int(eligible_ordinal * CONTEXT_FRACTION)
    after = int((eligible_ordinal + 1) * CONTEXT_FRACTION)
    return after > before


@dataclass(frozen=True)
class FrozenBatchSpec:
    selection_ordinal: int
    physical_batch_index: int
    row_indices: tuple[int, ...]
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    input_lengths: tuple[int, ...]
    padded_frames: int
    e_eligible: bool
    d_eligible: bool
    d_eligible_ordinal: int
    d_quota_selected: bool
    route_mass: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class FrozenCalibrationSchedule:
    path: Path
    sha256: str
    physical_batch_count: int
    selected_physical_batch_indices: tuple[int, ...]
    batches: tuple[FrozenBatchSpec, ...]
    payload: Mapping[str, Any]


def _validate_mass(
    value: Any,
    route: str,
    *,
    sample_count: int,
    source_count: int,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"route_mass.{route} must be an object")
    _require_exact_keys(value, _MASS_KEYS, f"route_mass.{route}")
    row = {
        name: _strict_int(value[name], f"route_mass.{route}.{name}")
        for name in sorted(_MASS_KEYS)
    }
    if row["batch_count"] not in (0, 1):
        raise ValueError(f"route_mass.{route}.batch_count is invalid")
    if row["sample_count"] > sample_count or row["source_count"] > source_count:
        raise ValueError(f"route_mass.{route} exceeds the physical batch")
    if row["source_count"] > row["sample_count"]:
        raise ValueError(f"route_mass.{route} source/sample count is invalid")
    if bool(row["event_count"]) != bool(row["batch_count"]):
        raise ValueError(f"route_mass.{route} event/batch eligibility mismatch")
    if bool(row["sample_count"]) != bool(row["batch_count"]):
        raise ValueError(f"route_mass.{route} sample/batch eligibility mismatch")
    return row


def load_frozen_schedule(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> FrozenCalibrationSchedule:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed_sha256 = sha256_file(resolved)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("calibration schedule hash mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("calibration schedule must be a JSON object")
    _require_exact_keys(payload, _TOP_KEYS, "calibration schedule")
    if (
        payload["schema_version"] != SCHEMA
        or payload["protocol_name"] != PROTOCOL
        or payload["status"] != STATUS
        or payload["schedule"] != _SCHEDULE
        or payload["authorization"] != _AUTHORIZATION
    ):
        raise ValueError("calibration schedule semantic mismatch")
    physical_count = _strict_int(
        payload["physical_batch_count"],
        "physical_batch_count",
        minimum=1,
    )
    if physical_count != PHYSICAL_BATCH_COUNT:
        raise ValueError("physical batch count disagrees with the G3 contract")
    selected_indices = _strict_int_tuple(
        payload["selected_physical_batch_indices"],
        "selected_physical_batch_indices",
    )
    if (
        len(selected_indices) != SELECTED_BATCH_COUNT
        or len(set(selected_indices)) != SELECTED_BATCH_COUNT
        or any(index >= physical_count for index in selected_indices)
    ):
        raise ValueError("selected physical batch indices are invalid")
    rows = payload["selected_batches"]
    if not isinstance(rows, list) or len(rows) != SELECTED_BATCH_COUNT:
        raise ValueError("selected batch inventory is invalid")

    batches: list[FrozenBatchSpec] = []
    next_d_ordinal = 0
    all_row_indices: set[int] = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"selected batch {ordinal} is not an object")
        _require_exact_keys(row, _ENTRY_KEYS, f"selected batch {ordinal}")
        selection_ordinal = _strict_int(
            row["selection_ordinal_zero_based"],
            f"selected batch {ordinal}.selection_ordinal",
        )
        physical_index = _strict_int(
            row["physical_batch_index"],
            f"selected batch {ordinal}.physical_batch_index",
        )
        if (
            selection_ordinal != ordinal
            or physical_index != selected_indices[ordinal]
        ):
            raise ValueError(f"selected batch {ordinal} order drifted")
        row_indices = _strict_int_tuple(
            row["row_indices"],
            f"selected batch {ordinal}.row_indices",
        )
        input_lengths = _strict_int_tuple(
            row["input_lengths"],
            f"selected batch {ordinal}.input_lengths",
            minimum=1,
        )
        sample_ids_raw = row["sample_ids"]
        source_ids_raw = row["source_ids"]
        if not isinstance(sample_ids_raw, list) or not isinstance(
            source_ids_raw, list
        ):
            raise ValueError(f"selected batch {ordinal} IDs must be lists")
        sample_ids = tuple(map(str, sample_ids_raw))
        source_ids = tuple(map(str, source_ids_raw))
        sample_count = _strict_int(
            row["sample_count"],
            f"selected batch {ordinal}.sample_count",
            minimum=1,
        )
        source_count = _strict_int(
            row["source_count"],
            f"selected batch {ordinal}.source_count",
            minimum=1,
        )
        if not (
            len(row_indices)
            == len(input_lengths)
            == len(sample_ids)
            == len(source_ids)
            == sample_count
        ):
            raise ValueError(f"selected batch {ordinal} alignment drifted")
        if (
            len(set(row_indices)) != sample_count
            or len(set(sample_ids)) != sample_count
            or source_count != len(set(source_ids))
            or any(not value.startswith("vicocktail:train:") for value in sample_ids)
            or any(not value for value in source_ids)
        ):
            raise ValueError(f"selected batch {ordinal} IDs are invalid")
        overlap = all_row_indices.intersection(row_indices)
        if overlap:
            raise ValueError(
                f"selected physical batches overlap at row {min(overlap)}"
            )
        all_row_indices.update(row_indices)
        padded_frames = _strict_int(
            row["padded_frames"],
            f"selected batch {ordinal}.padded_frames",
            minimum=1,
        )
        if (
            padded_frames != sample_count * max(input_lengths)
            or padded_frames > MAX_PADDED_FRAMES
        ):
            raise ValueError(f"selected batch {ordinal} padded workload drifted")
        if not isinstance(row["route_mass"], Mapping) or set(
            row["route_mass"]
        ) != {"E", "D"}:
            raise ValueError(f"selected batch {ordinal} route mass drifted")
        mass = {
            route: _validate_mass(
                row["route_mass"][route],
                route,
                sample_count=sample_count,
                source_count=source_count,
            )
            for route in ("E", "D")
        }
        e_eligible = row["E_eligible"]
        d_eligible = row["D_eligible"]
        if not isinstance(e_eligible, bool) or not isinstance(d_eligible, bool):
            raise ValueError(f"selected batch {ordinal} eligibility is not boolean")
        if e_eligible != bool(mass["E"]["batch_count"]):
            raise ValueError(f"selected batch {ordinal} E eligibility drifted")
        if d_eligible != bool(mass["D"]["batch_count"]):
            raise ValueError(f"selected batch {ordinal} D eligibility drifted")
        d_ordinal = _strict_int(
            row["D_eligible_ordinal_zero_based"],
            f"selected batch {ordinal}.D_eligible_ordinal",
        )
        d_rank = _strict_int(
            row["D_eligible_rank_one_based"],
            f"selected batch {ordinal}.D_eligible_rank",
            minimum=1,
        )
        if not d_eligible or d_ordinal != next_d_ordinal or d_rank != d_ordinal + 1:
            raise ValueError(f"selected batch {ordinal} D ordinal drifted")
        d_selected = row["D_quota_selected"]
        if (
            not isinstance(d_selected, bool)
            or d_selected != _quota_selected(d_ordinal)
        ):
            raise ValueError(f"selected batch {ordinal} D quota drifted")
        next_d_ordinal += 1
        batches.append(
            FrozenBatchSpec(
                selection_ordinal=selection_ordinal,
                physical_batch_index=physical_index,
                row_indices=row_indices,
                sample_ids=sample_ids,
                source_ids=source_ids,
                input_lengths=input_lengths,
                padded_frames=padded_frames,
                e_eligible=e_eligible,
                d_eligible=d_eligible,
                d_eligible_ordinal=d_ordinal,
                d_quota_selected=d_selected,
                route_mass=mass,
            )
        )
    if next_d_ordinal != SELECTED_BATCH_COUNT or not all(
        batch.e_eligible and batch.d_eligible for batch in batches
    ):
        raise ValueError("frozen G3 schedule lacks the exact E/D activity contract")
    if sum(batch.d_quota_selected for batch in batches) != 32:
        raise ValueError("frozen G3 D quota count drifted")
    return FrozenCalibrationSchedule(
        path=resolved,
        sha256=observed_sha256,
        physical_batch_count=physical_count,
        selected_physical_batch_indices=selected_indices,
        batches=tuple(batches),
        payload=payload,
    )


def validate_bucket_dataset(
    bucket_dataset: CustomBucketDataset,
    schedule: FrozenCalibrationSchedule,
) -> None:
    if len(bucket_dataset) != schedule.physical_batch_count:
        raise ValueError("reconstructed physical batch count drifted")
    base = bucket_dataset.dataset
    if not isinstance(base, AVDataset):
        raise TypeError("G3 requires an Auto-AVSR AVDataset")
    for spec in schedule.batches:
        row_indices = tuple(
            int(value) for value in bucket_dataset.batches[spec.physical_batch_index]
        )
        if row_indices != spec.row_indices:
            raise ValueError(
                f"physical batch {spec.physical_batch_index} row order drifted"
            )
        input_lengths = tuple(int(base.input_lengths[index]) for index in row_indices)
        sample_ids = tuple(
            str(base.sample_id(base.list[index][1])) for index in row_indices
        )
        if input_lengths != spec.input_lengths:
            raise ValueError(
                f"physical batch {spec.physical_batch_index} lengths drifted"
            )
        if sample_ids != spec.sample_ids:
            raise ValueError(
                f"physical batch {spec.physical_batch_index} sample IDs drifted"
            )


class FrozenPhysicalBatchSampler(torch.utils.data.Sampler[int]):
    def __init__(self, indices: Sequence[int]):
        self._indices = tuple(int(value) for value in indices)

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def build_vicocktail_schedule_loader(
    *,
    data_root: str | Path,
    train_labels: str | Path,
    canonical_ids: str | Path,
    schedule: FrozenCalibrationSchedule,
    num_workers: int = NUM_WORKERS,
    emit_transformed_input_sha256: bool = False,
) -> torch.utils.data.DataLoader:
    if num_workers != NUM_WORKERS:
        raise ValueError(f"G3 requires exactly {NUM_WORKERS} loader workers")
    canonical = CanonicalIDMap.load(canonical_ids)
    base = AVDataset(
        root_dir=str(Path(data_root).resolve()),
        label_path=str(Path(train_labels).resolve()),
        subset="train",
        modality="video",
        audio_transform=None,
        video_transform=VideoTransform("train"),
        sample_id_map=canonical,
        emit_transformed_input_sha256=emit_transformed_input_sha256,
    )
    bucket_dataset = CustomBucketDataset(
        base,
        base.input_lengths,
        MAX_PADDED_FRAMES,
        NUM_BUCKETS,
        shuffle=False,
        batch_size=None,
    )
    validate_bucket_dataset(bucket_dataset, schedule)
    generator = torch.Generator().manual_seed(SEED)
    return torch.utils.data.DataLoader(
        bucket_dataset,
        batch_size=None,
        sampler=FrozenPhysicalBatchSampler(
            schedule.selected_physical_batch_indices
        ),
        num_workers=num_workers,
        collate_fn=collate_pad,
        persistent_workers=True,
        pin_memory=True,
        generator=generator,
    )


def validate_emitted_batch(
    batch: Mapping[str, Any],
    spec: FrozenBatchSpec,
) -> dict[str, Any]:
    required = {
        "inputs",
        "input_lengths",
        "targets",
        "sample_ids",
    }
    if not required.issubset(batch):
        raise ValueError("emitted batch is missing required tensors/IDs")
    sample_ids = tuple(map(str, batch["sample_ids"]))
    lengths_tensor = batch["input_lengths"]
    if not isinstance(lengths_tensor, torch.Tensor) or lengths_tensor.ndim != 1:
        raise ValueError("emitted input_lengths is invalid")
    input_lengths = tuple(int(value) for value in lengths_tensor.tolist())
    inputs = batch["inputs"]
    targets = batch["targets"]
    if not isinstance(inputs, torch.Tensor) or not isinstance(
        targets, torch.Tensor
    ):
        raise ValueError("emitted inputs/targets must be tensors")
    padded_frames = len(sample_ids) * int(inputs.size(1))
    if (
        sample_ids != spec.sample_ids
        or input_lengths != spec.input_lengths
        or len(sample_ids) != len(spec.row_indices)
        or padded_frames != spec.padded_frames
        or int(inputs.size(0)) != len(sample_ids)
        or int(targets.size(0)) != len(sample_ids)
    ):
        raise ValueError(
            f"runtime batch {spec.selection_ordinal} disagrees with frozen schedule"
        )
    return {
        "selection_ordinal_zero_based": spec.selection_ordinal,
        "physical_batch_index": spec.physical_batch_index,
        "sample_count": len(sample_ids),
        "sample_ids_sha256": hashlib.sha256(
            ("\n".join(sample_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "padded_frames": padded_frames,
        "D_eligible_ordinal_zero_based": spec.d_eligible_ordinal,
        "D_quota_selected": spec.d_quota_selected,
    }


__all__ = [
    "FrozenBatchSpec",
    "FrozenCalibrationSchedule",
    "FrozenPhysicalBatchSampler",
    "build_vicocktail_schedule_loader",
    "load_frozen_schedule",
    "sha256_file",
    "validate_bucket_dataset",
    "validate_emitted_batch",
]
