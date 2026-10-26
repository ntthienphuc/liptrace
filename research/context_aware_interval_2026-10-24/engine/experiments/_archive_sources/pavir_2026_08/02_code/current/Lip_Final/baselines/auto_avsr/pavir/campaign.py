"""Locked campaign construction and gate controls for PAViR-VSR.

This module contains the filesystem-light, deterministic parts of campaign
orchestration.  Long-running training and decoding remain shell entry points,
but consume the registry produced here instead of assembling arm-specific
arguments ad hoc.

The route-shuffled control follows the campaign precommit exactly:

* only the frozen E/D labels are permuted (A remains A);
* the number of E/D/A edges is unchanged;
* every edge that may become E already has both endpoint prototypes;
* occurrences, canonical IDs, forbidden IDs and prototypes are byte-identical
  across all arms.

Confusion-support totals before and after shuffling are recorded as diagnostics,
not used to search for a favorable or specially balanced permutation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import (
    ArtifactStore,
    CanonicalIDMap,
    PAViRConfig,
    canonical_json_bytes,
    sha256_file,
)


ARM_SCHEMA = "pavir-campaign-arms-v1"
GATE_SCHEMA = "pavir-gate-authorization-v1"
STATUS_SCHEMA = "pavir-job-status-v1"
PRESCREEN_SCHEMA = "pavir-prescreen-pass-v1"
ROUTE_SHUFFLE_SCHEMA = "pavir-route-shuffle-v1"
ANALYSIS_SCHEMA = "pavir-analysis-v1"

ARM_NAMES = (
    "native",
    "encoder_only",
    "decoder_only",
    "full",
    "route_shuffled",
)
PAVIR_ARMS = frozenset(ARM_NAMES) - {"native"}
SCREEN_ONLY_ARMS = frozenset(
    {"encoder_only", "decoder_only", "route_shuffled"}
)

LOCKED_SEED = 42
LOCKED_LR = 1e-5
LOCKED_WEIGHT_DECAY = 0.03
LOCKED_PRECISION = "bf16-mixed"
LOCKED_MAX_FRAMES = 6000
LOCKED_MAX_EPOCHS = 5
LOCKED_WARMUP_EPOCHS = 1
LOCKED_CTC_WEIGHT = 0.1
LOCKED_BEAM_PRIMARY = 1
LOCKED_BEAM_SECONDARY = 10
LOCKED_TRAIN_BUCKETS = 400
LOCKED_OVERFIT_CLIPS = 32
LOCKED_OVERFIT_UPDATES = 64
LOCKED_SMOKE_BATCHES = 200
LOCKED_VRAM_LIMIT_GIB = 72.0
LOCKED_BLANK_COLLAPSE_MEAN_LIMIT = 0.99


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    config_path: Path
    config_payload: Mapping[str, Any]
    config: PAViRConfig
    store: ArtifactStore


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_beneath(root: Path, relative: str, context: str) -> Path:
    value = (root / relative).resolve()
    try:
        value.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{context} escapes its locked directory") from error
    return value


def _require_file(path: str | Path, context: str) -> Path:
    value = Path(path).resolve()
    if not value.is_file():
        raise FileNotFoundError(f"{context} is missing: {value}")
    if "\n" in str(value) or "\r" in str(value):
        raise ValueError(f"{context} path contains a newline")
    return value


def _require_directory(path: str | Path, context: str) -> Path:
    value = Path(path).resolve()
    if not value.is_dir():
        raise NotADirectoryError(f"{context} is missing: {value}")
    if "\n" in str(value) or "\r" in str(value):
        raise ValueError(f"{context} path contains a newline")
    return value


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _relative_file_record(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "sha256": sha256_file(path),
    }


def _write_once(path: Path, payload: bytes) -> None:
    """Create an immutable generated file, or verify an identical prior run."""

    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to overwrite a different locked campaign file: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _copy_once(source: Path, destination: Path) -> None:
    """Copy a locked artifact without ever replacing an existing file."""

    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise FileExistsError(
                f"refusing to overwrite a different locked artifact: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != sha256_file(source):
        raise RuntimeError(f"artifact copy changed bytes: {source}")


def load_source_bundle(bundle_dir: str | Path) -> SourceBundle:
    """Load and fully validate one output of ``freeze_pavir_config.py``."""

    root = _require_directory(bundle_dir, "frozen bundle")
    manifest_path = _require_file(root / "BUNDLE_MANIFEST.json", "bundle manifest")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "pavir-frozen-bundle-v1":
        raise ValueError("unsupported frozen PAViR bundle schema")
    config_record = manifest.get("config")
    if not isinstance(config_record, dict):
        raise ValueError("bundle manifest lacks its config record")
    config_path = _resolve_beneath(root, str(config_record.get("path", "")), "bundle config")
    _require_file(config_path, "bundle config")
    if sha256_file(config_path) != str(config_record.get("sha256", "")).lower():
        raise ValueError("frozen bundle config hash mismatch")
    config_payload = _read_json(config_path)
    config = PAViRConfig.load(config_path)
    store = ArtifactStore(config)
    if int(manifest.get("total_updates", -1)) <= 0:
        raise ValueError("bundle manifest total_updates must be positive")
    if int(manifest.get("ramp_updates", -1)) != config.ramp_updates:
        raise ValueError("bundle manifest/config ramp_updates disagree")
    expected_counts = {
        "edge_count": len(store.edges),
        "occurrence_sample_count": len(store.events),
        "prototype_count": len(store.prototype_keys),
    }
    for name, observed in expected_counts.items():
        if int(manifest.get(name, -1)) != observed:
            raise ValueError(
                f"bundle manifest {name} mismatch: "
                f"{manifest.get(name)!r} != {observed}"
            )
    return SourceBundle(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        config_path=config_path,
        config_payload=config_payload,
        config=config,
        store=store,
    )


def route_counts(routes: Mapping[str, Any]) -> dict[str, int]:
    rows = routes.get("edges")
    if not isinstance(rows, list):
        raise ValueError("route artifact must contain an edges list")
    result = {"E": 0, "D": 0, "A": 0}
    for row in rows:
        route = str(row.get("route", ""))
        if route not in result:
            raise ValueError(f"route artifact contains invalid route: {route!r}")
        result[route] += 1
    return result


def route_support(routes: Mapping[str, Any]) -> dict[str, int]:
    result = {"E": 0, "D": 0, "A": 0}
    for row in routes.get("edges", []):
        route = str(row.get("route", ""))
        support = row.get("support")
        if route not in result or isinstance(support, bool):
            raise ValueError("route artifact contains an invalid route/support")
        support = int(support)
        if support < 0:
            raise ValueError("route support must be non-negative")
        result[route] += support
    return result


def _seeded_route_assignment(
    active_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> set[int]:
    """Shuffle the frozen E/D labels with deterministic nonidentity retry."""

    original_e = {
        index for index, row in enumerate(active_rows) if row["route"] == "E"
    }
    original_d = set(range(len(active_rows))) - original_e
    if not original_e or not original_d:
        raise ValueError("route-shuffled control requires at least one E and one D edge")
    labels = [str(row["route"]) for row in active_rows]
    generator = random.Random(int(seed))
    for retry in range(1, 4097):
        shuffled = list(labels)
        generator.shuffle(shuffled)
        if shuffled != labels:
            return {
                index for index, route in enumerate(shuffled) if route == "E"
            }
    # With both labels present a nonidentity permutation exists.  This guard is
    # therefore about deterministic implementation failure, not data tuning.
    raise RuntimeError("seeded route permutation remained identity after 4096 retries")


def build_route_shuffled_payload(
    routes: Mapping[str, Any],
    *,
    prototype_keys: Iterable[str],
    seed: int = LOCKED_SEED,
) -> dict[str, Any]:
    """Return the precommitted deterministic E/D route-label permutation."""

    if routes.get("schema_version") != "pavir-routes-v1":
        raise ValueError("unsupported route schema")
    rows = routes.get("edges")
    if not isinstance(rows, list) or not rows:
        raise ValueError("route file has no edges")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        edge_id = str(row.get("edge_id", ""))
        if not edge_id or edge_id in by_id:
            raise ValueError("route file contains an empty/duplicate edge_id")
        by_id[edge_id] = row
    active_rows = sorted(
        (row for row in rows if row.get("route") in {"E", "D"}),
        key=lambda row: str(row["edge_id"]),
    )
    keys = frozenset(str(key) for key in prototype_keys)
    missing = sorted(
        key
        for row in active_rows
        for key in (
            f"{row['edge_id']}:target",
            f"{row['edge_id']}:confusion",
        )
        if key not in keys
    )
    if missing:
        raise ValueError(
            "route shuffle would require post-hoc prototype extraction; "
            f"first missing key={missing[0]}"
        )

    original_counts = route_counts(routes)
    original_support = route_support(routes)
    selected_e = _seeded_route_assignment(active_rows, seed=seed)
    assignment = {
        str(row["edge_id"]): ("E" if index in selected_e else "D")
        for index, row in enumerate(active_rows)
    }
    shuffled = copy.deepcopy(dict(routes))
    changed = []
    for row in shuffled["edges"]:
        edge_id = str(row["edge_id"])
        if edge_id in assignment:
            old = str(row["route"])
            row["route"] = assignment[edge_id]
            if row["route"] != old:
                changed.append(edge_id)
    shuffled["route_counts"] = route_counts(shuffled)
    shuffled["route_shuffle_control"] = {
        "schema_version": ROUTE_SHUFFLE_SCHEMA,
        "seed": int(seed),
        "method": "seeded-label-permutation-with-nonidentity-retry",
        "population": "evidence-sufficient frozen E/D edges only",
        "abstain_edges_unchanged": True,
        "edge_metadata_unchanged": True,
        "existing_prototypes_only": True,
        "lambda_and_ramp_budget_unchanged": True,
        "support_totals_are_diagnostic_not_a_selection_constraint": True,
        "reassigned_edge_count": len(changed),
        "reassigned_edge_ids": sorted(changed),
        "original_route_counts": original_counts,
        "shuffled_route_counts": route_counts(shuffled),
        "original_route_support": original_support,
        "shuffled_route_support": route_support(shuffled),
    }
    if not changed:
        raise RuntimeError("route shuffle is trivial")
    if route_counts(shuffled) != original_counts:
        raise AssertionError("route shuffle changed route counts")

    original_by_id = {str(row["edge_id"]): row for row in rows}
    for row in shuffled["edges"]:
        original = original_by_id[str(row["edge_id"])]
        left = {key: value for key, value in original.items() if key != "route"}
        right = {key: value for key, value in row.items() if key != "route"}
        if left != right:
            raise AssertionError(f"route shuffle changed edge metadata: {row['edge_id']}")
    return shuffled


def read_label_lengths(path: str | Path) -> list[int]:
    """Read Auto-AVSR's headerless CSV input lengths without touching videos."""

    label_path = _require_file(path, "training label file")
    lengths: list[int] = []
    with label_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split(",")
            if len(fields) not in (4, 5, 6, 8):
                raise ValueError(
                    f"training label line {line_number} has {len(fields)} fields"
                )
            try:
                length = int(fields[2])
            except ValueError as error:
                raise ValueError(
                    f"training label line {line_number} has invalid length"
                ) from error
            if length <= 0:
                raise ValueError(
                    f"training label line {line_number} has non-positive length"
                )
            lengths.append(length)
    if not lengths:
        raise ValueError("training label file is empty")
    return lengths


def build_overfit32_lines(
    train_label: str | Path,
    *,
    canonical_ids: str | Path,
    store: ArtifactStore,
    seed: int = LOCKED_SEED,
) -> tuple[list[str], dict[str, Any]]:
    """Select 32 frozen train rows and guarantee both auxiliaries are exercised."""

    label_path = _require_file(train_label, "training label")
    canonical_path = _require_file(canonical_ids, "canonical IDs")
    canonical = CanonicalIDMap.load(canonical_path)
    rows = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) < LOCKED_OVERFIT_CLIPS:
        raise ValueError("training label has fewer than 32 clips")
    metadata: list[tuple[int, str, frozenset[str]]] = []
    for index, line in enumerate(rows):
        fields = line.split(",")
        if len(fields) not in (4, 5, 6, 8):
            raise ValueError(f"training label row {index + 1} is malformed")
        sample_id = canonical.lookup("train", fields[1])
        routes = frozenset(
            store.edges[event.edge_id].route
            for event in store.events_for(sample_id)
            if store.edges[event.edge_id].route in {"E", "D"}
        )
        metadata.append((index, sample_id, routes))
    e_candidates = [row for row in metadata if "E" in row[2]]
    d_candidates = [row for row in metadata if "D" in row[2]]
    if not e_candidates or not d_candidates:
        raise ValueError("overfit32 requires at least one train occurrence for E and D")
    generator = random.Random(int(seed))
    chosen: set[int] = set()
    chosen.add(generator.choice(e_candidates)[0])
    remaining_d = [row for row in d_candidates if row[0] not in chosen]
    if not remaining_d:
        raise ValueError("overfit32 cannot select distinct E- and D-active rows")
    chosen.add(generator.choice(remaining_d)[0])
    remaining = [index for index in range(len(rows)) if index not in chosen]
    chosen.update(
        generator.sample(remaining, LOCKED_OVERFIT_CLIPS - len(chosen))
    )
    selected_indices = sorted(chosen)
    selected = [rows[index] for index in selected_indices]
    selected_meta = [metadata[index] for index in selected_indices]
    routes_hash = sha256_file(store.config.routes.path)
    summary = {
        "schema_version": "pavir-overfit32-selection-v1",
        "seed": int(seed),
        "selection": (
            "one seeded E-active row, one distinct seeded D-active row, then "
            "seeded fill from the frozen train label; original row order restored"
        ),
        "clips": len(selected),
        "E_active_clips": sum("E" in routes for _, _, routes in selected_meta),
        "D_active_clips": sum("D" in routes for _, _, routes in selected_meta),
        "sample_ids": [sample_id for _, sample_id, _ in selected_meta],
        "source_train_label_sha256": sha256_file(label_path),
        "canonical_ids_sha256": sha256_file(canonical_path),
        "routes_sha256": routes_hash,
    }
    return selected, summary


def count_padded_batches(
    lengths: Sequence[int],
    *,
    max_frames: int = LOCKED_MAX_FRAMES,
    num_buckets: int = LOCKED_TRAIN_BUCKETS,
) -> int:
    """Mirror ``CustomBucketDataset`` and return deterministic updates/epoch."""

    if not lengths or any(int(value) <= 0 for value in lengths):
        raise ValueError("lengths must be a non-empty positive sequence")
    if max_frames < max(int(value) for value in lengths):
        raise ValueError("max_frames is smaller than the longest clip")
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    values = np.asarray(lengths, dtype=np.int64)
    boundaries = np.linspace(
        int(values.min()), int(values.max()), num_buckets, dtype=np.float64
    )
    # torch.bucketize(..., right=False) is equivalent to left searchsorted.
    assignments = np.searchsorted(boundaries, values, side="left")
    order = sorted(range(len(values)), key=lambda index: int(assignments[index]))
    batches = 0
    current_size = 0
    current_max = 0
    for index in order:
        length = int(values[index])
        candidate_max = max(current_max, length)
        if current_size and (current_size + 1) * candidate_max > max_frames:
            batches += 1
            current_size = 1
            current_max = length
        else:
            current_size += 1
            current_max = candidate_max
    return batches + bool(current_size)


def _runtime_record(path: Path, *, kind: str) -> dict[str, str]:
    record = _file_record(path)
    record["kind"] = kind
    return record


def _config_payload(
    source: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, str]],
    lambda_visual: float,
    lambda_context: float,
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(source))
    payload["lambda_visual"] = float(lambda_visual)
    payload["lambda_context"] = float(lambda_context)
    payload["artifacts"] = copy.deepcopy(dict(artifacts))
    return payload


def build_campaign_bundle(
    *,
    source_bundle_dir: str | Path,
    output_dir: str | Path,
    precommit: str | Path,
    data_registry: str | Path,
    implementation_lock: str | Path,
    code_snapshot: str | Path,
    w0_checkpoint: str | Path,
    data_root: str | Path,
    train_label: str | Path,
    vc_calib_label: str | Path,
    vc_calib_manifest: str | Path,
    vc_promote_label: str | Path,
    vc_promote_manifest: str | Path,
    tokenizer_model: str | Path,
    tokenizer_units: str | Path,
) -> dict[str, Any]:
    """Build five deterministic arm configs and their locked run registry."""

    source = load_source_bundle(source_bundle_dir)
    precommit_path = _require_file(precommit, "campaign precommit")
    precommit_payload = _read_json(precommit_path)
    if precommit_payload.get("schema_version") != "pavir-campaign-precommit-v1":
        raise ValueError("unsupported PAViR campaign precommit schema")
    data_registry_path = _require_file(data_registry, "campaign data registry")
    data_registry_payload = _read_json(data_registry_path)
    if data_registry_payload.get("schema_version") != "pavir-data-registry-v1":
        raise ValueError("unsupported PAViR data registry schema")
    implementation_lock_path = _require_file(
        implementation_lock, "campaign implementation lock"
    )
    implementation_lock_payload = _read_json(implementation_lock_path)
    if (
        not isinstance(implementation_lock_payload, dict)
        or implementation_lock_payload.get("schema_version")
        != "pavir-implementation-lock-v1"
        or implementation_lock_payload.get("status")
        != "LOCKED_BEFORE_W0_CONFUSION_GRAPH_RESULTS"
    ):
        raise ValueError("unsupported or unlocked implementation lock")
    code_snapshot_path = _require_file(code_snapshot, "campaign code snapshot")
    code_snapshot_payload = _read_json(code_snapshot_path)
    code_aggregate = _validate_code_snapshot(
        code_snapshot_path,
        code_snapshot_payload,
        implementation_lock_path=implementation_lock_path,
    )
    protocol_family_id = _protocol_family_id(
        precommit_path=precommit_path,
        data_registry_path=data_registry_path,
        implementation_lock_path=implementation_lock_path,
        code_aggregate=code_aggregate,
    )
    precommit_training = precommit_payload.get("training", {})
    locked_precommit_training = {
        "seed": LOCKED_SEED,
        "optimizer": "AdamW",
        "learning_rate": LOCKED_LR,
        "weight_decay": LOCKED_WEIGHT_DECAY,
        "precision": LOCKED_PRECISION,
        "max_padded_frames": LOCKED_MAX_FRAMES,
        "epochs": LOCKED_MAX_EPOCHS,
        "warmup_epochs": LOCKED_WARMUP_EPOCHS,
        "early_stopping": False,
        "checkpoint_selection": "fixed final",
        "initialization": "weights-only corpus-specific W0",
        "optimizer_scheduler_reset": True,
    }
    for name, expected in locked_precommit_training.items():
        if precommit_training.get(name) != expected:
            raise ValueError(f"precommit training lock changed: {name}")
    short_screen = precommit_payload.get("short_screen", {})
    if (
        short_screen.get("arms") != list(ARM_NAMES)
        or int(short_screen.get("updates", -1)) != 2000
        or short_screen.get("restart_each_arm_from_w0") is not True
    ):
        raise ValueError("precommit short-screen arm/update lock changed")
    shuffle_lock = short_screen.get("route_shuffle", {})
    expected_shuffle = {
        "population": "evidence-sufficient frozen E/D edges only",
        "preserve_exact_E_and_D_edge_counts": True,
        "preserve_prototypes_and_gradient_budget": True,
        "seed": LOCKED_SEED,
        "reject_identity_permutation": True,
    }
    for name, expected in expected_shuffle.items():
        if shuffle_lock.get(name) != expected:
            raise ValueError(f"precommit route-shuffle lock changed: {name}")
    output = Path(output_dir).resolve()
    if output == source.root or source.root in output.parents:
        raise ValueError("campaign arm bundle must not be inside its frozen source bundle")
    output.mkdir(parents=True, exist_ok=True)

    if source.config.seed != LOCKED_SEED:
        raise ValueError(f"frozen PAViR seed must be {LOCKED_SEED}")
    if source.config.block_index != 6:
        raise ValueError("frozen PAViR block index must be 6")
    if not math.isclose(source.config.visual_margin, 0.15, abs_tol=1e-12):
        raise ValueError("frozen visual margin must be 0.15")
    if not math.isclose(source.config.context_margin, 0.10, abs_tol=1e-12):
        raise ValueError("frozen context margin must be 0.10")
    if not math.isclose(source.config.context_batch_fraction, 0.50, abs_tol=1e-12):
        raise ValueError("frozen context batch fraction must be exactly 0.50")
    if source.config.lambda_visual <= 0 or source.config.lambda_context <= 0:
        raise ValueError("full PAViR bundle must contain positive calibrated lambdas")

    runtime_paths = {
        "w0_checkpoint": _require_file(w0_checkpoint, "W0 checkpoint"),
        "train_label": _require_file(train_label, "train label"),
        "vc_calib_label": _require_file(vc_calib_label, "VC-calib label"),
        "vc_calib_manifest": _require_file(
            vc_calib_manifest, "VC-calib manifest"
        ),
        "vc_promote_label": _require_file(vc_promote_label, "VC-promote label"),
        "vc_promote_manifest": _require_file(
            vc_promote_manifest, "VC-promote manifest"
        ),
        "tokenizer_model": _require_file(tokenizer_model, "tokenizer model"),
        "tokenizer_units": _require_file(tokenizer_units, "tokenizer units"),
    }
    data = _require_directory(data_root, "Auto-AVSR data root")
    corpus_registry = data_registry_payload.get("corpora", {}).get(
        source.config.corpus
    )
    if not isinstance(corpus_registry, dict):
        raise ValueError(
            f"data registry lacks corpus {source.config.corpus!r}"
        )
    if source.config.corpus == "vicocktail":
        if source.config.forbidden_ids is None:
            raise ValueError("frozen bundle lacks forbidden promote/test IDs")
        expected_records = {
            "w0_checkpoint": corpus_registry["w0_checkpoint"],
            "train_label": {
                "path": corpus_registry["training"]["auto_avsr_labels_path"],
                "sha256": corpus_registry["training"]["auto_avsr_labels_sha256"],
            },
            "vc_calib_manifest": {
                "path": corpus_registry["firewall"]["vc_calib"]["manifest_path"],
                "sha256": corpus_registry["firewall"]["vc_calib"][
                    "manifest_sha256"
                ],
            },
            "vc_promote_manifest": {
                "path": corpus_registry["firewall"]["vc_promote"][
                    "manifest_path"
                ],
                "sha256": corpus_registry["firewall"]["vc_promote"][
                    "manifest_sha256"
                ],
            },
            "canonical_ids": corpus_registry["canonical_ids"],
            "forbidden_ids": corpus_registry["forbidden_promote_test_ids"],
        }
        for name in (
            "w0_checkpoint",
            "train_label",
            "vc_calib_manifest",
            "vc_promote_manifest",
        ):
            supplied = runtime_paths[name]
            locked = expected_records[name]
            if (
                supplied != Path(str(locked["path"])).resolve()
                or sha256_file(supplied) != str(locked["sha256"]).lower()
            ):
                raise ValueError(f"{name} disagrees with DATA_REGISTRY.json")
        for name, locked in (
            ("canonical_ids", expected_records["canonical_ids"]),
            ("forbidden_ids", expected_records["forbidden_ids"]),
        ):
            supplied = (
                source.config.canonical_ids.path
                if name == "canonical_ids"
                else source.config.forbidden_ids.path
            )
            assert supplied is not None
            if (
                supplied != Path(str(locked["path"])).resolve()
                or sha256_file(supplied) != str(locked["sha256"]).lower()
            ):
                raise ValueError(
                    f"frozen bundle {name} disagrees with DATA_REGISTRY.json"
                )
        label_hashes = {
            "vc_calib_label": corpus_registry["firewall"]["vc_calib"][
                "auto_avsr_labels_sha256"
            ],
            "vc_promote_label": corpus_registry["firewall"]["vc_promote"][
                "auto_avsr_labels_sha256"
            ],
        }
        for name, expected_hash in label_hashes.items():
            if sha256_file(runtime_paths[name]) != str(expected_hash).lower():
                raise ValueError(f"{name} disagrees with DATA_REGISTRY.json")
        tokenizer_registry = data_registry_payload.get("tokenizer", {})
        tokenizer_records = {
            "tokenizer_model": tokenizer_registry.get("sentencepiece_model", {}),
            "tokenizer_units": tokenizer_registry.get("auto_avsr_units", {}),
        }
        for name, locked in tokenizer_records.items():
            if sha256_file(runtime_paths[name]) != str(locked.get("sha256", "")).lower():
                raise ValueError(f"{name} disagrees with DATA_REGISTRY.json")

    lengths = read_label_lengths(runtime_paths["train_label"])
    updates_per_epoch = count_padded_batches(lengths)
    full_updates = updates_per_epoch * LOCKED_MAX_EPOCHS
    if full_updates % 10:
        raise ValueError(
            "five-epoch full budget is not exactly divisible into a 10% screen"
        )
    screen_updates = full_updates // 10
    if source.config.corpus == "vicocktail" and (
        updates_per_epoch != 4000
        or full_updates != 20_000
        or screen_updates != 2_000
    ):
        raise ValueError(
            "locked ViCocktail campaign requires exactly 4000 updates/epoch, "
            "20000 full updates, and 2000 screen updates"
        )
    if int(source.manifest["total_updates"]) != full_updates:
        raise ValueError(
            "frozen bundle total_updates disagrees with the actual five-epoch "
            f"loader budget: {source.manifest['total_updates']} != {full_updates}"
        )
    if source.config.ramp_updates != screen_updates:
        raise ValueError(
            "PAViR ramp must span exactly the first 10% of the full update budget"
        )

    artifact_dir = output / "artifacts"
    copied_paths = {
        "routes": artifact_dir / "routes.json",
        "occurrences": artifact_dir / "occurrences.jsonl",
        "prototypes": artifact_dir / "prototypes.npz",
        "canonical_ids": artifact_dir / "canonical_ids.jsonl",
    }
    source_paths = {
        "routes": source.config.routes.path,
        "occurrences": source.config.occurrences.path,
        "prototypes": source.config.prototypes.path,
        "canonical_ids": source.config.canonical_ids.path,
    }
    if source.config.forbidden_ids is not None:
        copied_paths["forbidden_ids"] = artifact_dir / "forbidden_ids.json"
        source_paths["forbidden_ids"] = source.config.forbidden_ids.path
    else:
        raise ValueError("campaign bundle must lock forbidden promote/test IDs")
    for name, source_path in source_paths.items():
        _copy_once(source_path, copied_paths[name])

    overfit_lines, overfit_selection = build_overfit32_lines(
        runtime_paths["train_label"],
        canonical_ids=copied_paths["canonical_ids"],
        store=source.store,
        seed=LOCKED_SEED,
    )
    preflight_dir = output / "preflight"
    overfit_label = preflight_dir / "overfit32.csv"
    overfit_summary = preflight_dir / "overfit32_selection.json"
    _write_once(
        overfit_label,
        ("\n".join(overfit_lines) + "\n").encode("utf-8"),
    )
    overfit_selection = {
        **overfit_selection,
        "selected_label_sha256": sha256_file(overfit_label),
    }
    _write_once(overfit_summary, canonical_json_bytes(overfit_selection))

    artifact_records = {
        name: _relative_file_record(path, output)
        for name, path in copied_paths.items()
    }
    routes_payload = _read_json(copied_paths["routes"])
    shuffled_payload = build_route_shuffled_payload(
        routes_payload,
        prototype_keys=source.store.prototype_keys,
        seed=LOCKED_SEED,
    )
    shuffled_routes = artifact_dir / "routes.route_shuffled.json"
    _write_once(shuffled_routes, canonical_json_bytes(shuffled_payload))
    occurrence_lines = copied_paths["occurrences"].read_bytes().splitlines(
        keepends=True
    )
    if not occurrence_lines:
        raise ValueError("occurrence artifact is empty")
    occurrence_header = json.loads(occurrence_lines[0])
    occurrence_inputs = occurrence_header.get("inputs")
    if not isinstance(occurrence_inputs, dict):
        raise ValueError("occurrence artifact lacks its routes hash contract")
    occurrence_header["inputs"]["routes_sha256"] = sha256_file(shuffled_routes)
    shuffled_occurrences = artifact_dir / "occurrences.route_shuffled.jsonl"
    _write_once(
        shuffled_occurrences,
        canonical_json_bytes(occurrence_header) + b"".join(occurrence_lines[1:]),
    )
    shuffled_artifacts = copy.deepcopy(artifact_records)
    shuffled_artifacts["routes"] = _relative_file_record(shuffled_routes, output)
    shuffled_artifacts["occurrences"] = _relative_file_record(
        shuffled_occurrences, output
    )
    shuffled_payload["route_shuffle_control"][
        "occurrence_rows_byte_identical_except_hash_bound_header"
    ] = True

    lambdas = {
        "encoder_only": (source.config.lambda_visual, 0.0),
        "decoder_only": (0.0, source.config.lambda_context),
        "full": (source.config.lambda_visual, source.config.lambda_context),
        "route_shuffled": (
            source.config.lambda_visual,
            source.config.lambda_context,
        ),
    }
    arm_records: dict[str, Any] = {
        "native": {
            "pavir_config": None,
            "loss": "native",
            "allowed_training_stages": ["screen", "full"],
        }
    }
    for arm in ("encoder_only", "decoder_only", "full", "route_shuffled"):
        artifacts = (
            shuffled_artifacts if arm == "route_shuffled" else artifact_records
        )
        config_payload = _config_payload(
            source.config_payload,
            artifacts=artifacts,
            lambda_visual=lambdas[arm][0],
            lambda_context=lambdas[arm][1],
        )
        config_path = output / f"pavir_{arm}.json"
        _write_once(config_path, canonical_json_bytes(config_payload))
        loaded = PAViRConfig.load(config_path)
        ArtifactStore(loaded)
        allowed = ["screen"] if arm in SCREEN_ONLY_ARMS else ["screen", "full"]
        arm_records[arm] = {
            "pavir_config": _relative_file_record(config_path, output),
            "loss": arm,
            "lambda_visual": loaded.lambda_visual,
            "lambda_context": loaded.lambda_context,
            "allowed_training_stages": allowed,
        }

    calibration_hash = str(source.manifest.get("calibration_sha256") or "").lower()
    if len(calibration_hash) != 64:
        raise ValueError("final frozen bundle lacks gradient-calibration provenance")
    campaign_content_inputs = {
        "protocol_family_id": protocol_family_id,
        "corpus": source.config.corpus,
        "final_frozen_bundle_manifest": sha256_file(source.manifest_path),
        "final_pavir_config": sha256_file(source.config_path),
        "gradient_calibration": calibration_hash,
        "w0_checkpoint": sha256_file(runtime_paths["w0_checkpoint"]),
        "train_label": sha256_file(runtime_paths["train_label"]),
        "vc_calib_label": sha256_file(runtime_paths["vc_calib_label"]),
        "vc_calib_manifest": sha256_file(runtime_paths["vc_calib_manifest"]),
        "vc_promote_label": sha256_file(runtime_paths["vc_promote_label"]),
        "vc_promote_manifest": sha256_file(runtime_paths["vc_promote_manifest"]),
        "tokenizer_model": sha256_file(runtime_paths["tokenizer_model"]),
        "tokenizer_units": sha256_file(runtime_paths["tokenizer_units"]),
        "canonical_ids": sha256_file(copied_paths["canonical_ids"]),
        "forbidden_ids": sha256_file(copied_paths["forbidden_ids"]),
    }
    campaign_content_id = hashlib.sha256(
        canonical_json_bytes(campaign_content_inputs)
    ).hexdigest()
    registry = {
        "schema_version": ARM_SCHEMA,
        "corpus": source.config.corpus,
        "source_bundle": {
            "path": str(source.root),
            "manifest_path": str(source.manifest_path),
            "manifest_sha256": sha256_file(source.manifest_path),
            "config_path": str(source.config_path),
            "config_sha256": sha256_file(source.config_path),
        },
        "precommit": _file_record(precommit_path),
        "data_registry": _file_record(data_registry_path),
        "implementation_lock": _file_record(implementation_lock_path),
        "code_snapshot": {
            **_file_record(code_snapshot_path),
            "aggregate_sha256": code_aggregate,
        },
        "protocol_family_id": protocol_family_id,
        "campaign_content_id": campaign_content_id,
        "campaign_content_inputs": campaign_content_inputs,
        "scientific_lock": {
            "seed": LOCKED_SEED,
            "optimizer": "AdamW",
            "learning_rate": LOCKED_LR,
            "weight_decay": LOCKED_WEIGHT_DECAY,
            "precision": LOCKED_PRECISION,
            "max_frames": LOCKED_MAX_FRAMES,
            "max_epochs": LOCKED_MAX_EPOCHS,
            "warmup_epochs": LOCKED_WARMUP_EPOCHS,
            "ctc_weight": LOCKED_CTC_WEIGHT,
            "early_stopping": False,
            "checkpoint_selection": "fixed_last_checkpoint",
            "initialization": "weights_only_W0_optimizer_scheduler_reset",
            "updates_per_epoch": updates_per_epoch,
            "full_updates": full_updates,
            "screen_fraction": 0.10,
            "screen_updates": screen_updates,
            "primary_decode": {
                "beam_size": LOCKED_BEAM_PRIMARY,
                "ctc_weight": LOCKED_CTC_WEIGHT,
                "external_lm": False,
                "length_penalty": 0.0,
            },
            "secondary_decode": {
                "beam_size": LOCKED_BEAM_SECONDARY,
                "ctc_weight": LOCKED_CTC_WEIGHT,
                "external_lm": False,
                "length_penalty": 0.0,
            },
        },
        "runtime_inputs": {
            "data_root": str(data),
            "w0_checkpoint": _runtime_record(
                runtime_paths["w0_checkpoint"], kind="weights_only_initialization"
            ),
            "train_label": _runtime_record(
                runtime_paths["train_label"], kind="train"
            ),
            "vc_calib_label": _runtime_record(
                runtime_paths["vc_calib_label"], kind="method_calibration"
            ),
            "vc_calib_manifest": _runtime_record(
                runtime_paths["vc_calib_manifest"], kind="method_calibration"
            ),
            "vc_promote_label": _runtime_record(
                runtime_paths["vc_promote_label"], kind="sealed_until_screen_pass"
            ),
            "vc_promote_manifest": _runtime_record(
                runtime_paths["vc_promote_manifest"],
                kind="sealed_until_screen_pass",
            ),
            "canonical_ids": _relative_file_record(
                copied_paths["canonical_ids"], output
            ),
            "tokenizer_model": _runtime_record(
                runtime_paths["tokenizer_model"], kind="tokenizer"
            ),
            "tokenizer_units": _runtime_record(
                runtime_paths["tokenizer_units"], kind="tokenizer"
            ),
            "overfit32_label": _relative_file_record(overfit_label, output),
            "overfit32_selection": _relative_file_record(
                overfit_summary, output
            ),
        },
        "shared_artifacts": artifact_records,
        "route_shuffle_invariants": shuffled_payload["route_shuffle_control"],
        "arm_order": list(ARM_NAMES),
        "arms": arm_records,
        "evaluation_firewall": {
            "vc_calib_requires_gate": None,
            "vc_promote_requires_gate": "short_screen",
            "clean_test_requires_gate": "promotion",
            "clean_test_path_recorded": False,
            "clean_test_is_never_chained_automatically": True,
        },
        "pre_screen_gate": {
            "required_before_five_arm_screen": True,
            "overfit32": {
                "arms": ["native", "full"],
                "clips": LOCKED_OVERFIT_CLIPS,
                "updates_per_arm": LOCKED_OVERFIT_UPDATES,
                "label": _relative_file_record(overfit_label, output),
                "require_finite": True,
                "require_loss_decrease": True,
            },
            "smoke200": {
                "arm": "full",
                "physical_train_batches": LOCKED_SMOKE_BATCHES,
                "optimizer_accumulation": 1,
                "precision": LOCKED_PRECISION,
                "maximum_peak_allocated_vram_gib_exclusive": LOCKED_VRAM_LIMIT_GIB,
                "maximum_mean_ctc_blank_argmax_ratio_exclusive": (
                    LOCKED_BLANK_COLLAPSE_MEAN_LIMIT
                ),
                "require_finite": True,
                "require_no_oom": True,
                "require_native_state_dict_reload": True,
            },
        },
    }
    registry_path = output / "ARM_REGISTRY.json"
    _write_once(registry_path, canonical_json_bytes(registry))
    registry = validate_registry(registry_path)
    return registry


def _locked_record_path(
    registry_path: Path,
    record: Mapping[str, Any],
    *,
    relative: bool,
    context: str,
) -> Path:
    raw = str(record.get("path", ""))
    path = (
        _resolve_beneath(registry_path.parent, raw, context)
        if relative
        else Path(raw).resolve()
    )
    _require_file(path, context)
    expected = str(record.get("sha256", "")).lower()
    if sha256_file(path) != expected:
        raise ValueError(f"{context} hash mismatch")
    return path


def _validate_code_snapshot(
    path: Path,
    payload: Mapping[str, Any],
    *,
    implementation_lock_path: Path,
) -> str:
    if payload.get("schema_version") != "pavir-code-snapshot-v1":
        raise ValueError("unsupported PAViR code snapshot schema")
    implementation = payload.get("implementation_lock")
    if not isinstance(implementation, dict):
        raise ValueError("code snapshot lacks its implementation-lock record")
    if (
        str(implementation.get("sha256", "")).lower()
        != sha256_file(implementation_lock_path)
    ):
        raise ValueError("code snapshot is bound to another implementation lock")
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("code snapshot contains no source records")
    if int(payload.get("file_count", -1)) != len(rows):
        raise ValueError("code snapshot file count changed")
    project_root = Path(str(payload.get("project_root", ""))).resolve()
    if not project_root.is_dir():
        raise ValueError("code snapshot project root is missing")
    digest = hashlib.sha256()
    observed_paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("code snapshot source record is not an object")
        relative = str(row.get("path", ""))
        expected_hash = str(row.get("sha256", "")).lower()
        source = _resolve_beneath(project_root, relative, "code snapshot source")
        _require_file(source, "code snapshot source")
        if sha256_file(source) != expected_hash:
            raise ValueError(f"code snapshot source changed: {relative}")
        observed_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(expected_hash.encode("ascii"))
        digest.update(b"\n")
    if observed_paths != sorted(set(observed_paths)):
        raise ValueError("code snapshot source paths must be unique and sorted")
    aggregate = digest.hexdigest()
    if str(payload.get("aggregate_sha256", "")).lower() != aggregate:
        raise ValueError("code snapshot aggregate changed")
    return aggregate


def _protocol_family_id(
    *,
    precommit_path: Path,
    data_registry_path: Path,
    implementation_lock_path: Path,
    code_aggregate: str,
) -> str:
    logical_inputs = [
        "pavir-protocol-family-v1",
        sha256_file(precommit_path),
        sha256_file(data_registry_path),
        sha256_file(implementation_lock_path),
        code_aggregate,
    ]
    return hashlib.sha256(canonical_json_bytes(logical_inputs)).hexdigest()


def validate_registry(path: str | Path) -> dict[str, Any]:
    """Validate every precommitted input without accessing clean test."""

    registry_path = _require_file(path, "campaign registry")
    registry = _read_json(registry_path)
    if registry.get("schema_version") != ARM_SCHEMA:
        raise ValueError("unsupported PAViR campaign registry schema")
    if (
        set(registry.get("arms", {})) != set(ARM_NAMES)
        or registry.get("arm_order") != list(ARM_NAMES)
    ):
        raise ValueError(
            f"campaign arms/order must be exactly {ARM_NAMES}"
        )
    provenance_paths = {}
    for name in (
        "precommit",
        "data_registry",
        "implementation_lock",
        "code_snapshot",
    ):
        provenance_paths[name] = _locked_record_path(
            registry_path,
            registry.get(name, {}),
            relative=False,
            context=name.replace("_", " "),
        )
    code_payload = _read_json(provenance_paths["code_snapshot"])
    code_aggregate = _validate_code_snapshot(
        provenance_paths["code_snapshot"],
        code_payload,
        implementation_lock_path=provenance_paths["implementation_lock"],
    )
    if (
        str(registry.get("code_snapshot", {}).get("aggregate_sha256", "")).lower()
        != code_aggregate
    ):
        raise ValueError("campaign code snapshot aggregate changed")
    family = _protocol_family_id(
        precommit_path=provenance_paths["precommit"],
        data_registry_path=provenance_paths["data_registry"],
        implementation_lock_path=provenance_paths["implementation_lock"],
        code_aggregate=code_aggregate,
    )
    if registry.get("protocol_family_id") != family:
        raise ValueError("campaign protocol_family_id changed")
    source_record = registry.get("source_bundle", {})
    source_root = _require_directory(
        source_record.get("path", ""), "frozen source bundle"
    )
    source_manifest = _require_file(
        source_record.get("manifest_path", ""), "frozen source manifest"
    )
    source_config = _require_file(
        source_record.get("config_path", ""), "frozen source config"
    )
    if source_manifest.parent != source_root or source_config.parent != source_root:
        raise ValueError("frozen source files escaped the source bundle")
    if (
        sha256_file(source_manifest)
        != str(source_record.get("manifest_sha256", "")).lower()
        or sha256_file(source_config)
        != str(source_record.get("config_sha256", "")).lower()
    ):
        raise ValueError("frozen source bundle changed")
    source_manifest_payload = _read_json(source_manifest)
    lock = registry.get("scientific_lock", {})
    expected = {
        "seed": LOCKED_SEED,
        "optimizer": "AdamW",
        "learning_rate": LOCKED_LR,
        "weight_decay": LOCKED_WEIGHT_DECAY,
        "precision": LOCKED_PRECISION,
        "max_frames": LOCKED_MAX_FRAMES,
        "max_epochs": LOCKED_MAX_EPOCHS,
        "warmup_epochs": LOCKED_WARMUP_EPOCHS,
        "ctc_weight": LOCKED_CTC_WEIGHT,
        "early_stopping": False,
        "checkpoint_selection": "fixed_last_checkpoint",
        "initialization": "weights_only_W0_optimizer_scheduler_reset",
    }
    for name, value in expected.items():
        if lock.get(name) != value:
            raise ValueError(f"campaign scientific lock changed: {name}")
    full_updates = int(lock.get("full_updates", -1))
    screen_updates = int(lock.get("screen_updates", -1))
    updates_per_epoch = int(lock.get("updates_per_epoch", -1))
    if full_updates != updates_per_epoch * LOCKED_MAX_EPOCHS:
        raise ValueError("full update budget is not exactly five epochs")
    if screen_updates * 10 != full_updates:
        raise ValueError("screen is not exactly 10% of the full update budget")
    if registry.get("corpus") == "vicocktail" and (
        updates_per_epoch,
        full_updates,
        screen_updates,
    ) != (4000, 20_000, 2_000):
        raise ValueError("ViCocktail 4000/20000/2000 budget lock changed")

    runtime = registry.get("runtime_inputs", {})
    data_root = _require_directory(runtime.get("data_root", ""), "data root")
    for name in (
        "w0_checkpoint",
        "train_label",
        "vc_calib_label",
        "vc_calib_manifest",
        "vc_promote_label",
        "vc_promote_manifest",
        "tokenizer_model",
        "tokenizer_units",
    ):
        _locked_record_path(
            registry_path,
            runtime.get(name, {}),
            relative=False,
            context=f"runtime input {name}",
        )
    canonical = _locked_record_path(
        registry_path,
        runtime.get("canonical_ids", {}),
        relative=True,
        context="runtime canonical IDs",
    )
    overfit_label = _locked_record_path(
        registry_path,
        runtime.get("overfit32_label", {}),
        relative=True,
        context="overfit32 label",
    )
    _locked_record_path(
        registry_path,
        runtime.get("overfit32_selection", {}),
        relative=True,
        context="overfit32 selection",
    )
    if len([line for line in overfit_label.read_text(encoding="utf-8").splitlines() if line]) != LOCKED_OVERFIT_CLIPS:
        raise ValueError("overfit subset no longer contains exactly 32 clips")
    observed_updates = count_padded_batches(
        read_label_lengths(Path(runtime["train_label"]["path"]))
    )
    if observed_updates != updates_per_epoch:
        raise ValueError(
            "current train label no longer yields the locked update count: "
            f"{observed_updates} != {updates_per_epoch}"
        )

    content_inputs = {
        "protocol_family_id": family,
        "corpus": registry.get("corpus"),
        "final_frozen_bundle_manifest": sha256_file(source_manifest),
        "final_pavir_config": sha256_file(source_config),
        "gradient_calibration": str(
            source_manifest_payload.get("calibration_sha256") or ""
        ).lower(),
        "w0_checkpoint": str(runtime["w0_checkpoint"]["sha256"]).lower(),
        "train_label": str(runtime["train_label"]["sha256"]).lower(),
        "vc_calib_label": str(runtime["vc_calib_label"]["sha256"]).lower(),
        "vc_calib_manifest": str(
            runtime["vc_calib_manifest"]["sha256"]
        ).lower(),
        "vc_promote_label": str(runtime["vc_promote_label"]["sha256"]).lower(),
        "vc_promote_manifest": str(
            runtime["vc_promote_manifest"]["sha256"]
        ).lower(),
        "tokenizer_model": str(runtime["tokenizer_model"]["sha256"]).lower(),
        "tokenizer_units": str(runtime["tokenizer_units"]["sha256"]).lower(),
        "canonical_ids": str(runtime["canonical_ids"]["sha256"]).lower(),
        "forbidden_ids": str(
            registry.get("shared_artifacts", {})
            .get("forbidden_ids", {})
            .get("sha256", "")
        ).lower(),
    }
    if registry.get("campaign_content_inputs") != content_inputs:
        raise ValueError("campaign content inputs changed")
    content_id = hashlib.sha256(canonical_json_bytes(content_inputs)).hexdigest()
    if registry.get("campaign_content_id") != content_id:
        raise ValueError("campaign_content_id changed")

    for arm in ARM_NAMES:
        arm_record = registry["arms"][arm]
        allowed = arm_record.get("allowed_training_stages")
        expected_allowed = (
            ["screen"] if arm in SCREEN_ONLY_ARMS else ["screen", "full"]
        )
        if allowed != expected_allowed:
            raise ValueError(f"invalid allowed stages for arm {arm}")
        config_record = arm_record.get("pavir_config")
        if arm == "native":
            if config_record is not None:
                raise ValueError("native arm must omit PAViR config")
            continue
        config_path = _locked_record_path(
            registry_path,
            config_record,
            relative=True,
            context=f"{arm} PAViR config",
        )
        config = PAViRConfig.load(config_path)
        ArtifactStore(config)
        if config.seed != LOCKED_SEED or config.canonical_ids.path != canonical:
            raise ValueError(f"{arm} config disagrees with campaign seed/canonical IDs")
    firewall = registry.get("evaluation_firewall", {})
    if (
        firewall.get("clean_test_requires_gate") != "promotion"
        or firewall.get("clean_test_path_recorded") is not False
        or firewall.get("clean_test_is_never_chained_automatically") is not True
    ):
        raise ValueError("clean-test firewall changed")
    pre_screen = registry.get("pre_screen_gate", {})
    overfit = pre_screen.get("overfit32", {})
    smoke = pre_screen.get("smoke200", {})
    if (
        pre_screen.get("required_before_five_arm_screen") is not True
        or overfit.get("arms") != ["native", "full"]
        or int(overfit.get("clips", -1)) != LOCKED_OVERFIT_CLIPS
        or int(overfit.get("updates_per_arm", -1)) != LOCKED_OVERFIT_UPDATES
        or smoke.get("arm") != "full"
        or int(smoke.get("physical_train_batches", -1))
        != LOCKED_SMOKE_BATCHES
        or smoke.get("precision") != LOCKED_PRECISION
        or float(smoke.get("maximum_peak_allocated_vram_gib_exclusive", -1))
        != LOCKED_VRAM_LIMIT_GIB
        or float(
            smoke.get("maximum_mean_ctc_blank_argmax_ratio_exclusive", -1)
        )
        != LOCKED_BLANK_COLLAPSE_MEAN_LIMIT
    ):
        raise ValueError("pre-screen engineering gate changed")
    # Important: validation intentionally has no clean-test path to stat/read.
    return registry


def resolve_training(
    registry_path: str | Path,
    *,
    arm: str,
    stage: str,
) -> dict[str, Any]:
    registry_path = Path(registry_path).resolve()
    registry = validate_registry(registry_path)
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown campaign arm: {arm}")
    if stage not in {"overfit32", "smoke200", "screen", "full"}:
        raise ValueError(f"unsupported training stage: {stage}")
    record = registry["arms"][arm]
    if stage in {"screen", "full"} and stage not in record["allowed_training_stages"]:
        raise ValueError(f"arm {arm} is not authorized for {stage} training")
    if stage == "overfit32" and arm not in {"native", "full"}:
        raise ValueError("overfit32 is limited to native/full")
    if stage == "smoke200" and arm != "full":
        raise ValueError("smoke200 is limited to the full PAViR arm")
    runtime = registry["runtime_inputs"]
    config = record.get("pavir_config")
    if stage == "overfit32":
        target_updates = LOCKED_OVERFIT_UPDATES
        train_label = _resolve_beneath(
            registry_path.parent,
            runtime["overfit32_label"]["path"],
            "overfit32 label",
        )
        validation_label = train_label
    elif stage == "smoke200":
        target_updates = LOCKED_SMOKE_BATCHES
        train_label = Path(runtime["train_label"]["path"]).resolve()
        validation_label = _resolve_beneath(
            registry_path.parent,
            runtime["overfit32_label"]["path"],
            "overfit32 label",
        )
    else:
        target_updates = int(
            registry["scientific_lock"][
                "screen_updates" if stage == "screen" else "full_updates"
            ]
        )
        train_label = Path(runtime["train_label"]["path"]).resolve()
        validation_label = Path(runtime["vc_calib_label"]["path"]).resolve()
    return {
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "corpus": registry["corpus"],
        "arm": arm,
        "stage": stage,
        "target_updates": target_updates,
        "w0_checkpoint": str(Path(runtime["w0_checkpoint"]["path"]).resolve()),
        "data_root": str(Path(runtime["data_root"]).resolve()),
        "train_label": str(train_label),
        "validation_label": str(validation_label),
        "canonical_ids": str(
            _resolve_beneath(
                registry_path.parent,
                runtime["canonical_ids"]["path"],
                "canonical IDs",
            )
        ),
        "tokenizer_model": str(
            Path(runtime["tokenizer_model"]["path"]).resolve()
        ),
        "tokenizer_units": str(
            Path(runtime["tokenizer_units"]["path"]).resolve()
        ),
        "pavir_config": (
            None
            if config is None
            else str(
                _resolve_beneath(
                    registry_path.parent,
                    config["path"],
                    f"{arm} config",
                )
            )
        ),
        "lock": registry["scientific_lock"],
    }


def resolve_evaluation(
    registry_path: str | Path,
    *,
    arm: str,
    split: str,
    gate_marker: str | Path | None = None,
    clean_label: str | Path | None = None,
    clean_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one decode only after its firewall gate has been checked.

    For ``clean-test`` the authorization check deliberately runs before either
    clean path is passed to ``Path.is_file`` or hashed.
    """

    registry_path = Path(registry_path).resolve()
    assert_evaluation_authorized(
        registry_path=registry_path,
        split=split,
        gate_marker=gate_marker,
    )
    registry = validate_registry(registry_path)
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown campaign arm: {arm}")
    runtime = registry["runtime_inputs"]
    if split == "vc-calib":
        training_stage = "screen"
        label = Path(runtime["vc_calib_label"]["path"]).resolve()
        manifest = Path(runtime["vc_calib_manifest"]["path"]).resolve()
        logical_subset = "val"
    elif split == "vc-promote":
        if arm not in {"native", "full"}:
            raise ValueError("VC-promote is limited to matched native/full arms")
        training_stage = "full"
        label = Path(runtime["vc_promote_label"]["path"]).resolve()
        manifest = Path(runtime["vc_promote_manifest"]["path"]).resolve()
        logical_subset = "val"
    elif split == "clean-test":
        if arm not in {"native", "full"}:
            raise ValueError("clean test is limited to matched native/full arms")
        if clean_label is None or clean_manifest is None:
            raise ValueError(
                "authorized clean-test evaluation requires explicit label/manifest"
            )
        # These are intentionally the first clean-input filesystem accesses.
        label = _require_file(clean_label, "clean-test label")
        manifest = _require_file(clean_manifest, "clean-test manifest")
        training_stage = "full"
        logical_subset = "test"
    else:  # guarded in assert_evaluation_authorized, kept for type clarity.
        raise ValueError(f"unsupported campaign evaluation split: {split}")
    target_key = "screen_updates" if training_stage == "screen" else "full_updates"
    canonical = _resolve_beneath(
        registry_path.parent,
        runtime["canonical_ids"]["path"],
        "canonical IDs",
    )
    return {
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "corpus": registry["corpus"],
        "arm": arm,
        "split": split,
        "training_stage": training_stage,
        "target_updates": int(registry["scientific_lock"][target_key]),
        "data_root": str(Path(runtime["data_root"]).resolve()),
        "label": str(label),
        "label_sha256": sha256_file(label),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "logical_subset": logical_subset,
        "canonical_ids": str(canonical),
        "tokenizer_model": str(
            Path(runtime["tokenizer_model"]["path"]).resolve()
        ),
        "tokenizer_units": str(
            Path(runtime["tokenizer_units"]["path"]).resolve()
        ),
    }


def _gate_spec(gate: str) -> tuple[str, str]:
    if gate == "short_screen":
        return "vc-calib", "short_screen"
    if gate == "promotion":
        return "vc-promote", "promotion"
    raise ValueError(f"unsupported campaign gate: {gate}")


def authorize_gate(
    *,
    registry_path: str | Path,
    summary_path: str | Path,
    gate: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create a hash-bound marker only when the locked analyzer says PASS."""

    registry_path = _require_file(registry_path, "campaign registry")
    registry = validate_registry(registry_path)
    summary_path = _require_file(summary_path, f"{gate} analysis summary")
    summary = _read_json(summary_path)
    expected_split, gate_name = _gate_spec(gate)
    if summary.get("schema_version") != ANALYSIS_SCHEMA:
        raise ValueError("unsupported PAViR analysis summary schema")
    if summary.get("split") != expected_split:
        raise ValueError(
            f"{gate} summary split must be {expected_split}, "
            f"not {summary.get('split')!r}"
        )
    if summary.get("evaluation_status") != "complete":
        raise ValueError(f"{gate} evaluation is not complete")
    campaign = summary.get("campaign")
    summary_registry = (
        campaign.get("campaign_registry")
        if isinstance(campaign, dict)
        else None
    )
    if (
        not isinstance(campaign, dict)
        or campaign.get("campaign_identifier") != sha256_file(registry_path)
        or campaign.get("protocol_family_id")
        != registry["protocol_family_id"]
        or campaign.get("campaign_content_id")
        != registry["campaign_content_id"]
        or campaign.get("corpus") != "vicocktail"
        or not isinstance(summary_registry, dict)
        or Path(str(summary_registry.get("path", ""))).resolve()
        != registry_path.resolve()
        or str(summary_registry.get("sha256", "")).lower()
        != sha256_file(registry_path)
    ):
        raise ValueError(
            f"{gate} summary is not bound to this ViCocktail campaign registry"
        )
    observed = (
        summary.get("gates", {}).get(gate_name, {}).get("status")
    )
    if observed != "pass":
        raise PermissionError(f"{gate} gate is {observed!r}, not pass")
    marker_without_time = {
        "schema_version": GATE_SCHEMA,
        "gate": gate,
        "split": expected_split,
        "status": "pass",
        "campaign_registry": _file_record(registry_path),
        "analysis_summary": _file_record(summary_path),
        "authorizes": (
            ["full_training", "vc_promote_evaluation"]
            if gate == "short_screen"
            else ["clean_test_evaluation"]
        ),
    }
    output_path = Path(output_path).resolve()
    if output_path.is_file():
        existing = _read_json(output_path)
        comparable = {
            key: value for key, value in existing.items() if key != "created_utc"
        }
        if comparable != marker_without_time:
            raise FileExistsError(
                f"refusing to replace a different gate authorization: {output_path}"
            )
        marker = existing
    else:
        marker = {
            **marker_without_time,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_once(output_path, canonical_json_bytes(marker))
    return marker


def validate_gate_marker(
    *,
    registry_path: str | Path,
    marker_path: str | Path,
    required_gate: str,
) -> dict[str, Any]:
    registry_path = _require_file(registry_path, "campaign registry")
    validate_registry(registry_path)
    marker_path = _require_file(marker_path, f"{required_gate} gate marker")
    marker = _read_json(marker_path)
    if marker.get("schema_version") != GATE_SCHEMA:
        raise ValueError("unsupported campaign gate marker schema")
    if marker.get("gate") != required_gate or marker.get("status") != "pass":
        raise PermissionError(
            f"marker does not authorize required gate {required_gate}"
        )
    registry_record = marker.get("campaign_registry", {})
    if (
        Path(str(registry_record.get("path", ""))).resolve()
        != registry_path.resolve()
        or str(registry_record.get("sha256", "")).lower()
        != sha256_file(registry_path)
    ):
        raise ValueError("gate marker is bound to a different campaign registry")
    summary_record = marker.get("analysis_summary", {})
    summary_path = _require_file(
        str(summary_record.get("path", "")), "gate analysis summary"
    )
    if sha256_file(summary_path) != str(summary_record.get("sha256", "")).lower():
        raise ValueError("gate analysis summary changed after authorization")
    summary = _read_json(summary_path)
    expected_split, gate_name = _gate_spec(required_gate)
    if (
        summary.get("schema_version") != ANALYSIS_SCHEMA
        or summary.get("split") != expected_split
        or summary.get("evaluation_status") != "complete"
        or summary.get("gates", {}).get(gate_name, {}).get("status") != "pass"
        or summary.get("campaign", {}).get("campaign_identifier")
        != sha256_file(registry_path)
    ):
        raise PermissionError("gate analysis no longer proves a locked pass")
    return marker


def validate_prescreen_marker(
    *,
    registry_path: str | Path,
    marker_path: str | Path,
) -> dict[str, Any]:
    registry_path = _require_file(registry_path, "campaign registry")
    validate_registry(registry_path)
    marker_path = _require_file(marker_path, "pre-screen PASS marker")
    marker = _read_json(marker_path)
    if (
        marker.get("schema_version") != PRESCREEN_SCHEMA
        or marker.get("status") != "pass"
    ):
        raise PermissionError("pre-screen marker is not a locked pass")
    registry_record = marker.get("campaign_registry", {})
    if (
        Path(str(registry_record.get("path", ""))).resolve()
        != registry_path.resolve()
        or str(registry_record.get("sha256", "")).lower()
        != sha256_file(registry_path)
    ):
        raise ValueError("pre-screen marker is bound to another registry")
    audit_record = marker.get("audit", {})
    audit_path = _require_file(
        str(audit_record.get("path", "")), "pre-screen audit"
    )
    if sha256_file(audit_path) != str(audit_record.get("sha256", "")).lower():
        raise ValueError("pre-screen audit changed after authorization")
    audit = _read_json(audit_path)
    if (
        audit.get("schema_version") != "pavir-prescreen-audit-v1"
        or audit.get("status") != "pass"
    ):
        raise PermissionError("pre-screen audit no longer proves PASS")
    return marker


def assert_evaluation_authorized(
    *,
    registry_path: str | Path,
    split: str,
    gate_marker: str | Path | None,
) -> None:
    """Fail before any sealed evaluation input is inspected."""

    if split == "vc-calib":
        return
    if split == "vc-promote":
        required = "short_screen"
    elif split == "clean-test":
        required = "promotion"
    else:
        raise ValueError(f"unsupported campaign evaluation split: {split}")
    if gate_marker is None:
        raise PermissionError(f"{split} requires a hash-locked {required} PASS marker")
    validate_gate_marker(
        registry_path=registry_path,
        marker_path=gate_marker,
        required_gate=required,
    )


def checkpoint_step(path: str | Path) -> int:
    """Read only the Lightning global step from a checkpoint."""

    import torch

    checkpoint = torch.load(
        _require_file(path, "Lightning checkpoint"),
        map_location="cpu",
        weights_only=False,
    )
    step = checkpoint.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint lacks a valid global_step")
    return int(step)


def write_status(
    *,
    output_path: str | Path,
    state: str,
    registry_path: str | Path,
    stage: str,
    arm: str,
    target_updates: int,
    message: str = "",
    exit_code: int | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Write a small machine-readable job status (mutable by design)."""

    if state not in {"running", "complete", "failed", "blocked"}:
        raise ValueError(f"unsupported job state: {state}")
    registry_path = _require_file(registry_path, "campaign registry")
    payload: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA,
        "state": state,
        "stage": stage,
        "arm": arm,
        "target_updates": int(target_updates),
        "campaign_registry": _file_record(registry_path),
        "pid": os.getppid(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "message": str(message),
    }
    if exit_code is not None:
        payload["exit_code"] = int(exit_code)
    if checkpoint is not None and Path(checkpoint).is_file():
        payload["checkpoint"] = {
            **_file_record(Path(checkpoint)),
            "global_step": checkpoint_step(checkpoint),
        }
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, path)
    return payload


__all__ = [
    "ANALYSIS_SCHEMA",
    "ARM_NAMES",
    "ARM_SCHEMA",
    "GATE_SCHEMA",
    "LOCKED_BEAM_PRIMARY",
    "LOCKED_BEAM_SECONDARY",
    "LOCKED_CTC_WEIGHT",
    "LOCKED_LR",
    "LOCKED_MAX_EPOCHS",
    "LOCKED_MAX_FRAMES",
    "LOCKED_PRECISION",
    "LOCKED_SEED",
    "LOCKED_TRAIN_BUCKETS",
    "LOCKED_WARMUP_EPOCHS",
    "LOCKED_WEIGHT_DECAY",
    "PAVIR_ARMS",
    "PRESCREEN_SCHEMA",
    "SCREEN_ONLY_ARMS",
    "assert_evaluation_authorized",
    "authorize_gate",
    "build_campaign_bundle",
    "build_route_shuffled_payload",
    "checkpoint_step",
    "count_padded_batches",
    "load_source_bundle",
    "read_label_lengths",
    "resolve_evaluation",
    "resolve_training",
    "route_counts",
    "route_support",
    "sha256_file",
    "validate_gate_marker",
    "validate_prescreen_marker",
    "validate_registry",
    "write_status",
]
