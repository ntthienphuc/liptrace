"""Strict, immutable artifact loading for PAViR-VSR.

The training process is allowed to *consume* only the confusion graph, routes,
prototype vectors, and occurrence map that were frozen before the matched-arm
campaign.  This module validates their hashes and schemas up front so a missing
ID cannot silently turn PAViR into a different experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


SCHEMA_CONFIG = "pavir-vsr-config-v1"
SCHEMA_ROUTES = "pavir-routes-v1"
SCHEMA_OCCURRENCES = "pavir-occurrences-v1"
SCHEMA_CANONICAL_IDS = "canonical-id-map-v1"
VALID_ROUTES = frozenset({"E", "D", "A"})


def sha256_file(path: str | Path, *, chunk_size: int = 8 << 20) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for precommit hashes."""

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


def _strip_known_extension(value: str) -> str:
    lowered = value.lower()
    for extension in (".video", ".mp4", ".npy", ".npz", ".mkv", ".avi", ".mov"):
        if lowered.endswith(extension):
            return value[: -len(extension)]
    return value


def normalize_sample_component(value: str) -> str:
    """Normalize a machine-independent clip/member component.

    ViCocktail CSV rows may contain ``tar_path#member_prefix``.  The tar path
    is storage metadata, so only the member prefix belongs in the canonical ID.
    """

    normalized = unicodedata.normalize("NFC", str(value)).replace("\\", "/").strip()
    if "#" in normalized:
        normalized = normalized.rsplit("#", 1)[1]
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("/"):
        raise ValueError(f"sample component is absolute: {value!r}")
    normalized = "/".join(part for part in normalized.split("/") if part not in ("", "."))
    normalized = _strip_known_extension(normalized)
    if not normalized:
        raise ValueError("sample component is empty")
    if any(part == ".." for part in normalized.split("/")):
        raise ValueError(f"sample component escapes its logical root: {value!r}")
    if "\x00" in normalized:
        raise ValueError("sample component contains a NUL byte")
    return normalized


def canonical_sample_id(corpus: str, split: str, value: str) -> str:
    """Build ``corpus:split:clip`` IDs shared by CSV, tar, and artifacts."""

    corpus_id = str(corpus).strip().lower()
    split_id = str(split).strip().lower()
    if corpus_id not in {"vicocktail", "vlr"}:
        raise ValueError(f"unsupported corpus: {corpus!r}")
    if not corpus_id or not split_id or ":" in corpus_id or ":" in split_id:
        raise ValueError("corpus and split must be non-empty colon-free identifiers")
    return f"{corpus_id}:{split_id}:{normalize_sample_component(value)}"


def canonical_lookup_key(loader_path: str) -> str:
    """Recover the storage-independent key used by a canonical-ID map."""

    normalized = unicodedata.normalize("NFC", str(loader_path)).replace("\\", "/")
    if "#" in normalized:
        return normalize_sample_component(normalized.rsplit("#", 1)[1])
    for marker in (
        "/data/train/",
        "/data/val/",
        "/data/valid/",
        "/data/test/",
        "/train/",
        "/val/",
        "/valid/",
        "/test/",
    ):
        if marker in normalized:
            return normalize_sample_component(normalized.split(marker, 1)[1])
    return normalize_sample_component(normalized)


def _require_keys(payload: Mapping[str, Any], required: Iterable[str], context: str) -> None:
    missing = sorted(set(required) - set(payload))
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")


def _strict_int_list(value: Any, context: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty JSON integer list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{context} must contain only integers")
    return tuple(int(item) for item in value)


@dataclass(frozen=True)
class FrozenEdge:
    edge_id: str
    target_text: str
    confusion_text: str
    target_token_ids: Tuple[int, ...]
    confusion_token_ids: Tuple[int, ...]
    route: str
    tone_only: bool
    support: int
    source_count: int


@dataclass(frozen=True)
class TrainingEvent:
    edge_id: str
    target_start: int
    target_end: int
    # Legacy occurrence artifacts omit this field and therefore use the
    # event's own confusion prototype.  Event-level negative controls keep
    # ``edge_id`` fixed and set only this optional confusion endpoint.
    negative_edge_id: str | None = None


@dataclass(frozen=True)
class LockedFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class PAViRConfig:
    path: Path
    corpus: str
    seed: int
    block_index: int
    visual_margin: float
    context_margin: float
    lambda_visual: float
    lambda_context: float
    ramp_updates: int
    context_batch_fraction: float
    routes: LockedFile
    occurrences: LockedFile
    prototypes: LockedFile
    canonical_ids: LockedFile
    forbidden_ids: LockedFile | None

    @classmethod
    def load(cls, path: str | Path) -> "PAViRConfig":
        config_path = Path(path).resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        _require_keys(
            payload,
            (
                "schema_version",
                "corpus",
                "seed",
                "block_index",
                "visual_margin",
                "context_margin",
                "lambda_visual",
                "lambda_context",
                "ramp_updates",
                "context_batch_fraction",
                "artifacts",
            ),
            "PAViR config",
        )
        if payload["schema_version"] != SCHEMA_CONFIG:
            raise ValueError(f"unsupported PAViR config schema: {payload['schema_version']!r}")
        corpus = str(payload["corpus"]).lower()
        if corpus not in {"vicocktail", "vlr"}:
            raise ValueError(f"unsupported PAViR corpus: {corpus!r}")

        artifacts = payload["artifacts"]
        if not isinstance(artifacts, dict):
            raise ValueError("config.artifacts must be an object")
        _require_keys(
            artifacts,
            ("routes", "occurrences", "prototypes", "canonical_ids"),
            "config.artifacts",
        )

        def locked(name: str, *, optional: bool = False) -> LockedFile | None:
            entry = artifacts.get(name)
            if entry is None and optional:
                return None
            if not isinstance(entry, dict):
                raise ValueError(f"artifacts.{name} must be an object")
            _require_keys(entry, ("path", "sha256"), f"artifacts.{name}")
            artifact_path = (config_path.parent / str(entry["path"])).resolve()
            try:
                artifact_path.relative_to(config_path.parent)
            except ValueError as error:
                raise ValueError(f"artifacts.{name} escapes the config directory") from error
            expected = str(entry["sha256"]).lower()
            if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
                raise ValueError(f"artifacts.{name}.sha256 is invalid")
            if not artifact_path.is_file():
                raise FileNotFoundError(f"locked artifact is missing: {artifact_path}")
            actual = sha256_file(artifact_path)
            if actual != expected:
                raise ValueError(
                    f"hash mismatch for {name}: expected {expected}, observed {actual}"
                )
            return LockedFile(artifact_path, expected)

        numeric = {
            name: float(payload[name])
            for name in (
                "visual_margin",
                "context_margin",
                "lambda_visual",
                "lambda_context",
                "context_batch_fraction",
            )
        }
        if any(not math.isfinite(value) for value in numeric.values()):
            raise ValueError("PAViR numeric fields must be finite")
        if numeric["visual_margin"] < 0 or numeric["context_margin"] < 0:
            raise ValueError("margins must be non-negative")
        if numeric["lambda_visual"] < 0 or numeric["lambda_context"] < 0:
            raise ValueError("auxiliary weights must be non-negative")
        if not 0 <= numeric["context_batch_fraction"] <= 0.5:
            raise ValueError("context_batch_fraction must be in [0, 0.5]")
        block_index = int(payload["block_index"])
        if block_index < 1:
            raise ValueError("block_index is one-based and must be positive")
        ramp_updates = int(payload["ramp_updates"])
        if ramp_updates < 0:
            raise ValueError("ramp_updates must be non-negative")

        routes = locked("routes")
        occurrences = locked("occurrences")
        prototypes = locked("prototypes")
        canonical_ids = locked("canonical_ids")
        assert (
            routes is not None
            and occurrences is not None
            and prototypes is not None
            and canonical_ids is not None
        )
        return cls(
            path=config_path,
            corpus=corpus,
            seed=int(payload["seed"]),
            block_index=block_index,
            visual_margin=numeric["visual_margin"],
            context_margin=numeric["context_margin"],
            lambda_visual=numeric["lambda_visual"],
            lambda_context=numeric["lambda_context"],
            ramp_updates=ramp_updates,
            context_batch_fraction=numeric["context_batch_fraction"],
            routes=routes,
            occurrences=occurrences,
            prototypes=prototypes,
            canonical_ids=canonical_ids,
            forbidden_ids=locked("forbidden_ids", optional=True),
        )

    def ramp(self, update: int) -> float:
        """Linear auxiliary ramp, reaching one at ``ramp_updates``."""

        if update < 0:
            raise ValueError("update must be non-negative")
        if self.ramp_updates == 0:
            return 1.0
        return min(1.0, float(update + 1) / float(self.ramp_updates))


class ArtifactStore:
    """In-memory, read-only view of the hash-locked PAViR artifacts."""

    def __init__(self, config: PAViRConfig):
        self.config = config
        self.edges = self._load_edges(
            config.routes.path,
            expected_corpus=config.corpus,
        )
        self.events = self._load_events(
            config.occurrences.path,
            expected_corpus=config.corpus,
            expected_routes_sha256=config.routes.sha256,
        )
        self.canonical_ids = CanonicalIDMap.load(config.canonical_ids.path)
        self.prototype_keys, self.prototype_vectors = self._load_prototypes(
            config.prototypes.path
        )
        self.prototype_index = {
            key: index for index, key in enumerate(self.prototype_keys)
        }
        self._validate_cross_references()
        if config.forbidden_ids is not None:
            self.assert_no_forbidden_ids(self._load_forbidden(config.forbidden_ids.path))

    @staticmethod
    def _load_edges(
        path: Path,
        *,
        expected_corpus: str,
    ) -> Dict[str, FrozenEdge]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_ROUTES:
            raise ValueError(f"unsupported route schema in {path}")
        if str(payload.get("corpus", "")).lower() != expected_corpus:
            raise ValueError("route artifact corpus disagrees with config")
        rows = payload.get("edges")
        if not isinstance(rows, list):
            raise ValueError("route artifact must contain an edges list")
        edges: Dict[str, FrozenEdge] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"edge {index} is not an object")
            _require_keys(
                row,
                (
                    "edge_id",
                    "target_text",
                    "confusion_text",
                    "target_token_ids",
                    "confusion_token_ids",
                    "route",
                    "tone_only",
                    "support",
                    "source_count",
                ),
                f"edge {index}",
            )
            edge_id = str(row["edge_id"])
            route = str(row["route"])
            if not edge_id or edge_id in edges:
                raise ValueError(f"empty or duplicate edge_id: {edge_id!r}")
            if route not in VALID_ROUTES:
                raise ValueError(f"invalid route for edge {edge_id}: {route!r}")
            support, source_count = int(row["support"]), int(row["source_count"])
            if support < 0 or source_count < 0 or source_count > support:
                raise ValueError(f"invalid support for edge {edge_id}")
            edges[edge_id] = FrozenEdge(
                edge_id=edge_id,
                target_text=unicodedata.normalize("NFC", str(row["target_text"])),
                confusion_text=unicodedata.normalize("NFC", str(row["confusion_text"])),
                target_token_ids=_strict_int_list(
                    row["target_token_ids"], f"{edge_id}.target_token_ids"
                ),
                confusion_token_ids=_strict_int_list(
                    row["confusion_token_ids"], f"{edge_id}.confusion_token_ids"
                ),
                route=route,
                tone_only=bool(row["tone_only"]),
                support=support,
                source_count=source_count,
            )
            if any(token <= 0 for token in (
                *edges[edge_id].target_token_ids,
                *edges[edge_id].confusion_token_ids,
            )):
                raise ValueError(
                    f"edge {edge_id} token IDs must exclude blank/padding"
                )
            if (
                edges[edge_id].target_token_ids
                == edges[edge_id].confusion_token_ids
            ):
                raise ValueError(f"edge {edge_id} has identical token endpoints")
        return edges

    @staticmethod
    def _load_events(
        path: Path,
        *,
        expected_corpus: str,
        expected_routes_sha256: str,
    ) -> Dict[str, Tuple[TrainingEvent, ...]]:
        events: Dict[str, Tuple[TrainingEvent, ...]] = {}
        with path.open(encoding="utf-8") as stream:
            header = json.loads(stream.readline())
            if header.get("schema_version") != SCHEMA_OCCURRENCES:
                raise ValueError(f"unsupported occurrence schema in {path}")
            if str(header.get("corpus", "")).lower() != expected_corpus:
                raise ValueError("occurrence artifact corpus disagrees with config")
            inputs = header.get("inputs")
            if not isinstance(inputs, dict):
                raise ValueError("occurrence artifact has no inputs hash contract")
            if str(inputs.get("routes_sha256", "")).lower() != expected_routes_sha256:
                raise ValueError("occurrence artifact is not locked to supplied routes")
            for line_number, line in enumerate(stream, start=2):
                if not line.strip():
                    continue
                row = json.loads(line)
                _require_keys(row, ("sample_id", "events"), f"occurrence line {line_number}")
                sample_id = str(row["sample_id"])
                if sample_id in events:
                    raise ValueError(f"duplicate occurrence sample_id: {sample_id}")
                if not (
                    sample_id.startswith("vicocktail:")
                    or sample_id.startswith("vlr:")
                ):
                    raise ValueError(f"non-canonical sample_id: {sample_id!r}")
                parsed = []
                seen = set()
                for event_index, event in enumerate(row["events"]):
                    _require_keys(
                        event,
                        ("edge_id", "target_start", "target_end"),
                        f"occurrence {sample_id}[{event_index}]",
                    )
                    start, end = int(event["target_start"]), int(event["target_end"])
                    if start < 0 or end <= start:
                        raise ValueError(f"invalid target span for {sample_id}")
                    negative_edge_id = None
                    if "negative_edge_id" in event:
                        raw_negative_edge_id = event["negative_edge_id"]
                        if (
                            not isinstance(raw_negative_edge_id, str)
                            or not raw_negative_edge_id
                        ):
                            raise ValueError(
                                "negative_edge_id must be a non-empty string for "
                                f"{sample_id}[{event_index}]"
                            )
                        negative_edge_id = raw_negative_edge_id
                    parsed_event = TrainingEvent(
                        str(event["edge_id"]),
                        start,
                        end,
                        negative_edge_id,
                    )
                    event_key = (
                        parsed_event.edge_id,
                        parsed_event.target_start,
                        parsed_event.target_end,
                    )
                    if event_key in seen:
                        raise ValueError(
                            f"duplicate occurrence event for {sample_id}: {event_key}"
                        )
                    seen.add(event_key)
                    parsed.append(parsed_event)
                if parsed:
                    events[sample_id] = tuple(parsed)
        return events

    @staticmethod
    def _load_prototypes(path: Path) -> tuple[Tuple[str, ...], np.ndarray]:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"keys", "vectors"}:
                raise ValueError("prototype NPZ must contain exactly keys and vectors")
            keys = tuple(str(value) for value in payload["keys"].tolist())
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or len(keys) != len(vectors) or not len(keys):
            raise ValueError("prototype keys/vectors have incompatible shapes")
        if len(set(keys)) != len(keys) or any(not key for key in keys):
            raise ValueError("prototype keys must be unique and non-empty")
        if not np.isfinite(vectors).all():
            raise ValueError("prototype vectors contain non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(np.abs(norms - 1.0) > 5e-3):
            raise ValueError("prototype vectors must be L2-normalized")
        vectors.setflags(write=False)
        return keys, vectors

    @staticmethod
    def _load_forbidden(path: Path) -> frozenset[str]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("sample_ids", payload)
        if not isinstance(values, list):
            raise ValueError("forbidden-ID artifact must be a list or {sample_ids: list}")
        return frozenset(str(value) for value in values)

    def _validate_cross_references(self) -> None:
        canonical_samples = self.canonical_ids.sample_ids
        canonical_corpora = {
            sample_id.split(":", 1)[0] for sample_id in canonical_samples
        }
        if canonical_corpora != {self.config.corpus}:
            raise ValueError(
                "canonical-ID map corpus disagrees with config: "
                f"{sorted(canonical_corpora)}"
            )
        positive_e_edge_ids: Counter[str] = Counter()
        negative_e_edge_ids: Counter[str] = Counter()
        legacy_e_events = 0
        for sample_id, sample_events in self.events.items():
            if not sample_id.startswith(self.config.corpus + ":"):
                raise ValueError(
                    f"occurrence corpus disagrees with config: {sample_id}"
                )
            if sample_id not in canonical_samples:
                raise ValueError(
                    f"occurrence sample is absent from canonical-ID map: {sample_id}"
                )
            if self.canonical_ids.split_for(sample_id) != "train":
                raise ValueError(
                    f"occurrence sample is not in canonical train split: {sample_id}"
                )
            for event in sample_events:
                edge = self.edges.get(event.edge_id)
                if edge is None:
                    raise ValueError(f"event references unknown edge: {event.edge_id}")
                if edge.route == "A":
                    raise ValueError(
                        f"occurrence references inactive route-A edge: {event.edge_id}"
                    )
                if event.negative_edge_id is not None:
                    if edge.route != "E":
                        raise ValueError(
                            "event-level negative_edge_id is valid only for route-E "
                            f"events: {event.edge_id}"
                        )
                    negative_edge = self.edges.get(event.negative_edge_id)
                    if negative_edge is None:
                        raise ValueError(
                            "event references unknown negative edge: "
                            f"{event.negative_edge_id}"
                        )
                    if negative_edge.route != "E":
                        raise ValueError(
                            "event-level negative edge must use route E: "
                            f"{event.negative_edge_id}"
                        )
                    if negative_edge.edge_id == edge.edge_id:
                        raise ValueError(
                            "event-level negative edge must be a zero-identity "
                            f"derangement: {edge.edge_id}"
                        )
                    if (
                        negative_edge.confusion_token_ids
                        == edge.target_token_ids
                    ):
                        raise ValueError(
                            "event-level negative confusion tokens must differ "
                            "from the positive target tokens: "
                            f"positive={edge.edge_id}, "
                            f"negative={negative_edge.edge_id}"
                        )
                    if (
                        negative_edge.confusion_token_ids
                        == edge.confusion_token_ids
                    ):
                        raise ValueError(
                            "event-level negative confusion tokens must differ "
                            "from the original confusion tokens: "
                            f"positive={edge.edge_id}, "
                            f"negative={negative_edge.edge_id}"
                        )
                if edge.route == "E":
                    positive_e_edge_ids[edge.edge_id] += 1
                    if event.negative_edge_id is None:
                        legacy_e_events += 1
                    else:
                        negative_e_edge_ids[event.negative_edge_id] += 1
                    missing = [
                        key
                        for key in (
                            f"{edge.edge_id}:target",
                            f"{edge.edge_id}:confusion",
                        )
                        if key not in self.prototype_index
                    ]
                    if missing:
                        raise ValueError(
                            f"route-E edge {edge.edge_id} is missing prototypes: {missing}"
                        )
                    if (
                        event.negative_edge_id is not None
                        and f"{event.negative_edge_id}:confusion"
                        not in self.prototype_index
                    ):
                        raise ValueError(
                            "event-level negative edge is missing its confusion "
                            f"prototype: {event.negative_edge_id}"
                        )
        if negative_e_edge_ids:
            if legacy_e_events:
                raise ValueError(
                    "event-level negative control must cover every route-E event"
                )
            if negative_e_edge_ids != positive_e_edge_ids:
                raise ValueError(
                    "event-level negative edge IDs are not an exact permutation "
                    "of positive route-E edge IDs"
                )
        for key in self.prototype_keys:
            try:
                edge_id, endpoint = key.rsplit(":", 1)
            except ValueError as error:
                raise ValueError(f"malformed prototype key: {key!r}") from error
            edge = self.edges.get(edge_id)
            if edge is None or endpoint not in {"target", "confusion"}:
                raise ValueError(f"prototype references unknown endpoint: {key}")
            if edge.route == "A":
                raise ValueError(f"prototype references inactive route-A edge: {key}")

    def assert_no_forbidden_ids(self, forbidden: Sequence[str] | set[str]) -> None:
        leakage = sorted(set(self.events).intersection(forbidden))
        if leakage:
            raise ValueError(
                f"frozen training artifacts leak {len(leakage)} forbidden IDs; "
                f"first={leakage[0]}"
            )

    def events_for(self, sample_id: str, route: str | None = None) -> Tuple[TrainingEvent, ...]:
        values = self.events.get(str(sample_id), ())
        if route is None:
            return values
        if route not in VALID_ROUTES:
            raise ValueError(f"unknown route: {route}")
        return tuple(event for event in values if self.edges[event.edge_id].route == route)

    def prototype(self, edge_id: str, endpoint: str) -> np.ndarray:
        if endpoint not in {"target", "confusion"}:
            raise ValueError("prototype endpoint must be target or confusion")
        key = f"{edge_id}:{endpoint}"
        try:
            return self.prototype_vectors[self.prototype_index[key]]
        except KeyError as error:
            raise KeyError(f"missing frozen prototype: {key}") from error


def normalize_split(value: str) -> str:
    split = str(value).strip().lower()
    aliases = {"validation": "val", "valid": "val"}
    split = aliases.get(split, split)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {value!r}")
    return split


class CanonicalIDMap:
    """Strict lookup from loader-relative keys to corpus-stable sample IDs."""

    def __init__(
        self,
        entries: Mapping[tuple[str, str], str],
        *,
        source_ids: Mapping[str, str] | None = None,
    ):
        self._entries = dict(entries)
        self._sample_splits: Dict[str, str] = {}
        for (split, _), sample_id in self._entries.items():
            previous = self._sample_splits.get(sample_id)
            if previous is not None and previous != split:
                raise ValueError(
                    f"canonical sample_id maps to multiple splits: {sample_id}"
                )
            self._sample_splits[sample_id] = split
        self._source_ids = dict(source_ids or {})

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalIDMap":
        entries: Dict[tuple[str, str], str] = {}
        source_ids: Dict[str, str] = {}
        seen_ids = set()
        with Path(path).open(encoding="utf-8") as stream:
            try:
                header = json.loads(stream.readline())
            except json.JSONDecodeError as error:
                raise ValueError("canonical-ID map has an invalid header") from error
            if header.get("schema_version") != SCHEMA_CANONICAL_IDS:
                raise ValueError("unsupported canonical-ID map schema")
            for line_number, line in enumerate(stream, start=2):
                if not line.strip():
                    continue
                row = json.loads(line)
                _require_keys(
                    row,
                    ("split", "lookup_key", "sample_id", "source_id"),
                    f"canonical-ID line {line_number}",
                )
                split = normalize_split(row["split"])
                lookup_key = normalize_sample_component(row["lookup_key"])
                sample_id = str(row["sample_id"])
                source_id = str(row["source_id"])
                if not source_id:
                    raise ValueError(f"empty source_id at canonical-ID line {line_number}")
                if not (
                    sample_id.startswith("vicocktail:")
                    or sample_id.startswith("vlr:")
                ):
                    raise ValueError(f"non-canonical sample_id: {sample_id!r}")
                key = (split, lookup_key)
                if key in entries:
                    raise ValueError(f"duplicate canonical lookup key: {key}")
                if sample_id in seen_ids:
                    raise ValueError(f"duplicate canonical sample_id: {sample_id}")
                entries[key] = sample_id
                source_ids[sample_id] = source_id
                seen_ids.add(sample_id)
        if not entries:
            raise ValueError("canonical-ID map is empty")
        return cls(entries, source_ids=source_ids)

    def lookup(self, split: str, loader_path: str) -> str:
        key = (normalize_split(split), canonical_lookup_key(loader_path))
        try:
            return self._entries[key]
        except KeyError as error:
            raise KeyError(
                f"canonical-ID map has no entry for split/key {key}"
            ) from error

    @property
    def sample_ids(self) -> frozenset[str]:
        return frozenset(self._sample_splits)

    def split_for(self, sample_id: str) -> str:
        try:
            return self._sample_splits[str(sample_id)]
        except KeyError as error:
            raise KeyError(f"canonical-ID map has no sample_id {sample_id!r}") from error

    def source_for(self, sample_id: str) -> str:
        try:
            return self._source_ids[str(sample_id)]
        except KeyError as error:
            raise KeyError(
                f"canonical-ID map has no source_id for {sample_id!r}"
            ) from error

    def __len__(self) -> int:
        return len(self._entries)
