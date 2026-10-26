import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pavir.artifacts import (
    ArtifactStore,
    CanonicalIDMap,
    PAViRConfig,
    canonical_json_bytes,
    canonical_lookup_key,
    canonical_sample_id,
    normalize_sample_component,
    sha256_file,
)


def _write_fixture(root: Path, *, forbidden=False):
    routes = {
        "schema_version": "pavir-routes-v1",
        "corpus": "vicocktail",
        "edges": [
            {
                "edge_id": "e1",
                "target_text": "má",
                "confusion_text": "ma",
                "target_token_ids": [11],
                "confusion_token_ids": [12],
                "route": "E",
                "tone_only": False,
                "support": 25,
                "source_count": 12,
            },
            {
                "edge_id": "d1",
                "target_text": "má",
                "confusion_text": "mà",
                "target_token_ids": [11],
                "confusion_token_ids": [13],
                "route": "D",
                "tone_only": True,
                "support": 30,
                "source_count": 15,
            },
        ],
    }
    route_path = root / "routes.json"
    route_path.write_bytes(canonical_json_bytes(routes))
    occurrence_path = root / "occurrences.jsonl"
    occurrence_path.write_text(
        json.dumps(
            {
                "schema_version": "pavir-occurrences-v1",
                "corpus": "vicocktail",
                "inputs": {"routes_sha256": sha256_file(route_path)},
            }
        )
        + "\n"
        + json.dumps(
            {
                "sample_id": "vicocktail:train:source/0001",
                "events": [
                    {"edge_id": "e1", "target_start": 0, "target_end": 1},
                    {"edge_id": "d1", "target_start": 0, "target_end": 1},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prototype_path = root / "prototypes.npz"
    np.savez(
        prototype_path,
        keys=np.array(["e1:target", "e1:confusion"]),
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
    )
    canonical_path = root / "canonical_ids.jsonl"
    canonical_path.write_text(
        json.dumps({"schema_version": "canonical-id-map-v1"}) + "\n"
        + json.dumps(
            {
                "split": "train",
                "lookup_key": "0000000001",
                "sample_id": "vicocktail:train:source/0001",
                "source_id": "source",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = {
        name: {"path": path.name, "sha256": sha256_file(path)}
        for name, path in (
            ("routes", route_path),
            ("occurrences", occurrence_path),
            ("prototypes", prototype_path),
            ("canonical_ids", canonical_path),
        )
    }
    if forbidden:
        forbidden_path = root / "forbidden.json"
        forbidden_path.write_text(
            json.dumps({"sample_ids": ["vicocktail:train:source/0001"]}),
            encoding="utf-8",
        )
        artifacts["forbidden_ids"] = {
            "path": forbidden_path.name,
            "sha256": sha256_file(forbidden_path),
        }
    config = {
        "schema_version": "pavir-vsr-config-v1",
        "corpus": "vicocktail",
        "seed": 42,
        "block_index": 6,
        "visual_margin": 0.15,
        "context_margin": 0.10,
        "lambda_visual": 0.02,
        "lambda_context": 0.01,
        "ramp_updates": 100,
        "context_batch_fraction": 0.5,
        "artifacts": artifacts,
    }
    config_path = root / "config.json"
    config_path.write_bytes(canonical_json_bytes(config))
    return config_path


def test_canonical_ids_handle_vicocktail_tar_members():
    assert (
        canonical_sample_id(
            "ViCocktail", "train", "/cache/train-00003.tar#VTV24/show/clip.video"
        )
        == "vicocktail:train:VTV24/show/clip"
    )
    assert normalize_sample_component(r".\BAIGI1\clip.npy") == "BAIGI1/clip"
    assert (
        canonical_lookup_key(
            "/root/workspace/data/vlr/cloud_dataset_signer_split/"
            "data/train/part1/BAIGI3/clip.mp4"
        )
        == "part1/BAIGI3/clip"
    )
    with pytest.raises(ValueError):
        normalize_sample_component("../../escape.mp4")
    with pytest.raises(ValueError):
        normalize_sample_component("/absolute/clip.mp4")


def test_strict_artifact_load_and_ramp(tmp_path):
    config = PAViRConfig.load(_write_fixture(tmp_path))
    store = ArtifactStore(config)
    assert store.edges["e1"].route == "E"
    assert [event.edge_id for event in store.events_for(
        "vicocktail:train:source/0001", "D"
    )] == ["d1"]
    np.testing.assert_array_equal(store.prototype("e1", "target"), [1.0, 0.0])
    assert config.ramp(0) == pytest.approx(0.01)
    assert config.ramp(99) == 1.0
    assert config.ramp(1000) == 1.0
    canonical = CanonicalIDMap.load(config.canonical_ids.path)
    assert (
        canonical.lookup("train", "/old/cache/shard.tar#0000000001.video")
        == "vicocktail:train:source/0001"
    )


def test_hash_mismatch_fails_closed(tmp_path):
    config_path = _write_fixture(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["artifacts"]["routes"]["sha256"] = "0" * 64
    config_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        PAViRConfig.load(config_path)


def test_forbidden_id_leakage_fails_closed(tmp_path):
    config = PAViRConfig.load(_write_fixture(tmp_path, forbidden=True))
    with pytest.raises(ValueError, match="leak"):
        ArtifactStore(config)


def test_occurrences_must_be_canonical_train_members_locked_to_routes(tmp_path):
    config_path = _write_fixture(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    occurrence_path = tmp_path / payload["artifacts"]["occurrences"]["path"]
    lines = occurrence_path.read_text(encoding="utf-8").splitlines()

    header = json.loads(lines[0])
    header["inputs"]["routes_sha256"] = "0" * 64
    occurrence_path.write_text(
        json.dumps(header) + "\n" + "\n".join(lines[1:]) + "\n",
        encoding="utf-8",
    )
    payload["artifacts"]["occurrences"]["sha256"] = sha256_file(occurrence_path)
    config_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="locked to supplied routes"):
        ArtifactStore(PAViRConfig.load(config_path))

    config_path = _write_fixture(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    occurrence_path = tmp_path / payload["artifacts"]["occurrences"]["path"]
    lines = occurrence_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["sample_id"] = "vicocktail:train:not-in-map"
    occurrence_path.write_text(
        lines[0] + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    payload["artifacts"]["occurrences"]["sha256"] = sha256_file(occurrence_path)
    config_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="absent from canonical-ID map"):
        ArtifactStore(PAViRConfig.load(config_path))


def test_canonical_json_is_stable():
    left = canonical_json_bytes({"b": 2, "a": "má"})
    right = canonical_json_bytes({"a": "má", "b": 2})
    assert left == right
    assert hashlib.sha256(left).hexdigest() == hashlib.sha256(right).hexdigest()
