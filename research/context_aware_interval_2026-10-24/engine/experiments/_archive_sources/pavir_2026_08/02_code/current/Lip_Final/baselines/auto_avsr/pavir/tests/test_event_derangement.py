import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pavir.artifacts import (
    ArtifactStore,
    FrozenEdge,
    PAViRConfig,
    TrainingEvent,
    canonical_json_bytes,
    sha256_file,
)
from pavir.routing import visual_margin_loss


def _write_store_fixture(
    root: Path,
    *,
    event_negative: str | None,
    d_event_negative: str | None = None,
    cover_second_e_event: bool = True,
    target_collision: bool = False,
    original_confusion_collision: bool = False,
) -> Path:
    routes = {
        "schema_version": "pavir-routes-v1",
        "corpus": "vicocktail",
        "edges": [
            {
                "edge_id": "e1",
                "target_text": "a",
                "confusion_text": "b",
                "target_token_ids": [1],
                "confusion_token_ids": [2],
                "route": "E",
                "tone_only": False,
                "support": 10,
                "source_count": 5,
            },
            {
                "edge_id": "e2",
                "target_text": "a",
                "confusion_text": "c",
                "target_token_ids": [4]
                if (target_collision or original_confusion_collision)
                else [1],
                "confusion_token_ids": [
                    1
                    if target_collision
                    else (2 if original_confusion_collision else 3)
                ],
                "route": "E",
                "tone_only": False,
                "support": 8,
                "source_count": 4,
            },
            {
                "edge_id": "d1",
                "target_text": "a",
                "confusion_text": "d",
                "target_token_ids": [1],
                "confusion_token_ids": [4],
                "route": "D",
                "tone_only": False,
                "support": 6,
                "source_count": 3,
            },
        ],
    }
    route_path = root / "routes.json"
    route_path.write_bytes(canonical_json_bytes(routes))

    e_event = {"edge_id": "e1", "target_start": 0, "target_end": 1}
    if event_negative is not None:
        e_event["negative_edge_id"] = event_negative
    second_e_event = {"edge_id": "e2", "target_start": 1, "target_end": 2}
    if event_negative is not None and cover_second_e_event:
        second_e_event["negative_edge_id"] = "e1"
    d_event = {"edge_id": "d1", "target_start": 0, "target_end": 1}
    if d_event_negative is not None:
        d_event["negative_edge_id"] = d_event_negative
    occurrence_path = root / "occurrences.jsonl"
    occurrence_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "pavir-occurrences-v1",
                "corpus": "vicocktail",
                "inputs": {"routes_sha256": sha256_file(route_path)},
            }
        )
        + canonical_json_bytes(
            {
                "sample_id": "vicocktail:train:s1",
                "events": [e_event, second_e_event, d_event],
            }
        )
    )

    prototype_path = root / "prototypes.npz"
    np.savez(
        prototype_path,
        keys=np.array(
            ["e1:target", "e1:confusion", "e2:target", "e2:confusion"]
        ),
        vectors=np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0]],
            dtype=np.float16,
        ),
    )
    canonical_path = root / "canonical.jsonl"
    canonical_path.write_bytes(
        canonical_json_bytes({"schema_version": "canonical-id-map-v1"})
        + canonical_json_bytes(
            {
                "split": "train",
                "lookup_key": "s1",
                "sample_id": "vicocktail:train:s1",
                "source_id": "source",
            }
        )
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
    config_path = root / "config.json"
    config_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "pavir-vsr-config-v1",
                "corpus": "vicocktail",
                "seed": 42,
                "block_index": 6,
                "visual_margin": 0.15,
                "context_margin": 0.1,
                "lambda_visual": 1.0,
                "lambda_context": 0.0,
                "ramp_updates": 0,
                "context_batch_fraction": 0.5,
                "artifacts": artifacts,
            }
        )
    )
    return config_path


def test_artifact_store_loads_event_negative_and_legacy_occurrences(
    tmp_path: Path,
) -> None:
    controlled = tmp_path / "controlled"
    controlled.mkdir()
    controlled_store = ArtifactStore(
        PAViRConfig.load(
            _write_store_fixture(controlled, event_negative="e2")
        )
    )
    controlled_events = controlled_store.events_for(
        "vicocktail:train:s1", "E"
    )
    assert controlled_events == (
        TrainingEvent("e1", 0, 1, "e2"),
        TrainingEvent("e2", 1, 2, "e1"),
    )

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    legacy_store = ArtifactStore(
        PAViRConfig.load(_write_store_fixture(legacy, event_negative=None))
    )
    assert legacy_store.events_for("vicocktail:train:s1", "E") == (
        TrainingEvent("e1", 0, 1),
        TrainingEvent("e2", 1, 2),
    )


def test_artifact_store_rejects_identity_or_non_e_negative(tmp_path: Path) -> None:
    identity = tmp_path / "identity"
    identity.mkdir()
    with pytest.raises(ValueError, match="zero-identity"):
        ArtifactStore(
            PAViRConfig.load(_write_store_fixture(identity, event_negative="e1"))
        )

    wrong_route = tmp_path / "wrong-route"
    wrong_route.mkdir()
    with pytest.raises(ValueError, match="negative edge must use route E"):
        ArtifactStore(
            PAViRConfig.load(_write_store_fixture(wrong_route, event_negative="d1"))
        )

    d_event = tmp_path / "d-event"
    d_event.mkdir()
    with pytest.raises(ValueError, match="valid only for route-E"):
        ArtifactStore(
            PAViRConfig.load(
                _write_store_fixture(
                    d_event,
                    event_negative="e2",
                    d_event_negative="e1",
                )
            )
        )

    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(ValueError, match="cover every route-E event"):
        ArtifactStore(
            PAViRConfig.load(
                _write_store_fixture(
                    partial,
                    event_negative="e2",
                    cover_second_e_event=False,
                )
            )
        )

    target_collision = tmp_path / "target-collision"
    target_collision.mkdir()
    with pytest.raises(
        ValueError,
        match="negative confusion tokens must differ from the positive target",
    ):
        ArtifactStore(
            PAViRConfig.load(
                _write_store_fixture(
                    target_collision,
                    event_negative="e2",
                    target_collision=True,
                )
            )
        )

    original_confusion_collision = tmp_path / "original-confusion-collision"
    original_confusion_collision.mkdir()
    with pytest.raises(
        ValueError,
        match="negative confusion tokens must differ from the original confusion",
    ):
        ArtifactStore(
            PAViRConfig.load(
                _write_store_fixture(
                    original_confusion_collision,
                    event_negative="e2",
                    original_confusion_collision=True,
                )
            )
        )


class _LookupStore:
    def __init__(self) -> None:
        self.edges = {
            "e1": FrozenEdge("e1", "a", "b", (1,), (2,), "E", False, 2, 2),
            "e2": FrozenEdge("e2", "a", "c", (1,), (3,), "E", False, 2, 2),
        }
        self.calls: list[tuple[str, str]] = []

    def events_for(self, sample_id: str, route: str | None = None):
        assert sample_id == "vicocktail:train:s1"
        assert route == "E"
        return (TrainingEvent("e1", 0, 1, "e2"),)

    def prototype(self, edge_id: str, endpoint: str) -> np.ndarray:
        self.calls.append((edge_id, endpoint))
        values = {
            ("e1", "target"): [1.0, 0.0],
            ("e1", "confusion"): [0.0, 1.0],
            ("e2", "confusion"): [1.0, 0.0],
        }
        return np.asarray(values[(edge_id, endpoint)], dtype=np.float32)


def test_visual_loss_keeps_positive_edge_and_uses_deranged_negative() -> None:
    store = _LookupStore()
    features = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
        requires_grad=True,
    )
    logits = torch.tensor(
        [[[0.0, 3.0, -2.0], [3.0, 0.0, -2.0], [0.0, 3.0, -2.0]]]
    )
    result = visual_margin_loss(
        features,
        logits.log_softmax(-1),
        torch.tensor([[1]]),
        [3],
        ["vicocktail:train:s1"],
        store,
        margin=0.15,
    )
    assert store.calls == [("e1", "target"), ("e2", "confusion")]
    assert result.loss.item() == pytest.approx(0.15)
