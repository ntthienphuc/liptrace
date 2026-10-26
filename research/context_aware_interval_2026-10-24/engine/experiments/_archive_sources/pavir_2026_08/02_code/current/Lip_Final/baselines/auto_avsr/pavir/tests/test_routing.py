from types import SimpleNamespace

import numpy as np
import torch

from pavir.artifacts import FrozenEdge, TrainingEvent
from pavir.routing import (
    build_context_replacements,
    context_batch_selected,
    context_margin_loss,
    visual_margin_loss,
)


class FakeStore:
    def __init__(self):
        self.edges = {
            "e1": FrozenEdge(
                "e1", "a", "b", (1,), (2,), "E", False, 25, 12
            ),
            "d1": FrozenEdge(
                "d1", "a", "c", (1,), (3,), "D", True, 25, 12
            ),
        }
        self._events = {
            "vicocktail:train:s1": (
                TrainingEvent("e1", 0, 1),
                TrainingEvent("d1", 0, 1),
            )
        }

    def events_for(self, sample_id, route=None):
        values = self._events.get(sample_id, ())
        if route is None:
            return values
        return tuple(item for item in values if self.edges[item.edge_id].route == route)

    def prototype(self, edge_id, endpoint):
        assert edge_id == "e1"
        return np.array(
            [1.0, 0.0] if endpoint == "target" else [0.0, 1.0],
            dtype=np.float32,
        )


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.dropout = torch.nn.Dropout(0.5)
        self.output = torch.nn.Linear(4, 8)

    def forward(self, tokens, token_mask, memory, memory_mask):
        del token_mask, memory_mask
        value = self.embedding(tokens) + memory.mean(dim=1, keepdim=True)
        return self.output(self.dropout(value)), None


def test_visual_route_updates_features_not_alignment():
    store = FakeStore()
    features = torch.tensor(
        [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], requires_grad=True
    )
    logits = torch.tensor(
        [[[0.0, 3.0, -2.0], [3.0, 0.0, -2.0], [0.0, 3.0, -2.0]]],
        requires_grad=True,
    )
    log_probs = logits.log_softmax(-1)
    result = visual_margin_loss(
        features,
        log_probs,
        torch.tensor([[1]]),
        [3],
        ["vicocktail:train:s1"],
        store,
        margin=0.15,
    )
    assert result.active == 1
    assert result.loss.item() > 0
    result.loss.backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert logits.grad is None


def test_visual_active_counts_only_positive_hinges():
    store = FakeStore()
    features = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]], requires_grad=True
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
    assert result.eligible == 1
    assert result.active == 0
    assert result.loss.item() == 0.0


def test_context_replacement_is_one_per_sentence_and_deterministic():
    store = FakeStore()
    first = build_context_replacements(
        torch.tensor([[1, 4]]),
        ["vicocktail:train:s1"],
        store,
        seed=42,
        update=7,
    )
    second = build_context_replacements(
        torch.tensor([[1, 4]]),
        ["vicocktail:train:s1"],
        store,
        seed=42,
        update=7,
    )
    assert first is not None and second is not None
    torch.testing.assert_close(first.negative_targets, torch.tensor([[3, 4]]))
    torch.testing.assert_close(first.negative_targets, second.negative_targets)
    assert first.edge_ids == ("d1",)


def test_context_route_isolates_encoder_and_preserves_rng():
    store = FakeStore()
    config = SimpleNamespace(
        seed=42,
        context_batch_fraction=0.5,
        context_margin=10.0,
    )
    sample_ids = ["vicocktail:train:s1"]
    update = next(
        value
        for value in range(1000)
        if context_batch_selected(sample_ids, seed=42, update=value, fraction=0.5)
    )
    decoder = TinyDecoder()
    memory = torch.randn(1, 3, 4, requires_grad=True)
    targets = torch.tensor([[1, 4]])
    cpu_state = torch.random.get_rng_state().clone()
    result = context_margin_loss(
        decoder,
        memory,
        None,
        targets,
        sample_ids,
        store,
        config,
        update=update,
        sos=7,
        eos=7,
    )
    assert torch.equal(cpu_state, torch.random.get_rng_state())
    assert result.active == 1
    assert result.loss.item() > 0
    result.loss.backward()
    assert memory.grad is None
    assert sum(
        parameter.grad.abs().sum().item()
        for parameter in decoder.parameters()
        if parameter.grad is not None
    ) > 0


def test_context_batch_gate_never_exceeds_locked_half():
    sample_ids = ["vicocktail:train:s1"]
    cumulative = 0
    for update in range(10000):
        cumulative += context_batch_selected(
            sample_ids, seed=42, update=update, fraction=0.5
        )
        assert cumulative <= (update + 1) * 0.5
    assert cumulative == 5000
