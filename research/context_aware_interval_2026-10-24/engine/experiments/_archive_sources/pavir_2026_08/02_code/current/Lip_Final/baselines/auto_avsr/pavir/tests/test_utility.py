import os

import pytest
import torch

from pavir.artifacts import FrozenEdge, TrainingEvent
from pavir.utility import (
    ctc_confusion_utility_loss,
    ctc_confusion_utility_loss_cpu_vjp,
    ctc_native_loss_cpu_vjp,
    visual_pair_diagnostics,
)


class FakeStore:
    def __init__(self):
        self.edges = {
            "e1": FrozenEdge(
                "e1", "a", "b", (1,), (2,), "E", False, 25, 12
            )
        }
        self._events = {"vicocktail:train:s1": (TrainingEvent("e1", 0, 1),)}

    def events_for(self, sample_id, route=None):
        values = self._events.get(sample_id, ())
        if route is None:
            return values
        return tuple(item for item in values if self.edges[item.edge_id].route == route)

    def prototype(self, edge_id, endpoint):
        assert edge_id == "e1"
        return torch.tensor(
            [1.0, 0.0] if endpoint == "target" else [0.0, 1.0]
        ).numpy()


def test_ctc_utility_gradient_favors_reference_over_confusion():
    store = FakeStore()
    logits = torch.tensor(
        [[[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
        requires_grad=True,
    )
    result = ctc_confusion_utility_loss(
        logits,
        torch.tensor([[1]]),
        [3],
        ["vicocktail:train:s1"],
        store,
    )
    assert result.eligible == result.feasible == 1
    assert result.infeasible == 0
    assert result.loss.item() > 0.0
    assert result.mean_margin.item() > 0.0
    assert result.hard == 0
    result.loss.backward()
    # Gradient descent subtracts this gradient: it raises the correct token's
    # score and lowers the frozen confusion token's score.
    assert logits.grad is not None
    assert logits.grad[:, :, 1].sum().item() < 0.0
    assert logits.grad[:, :, 2].sum().item() > 0.0


def test_ctc_utility_ignores_samples_without_frozen_events():
    store = FakeStore()
    logits = torch.zeros((1, 3, 3), requires_grad=True)
    result = ctc_confusion_utility_loss(
        logits,
        torch.tensor([[1]]),
        [3],
        ["vicocktail:train:absent"],
        store,
    )
    assert result.eligible == result.feasible == result.infeasible == 0
    assert result.loss.item() == 0.0


def test_cpu_ctc_vjp_matches_confusion_objective_value_and_logit_gradient():
    store = FakeStore()
    values = torch.tensor(
        [[[0.2, 1.3, -0.4], [1.1, -0.2, 0.4], [0.1, 0.9, -0.3]]]
    )
    direct_logits = values.clone().requires_grad_(True)
    direct = ctc_confusion_utility_loss(
        direct_logits,
        torch.tensor([[1]]),
        [3],
        ["vicocktail:train:s1"],
        store,
    )
    direct_gradient = torch.autograd.grad(direct.loss, direct_logits)[0]
    vjp_logits = values.clone().requires_grad_(True)
    vjp = ctc_confusion_utility_loss_cpu_vjp(
        vjp_logits,
        torch.tensor([[1]]),
        [3],
        ["vicocktail:train:s1"],
        store,
    )
    vjp_gradient = torch.autograd.grad(vjp.loss, vjp_logits)[0]
    assert vjp.loss.item() == pytest.approx(direct.loss.item(), abs=1e-6)
    assert vjp.mean_margin.item() == pytest.approx(direct.mean_margin.item(), abs=1e-6)
    assert torch.allclose(vjp_gradient, direct_gradient, atol=1e-6, rtol=1e-5)


def test_cpu_ctc_vjp_matches_native_ctc_value_and_logit_gradient():
    values = torch.tensor(
        [
            [[0.2, 1.3, -0.4], [1.1, -0.2, 0.4], [0.1, 0.9, -0.3]],
            [[0.4, -0.3, 0.7], [0.6, 0.1, -0.1], [0.2, 1.0, -0.5]],
        ]
    )
    targets = torch.tensor([[1, -1], [2, -1]])
    lengths = torch.tensor([3, 3])
    direct_logits = values.clone().requires_grad_(True)
    direct_loss = torch.nn.functional.ctc_loss(
        direct_logits.log_softmax(-1).transpose(0, 1),
        torch.tensor([1, 2]),
        lengths,
        torch.tensor([1, 1]),
        blank=0,
        reduction="sum",
        zero_infinity=True,
    ) / 2
    direct_gradient = torch.autograd.grad(direct_loss, direct_logits)[0]
    vjp_logits = values.clone().requires_grad_(True)
    vjp = ctc_native_loss_cpu_vjp(vjp_logits, targets, lengths)
    vjp_gradient = torch.autograd.grad(vjp.loss, vjp_logits)[0]
    assert vjp.loss.item() == pytest.approx(direct_loss.item(), abs=1e-6)
    assert torch.allclose(vjp_gradient, direct_gradient, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_ctc_vjp_uses_only_cpu_ctc_with_strict_gpu_vjp(monkeypatch):
    """A CUDA VJP must never dispatch a CUDA CTC backward kernel."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    features = torch.tensor(
        [
            [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.6], [0.1, 0.5]],
            [[-0.3, 0.2], [0.5, -0.4], [0.7, 0.1], [-0.2, -0.5]],
        ],
        device=device,
    )
    weight = torch.nn.Parameter(
        torch.tensor(
            [[0.1, 0.4, -0.2, 0.3, -0.5], [0.2, -0.3, 0.6, -0.1, 0.5]],
            device=device,
        )
    )
    logits = torch.einsum("btf,fv->btv", features, weight)
    ctc_devices = []
    real_ctc_loss = torch.nn.functional.ctc_loss

    def cpu_ctc_only(*args, **kwargs):
        ctc_devices.append(args[0].device.type)
        if args[0].device.type != "cpu":
            raise AssertionError("synthetic G1R test attempted CUDA CTC")
        return real_ctc_loss(*args, **kwargs)

    monkeypatch.setattr("pavir.utility.F.ctc_loss", cpu_ctc_only)
    observed_logit_vjps = []
    logits.register_hook(lambda gradient: observed_logit_vjps.append(gradient.detach()))
    result = ctc_native_loss_cpu_vjp(
        logits,
        torch.tensor([[1, 2], [3, -1]], device=device),
        torch.tensor([4, 4], device=device),
    )
    gradient = torch.autograd.grad(result.loss, weight)[0]
    assert ctc_devices and set(ctc_devices) == {"cpu"}
    assert len(observed_logit_vjps) == 1
    manual = torch.einsum("btf,btv->fv", features, observed_logit_vjps[0])
    assert torch.allclose(gradient, manual, atol=3e-6, rtol=3e-6)
    assert result.cpu_logit_gradient_sha256 is not None
    assert result.cpu_logit_gradient_l2_norm > 0.0


def test_ctc_utility_rejects_tokenizer_mismatch():
    store = FakeStore()
    logits = torch.zeros((1, 3, 3), requires_grad=True)
    try:
        ctc_confusion_utility_loss(
            logits,
            torch.tensor([[2]]),
            [3],
            ["vicocktail:train:s1"],
            store,
        )
    except ValueError as error:
        assert "artifact/tokenizer mismatch" in str(error)
    else:
        raise AssertionError("tokenizer mismatch did not fail closed")


def test_visual_pair_diagnostic_rejects_unshuffled_control():
    candidate = FakeStore()
    control = FakeStore()
    features = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]])
    logits = torch.tensor(
        [[[0.0, 3.0, -2.0], [3.0, 0.0, -2.0], [0.0, 3.0, -2.0]]]
    )
    try:
        visual_pair_diagnostics(
            features,
            logits.log_softmax(-1),
            torch.tensor([[1]]),
            [3],
            ["vicocktail:train:s1"],
            candidate,
            control,
            margin=0.15,
        )
    except ValueError as error:
        assert "zero-identity derangement" in str(error)
    else:
        raise AssertionError("unshuffled control did not fail closed")
