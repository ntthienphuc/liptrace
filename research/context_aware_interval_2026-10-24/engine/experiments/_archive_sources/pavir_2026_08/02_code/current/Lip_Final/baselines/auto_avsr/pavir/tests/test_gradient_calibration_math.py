from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "calibrate_pavir_gradients",
    ROOT / "tools" / "calibrate_pavir_gradients.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gradient_norm_is_side_effect_free_and_exact():
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    loss = (parameter.square()).sum()
    norm = MODULE.gradient_norm(loss, [parameter], retain_graph=False)
    assert abs(norm - 10.0) < 1e-10
    assert parameter.grad is None


def test_paired_gradient_statistics_measure_conflict_without_side_effects():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    native = parameter.sum()
    auxiliary = -parameter.sum()
    result = MODULE.paired_gradient_statistics(
        native,
        auxiliary,
        [parameter],
        retain_graph=False,
    )
    assert result["native_norm"] == pytest.approx(2.0**0.5)
    assert result["auxiliary_norm"] == pytest.approx(2.0**0.5)
    assert result["cosine"] == pytest.approx(-1.0)
    assert parameter.grad is None
