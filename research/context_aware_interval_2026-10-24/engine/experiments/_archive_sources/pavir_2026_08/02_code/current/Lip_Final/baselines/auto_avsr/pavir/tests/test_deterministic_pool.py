import os

import pytest
import torch
import torch.nn.functional as F

from pavir.deterministic_pool import DeterministicMaxPool3dCPUBackward


def _single_cpu_thread() -> None:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _native_pool(inputs):
    return F.max_pool3d(
        inputs,
        kernel_size=(1, 3, 3),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_deterministic_pool_matches_native_cpu_forward_and_vjp(dtype):
    _single_cpu_thread()
    # Repeated values force ties, and stride < kernel creates overlapping
    # windows; the native forward indices must still define the exact VJP.
    values = torch.tensor(
        [
            [
                [
                    [[1.0, 1.0, 0.0, 2.0], [1.0, 1.0, 0.0, 2.0], [0.0, 0.0, 3.0, 3.0], [0.0, 0.0, 3.0, 3.0]],
                    [[2.0, 2.0, 0.0, 1.0], [2.0, 2.0, 0.0, 1.0], [0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0]],
                ]
            ]
        ],
        dtype=dtype,
    )
    native_inputs = values.clone().requires_grad_(True)
    recovered_inputs = values.clone().requires_grad_(True)
    recovered = DeterministicMaxPool3dCPUBackward(
        kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
    )
    native_outputs = _native_pool(native_inputs)
    recovered_outputs = recovered(recovered_inputs)
    weights = torch.arange(
        native_outputs.numel(), dtype=dtype
    ).reshape_as(native_outputs)
    native_gradient = torch.autograd.grad((native_outputs * weights).sum(), native_inputs)[0]
    recovered_gradient = torch.autograd.grad(
        (recovered_outputs * weights).sum(), recovered_inputs
    )[0]
    assert torch.equal(recovered_outputs, native_outputs)
    assert torch.equal(recovered_gradient, native_gradient)
    assert recovered.tracker["forward_calls"] == 1
    assert recovered.tracker["backward_calls"] == 1
    assert recovered.tracker["cpu_native_backward_calls"] == 1


def test_deterministic_pool_reuses_indices_across_retained_vjps():
    _single_cpu_thread()
    inputs = torch.randn(1, 2, 2, 5, 5, requires_grad=True)
    recovered = DeterministicMaxPool3dCPUBackward(
        kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
    )
    outputs = recovered(inputs)
    for ordinal in range(4):
        torch.autograd.grad(
            (ordinal + 1) * outputs.sum(),
            inputs,
            retain_graph=ordinal < 3,
        )
    assert recovered.tracker["forward_calls"] == 1
    assert recovered.tracker["backward_calls"] == 4
    assert recovered.tracker["cpu_index_copies"] == 1
    assert recovered.tracker["cpu_index_reuses"] == 3


def test_deterministic_pool_can_hash_saved_native_indices(monkeypatch):
    _single_cpu_thread()
    monkeypatch.setenv("PAVIR_POOL_RECORD_INDEX_SHA256", "1")
    recovered = DeterministicMaxPool3dCPUBackward(
        kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
    )
    recovered(torch.randn(1, 1, 1, 4, 4, requires_grad=True))
    digest = recovered.tracker["cpu_index_sha256"]
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deterministic_pool_uses_cpu_vjp_under_strict_cuda_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _single_cpu_thread()
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    inputs = torch.tensor(
        [[[[[1.0, 1.0, 0.0, 2.0], [1.0, 1.0, 0.0, 2.0], [0.0, 0.0, 3.0, 3.0], [0.0, 0.0, 3.0, 3.0]]]]],
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    recovered = DeterministicMaxPool3dCPUBackward(
        kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
    ).to(device)
    outputs = recovered(inputs)
    gradient = torch.autograd.grad(outputs.float().square().sum(), inputs)[0]
    native_forward = _native_pool(inputs.detach())
    cpu_inputs = inputs.detach().cpu().requires_grad_(True)
    cpu_outputs = _native_pool(cpu_inputs)
    cpu_gradient = torch.autograd.grad(cpu_outputs.float().square().sum(), cpu_inputs)[0]
    assert torch.equal(outputs.detach().cpu(), native_forward.cpu())
    assert torch.equal(gradient.detach().cpu(), cpu_gradient.detach())
    assert recovered.tracker["forward_calls"] == 1
    assert recovered.tracker["backward_calls"] == 1
    assert recovered.tracker["cpu_native_backward_calls"] == 1
