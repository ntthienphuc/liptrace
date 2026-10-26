"""Deterministic CPU VJP for the exact native MaxPool3d forward.

PyTorch's CUDA MaxPool3d backward has no deterministic implementation.  This
module deliberately retains the native CUDA forward (including its selected
tie indices), then invokes PyTorch's native CPU MaxPool3d backward on a single
CPU thread with the saved indices.  The resulting input derivative is copied
back to the original CUDA graph.  It is useful for strict zero-step diagnostics
only; it does not alter model parameters or the forward activations.
"""

from __future__ import annotations

from collections.abc import MutableMapping
import hashlib
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch import Tensor, nn


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value),) * 3
    if len(value) != 3:
        raise ValueError("MaxPool3d argument must be an int or length-three tuple")
    return tuple(int(item) for item in value)


class _NativeForwardCPUBackwardMaxPool3d(torch.autograd.Function):
    @staticmethod
    def forward(
        context,
        inputs: Tensor,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        dilation: tuple[int, int, int],
        ceil_mode: bool,
        tracker: MutableMapping[str, Any],
    ) -> Tensor:
        outputs, indices = F.max_pool3d(
            inputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )
        # Retain the exact native CUDA tie/argmax choices once on CPU.  G1R2
        # asks four VJPs through one frozen forward, so copying here both
        # removes the GPU-only backward dependency and avoids repeating a
        # potentially GiB-scale index transfer for each VJP.
        cpu_indices = (
            indices.detach()
            .to(device="cpu", dtype=torch.long, non_blocking=False)
            .contiguous()
        )
        context.save_for_backward(cpu_indices)
        context.input_shape = tuple(int(value) for value in inputs.shape)
        context.input_dtype = inputs.dtype
        context.kernel_size = kernel_size
        context.stride = stride
        context.padding = padding
        context.dilation = dilation
        context.ceil_mode = ceil_mode
        context.tracker = tracker
        tracker["forward_calls"] = int(tracker.get("forward_calls", 0)) + 1
        tracker["last_input_shape"] = list(context.input_shape)
        tracker["last_output_shape"] = [int(value) for value in outputs.shape]
        tracker["cpu_index_copies"] = int(tracker.get("cpu_index_copies", 0)) + 1
        tracker["cpu_index_copied_elements"] = int(
            tracker.get("cpu_index_copied_elements", 0)
        ) + int(cpu_indices.numel())
        if os.environ.get("PAVIR_POOL_RECORD_INDEX_SHA256", "0") == "1":
            tracker["cpu_index_sha256"] = hashlib.sha256(
                cpu_indices.numpy().tobytes()
            ).hexdigest()
        return outputs

    @staticmethod
    @once_differentiable
    def backward(context, upstream: Tensor):
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError(
                "deterministic MaxPool3d CPU VJP requires one PyTorch CPU/inter-op thread"
            )
        (cpu_indices,) = context.saved_tensors
        if upstream.dtype != context.input_dtype:
            raise RuntimeError(
                "deterministic MaxPool3d CPU VJP received a gradient with a different dtype"
            )
        # The saved CPU indices are invariant across retained VJPs; upstream
        # derivatives themselves are intentionally never cached.
        if int(context.tracker.get("backward_calls", 0)) > 0:
            context.tracker["cpu_index_reuses"] = int(
                context.tracker.get("cpu_index_reuses", 0)
            ) + 1
        cpu_upstream = (
            upstream.detach()
            .to(device="cpu", non_blocking=False)
            .contiguous()
        )
        # The native backward depends on the input's shape/dtype and on the
        # exact forward-selected indices, not on input values.  An uninitialised
        # CPU tensor avoids an unnecessary GPU->CPU copy of frontend activations.
        cpu_shape_reference = torch.empty(
            context.input_shape, dtype=context.input_dtype, device="cpu"
        )
        cpu_input_gradient = torch.ops.aten.max_pool3d_with_indices_backward.default(
            cpu_upstream,
            cpu_shape_reference,
            context.kernel_size,
            context.stride,
            context.padding,
            context.dilation,
            context.ceil_mode,
            cpu_indices,
        )
        tracker = context.tracker
        tracker["backward_calls"] = int(tracker.get("backward_calls", 0)) + 1
        tracker["cpu_native_backward_calls"] = int(
            tracker.get("cpu_native_backward_calls", 0)
        ) + 1
        tracker["cpu_index_vjp_elements"] = int(
            tracker.get("cpu_index_vjp_elements", 0)
        ) + int(cpu_indices.numel())
        tracker["cpu_input_gradient_elements"] = int(
            tracker.get("cpu_input_gradient_elements", 0)
        ) + int(cpu_input_gradient.numel())
        gradient = cpu_input_gradient.reshape(context.input_shape).to(
            device=upstream.device,
            dtype=upstream.dtype,
            non_blocking=False,
        )
        # This GPU scan is equivalent to a CPU scan after the copy but is tiny
        # relative to the single-thread CPU native backward on real batches.
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("deterministic MaxPool3d CPU VJP is non-finite")
        return gradient, None, None, None, None, None, None


class DeterministicMaxPool3dCPUBackward(nn.Module):
    """Native MaxPool3d forward with CPU native-backward input VJP.

    ``tracker`` is plain Python state, not a model buffer or parameter.  A
    guarded diagnostic runner can inspect it after each autograd bundle to
    prove this backend, rather than CUDA MaxPool3d backward, was used.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] | None = None,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(kernel_size if stride is None else stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.ceil_mode = bool(ceil_mode)
        self.tracker: dict[str, Any] = {}

    @classmethod
    def from_native(cls, native: nn.MaxPool3d) -> "DeterministicMaxPool3dCPUBackward":
        if not isinstance(native, nn.MaxPool3d):
            raise TypeError("expected the native nn.MaxPool3d module")
        return cls(
            native.kernel_size,
            native.stride,
            native.padding,
            native.dilation,
            native.ceil_mode,
        )

    def reset_tracker(self) -> None:
        self.tracker.clear()

    def forward(self, inputs: Tensor) -> Tensor:
        return _NativeForwardCPUBackwardMaxPool3d.apply(
            inputs,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.tracker,
        )
