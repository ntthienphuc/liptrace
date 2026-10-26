"""Zero-step, confusion-specific utility diagnostics for PAViR.

This module is deliberately outside the training forward path.  It supplies a
prospective diagnostic loss whose gradient answers whether an auxiliary update
would favor the frozen reference sequence over the exact frozen confusion
sequence under CTC.  No optimizer step consumes this loss.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .artifacts import ArtifactStore, TrainingEvent
from .ctc_posterior import ctc_forward_backward, pool_token_span


@dataclass(frozen=True)
class CTCUtilityResult:
    """Confusion-specific CTC loss-ratio diagnostic and accounting."""

    loss: Tensor
    eligible: int
    feasible: int
    infeasible: int
    hard: int
    mean_reference_nll: Tensor
    mean_confusion_nll: Tensor
    mean_margin: Tensor
    margins: Tensor
    cpu_logit_gradient_sha256: str | None = None
    cpu_logit_gradient_l2_norm: float | None = None


@dataclass(frozen=True)
class CTCNativeVJPResult:
    """Deterministic-reference CTC value with a GPU-logit VJP surrogate."""

    loss: Tensor
    value: Tensor
    cpu_logit_gradient_sha256: str
    cpu_logit_gradient_l2_norm: float


@dataclass(frozen=True)
class VisualPairDiagnostic:
    """One exact event's candidate/control geometry at frozen W0."""

    sample_id: str
    source_id: str
    edge_id: str
    target_text: str
    control_negative_edge_id: str
    candidate_pre_hinge: float
    control_pre_hinge: float
    candidate_hinge: float
    control_hinge: float


def _zero_from(tensor: Tensor) -> Tensor:
    return tensor.reshape(-1)[:1].sum() * 0.0


def _target_row(targets: Tensor, index: int, ignore_id: int) -> Tensor:
    return targets[index][targets[index].ne(ignore_id)]


def _minimum_ctc_frames(tokens: Tensor) -> int:
    """Minimum frames needed by CTC, including repeated-label separators."""

    if not tokens.numel():
        return 0
    repeats = int(tokens[1:].eq(tokens[:-1]).sum().item())
    return int(tokens.numel()) + repeats


def _verify_original_target(
    target: Tensor,
    event: TrainingEvent,
    expected: Sequence[int],
    sample_id: str,
) -> None:
    if event.target_end > target.numel():
        raise ValueError(
            f"frozen utility span exceeds target for {sample_id}: "
            f"[{event.target_start},{event.target_end}) > {target.numel()}"
        )
    observed = tuple(
        int(value)
        for value in target[event.target_start : event.target_end]
        .detach()
        .cpu()
        .tolist()
    )
    if observed != tuple(expected):
        raise ValueError(
            f"utility artifact/tokenizer mismatch for {sample_id}/{event.edge_id}: "
            f"expected={tuple(expected)}, observed={observed}"
        )


def _packed_ctc_nll(
    log_probs_tnv: Tensor,
    rows: Sequence[Tensor],
    input_lengths: Tensor,
    *,
    blank: int,
) -> Tensor:
    lengths = torch.tensor(
        [int(row.numel()) for row in rows],
        dtype=torch.long,
        device=log_probs_tnv.device,
    )
    packed = torch.cat(tuple(rows)).long()
    losses = F.ctc_loss(
        log_probs_tnv,
        packed,
        input_lengths.long(),
        lengths,
        blank=int(blank),
        reduction="none",
        zero_infinity=False,
    )
    if not torch.isfinite(losses).all():
        raise FloatingPointError("confusion-specific CTC utility produced non-finite NLL")
    return losses


def _cpu_ctc_nll_graph(
    gpu_logits_btv: Tensor,
    rows: Sequence[Tensor],
    input_lengths: Tensor,
    *,
    blank: int,
    zero_infinity: bool,
) -> tuple[Tensor, Tensor]:
    """Build a deterministic CPU CTC graph from detached GPU logits.

    CUDA currently lacks a deterministic CTC backward implementation.  The
    caller takes the exact CPU derivative with respect to this leaf and uses a
    value-preserving GPU surrogate to propagate that derivative through the
    original logits.  This function deliberately performs no model backward.
    """

    if gpu_logits_btv.ndim != 3 or not rows:
        raise ValueError("CPU CTC VJP requires rank-three logits and nonempty rows")
    cpu_logits = gpu_logits_btv.detach().float().to("cpu").contiguous()
    cpu_logits.requires_grad_(True)
    cpu_rows = tuple(
        row.detach().to(device="cpu", dtype=torch.long).contiguous()
        for row in rows
    )
    if any(not row.numel() for row in cpu_rows):
        raise ValueError("CPU CTC VJP received an empty target row")
    cpu_lengths = input_lengths.detach().to(device="cpu", dtype=torch.long)
    target_lengths = torch.tensor(
        [int(row.numel()) for row in cpu_rows], dtype=torch.long, device="cpu"
    )
    packed = torch.cat(cpu_rows)
    nll = F.ctc_loss(
        cpu_logits.log_softmax(dim=-1).transpose(0, 1),
        packed,
        cpu_lengths,
        target_lengths,
        blank=int(blank),
        reduction="none",
        zero_infinity=bool(zero_infinity),
    )
    if not torch.isfinite(nll).all():
        raise FloatingPointError("deterministic CPU CTC VJP produced non-finite NLL")
    return cpu_logits, nll


def _value_preserving_gpu_vjp_surrogate(
    gpu_logits_btv: Tensor,
    cpu_logits_btv: Tensor,
    cpu_objective: Tensor,
) -> tuple[Tensor, Tensor, str, float]:
    """Return the CPU objective value with its exact CPU logit derivative.

    If ``q = d objective / d logits`` is calculated on CPU, then
    ``<gpu_logits, q> + stopgrad(objective - <gpu_logits, q>)`` has the same
    scalar value as the reference objective and the required derivative with
    respect to GPU logits.  Its subsequent VJP through the model uses only
    standard deterministic GPU operators, not CUDA CTC backward.
    """

    if cpu_objective.ndim != 0 or not torch.isfinite(cpu_objective):
        raise FloatingPointError("CPU CTC VJP objective is non-finite or nonscalar")
    cpu_gradient = torch.autograd.grad(cpu_objective, cpu_logits_btv)[0]
    if not torch.isfinite(cpu_gradient).all():
        raise FloatingPointError("CPU CTC VJP gradient is non-finite")
    canonical_cpu_gradient = cpu_gradient.detach().float().contiguous()
    gradient_sha256 = hashlib.sha256(
        canonical_cpu_gradient.numpy().tobytes()
    ).hexdigest()
    gradient_l2_norm = float(
        torch.linalg.vector_norm(canonical_cpu_gradient.double()).item()
    )
    logit_gradient = cpu_gradient.to(device=gpu_logits_btv.device, dtype=torch.float32)
    tangent = (gpu_logits_btv.float() * logit_gradient).sum()
    value = cpu_objective.detach().to(device=gpu_logits_btv.device, dtype=tangent.dtype)
    # Put the value first: ``tangent - stopgrad(tangent)`` is exactly zero in
    # the forward pass, avoiding a large cancellation around the CPU scalar.
    return (
        value + (tangent - tangent.detach()),
        value,
        gradient_sha256,
        gradient_l2_norm,
    )


def ctc_confusion_utility_loss(
    ctc_logits: Tensor,
    targets: Tensor,
    input_lengths: Sequence[int] | Tensor,
    sample_ids: Sequence[str],
    store: ArtifactStore,
    *,
    route: str = "E",
    blank: int = 0,
    ignore_id: int = -1,
) -> CTCUtilityResult:
    """Return a softplus CTC likelihood-ratio oracle over frozen events.

    A training-loss gradient that has positive cosine with this diagnostic
    gradient shares its local descent direction: both favor the reference
    sequence over the exact W0 confusion.  The function never substitutes a
    deranged control endpoint here; controls are always judged against the
    same true-confusion utility target.
    """

    if ctc_logits.ndim != 3:
        raise ValueError("ctc_logits must be batch-first [B,T,V]")
    if targets.ndim != 2 or targets.size(0) != ctc_logits.size(0):
        raise ValueError("targets must be [B,U]")
    if len(sample_ids) != ctc_logits.size(0):
        raise ValueError("sample_ids does not match batch size")
    lengths = torch.as_tensor(
        input_lengths, dtype=torch.long, device=ctc_logits.device
    )
    if lengths.ndim != 1 or lengths.numel() != ctc_logits.size(0):
        raise ValueError("input_lengths does not match batch size")
    if torch.any(lengths <= 0) or torch.any(lengths > ctc_logits.size(1)):
        raise ValueError("input_lengths are outside the CTC logit time axis")

    event_sample_indices: list[int] = []
    reference_rows: list[Tensor] = []
    confusion_rows: list[Tensor] = []
    eligible = 0
    infeasible = 0
    for sample_index, sample_id in enumerate(sample_ids):
        target = _target_row(targets, sample_index, ignore_id)
        for event in store.events_for(str(sample_id), route):
            eligible += 1
            edge = store.edges[event.edge_id]
            _verify_original_target(
                target, event, edge.target_token_ids, str(sample_id)
            )
            confusion = torch.cat(
                (
                    target[: event.target_start],
                    target.new_tensor(edge.confusion_token_ids),
                    target[event.target_end :],
                )
            )
            if not confusion.numel() or torch.equal(target, confusion):
                raise ValueError(
                    f"invalid exact-confusion utility event: {sample_id}/{edge.edge_id}"
                )
            available = int(lengths[sample_index].item())
            if (
                _minimum_ctc_frames(target) > available
                or _minimum_ctc_frames(confusion) > available
            ):
                infeasible += 1
                continue
            event_sample_indices.append(sample_index)
            reference_rows.append(target)
            confusion_rows.append(confusion)

    if not event_sample_indices:
        zero = _zero_from(ctc_logits)
        return CTCUtilityResult(
            loss=zero,
            eligible=eligible,
            feasible=0,
            infeasible=infeasible,
            hard=0,
            mean_reference_nll=zero,
            mean_confusion_nll=zero,
            mean_margin=zero,
            margins=ctc_logits.new_empty((0,), dtype=torch.float32),
        )

    indices = torch.tensor(
        event_sample_indices, dtype=torch.long, device=ctc_logits.device
    )
    selected_lengths = lengths.index_select(0, indices)
    log_probs = (
        ctc_logits.index_select(0, indices).float().log_softmax(dim=-1).transpose(0, 1)
    )
    reference_nll = _packed_ctc_nll(
        log_probs,
        reference_rows,
        selected_lengths,
        blank=blank,
    )
    confusion_nll = _packed_ctc_nll(
        log_probs,
        confusion_rows,
        selected_lengths,
        blank=blank,
    )
    reference_lengths = torch.tensor(
        [int(row.numel()) for row in reference_rows],
        dtype=reference_nll.dtype,
        device=reference_nll.device,
    )
    confusion_lengths = torch.tensor(
        [int(row.numel()) for row in confusion_rows],
        dtype=confusion_nll.dtype,
        device=confusion_nll.device,
    )
    denominators = torch.maximum(reference_lengths, confusion_lengths)
    # r = (log P(reference) - log P(confusion)) / sequence length.
    # Since CTCLoss is -log P, this is (NLL_confusion - NLL_reference) / L.
    margins = (confusion_nll - reference_nll) / denominators
    loss = F.softplus(-margins).mean()
    mean_reference = (reference_nll / reference_lengths).mean()
    mean_confusion = (confusion_nll / confusion_lengths).mean()
    return CTCUtilityResult(
        loss=loss,
        eligible=eligible,
        feasible=len(event_sample_indices),
        infeasible=infeasible,
        hard=int(torch.count_nonzero(margins.detach().le(0)).item()),
        mean_reference_nll=mean_reference,
        mean_confusion_nll=mean_confusion,
        mean_margin=margins.mean(),
        margins=margins,
    )


def ctc_confusion_utility_loss_cpu_vjp(
    ctc_logits: Tensor,
    targets: Tensor,
    input_lengths: Sequence[int] | Tensor,
    sample_ids: Sequence[str],
    store: ArtifactStore,
    *,
    route: str = "E",
    blank: int = 0,
    ignore_id: int = -1,
) -> CTCUtilityResult:
    """CTC confusion oracle with deterministic CPU logit derivatives.

    This is mathematically the same full-sequence likelihood-ratio objective
    as :func:`ctc_confusion_utility_loss`.  Only its CTC derivative backend is
    changed: CPU computes ``dL/dlogits`` deterministically, then a
    value-preserving VJP surrogate carries that derivative through the
    original GPU model graph.  It is intended for zero-step diagnostics when
    CUDA CTC backward is unavailable under a fail-closed determinism policy.
    """

    if ctc_logits.ndim != 3:
        raise ValueError("ctc_logits must be batch-first [B,T,V]")
    if targets.ndim != 2 or targets.size(0) != ctc_logits.size(0):
        raise ValueError("targets must be [B,U]")
    if len(sample_ids) != ctc_logits.size(0):
        raise ValueError("sample_ids does not match batch size")
    lengths = torch.as_tensor(
        input_lengths, dtype=torch.long, device=ctc_logits.device
    )
    if lengths.ndim != 1 or lengths.numel() != ctc_logits.size(0):
        raise ValueError("input_lengths does not match batch size")
    if torch.any(lengths <= 0) or torch.any(lengths > ctc_logits.size(1)):
        raise ValueError("input_lengths are outside the CTC logit time axis")

    event_sample_indices: list[int] = []
    reference_rows: list[Tensor] = []
    confusion_rows: list[Tensor] = []
    eligible = 0
    infeasible = 0
    for sample_index, sample_id in enumerate(sample_ids):
        target = _target_row(targets, sample_index, ignore_id)
        for event in store.events_for(str(sample_id), route):
            eligible += 1
            edge = store.edges[event.edge_id]
            _verify_original_target(
                target, event, edge.target_token_ids, str(sample_id)
            )
            confusion = torch.cat(
                (
                    target[: event.target_start],
                    target.new_tensor(edge.confusion_token_ids),
                    target[event.target_end :],
                )
            )
            if not confusion.numel() or torch.equal(target, confusion):
                raise ValueError(
                    f"invalid exact-confusion utility event: {sample_id}/{edge.edge_id}"
                )
            available = int(lengths[sample_index].item())
            if (
                _minimum_ctc_frames(target) > available
                or _minimum_ctc_frames(confusion) > available
            ):
                infeasible += 1
                continue
            event_sample_indices.append(sample_index)
            reference_rows.append(target)
            confusion_rows.append(confusion)

    if not event_sample_indices:
        zero = _zero_from(ctc_logits)
        return CTCUtilityResult(
            loss=zero,
            eligible=eligible,
            feasible=0,
            infeasible=infeasible,
            hard=0,
            mean_reference_nll=zero,
            mean_confusion_nll=zero,
            mean_margin=zero,
            margins=ctc_logits.new_empty((0,), dtype=torch.float32),
        )

    indices = torch.tensor(
        event_sample_indices, dtype=torch.long, device=ctc_logits.device
    )
    selected_logits = ctc_logits.index_select(0, indices)
    selected_lengths = lengths.index_select(0, indices)
    cpu_logits, reference_nll = _cpu_ctc_nll_graph(
        selected_logits,
        reference_rows,
        selected_lengths,
        blank=blank,
        zero_infinity=False,
    )
    # Reuse the exact same detached CPU logits to build both NLL collections.
    # CTC gradients of their difference therefore share one numerical backend.
    cpu_log_probs = cpu_logits.log_softmax(dim=-1).transpose(0, 1)
    cpu_confusion_rows = tuple(
        row.detach().to(device="cpu", dtype=torch.long).contiguous()
        for row in confusion_rows
    )
    cpu_confusion_lengths = torch.tensor(
        [int(row.numel()) for row in cpu_confusion_rows],
        dtype=torch.long,
        device="cpu",
    )
    confusion_nll = F.ctc_loss(
        cpu_log_probs,
        torch.cat(cpu_confusion_rows),
        selected_lengths.detach().to(device="cpu", dtype=torch.long),
        cpu_confusion_lengths,
        blank=int(blank),
        reduction="none",
        zero_infinity=False,
    )
    if not torch.isfinite(confusion_nll).all():
        raise FloatingPointError("deterministic CPU CTC VJP produced non-finite NLL")
    reference_lengths = torch.tensor(
        [int(row.numel()) for row in reference_rows],
        dtype=reference_nll.dtype,
        device="cpu",
    )
    confusion_lengths = torch.tensor(
        [int(row.numel()) for row in confusion_rows],
        dtype=confusion_nll.dtype,
        device="cpu",
    )
    denominators = torch.maximum(reference_lengths, confusion_lengths)
    margins = (confusion_nll - reference_nll) / denominators
    cpu_objective = F.softplus(-margins).mean()
    loss, value, gradient_sha256, gradient_l2_norm = _value_preserving_gpu_vjp_surrogate(
        selected_logits, cpu_logits, cpu_objective
    )
    mean_reference = (reference_nll / reference_lengths).mean().detach().to(
        device=ctc_logits.device, dtype=value.dtype
    )
    mean_confusion = (confusion_nll / confusion_lengths).mean().detach().to(
        device=ctc_logits.device, dtype=value.dtype
    )
    mean_margin = margins.mean().detach().to(
        device=ctc_logits.device, dtype=value.dtype
    )
    return CTCUtilityResult(
        loss=loss,
        eligible=eligible,
        feasible=len(event_sample_indices),
        infeasible=infeasible,
        hard=int(torch.count_nonzero(margins.detach().le(0)).item()),
        mean_reference_nll=mean_reference,
        mean_confusion_nll=mean_confusion,
        mean_margin=mean_margin,
        margins=margins.detach().to(device=ctc_logits.device, dtype=torch.float32),
        cpu_logit_gradient_sha256=gradient_sha256,
        cpu_logit_gradient_l2_norm=gradient_l2_norm,
    )


def ctc_native_loss_cpu_vjp(
    ctc_logits: Tensor,
    targets: Tensor,
    input_lengths: Sequence[int] | Tensor,
    *,
    blank: int = 0,
    ignore_id: int = -1,
) -> CTCNativeVJPResult:
    """Match Auto-AVSR's native CTC loss using deterministic CPU CTC VJP."""

    if ctc_logits.ndim != 3:
        raise ValueError("ctc_logits must be batch-first [B,T,V]")
    if targets.ndim != 2 or targets.size(0) != ctc_logits.size(0):
        raise ValueError("targets must be [B,U]")
    lengths = torch.as_tensor(
        input_lengths, dtype=torch.long, device=ctc_logits.device
    )
    if lengths.ndim != 1 or lengths.numel() != ctc_logits.size(0):
        raise ValueError("input_lengths does not match batch size")
    target_rows = tuple(
        _target_row(targets, index, ignore_id) for index in range(targets.size(0))
    )
    cpu_logits, nll = _cpu_ctc_nll_graph(
        ctc_logits,
        target_rows,
        lengths,
        blank=blank,
        zero_infinity=True,
    )
    cpu_objective = nll.sum() / ctc_logits.size(0)
    loss, value, gradient_sha256, gradient_l2_norm = _value_preserving_gpu_vjp_surrogate(
        ctc_logits, cpu_logits, cpu_objective
    )
    return CTCNativeVJPResult(
        loss=loss,
        value=value,
        cpu_logit_gradient_sha256=gradient_sha256,
        cpu_logit_gradient_l2_norm=gradient_l2_norm,
    )


@torch.no_grad()
def visual_pair_diagnostics(
    block_features: Tensor,
    ctc_log_probs: Tensor,
    targets: Tensor,
    input_lengths: Sequence[int] | Tensor,
    sample_ids: Sequence[str],
    candidate_store: ArtifactStore,
    control_store: ArtifactStore,
    *,
    margin: float,
    blank: int = 0,
    ignore_id: int = -1,
) -> tuple[tuple[VisualPairDiagnostic, ...], int]:
    """Measure true-pair and deranged-pair hinge geometry with one pooling pass."""

    if block_features.ndim != 3 or ctc_log_probs.ndim != 3:
        raise ValueError("block_features and ctc_log_probs must be [B,T,D/V]")
    if block_features.shape[:2] != ctc_log_probs.shape[:2]:
        raise ValueError("block features and CTC posteriors disagree on B/T")
    if targets.ndim != 2 or targets.size(0) != block_features.size(0):
        raise ValueError("targets must be [B,U]")
    if len(sample_ids) != block_features.size(0):
        raise ValueError("sample_ids does not match batch size")
    lengths = [int(value) for value in input_lengths]
    if len(lengths) != block_features.size(0):
        raise ValueError("input_lengths does not match batch size")

    metadata: list[tuple[str, str, str, str, str]] = []
    candidate_values: list[Tensor] = []
    control_values: list[Tensor] = []
    infeasible = 0
    for sample_index, raw_sample_id in enumerate(sample_ids):
        sample_id = str(raw_sample_id)
        candidate_events = candidate_store.events_for(sample_id, "E")
        control_events = control_store.events_for(sample_id, "E")
        candidate_projection = tuple(
            (event.edge_id, event.target_start, event.target_end)
            for event in candidate_events
        )
        control_projection = tuple(
            (event.edge_id, event.target_start, event.target_end)
            for event in control_events
        )
        if candidate_projection != control_projection:
            raise ValueError(f"candidate/control event projection differs: {sample_id}")
        if not candidate_events:
            continue
        target = _target_row(targets, sample_index, ignore_id)
        posterior = ctc_forward_backward(
            ctc_log_probs[sample_index, : lengths[sample_index]],
            target,
            blank=blank,
            detach=True,
        )
        if not posterior.feasible:
            infeasible += len(candidate_events)
            continue
        for candidate_event, control_event in zip(
            candidate_events, control_events
        ):
            edge = candidate_store.edges[candidate_event.edge_id]
            _verify_original_target(
                target, candidate_event, edge.target_token_ids, sample_id
            )
            if candidate_event.negative_edge_id is not None:
                raise ValueError("candidate event unexpectedly contains a deranged negative")
            control_negative = control_event.negative_edge_id
            if control_negative is None or control_negative == edge.edge_id:
                raise ValueError("control event is not a zero-identity derangement")
            pooled = pool_token_span(
                block_features[sample_index, : lengths[sample_index]],
                posterior.token_occupancy,
                candidate_event.target_start,
                candidate_event.target_end,
            )
            representation = F.normalize(pooled.float(), dim=0)

            def endpoint(edge_id: str, name: str) -> Tensor:
                value = torch.as_tensor(
                    candidate_store.prototype(edge_id, name),
                    dtype=torch.float32,
                    device=representation.device,
                )
                return F.normalize(value, dim=0)

            positive = endpoint(edge.edge_id, "target")
            candidate_negative = endpoint(edge.edge_id, "confusion")
            control_negative_tensor = endpoint(control_negative, "confusion")
            positive_distance = 1.0 - torch.dot(representation, positive)
            candidate_pre = positive_distance - (
                1.0 - torch.dot(representation, candidate_negative)
            )
            control_pre = positive_distance - (
                1.0 - torch.dot(representation, control_negative_tensor)
            )
            candidate_values.append(candidate_pre)
            control_values.append(control_pre)
            metadata.append(
                (
                    sample_id,
                    candidate_store.canonical_ids.source_for(sample_id),
                    edge.edge_id,
                    edge.target_text,
                    control_negative,
                )
            )

    if not metadata:
        return (), infeasible
    candidate_array = torch.stack(candidate_values).float().cpu().numpy()
    control_array = torch.stack(control_values).float().cpu().numpy()
    rows = []
    for fields, candidate_pre, control_pre in zip(
        metadata, candidate_array.tolist(), control_array.tolist()
    ):
        rows.append(
            VisualPairDiagnostic(
                *fields,
                candidate_pre_hinge=float(candidate_pre),
                control_pre_hinge=float(control_pre),
                candidate_hinge=max(0.0, float(candidate_pre) + float(margin)),
                control_hinge=max(0.0, float(control_pre) + float(margin)),
            )
        )
    return tuple(rows), infeasible
