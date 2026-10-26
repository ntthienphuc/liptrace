"""Gradient-isolated auxiliary routing losses for PAViR-VSR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .artifacts import ArtifactStore, PAViRConfig, TrainingEvent
from .ctc_posterior import ctc_forward_backward, pool_token_span


@dataclass(frozen=True)
class AuxiliaryResult:
    """One auxiliary route's scalar loss and accounting counters.

    ``eligible`` counts frozen events offered to the route.  ``active`` counts
    only events whose hinge is strictly positive and therefore contributes a
    non-zero loss gradient.  Keeping these meanings separate prevents an
    eligibility counter from masquerading as evidence that the auxiliary
    objective is still gradient-bearing.
    """

    loss: Tensor
    eligible: int
    active: int
    infeasible: int = 0
    selected_batch: bool = True


def _zero_from(tensor: Tensor) -> Tensor:
    """Differentiable scalar zero on the same device without new parameters."""

    return tensor.reshape(-1)[:1].sum() * 0.0


def _target_row(targets: Tensor, index: int, ignore_id: int) -> Tensor:
    return targets[index][targets[index].ne(ignore_id)]


def _verify_event_tokens(
    target: Tensor,
    event: TrainingEvent,
    expected: Sequence[int],
    sample_id: str,
) -> None:
    if event.target_end > target.numel():
        raise ValueError(
            f"frozen event span exceeds target for {sample_id}: "
            f"[{event.target_start},{event.target_end}) > {target.numel()}"
        )
    observed = tuple(
        int(value)
        for value in target[event.target_start : event.target_end].detach().cpu().tolist()
    )
    if observed != tuple(expected):
        raise ValueError(
            f"artifact/tokenizer mismatch for {sample_id}/{event.edge_id}: "
            f"expected={tuple(expected)}, observed={observed}"
        )


def visual_margin_loss(
    block_features: Tensor,
    ctc_log_probs: Tensor,
    targets: Tensor,
    input_lengths: Sequence[int] | Tensor,
    sample_ids: Sequence[str],
    store: ArtifactStore,
    *,
    margin: float,
    blank: int = 0,
    ignore_id: int = -1,
) -> AuxiliaryResult:
    """Compute route-E posterior-pooled prototype margin loss.

    Alignment occupancies are detached inside the forward-backward routine.
    Consequently this loss updates ``block_features`` (and therefore the
    visual frontend/encoder) but not the CTC head through the alignment.
    """

    if block_features.ndim != 3 or ctc_log_probs.ndim != 3:
        raise ValueError("block_features and ctc_log_probs must be batch-first rank 3")
    if block_features.shape[:2] != ctc_log_probs.shape[:2]:
        raise ValueError("block features and CTC posteriors have different B/T axes")
    if targets.ndim != 2 or targets.size(0) != block_features.size(0):
        raise ValueError("targets must be [B,U]")
    if len(sample_ids) != block_features.size(0):
        raise ValueError("sample_ids does not match batch size")
    lengths = [int(value) for value in input_lengths]
    if len(lengths) != block_features.size(0):
        raise ValueError("input_lengths does not match batch size")
    if margin < 0:
        raise ValueError("visual margin must be non-negative")

    losses = []
    eligible = 0
    infeasible = 0
    for index, sample_id in enumerate(sample_ids):
        events = store.events_for(str(sample_id), "E")
        if not events:
            continue
        target = _target_row(targets, index, ignore_id)
        posterior = ctc_forward_backward(
            ctc_log_probs[index, : lengths[index]],
            target,
            blank=blank,
            detach=True,
        )
        eligible += len(events)
        if not posterior.feasible:
            infeasible += len(events)
            continue
        for event in events:
            edge = store.edges[event.edge_id]
            _verify_event_tokens(
                target, event, edge.target_token_ids, str(sample_id)
            )
            pooled = pool_token_span(
                block_features[index, : lengths[index]],
                posterior.token_occupancy,
                event.target_start,
                event.target_end,
            )
            representation = F.normalize(pooled.float(), dim=0)
            positive = torch.tensor(
                store.prototype(edge.edge_id, "target"),
                dtype=torch.float32,
                device=representation.device,
            )
            negative = torch.tensor(
                store.prototype(
                    event.negative_edge_id or edge.edge_id,
                    "confusion",
                ),
                dtype=torch.float32,
                device=representation.device,
            )
            positive = F.normalize(positive, dim=0)
            negative = F.normalize(negative, dim=0)
            positive_distance = 1.0 - torch.dot(representation, positive)
            negative_distance = 1.0 - torch.dot(representation, negative)
            losses.append(
                F.relu(positive_distance - negative_distance + float(margin))
            )

    if losses:
        hinges = torch.stack(losses)
        loss = hinges.mean()
        active = int(torch.count_nonzero(hinges.detach().gt(0)).item())
    else:
        loss = _zero_from(block_features)
        active = 0
    return AuxiliaryResult(
        loss=loss,
        eligible=eligible,
        active=active,
        infeasible=infeasible,
    )


def _stable_fraction(seed: int, update: int, values: Sequence[str]) -> float:
    payload = "\x1f".join((str(seed), str(update), *map(str, values))).encode("utf-8")
    integer = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return integer / float(1 << 64)


def context_batch_selected(
    sample_ids: Sequence[str],
    *,
    seed: int,
    update: int,
    fraction: float,
) -> bool:
    """Deterministic quota gate that never exceeds the locked fraction.

    ``update`` is the zero-based eligible-batch ordinal in production.  The
    cumulative number selected after ``n`` calls is exactly
    ``floor(n * fraction)``; unlike a Bernoulli hash it can never drift above
    50 percent.  ``sample_ids`` and ``seed`` remain in the signature for the
    frozen API and audit trail, but replacement selection still uses them
    separately.
    """

    if not 0 <= fraction <= 0.5:
        raise ValueError("context batch fraction must be in [0,0.5]")
    if update < 0:
        raise ValueError("context batch ordinal must be non-negative")
    if fraction == 0:
        return False
    del sample_ids, seed
    before = int(update * fraction)
    after = int((update + 1) * fraction)
    return after > before


@dataclass(frozen=True)
class ContextReplacements:
    sample_indices: Tensor
    positive_targets: Tensor
    negative_targets: Tensor
    edge_ids: tuple[str, ...]


def _pad_target_rows(rows: Sequence[Tensor], ignore_id: int) -> Tensor:
    if not rows:
        raise ValueError("cannot pad an empty target collection")
    width = max(int(row.numel()) for row in rows)
    output = rows[0].new_full((len(rows), width), int(ignore_id))
    for index, row in enumerate(rows):
        output[index, : row.numel()] = row
    return output


def build_context_replacements(
    targets: Tensor,
    sample_ids: Sequence[str],
    store: ArtifactStore,
    *,
    seed: int,
    update: int,
    ignore_id: int = -1,
) -> ContextReplacements | None:
    """Choose at most one frozen route-D replacement per sentence."""

    if targets.ndim != 2 or targets.size(0) != len(sample_ids):
        raise ValueError("targets/sample_ids batch mismatch")
    selected_indices = []
    positives = []
    negatives = []
    edge_ids = []
    for index, sample_id in enumerate(sample_ids):
        events = store.events_for(str(sample_id), "D")
        if not events:
            continue
        choice_key = _stable_fraction(
            seed, update, (str(sample_id), *sorted(event.edge_id for event in events))
        )
        selected = events[min(int(choice_key * len(events)), len(events) - 1)]
        edge = store.edges[selected.edge_id]
        target = _target_row(targets, index, ignore_id)
        _verify_event_tokens(target, selected, edge.target_token_ids, str(sample_id))
        replacement = target.new_tensor(edge.confusion_token_ids)
        negative = torch.cat(
            (
                target[: selected.target_start],
                replacement,
                target[selected.target_end :],
            )
        )
        if torch.equal(target, negative):
            raise ValueError(f"context edge is an identity replacement: {edge.edge_id}")
        selected_indices.append(index)
        positives.append(target)
        negatives.append(negative)
        edge_ids.append(edge.edge_id)
    if not positives:
        return None
    return ContextReplacements(
        sample_indices=torch.tensor(
            selected_indices, dtype=torch.long, device=targets.device
        ),
        positive_targets=_pad_target_rows(positives, ignore_id),
        negative_targets=_pad_target_rows(negatives, ignore_id),
        edge_ids=tuple(edge_ids),
    )


def _decoder_io(
    targets: Tensor,
    *,
    sos: int,
    eos: int,
    ignore_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build padded teacher-forcing tensors and an explicit causal mask."""

    rows = [row[row.ne(ignore_id)] for row in targets]
    if any(not row.numel() for row in rows):
        raise ValueError("context scoring does not accept an empty sentence")
    lengths = torch.tensor(
        [int(row.numel()) for row in rows], dtype=torch.long, device=targets.device
    )
    width = int(lengths.max().item()) + 1
    inputs = targets.new_full((len(rows), width), int(eos))
    outputs = targets.new_full((len(rows), width), int(ignore_id))
    for index, row in enumerate(rows):
        inputs[index, 0] = int(sos)
        inputs[index, 1 : row.numel() + 1] = row
        outputs[index, : row.numel()] = row
        outputs[index, row.numel()] = int(eos)
    causal = torch.ones((width, width), dtype=torch.bool, device=targets.device).tril()
    valid_queries = (
        torch.arange(width, device=targets.device).unsqueeze(0)
        < (lengths + 1).unsqueeze(1)
    )
    decoder_mask = causal.unsqueeze(0) & valid_queries.unsqueeze(2)
    lexical_mask = (
        torch.arange(width, device=targets.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    return inputs, outputs, decoder_mask, lexical_mask


def _normalized_decoder_scores(
    decoder: Callable[[Tensor, Tensor, Tensor, Tensor | None], tuple[Tensor, Tensor]],
    targets: Tensor,
    memory: Tensor,
    memory_mask: Tensor | None,
    *,
    sos: int,
    eos: int,
    ignore_id: int,
) -> Tensor:
    inputs, outputs, decoder_mask, lexical_mask = _decoder_io(
        targets, sos=sos, eos=eos, ignore_id=ignore_id
    )
    logits, _ = decoder(inputs, decoder_mask, memory, memory_mask)
    log_probs = logits.float().log_softmax(dim=-1)
    safe_outputs = outputs.masked_fill(outputs.eq(ignore_id), 0)
    token_scores = log_probs.gather(-1, safe_outputs.unsqueeze(-1)).squeeze(-1)
    lexical_mask = lexical_mask.to(token_scores.dtype)
    return (token_scores * lexical_mask).sum(dim=1) / lexical_mask.sum(dim=1)


def context_margin_loss(
    decoder: Callable[[Tensor, Tensor, Tensor, Tensor | None], tuple[Tensor, Tensor]],
    memory: Tensor,
    memory_mask: Tensor | None,
    targets: Tensor,
    sample_ids: Sequence[str],
    store: ArtifactStore,
    config: PAViRConfig,
    *,
    update: int,
    gate_index: int | None = None,
    sos: int,
    eos: int,
    ignore_id: int = -1,
) -> AuxiliaryResult:
    """Compute route-D contextual margin loss with encoder/RNG isolation."""

    replacements = build_context_replacements(
        targets,
        sample_ids,
        store,
        seed=config.seed,
        update=update,
        ignore_id=ignore_id,
    )
    if replacements is None:
        return AuxiliaryResult(
            loss=_zero_from(memory),
            eligible=0,
            active=0,
            selected_batch=False,
        )
    eligible = int(replacements.sample_indices.numel())
    selected = context_batch_selected(
        sample_ids,
        seed=config.seed,
        update=update if gate_index is None else gate_index,
        fraction=config.context_batch_fraction,
    )
    if not selected:
        return AuxiliaryResult(
            loss=_zero_from(memory),
            eligible=eligible,
            active=0,
            selected_batch=False,
        )

    indices = replacements.sample_indices
    isolated_memory = memory.index_select(0, indices).detach()
    isolated_mask = (
        None if memory_mask is None else memory_mask.index_select(0, indices)
    )
    positive_width = replacements.positive_targets.size(1)
    negative_width = replacements.negative_targets.size(1)
    common_width = max(positive_width, negative_width)

    def extend(value: Tensor) -> Tensor:
        if value.size(1) == common_width:
            return value
        return F.pad(
            value,
            (0, common_width - value.size(1)),
            value=int(ignore_id),
        )

    combined_targets = torch.cat(
        (
            extend(replacements.positive_targets),
            extend(replacements.negative_targets),
        ),
        dim=0,
    )
    combined_memory = torch.cat((isolated_memory, isolated_memory), dim=0)
    combined_mask = (
        None
        if isolated_mask is None
        else torch.cat((isolated_mask, isolated_mask), dim=0)
    )

    # Auxiliary dropout may be stochastic, but it must not advance the outer
    # RNG stream and thereby change future native dropout between B0 and P.
    cuda_devices = [memory.device] if memory.is_cuda else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        scores = _normalized_decoder_scores(
            decoder,
            combined_targets,
            combined_memory,
            combined_mask,
            sos=sos,
            eos=eos,
            ignore_id=ignore_id,
        )
    positive, negative = scores[:eligible], scores[eligible:]
    hinges = F.relu(float(config.context_margin) - (positive - negative))
    loss = hinges.mean()
    return AuxiliaryResult(
        loss=loss,
        eligible=eligible,
        active=int(torch.count_nonzero(hinges.detach().gt(0)).item()),
        selected_batch=True,
    )
