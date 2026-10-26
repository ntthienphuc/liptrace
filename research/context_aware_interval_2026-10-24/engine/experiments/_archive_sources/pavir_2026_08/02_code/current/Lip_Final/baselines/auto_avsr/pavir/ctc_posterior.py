"""Numerically stable CTC forward-backward posterior occupancies.

The implementation intentionally stays in FP32 even when the surrounding model
uses mixed precision.  PAViR uses the detached token occupancies only as
alignment weights; gradients must not flow through the alignment itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class CTCPosterior:
    """Posterior for one unpadded example."""

    log_likelihood: Tensor
    state_occupancy: Tensor
    token_occupancy: Tensor
    extended_targets: Tensor
    feasible: bool


def minimum_ctc_frames(targets: Tensor) -> int:
    """Return the shortest alignment length, including repeated-token blanks."""

    targets = targets.reshape(-1)
    if targets.numel() < 2:
        return int(targets.numel())
    repeats = targets[1:].eq(targets[:-1]).sum()
    return int(targets.numel() + repeats.item())


def _extended_targets(targets: Tensor, blank: int) -> Tensor:
    extended = targets.new_full((2 * targets.numel() + 1,), int(blank))
    if targets.numel():
        extended[1::2] = targets
    return extended


def _impossible_result(
    log_probs: Tensor,
    targets: Tensor,
    blank: int,
    detach: bool,
) -> CTCPosterior:
    time = int(log_probs.size(0))
    extended = _extended_targets(targets, blank)
    state = log_probs.new_zeros((time, extended.numel()), dtype=torch.float32)
    token = log_probs.new_zeros((time, targets.numel()), dtype=torch.float32)
    likelihood = log_probs.new_tensor(float("-inf"), dtype=torch.float32)
    if detach:
        state, token, likelihood = state.detach(), token.detach(), likelihood.detach()
    return CTCPosterior(likelihood, state, token, extended, False)


def ctc_forward_backward(
    log_probs: Tensor,
    targets: Tensor,
    *,
    blank: int = 0,
    detach: bool = True,
) -> CTCPosterior:
    """Compute state and target-position CTC posterior occupancies.

    Args:
        log_probs: ``[T, V]`` log probabilities for one unpadded example.
        targets: ``[U]`` target token IDs, without blanks or padding.
        blank: CTC blank token ID.
        detach: detach likelihood and occupancies from the autograd graph.

    Returns:
        A :class:`CTCPosterior`.  ``token_occupancy[t, u]`` is the
        forward-backward probability that frame ``t`` occupies target position
        ``u``.  Impossible alignments return zero occupancies and ``-inf``.
    """

    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must have shape [T,V], got {tuple(log_probs.shape)}")
    if targets.ndim != 1:
        targets = targets.reshape(-1)
    if targets.dtype not in (torch.int32, torch.int64):
        raise TypeError("targets must contain integer token IDs")

    time, vocab = map(int, log_probs.shape)
    if not 0 <= blank < vocab:
        raise ValueError(f"blank={blank} is outside vocabulary size {vocab}")
    if time == 0:
        return _impossible_result(log_probs, targets, blank, detach)
    if targets.numel():
        if bool(targets.eq(blank).any()):
            raise ValueError("targets must not contain the CTC blank token")
        if int(targets.min()) < 0 or int(targets.max()) >= vocab:
            raise ValueError("targets contain an out-of-vocabulary token")
    if time < minimum_ctc_frames(targets):
        return _impossible_result(log_probs, targets, blank, detach)

    # Explicit conversion avoids BF16/FP16 DP instability under autocast.
    scores = log_probs.float()
    extended = _extended_targets(targets.to(device=scores.device), blank)
    states = int(extended.numel())
    emissions = scores.index_select(1, extended)
    neg_inf = scores.new_tensor(float("-inf"))

    # A skip into state s is valid only for a non-blank label different from
    # the label two states earlier.  This is the standard repeated-token rule.
    allow_skip_to = torch.zeros(states, dtype=torch.bool, device=scores.device)
    if states > 2:
        allow_skip_to[2:] = extended[2:].ne(blank) & extended[2:].ne(extended[:-2])

    alpha_start = scores.new_full((states,), neg_inf)
    alpha_start[0] = emissions[0, 0]
    if states > 1:
        alpha_start[1] = emissions[0, 1]

    # Keep the recurrences functional.  In-place writes into a preallocated
    # alpha/beta table are slightly faster, but invalidate saved views during
    # backward when ``detach=False`` (used by the mathematical preflight).
    alpha_rows = [alpha_start]
    for frame in range(1, time):
        previous = alpha_rows[-1]
        step_one = torch.cat((neg_inf.expand(1), previous[:-1]))
        if states > 2:
            step_two = torch.cat((neg_inf.expand(2), previous[:-2]))
            step_two = torch.where(allow_skip_to, step_two, neg_inf)
        else:
            step_two = neg_inf.expand(states)
        incoming = torch.logsumexp(
            torch.stack((previous, step_one, step_two), dim=0), dim=0
        )
        alpha_rows.append(incoming + emissions[frame])
    alpha = torch.stack(alpha_rows, dim=0)

    final_states = [states - 1] if states == 1 else [states - 2, states - 1]
    final_index = torch.tensor(final_states, device=scores.device)
    log_likelihood = torch.logsumexp(alpha[-1].index_select(0, final_index), dim=0)
    if not bool(torch.isfinite(log_likelihood)):
        return _impossible_result(log_probs, targets, blank, detach)

    # beta[t,s] excludes the emission at t and contains all future emissions.
    beta_end = scores.new_full((states,), neg_inf)
    beta_end = beta_end.index_fill(0, final_index, 0.0)
    allow_skip_from = torch.zeros(states, dtype=torch.bool, device=scores.device)
    if states > 2:
        allow_skip_from[:-2] = allow_skip_to[2:]

    beta_rows_reversed = [beta_end]
    for frame in range(time - 2, -1, -1):
        future = beta_rows_reversed[-1] + emissions[frame + 1]
        step_one = torch.cat((future[1:], neg_inf.expand(1)))
        if states > 2:
            step_two = torch.cat((future[2:], neg_inf.expand(2)))
            step_two = torch.where(allow_skip_from, step_two, neg_inf)
        else:
            step_two = neg_inf.expand(states)
        beta_rows_reversed.append(
            torch.logsumexp(
                torch.stack((future, step_one, step_two), dim=0), dim=0
            )
        )
    beta = torch.stack(list(reversed(beta_rows_reversed)), dim=0)

    log_gamma = alpha + beta - log_likelihood
    # Tiny positive log-gamma values can occur from rounding; probabilities
    # must remain bounded by one.
    state_occupancy = log_gamma.clamp_max(0.0).exp()
    token_occupancy = state_occupancy[:, 1::2]

    if detach:
        log_likelihood = log_likelihood.detach()
        state_occupancy = state_occupancy.detach()
        token_occupancy = token_occupancy.detach()
    return CTCPosterior(
        log_likelihood,
        state_occupancy,
        token_occupancy,
        extended,
        True,
    )


def batched_ctc_forward_backward(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Sequence[int] | Tensor,
    target_lengths: Optional[Sequence[int] | Tensor] = None,
    *,
    blank: int = 0,
    padding_value: int = -1,
    detach: bool = True,
) -> List[CTCPosterior]:
    """Run :func:`ctc_forward_backward` over a padded batch.

    ``log_probs`` is batch-first ``[B,T,V]``.  If ``target_lengths`` is
    omitted, target rows are truncated at the first ``padding_value`` and
    padding must form one contiguous suffix.  A non-padding token after the
    first padding value is rejected rather than silently changing the target.
    """

    if log_probs.ndim != 3:
        raise ValueError(
            f"batched log_probs must have shape [B,T,V], got {tuple(log_probs.shape)}"
        )
    if targets.ndim != 2 or targets.size(0) != log_probs.size(0):
        raise ValueError("targets must have shape [B,U] with the same batch size")

    input_lengths_list = [int(value) for value in input_lengths]
    if len(input_lengths_list) != log_probs.size(0):
        raise ValueError("input_lengths does not match batch size")
    if target_lengths is None:
        target_lengths_list = []
        for index, row in enumerate(targets):
            padding_positions = row.eq(padding_value).nonzero(as_tuple=False)
            if padding_positions.numel() == 0:
                target_lengths_list.append(int(row.numel()))
                continue
            first_padding = int(padding_positions[0].item())
            if bool(row[first_padding:].ne(padding_value).any()):
                raise ValueError(
                    "targets contain a non-padding token after the first "
                    f"padding value at batch index {index}"
                )
            target_lengths_list.append(first_padding)
    else:
        target_lengths_list = [int(value) for value in target_lengths]
    if len(target_lengths_list) != log_probs.size(0):
        raise ValueError("target_lengths does not match batch size")

    output: List[CTCPosterior] = []
    for index, (input_length, target_length) in enumerate(
        zip(input_lengths_list, target_lengths_list)
    ):
        if not 0 <= input_length <= log_probs.size(1):
            raise ValueError(f"invalid input length at batch index {index}")
        if not 0 <= target_length <= targets.size(1):
            raise ValueError(f"invalid target length at batch index {index}")
        output.append(
            ctc_forward_backward(
                log_probs[index, :input_length],
                targets[index, :target_length],
                blank=blank,
                detach=detach,
            )
        )
    return output


def pool_token_span(
    features: Tensor,
    token_occupancy: Tensor,
    start: int,
    end: int,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Posterior-pool ``features[T,D]`` over target positions ``[start,end)``."""

    if features.ndim != 2 or token_occupancy.ndim != 2:
        raise ValueError("features and token_occupancy must both be rank two")
    if features.size(0) != token_occupancy.size(0):
        raise ValueError("features and token occupancies have different time axes")
    if not 0 <= start < end <= token_occupancy.size(1):
        raise ValueError(f"invalid target span [{start},{end})")
    weights = token_occupancy[:, start:end].sum(dim=1).to(features.dtype)
    mass = weights.sum()
    if not bool(torch.isfinite(mass)) or float(mass) <= eps:
        raise ValueError("target span has no finite posterior occupancy")
    return (features * (weights / mass.clamp_min(eps)).unsqueeze(1)).sum(dim=0)
