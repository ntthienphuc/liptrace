import pytest
import torch

from pavir.ctc_posterior import (
    batched_ctc_forward_backward,
    ctc_forward_backward,
    minimum_ctc_frames,
    pool_token_span,
)


def _brute_state_paths(log_probs, targets, blank=0):
    extended = [blank]
    for token in targets:
        extended.extend((int(token), blank))
    states = len(extended)
    valid = []

    def visit(path):
        if len(path) == log_probs.size(0):
            if path[-1] in ({states - 1} if states == 1 else {states - 2, states - 1}):
                valid.append(tuple(path))
            return
        state = path[-1]
        candidates = [state]
        if state + 1 < states:
            candidates.append(state + 1)
        if (
            state + 2 < states
            and extended[state + 2] != blank
            and extended[state + 2] != extended[state]
        ):
            candidates.append(state + 2)
        for candidate in candidates:
            visit(path + [candidate])

    starts = [0] if states == 1 else [0, 1]
    for start in starts:
        visit([start])

    path_scores = []
    for path in valid:
        path_scores.append(
            sum(log_probs[frame, extended[state]] for frame, state in enumerate(path))
        )
    log_z = torch.logsumexp(torch.stack(path_scores), dim=0)
    occupancy = log_probs.new_zeros((log_probs.size(0), states))
    for path, score in zip(valid, path_scores):
        probability = (score - log_z).exp()
        for frame, state in enumerate(path):
            occupancy[frame, state] += probability
    return log_z, occupancy


@pytest.mark.parametrize(
    "targets,time",
    [
        ([1], 1),
        ([1], 3),
        ([1, 2], 3),
        ([1, 1], 3),
        ([1, 2, 1], 4),
    ],
)
def test_matches_brute_force(targets, time):
    torch.manual_seed(7 + time + len(targets))
    log_probs = torch.randn(time, 4, dtype=torch.float64).log_softmax(-1)
    expected_log_z, expected_occupancy = _brute_state_paths(log_probs, targets)
    actual = ctc_forward_backward(
        log_probs, torch.tensor(targets), detach=False
    )
    assert actual.feasible
    torch.testing.assert_close(
        actual.log_likelihood, expected_log_z.float(), rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        actual.state_occupancy,
        expected_occupancy.float(),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("targets,time", [([1], 2), ([1, 2], 4), ([1, 1], 4)])
def test_likelihood_matches_torch_ctc(targets, time):
    torch.manual_seed(19 + time)
    log_probs = torch.randn(time, 1, 5).log_softmax(-1)
    target = torch.tensor(targets, dtype=torch.long)
    loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)(
        log_probs,
        target.unsqueeze(0),
        torch.tensor([time]),
        torch.tensor([len(targets)]),
    )
    actual = ctc_forward_backward(log_probs[:, 0], target)
    torch.testing.assert_close(actual.log_likelihood, -loss[0], rtol=2e-5, atol=2e-5)


def test_repeated_token_minimum_and_impossible_alignment():
    target = torch.tensor([1, 1, 2, 2, 2])
    assert minimum_ctc_frames(target) == 8
    result = ctc_forward_backward(
        torch.randn(7, 4).log_softmax(-1), target
    )
    assert not result.feasible
    assert torch.isneginf(result.log_likelihood)
    assert result.token_occupancy.count_nonzero() == 0


def test_empty_target_uses_all_blank_path():
    log_probs = torch.randn(4, 3).log_softmax(-1)
    result = ctc_forward_backward(log_probs, torch.empty(0, dtype=torch.long))
    assert result.feasible
    torch.testing.assert_close(result.log_likelihood, log_probs[:, 0].sum())
    assert result.token_occupancy.shape == (4, 0)
    torch.testing.assert_close(result.state_occupancy, torch.ones(4, 1))


def test_detached_by_default_and_optional_gradient():
    logits = torch.randn(3, 4, requires_grad=True)
    log_probs = logits.log_softmax(-1)
    detached = ctc_forward_backward(log_probs, torch.tensor([1]))
    assert not detached.log_likelihood.requires_grad
    assert not detached.token_occupancy.requires_grad
    attached = ctc_forward_backward(log_probs, torch.tensor([1]), detach=False)
    assert attached.log_likelihood.requires_grad
    attached.log_likelihood.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_each_frame_has_unit_state_mass():
    result = ctc_forward_backward(
        torch.randn(7, 5).log_softmax(-1), torch.tensor([1, 2, 2])
    )
    assert result.feasible
    torch.testing.assert_close(
        result.state_occupancy.sum(dim=1),
        torch.ones(7),
        rtol=2e-5,
        atol=2e-5,
    )


def test_batched_lengths_and_posterior_pooling():
    logits = torch.randn(2, 5, 4).log_softmax(-1)
    targets = torch.tensor([[1, 2, -1], [2, -1, -1]])
    results = batched_ctc_forward_backward(logits, targets, [5, 3])
    assert [item.token_occupancy.shape for item in results] == [(5, 2), (3, 1)]
    features = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    pooled = pool_token_span(features, results[0].token_occupancy, 0, 2)
    assert pooled.shape == (3,)
    assert torch.isfinite(pooled).all()


def test_batched_target_padding_must_be_a_contiguous_suffix():
    logits = torch.randn(1, 4, 4).log_softmax(dim=-1)
    targets = torch.tensor([[1, -1, 2]])
    with pytest.raises(
        ValueError, match="non-padding token after the first padding"
    ):
        batched_ctc_forward_backward(logits, targets, [4])


@pytest.mark.parametrize("bad_target", [torch.tensor([0]), torch.tensor([-1]), torch.tensor([9])])
def test_rejects_invalid_targets(bad_target):
    with pytest.raises(ValueError):
        ctc_forward_backward(torch.randn(3, 4).log_softmax(-1), bad_target)
