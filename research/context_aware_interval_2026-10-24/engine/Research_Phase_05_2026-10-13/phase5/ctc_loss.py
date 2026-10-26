"""Differentiable CPU CTC avoids the nondeterministic CUDA backward kernel."""
import torch


def cpu_ctc(log_probabilities, targets, input_lengths, target_lengths, zero_infinity=True):
    if log_probabilities.dtype != torch.float32:
        raise ValueError('CTC probabilities must use FP32')
    loss = torch.nn.functional.ctc_loss(
        log_probabilities.to('cpu'), targets.to('cpu'),
        torch.as_tensor(input_lengths, device='cpu', dtype=torch.long),
        torch.as_tensor(target_lengths, device='cpu', dtype=torch.long),
        blank=0, reduction='sum', zero_infinity=zero_infinity)
    return loss.to(log_probabilities.device)
