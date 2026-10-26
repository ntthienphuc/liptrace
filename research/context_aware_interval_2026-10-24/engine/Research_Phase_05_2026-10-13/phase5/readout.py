"""Small temporal readout; initialization preserves the native core representation exactly."""

import math
import torch
from torch import nn


class BoundaryReadout(nn.Module):
    def __init__(self, dimension=768, bottleneck=128):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.query = nn.Linear(dimension, bottleneck)
        self.key = nn.Linear(dimension, bottleneck)
        self.value = nn.Linear(dimension, bottleneck)
        self.output = nn.Linear(bottleneck, dimension)
        self.penalties = nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, encoded, start, end, boundary=True):
        core = encoded[:, start:end]
        normalized = self.norm(encoded)
        query = self.query(normalized[:, start:end])
        logits = query @ self.key(normalized).transpose(-1, -2)
        logits = logits / math.sqrt(query.size(-1))
        if boundary:
            indices = torch.arange(encoded.size(1), device=encoded.device)
            penalties = torch.nn.functional.softplus(self.penalties)
            bias = penalties[0] * (indices < start) + penalties[1] * (indices >= end)
            logits = logits - bias
        weights = torch.softmax(logits.float(), dim=-1).to(encoded.dtype)
        return core + self.output(weights @ self.value(normalized))
