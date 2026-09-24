"""Feature-Only branch: a small MLP over the derived measurements."""

from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN_SIZES = (128, 128, 64, 32)


class FeatureOnlyNet(nn.Module):
    def __init__(self, in_dim: int, hidden=HIDDEN_SIZES, dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        previous = in_dim
        for size in hidden:
            layers += [nn.Linear(previous, size), nn.BatchNorm1d(size),
                       nn.ReLU(), nn.Dropout(dropout)]
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
