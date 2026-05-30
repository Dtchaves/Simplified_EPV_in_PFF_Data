from __future__ import annotations

import torch
import torch.nn as nn


class ShotEPVNet(nn.Module):
    def __init__(self, n_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
