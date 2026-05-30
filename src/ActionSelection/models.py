from __future__ import annotations

import torch
import torch.nn as nn


class ActionSelectionNet(nn.Module):
    def __init__(self, n_features: int = 8, n_actions: int = 3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(n_features, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.feature_extractor(x))
        return logits
