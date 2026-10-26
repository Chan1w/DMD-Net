"""z_r condition encoder and receiver-SNR embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionEncoder(nn.Module):
    """Map a configurable latent condition to same-resolution features."""

    def __init__(self, in_channels: int = 4, out_channels: int = 320,
                 hidden_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class CSIEmbedding(nn.Module):
    """Map per-sample CSI (``[B, csi_dim]``) to the time-embedding dimension."""

    def __init__(self, csi_dim: int = 1, out_dim: int = 1280):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(csi_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        return self.mlp(csi.float())

