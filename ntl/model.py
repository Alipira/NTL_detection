"""
Base model: a single-branch 1D-CNN temporal classifier.

Design intent: the encoder and head are separate, swappable modules. To grow
into PI-STGAT you replace `CNNEncoder` with a dilated-TCN + self-attention
encoder and add a graph branch + static-MLP branch + an energy-conservation
head -- the training loop, data pipeline, and metrics all stay the same.

PI-STGAT HOOKS are marked in comments below.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Stack of 1-D conv blocks over a customer's consumption series.

    Input:  (B, 1, T)
    Output: (B, out_dim)   -- a fixed-length representation of the load curve.

    Dilations widen the receptive field cheaply so intra-week and seasonal
    structure are both visible. Global avg+max pooling makes it length-agnostic.
    """

    def __init__(self, channels=(32, 64, 128), kernel: int = 5, dropout: float = 0.2):
        super().__init__()
        dilations = [1, 2, 4][: len(channels)]
        blocks = []
        in_c = 1
        for c, d in zip(channels, dilations):
            pad = (kernel - 1) // 2 * d
            blocks += [
                nn.Conv1d(in_c, c, kernel_size=kernel, padding=pad, dilation=d),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_c = c
        self.conv = nn.Sequential(*blocks)
        self.out_dim = channels[-1] * 2  # avg-pool ++ max-pool concat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)                         # (B, C, T)
        avg = h.mean(dim=-1)                     # (B, C)
        mx = h.max(dim=-1).values                # (B, C)
        return torch.cat([avg, mx], dim=-1)      # (B, 2C)


class ClassifierHead(nn.Module):
    """MLP -> single logit. Accepts an optional static-feature vector so the
    PI-STGAT static branch concatenates here with zero rewiring."""

    def __init__(self, in_dim: int, hidden: int = 64, static_dim: int = 0, dropout: float = 0.3):
        super().__init__()
        self.static_dim = static_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim + static_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor, static: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.static_dim and static is not None:
            feat = torch.cat([feat, static], dim=-1)
        return self.net(feat).squeeze(-1)        # (B,)


class ETDClassifier(nn.Module):
    """Encoder + head. `encoder` is injected so it can be swapped later."""

    def __init__(self, encoder: nn.Module, head: ClassifierHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor, static: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.encoder(x), static)
        # PI-STGAT HOOK (graph branch): combine self.encoder(x) with a GAT over
        # transformer-neighbours before the head; add a second forward that also
        # returns the per-transformer energy-conservation residual.


def build_base_model(channels=(32, 64, 128), kernel: int = 5, dropout: float = 0.2) -> ETDClassifier:
    enc = CNNEncoder(channels=channels, kernel=kernel, dropout=dropout)
    head = ClassifierHead(in_dim=enc.out_dim, static_dim=0, dropout=max(dropout, 0.3))
    return ETDClassifier(enc, head)
