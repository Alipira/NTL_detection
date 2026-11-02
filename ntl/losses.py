"""Loss functions for class-imbalanced theft detection."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017) on logits.

    Down-weights easy negatives so the rare theft class drives learning. This
    is the recommended default for the heavy imbalance in NTL detection and
    tends to behave better than plain oversampling.

    alpha: weight on the positive class in [0,1]. gamma: focusing strength.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        target = target.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return loss.mean()


def weighted_bce(pos_weight: float) -> nn.Module:
    """BCEWithLogits with a scalar positive-class weight (= n_neg / n_pos)."""
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
