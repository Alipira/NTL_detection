"""
Positive-Unlabeled learning (non-negative PU risk, Kiryo et al. 2017).

In NTL detection the only confirmed labels are caught thieves; everyone else is
*un-inspected*, not innocent. Treating un-inspected customers as negatives biases
the model. nnPU instead treats them as a mix of positives and negatives with a
known class prior pi, using only confirmed positives and the unlabeled pool.

Sigmoid surrogate: l(z, +1) = sigmoid(-z),  l(z, -1) = sigmoid(z).
"""
from __future__ import annotations

import torch


def nnpu_loss(logits, pu_known, mask, prior, beta: float = 0.0, gamma: float = 1.0):
    z = logits[mask]                       # real customers only
    k = pu_known[mask]
    pos = z[k == 1]
    unl = z[k == 0]

    if pos.numel() > 0:
        Rp_plus = torch.sigmoid(-pos).mean()
        Rp_minus = torch.sigmoid(pos).mean()
    else:
        Rp_plus = z.sum() * 0.0
        Rp_minus = z.sum() * 0.0
    Ru_minus = torch.sigmoid(unl).mean() if unl.numel() > 0 else z.sum() * 0.0

    pos_risk = prior * Rp_plus
    neg_risk = Ru_minus - prior * Rp_minus

    if neg_risk < -beta:
        # non-negative correction: stop the estimator going negative, push back
        return pos_risk - gamma * neg_risk
    return pos_risk + neg_risk
