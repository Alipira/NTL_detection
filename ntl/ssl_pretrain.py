"""
Self-supervised pretraining: masked span reconstruction.

Labels are scarce in NTL detection but unlabeled load curves are abundant.
Before any supervised training we can teach the temporal encoder what *normal
consumption dynamics* look like: hide random spans of each series (set to the
normalized global mean = 0) and train the encoder + a light decoder to
reconstruct the hidden values. The encoder weights then initialize PI-STGAT's
temporal branch.

This is the masked-autoencoder idea adapted to load curves, and the
semi-supervised cousin of prediction-residual methods (Sharma & Majumdar):
a model that can fill in a customer's missing week has internalized their
seasonality, weekly rhythm, and noise level -- exactly the structure theft
violates.

Span masking (not single points) matters: load curves are smooth enough that
single missing points are trivially interpolated; whole weeks are not.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ntl.pistgat import TemporalEncoder


def random_span_mask(n: int, T: int, ratio: float, rng,
                     min_span: int = 5, max_span: int = 21) -> np.ndarray:
    """Boolean (n, T) mask, True = hidden. Contiguous spans until ~ratio covered."""
    mask = np.zeros((n, T), dtype=bool)
    target = int(ratio * T)
    for i in range(n):
        covered = 0
        while covered < target:
            span = int(rng.integers(min_span, max_span + 1))
            start = int(rng.integers(0, max(1, T - span)))
            mask[i, start:start + span] = True
            covered = int(mask[i].sum())
    return mask


class MaskedReconstructor(nn.Module):
    """TemporalEncoder + per-token linear decoder back to the time axis.

    The encoder pools time by `pool_size`, so each token decodes to `pool_size`
    consecutive values; reconstruction covers the first T' * pool_size steps.
    """

    def __init__(self, d_model: int = 64, pos_enc: bool = False):
        super().__init__()
        self.encoder = TemporalEncoder(d_model=d_model, pos_enc=pos_enc)
        self.decoder = nn.Linear(d_model, self.encoder.pool_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (N,1,T); mask: (N,T) bool. Returns scalar masked-MSE loss."""
        x_corrupt = x.masked_fill(mask.unsqueeze(1), 0.0)
        tok = self.encoder.tokens(x_corrupt)                 # (N, T', d)
        rec = self.decoder(tok).reshape(x.shape[0], -1)      # (N, T'*pool)
        L = rec.shape[1]
        target = x[:, 0, :L]
        m = mask[:, :L]
        if m.sum() == 0:
            return F.mse_loss(rec, target)
        return F.mse_loss(rec[m], target[m])


def pretrain_temporal(
    series: np.ndarray,
    d_model: int = 64,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    mask_ratio: float = 0.3,
    seed: int = 0,
    verbose: bool = True,
    pos_enc: bool = False,
) -> dict:
    """Pretrain on normalized series (N, T); returns encoder state_dict.

    Use only TRAINING-split customers to avoid any leakage into evaluation.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.from_numpy(np.asarray(series, dtype=np.float32)).unsqueeze(1)  # (N,1,T)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = MaskedReconstructor(d_model=d_model, pos_enc=pos_enc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for ep in range(1, epochs + 1):
        tot, nb = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            mb = torch.from_numpy(
                random_span_mask(xb.shape[0], xb.shape[-1], mask_ratio, rng)
            ).to(device)
            loss = model(xb, mb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if verbose:
            print(f"[ssl {ep:02d}/{epochs}] masked-recon mse={tot / max(1, nb):.4f}")
    return {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}
