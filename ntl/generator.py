"""
Learned generative augmenter for theft samples (cVAE over the theft *mask*).

The fixed six-function augmentation encodes hand-written guesses about how
thieves behave. This module instead LEARNS the manipulation distribution from
confirmed cases and samples fresh, realistic thieves from it.

Key design choice -- generate the multiplicative mask, not the series:
    m_t = reported_t / true_t   in [0, 1]
    generated reported = m_hat * honest_series
A sigmoid output keeps m_hat in (0,1), so every generated thief is a physically
valid *under-reporter* by construction (no impossible profiles, unlike SMOTE in
raw kWh space). The honest profile conditions the decoder, so manipulation
adapts to the customer (e.g. steal on high-use days). Latent z carries attack
"style"; sampling z interpolates between styles, smoothing the decision
boundary beyond what six fixed rules give.

Why cVAE rather than full TimeGAN: we condition on the honest profile and have
(true, reported) pairs, which is exactly the conditional-VAE setting; it trains
stably in seconds on CPU. TimeGAN's adversarial machinery is the heavier
alternative when no pairs exist at all.

Where pairs come from:
  - simulator: true series are known (SimInfo.X_true);
  - real data: confirmed inspections, where the utility's post-inspection
    corrected consumption provides the (true, reported) pair -- standard
    practice in back-billing.

Known limitation (stated honestly): masks in (0,1) cannot represent
energy-PRESERVING attacks such as h6 (series reversal). The augmenter models
the under-reporting family, which is where the money is; pattern-only attacks
remain the temporal/graph branches' job.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = 16
COND_L = 32         # downsampled honest-shape length
MASK_L = 64         # downsampled mask length the VAE works in


def _down(x: torch.Tensor, L: int) -> torch.Tensor:
    return F.adaptive_avg_pool1d(x.unsqueeze(1), L).squeeze(1)


class TheftMaskCVAE(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        cond_dim = COND_L + 1                      # shape + log-level
        self.enc = nn.Sequential(
            nn.Linear(MASK_L + cond_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * LATENT),
        )
        self.dec = nn.Sequential(
            nn.Linear(LATENT + cond_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, MASK_L),
        )

    @staticmethod
    def make_cond(x_true: torch.Tensor) -> torch.Tensor:
        """Scale-free shape (mean-normalized, downsampled) + log mean level."""
        mu = x_true.mean(dim=1, keepdim=True).clamp(min=1e-3)
        shape = _down(x_true / mu, COND_L)
        return torch.cat([shape, torch.log(mu)], dim=1)

    def encode(self, m_ds, cond):
        h = self.enc(torch.cat([m_ds, cond], dim=1))
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar.clamp(-8, 8)

    def decode(self, z, cond):
        return torch.sigmoid(self.dec(torch.cat([z, cond], dim=1)))

    def forward(self, m_ds, cond):
        mu, logvar = self.encode(m_ds, cond)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, cond), mu, logvar


class GenAugmenter:
    """Fit on (true, reported) theft pairs; then call like the fixed attacks:
        y_new, attack_id = augmenter(x_honest, rng)   # attack_id = 9
    """

    ATTACK_ID = 9

    def __init__(self, hidden: int = 128, seed: int = 0):
        torch.manual_seed(seed)
        self.model = TheftMaskCVAE(hidden=hidden)
        self.fitted = False

    # ---------------- training ----------------
    def fit(self, X_true: np.ndarray, Y_rep: np.ndarray,
            epochs: int = 400, lr: float = 1e-3, beta: float = 0.01,
            verbose: bool = True) -> "GenAugmenter":
        X = torch.from_numpy(np.asarray(X_true, dtype=np.float32))
        Y = torch.from_numpy(np.asarray(Y_rep, dtype=np.float32))
        m = torch.where(X > 1e-3, (Y / X.clamp(min=1e-3)).clamp(0, 1),
                        torch.ones_like(X))
        m_ds = _down(m, MASK_L)
        cond = TheftMaskCVAE.make_cond(X)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            m_hat, mu, logvar = self.model(m_ds, cond)
            rec = F.mse_loss(m_hat, m_ds)
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and (ep % max(1, epochs // 4) == 0 or ep == 1):
                print(f"[cvae {ep:03d}/{epochs}] recon={rec.item():.4f} kl={kl.item():.3f}")
        self.fitted = True
        return self

    # ---------------- sampling ----------------
    @torch.no_grad()
    def sample_mask(self, x_honest: np.ndarray, rng) -> np.ndarray:
        x = torch.from_numpy(np.asarray(x_honest, dtype=np.float32)).unsqueeze(0)
        cond = TheftMaskCVAE.make_cond(x)
        z = torch.from_numpy(rng.normal(size=(1, LATENT)).astype(np.float32))
        m_ds = self.model.decode(z, cond)                       # (1, MASK_L)
        m = F.interpolate(m_ds.unsqueeze(1), size=x.shape[-1],
                          mode="linear", align_corners=False)
        return m.squeeze().numpy().astype(np.float32)

    def __call__(self, x_honest: np.ndarray, rng) -> tuple[np.ndarray, int]:
        assert self.fitted, "GenAugmenter.fit(...) before sampling"
        m = self.sample_mask(x_honest, rng)
        return (m * x_honest).astype(np.float32), self.ATTACK_ID

    # ---------------- persistence ----------------
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> "GenAugmenter":
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.fitted = True
        return self
