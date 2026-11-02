"""
PI-STGAT: Physics-Informed Spatio-Temporal Graph-Attention detector.

Per-customer representation built from three branches, fused, and read out by
two heads:

  temporal  : dilated TCN + self-attention over the load curve
  graph     : attention among customers that share a transformer (the feeder)
  static    : MLP over contract / meter / region features
       |
     fuse -> theft-risk head      (calibrated probability)
          -> conservation head    (per-customer stolen-energy estimate s_i >= 0)

The conservation head is tied to physics: for each feeder the predicted stolen
energies must sum to the observed head-balance residual
    target_f = head_f * (1 - tl_frac) - sum(reported_i)
so the model can only attribute theft when the transformer's books fail to
close. That residual estimate s_i doubles as the expected recovered energy used
for cost-aware ranking.

Ablation switches (use_temporal is always on): use_graph, use_static,
use_physics. Turning all extras off recovers a temporal-only classifier.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Dilated 1-D convolutions + a self-attention layer over time."""

    def __init__(self, channels=(32, 64), kernel=5, d_model=64, heads=4,
                 pool=4, dropout=0.2, pos_enc=False):
        super().__init__()
        self.pos_enc = pos_enc
        dilations = [1, 2, 4][: len(channels)]
        blocks, in_c = [], 1
        for c, d in zip(channels, dilations):
            pad = (kernel - 1) // 2 * d
            blocks += [nn.Conv1d(in_c, c, kernel, padding=pad, dilation=d),
                       nn.BatchNorm1d(c), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_c = c
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.MaxPool1d(pool)
        self.pool_size = pool
        self.proj = nn.Linear(channels[-1], d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model * 2

    def tokens(self, x):                        # x: (N, 1, T) -> (N, T', d)
        """Per-timestep token sequence (before pooling over time). Used both by
        forward() and by the masked-reconstruction pretrainer."""
        h = self.pool(self.conv(x))             # (N, C, T'), T' = T // pool
        h = h.transpose(1, 2)                   # (N, T', C)
        h = self.proj(h)                        # (N, T', d)
        if self.pos_enc:
            h = h + self._positions(h)
        a, _ = self.attn(h, h, h)               # self-attention over time
        return self.norm(h + a)

    @staticmethod
    def _positions(h):
        """Sinusoidal position code, so attention can see where in the series a token sits."""
        t = torch.arange(h.shape[1], device=h.device, dtype=h.dtype).unsqueeze(1)
        freq = torch.exp(torch.arange(0, h.shape[2], 2, device=h.device, dtype=h.dtype)
                         * (-math.log(10000.0) / h.shape[2]))
        pe = torch.zeros(h.shape[1], h.shape[2], device=h.device, dtype=h.dtype)
        pe[:, 0::2] = torch.sin(t * freq)
        pe[:, 1::2] = torch.cos(t * freq[: h.shape[2] // 2])
        return pe.unsqueeze(0)

    def forward(self, x):                       # x: (N, 1, T)
        h = self.tokens(x)
        return torch.cat([h.mean(1), h.max(1).values], dim=-1)   # (N, 2d)


class StaticEncoder(nn.Module):
    def __init__(self, d_in, d_out=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(64, d_out))
        self.out_dim = d_out

    def forward(self, s):
        return self.net(s)


class FeederGraphAttention(nn.Module):
    """Multi-head attention among customers on the same feeder (a GAT over a
    transformer's neighbourhood, implemented as masked self-attention)."""

    def __init__(self, d_model, heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, mask):                 # h: (B, K, d); mask: (B, K) bool
        a, _ = self.attn(h, h, h, key_padding_mask=~mask)
        return self.norm(h + a)


class PISTGAT(nn.Module):
    def __init__(self, d_static, d_model=64, tl_frac=0.06,
                 use_graph=True, use_static=True, use_physics=True, pos_enc=False):
        super().__init__()
        self.use_graph = use_graph
        self.use_static = use_static
        self.use_physics = use_physics
        self.tl_frac = tl_frac

        self.temporal = TemporalEncoder(d_model=d_model, pos_enc=pos_enc)
        fuse_in = self.temporal.out_dim
        if use_static:
            self.static = StaticEncoder(d_static)
            fuse_in += self.static.out_dim
        self.fuse = nn.Sequential(nn.Linear(fuse_in, d_model), nn.ReLU(inplace=True))
        if use_graph:
            self.graph = FeederGraphAttention(d_model)

        if use_physics:
            self.s_head = nn.Linear(d_model, 1)          # stolen-energy estimate (>=0)
            nn.init.constant_(self.s_head.bias, -4.0)    # start s ~ 0 so conservation loss is well-scaled
            self.cls_head = nn.Linear(d_model + 1, 1)    # uses fused + s
        else:
            self.cls_head = nn.Linear(d_model, 1)

    def forward(self, batch):
        x = batch["series"]                              # (B, K, 1, T)
        B, K = x.shape[0], x.shape[1]
        mask = batch["mask"]                             # (B, K)

        # encode real customers only: padding rows must not reach BatchNorm
        enc = self.temporal(x[mask])                     # (n_real, D)
        h = enc.new_zeros(B, K, enc.shape[-1])
        h[mask] = enc
        if self.use_static:
            h = torch.cat([h, self.static(batch["static"])], dim=-1)
        h = self.fuse(h)                                 # (B, K, d)
        if self.use_graph:
            h = self.graph(h, mask)

        if self.use_physics:
            s = F.softplus(self.s_head(h)).squeeze(-1) * mask    # (B, K) >= 0
            logit = self.cls_head(torch.cat([h, s.unsqueeze(-1)], dim=-1)).squeeze(-1)
        else:
            s = torch.zeros(B, K, device=x.device)
            logit = self.cls_head(h).squeeze(-1)
        return {"logit": logit, "s": s}

    def conservation_loss(self, s, batch):
        """Per-feeder: (sum_i s_i  -  [head*(1-tl) - sum_i reported_i])^2."""
        if not self.use_physics:
            return torch.tensor(0.0, device=s.device)
        mask = batch["mask"].float()
        reported_sum = (batch["cust_mean"] * mask).sum(1)        # (B,)
        target = batch["head"] * (1.0 - self.tl_frac) - reported_sum
        pred_sum = (s * mask).sum(1)                             # (B,)
        return F.mse_loss(pred_sum, target.clamp(min=0.0))


def build_pistgat(meta, **flags) -> PISTGAT:
    return PISTGAT(d_static=meta["d_static"], **flags)


def load_pistgat(path: str, device="cpu"):
    """Rebuild a saved model. Returns (model, checkpoint); the checkpoint also holds
    what scoring new data needs: norm_mean/norm_std, stat_mean/stat_std, unit,
    temperature and threshold."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = PISTGAT(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
