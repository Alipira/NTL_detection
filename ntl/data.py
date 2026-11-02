"""
Data pipeline for non-technical-loss (electricity-theft) detection.

This module is deliberately self-contained so the whole project runs with no
external dataset: `generate_synthetic` produces honest load profiles and turns
a fraction of them into thieves using the six canonical attack functions from
Jokar et al. (2016) -- the field-standard benchmark. To use real data instead
(e.g. SGCC), point `load_csv` at a file with one customer per row, daily
consumption columns, and a `label` column.

The base model uses only per-customer consumption series, so nothing here needs
feeder topology or head metering. Those are introduced later for PI-STGAT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Six canonical attack functions (Jokar et al., 2016).
# Each maps an honest 1-D series x -> a manipulated (lower-reporting) series.
# These are the standard way the literature creates positive (theft) samples
# when real labels are scarce.
# --------------------------------------------------------------------------- #

def h1(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale every reading by one constant factor a in [0.1, 0.8]."""
    a = rng.uniform(0.1, 0.8)
    return a * x


def h2(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Report zero for a random contiguous window (intermittent bypass)."""
    T = len(x)
    min_off = max(1, T // 20)
    start = rng.integers(0, max(1, T - min_off))
    dur = rng.integers(min_off, max(min_off + 1, T - start + 1))
    y = x.copy()
    y[start:start + dur] = 0.0
    return y


def h3(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiply each reading by its own random factor in [0.1, 0.8]."""
    g = rng.uniform(0.1, 0.8, size=len(x))
    return g * x


def h4(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Report a random fraction of the customer's own mean each step."""
    g = rng.uniform(0.1, 0.8, size=len(x))
    return g * float(np.mean(x))


def h5(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Report the flat daily mean every step (load-profile flattening)."""
    return np.full_like(x, float(np.mean(x)))


def h6(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reverse the series (shift usage to low-tariff periods).

    Most meaningful at sub-daily resolution; on daily data it still induces a
    detectable distributional shift, so it remains a useful benchmark attack.
    """
    return x[::-1].copy()


ATTACKS = [h1, h2, h3, h4, h5, h6]


def apply_random_attack(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Apply one randomly chosen attack; return (manipulated series, attack id 1..6)."""
    idx = int(rng.integers(0, len(ATTACKS)))
    return ATTACKS[idx](x, rng), idx + 1


# --------------------------------------------------------------------------- #
# Synthetic honest profiles
# --------------------------------------------------------------------------- #

def _honest_profile(T: int, rng: np.random.Generator) -> np.ndarray:
    """A plausible residential daily-kWh series: yearly + weekly pattern, noise,
    occasional vacancy gaps. Non-negative."""
    t = np.arange(T)
    yearly = 1.0 + 0.35 * np.sin(2 * np.pi * (t / 365.0) + rng.uniform(0, 2 * np.pi))
    dow = t % 7
    weekly = np.where(dow >= 5, rng.uniform(1.1, 1.4), 1.0)  # heavier weekends
    level = rng.lognormal(mean=1.6, sigma=0.5)               # per-customer scale
    noise = rng.normal(1.0, 0.12, size=T)
    x = level * yearly * weekly * noise
    # occasional vacancy / travel gaps (legitimate low usage)
    for _ in range(rng.integers(0, 3)):
        s = rng.integers(0, max(1, T - 14))
        d = rng.integers(3, 14)
        x[s:s + d] *= rng.uniform(0.0, 0.3)
    return np.clip(x, 0.0, None).astype(np.float32)


def generate_synthetic(
    n_customers: int = 4000,
    T: int = 365,
    theft_ratio: float = 0.09,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X[n,T] float32, y[n] int {0,1}, attack_type[n] int {0..6}).

    attack_type 0 = honest; 1..6 = which Jokar function was used.
    theft_ratio mimics real low prevalence (a few percent).
    """
    rng = np.random.default_rng(seed)
    X = np.stack([_honest_profile(T, rng) for _ in range(n_customers)])
    y = np.zeros(n_customers, dtype=np.int64)
    atk = np.zeros(n_customers, dtype=np.int64)
    n_theft = int(round(theft_ratio * n_customers))
    thieves = rng.choice(n_customers, size=n_theft, replace=False)
    for i in thieves:
        X[i], atk[i] = apply_random_attack(X[i], rng)
        y[i] = 1
    return X.astype(np.float32), y, atk


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

def impute(X: np.ndarray) -> np.ndarray:
    """Fill NaN/inf by linear interpolation along time, holding the nearest value at the edges.
    Real meter data (e.g. SGCC) is full of gaps; honest handling matters."""
    X = np.asarray(X, dtype=np.float32).copy()
    X[~np.isfinite(X)] = np.nan
    for i in range(X.shape[0]):
        row = X[i]
        mask = np.isnan(row)
        if mask.all():
            X[i] = 0.0
            continue
        if mask.any():
            idx = np.arange(len(row))
            row[mask] = np.interp(idx[mask], idx[~mask], row[~mask])
            X[i] = row
    return X


@dataclass
class Normalizer:
    """log1p + global standardization. Preserves cross-customer scale (theft
    lowers the *level*, not just the shape) while keeping values well-behaved."""
    mean: float = 0.0
    std: float = 1.0

    def fit(self, X: np.ndarray) -> "Normalizer":
        z = np.log1p(np.clip(X, 0.0, None))
        self.mean = float(z.mean())
        self.std = float(z.std() + 1e-6)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = np.log1p(np.clip(X, 0.0, None))
        return ((z - self.mean) / self.std).astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset + loaders
# --------------------------------------------------------------------------- #

class EnergyDataset(Dataset):
    """Per-customer series + binary label.

    On-the-fly augmentation (train only): with probability `aug_p`, an honest
    sample is converted into a synthetic thief via a random Jokar attack and
    relabeled 1. This generates *fresh* positives every epoch -- a stronger,
    more realistic alternative to interpolating raw series with SMOTE, and the
    natural place to later swap in a learned generative model.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalizer: Normalizer,
        augment: bool = False,
        aug_p: float = 0.15,
        seed: int = 0,
    ):
        self.X_raw = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.norm = normalizer
        self.augment = augment
        self.aug_p = aug_p
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.X_raw[i]
        label = int(self.y[i])
        if self.augment and label == 0 and self.rng.random() < self.aug_p:
            x, _ = apply_random_attack(x, self.rng)
            label = 1
        x = self.norm.transform(x[None, :])[0]      # (T,)
        x = torch.from_numpy(x).unsqueeze(0)         # (1, T) for Conv1d
        return x, torch.tensor(label, dtype=torch.float32)


def stratified_split(
    y: np.ndarray, fracs=(0.7, 0.15, 0.15), seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-stratified index split so every fold keeps the rare positives."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(fracs[0] * n)
        n_va = int(fracs[1] * n)
        train_idx += idx[:n_tr].tolist()
        val_idx += idx[n_tr:n_tr + n_va].tolist()
        test_idx += idx[n_tr + n_va:].tolist()
    return (np.array(train_idx), np.array(val_idx), np.array(test_idx))


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    augment: bool = True,
    aug_p: float = 0.15,
    seed: int = 0,
):
    """Build train/val/test DataLoaders. Normalizer is fit on TRAIN only."""
    X = impute(X)
    tr, va, te = stratified_split(y, seed=seed)
    norm = Normalizer().fit(X[tr])
    train_ds = EnergyDataset(X[tr], y[tr], norm, augment=augment, aug_p=aug_p, seed=seed)
    val_ds = EnergyDataset(X[va], y[va], norm, augment=False)
    test_ds = EnergyDataset(X[te], y[te], norm, augment=False)
    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        norm,
        (tr, va, te),
    )


def load_csv(
    path: str, label_col: str = "label", id_col: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Real-data hook. Expects one customer per row, consumption in the
    remaining numeric columns, and a binary `label` column.

    For SGCC: the public release is already in this shape (FLAG = label,
    CONS_NO = id, the rest are daily-kWh date columns).
    """
    df = pd.read_csv(path)
    y = df[label_col].astype(int).to_numpy()
    drop = [label_col] + ([id_col] if id_col else [])
    X = df.drop(columns=drop).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return X, y
