"""
Whole-network simulator for low-voltage NTL research.

This upgrades the simple per-feeder generator in data_graph.py to a physically
grounded simulation of an LV distribution network:

  weather ─► customer demand (3 archetypes with heating/cooling response)
                 │
        true consumption x_i,t  ──(thieves manipulate)──►  reported y_i,t
                 │
        line + transformer losses, computed from load (quadratic, I²R-style:
        loss_t grows with x_t², so the technical-loss *fraction* varies with
        load — a flat tl_frac is only an approximation, as on a real feeder)
                 │
        head reading H_t = Σ x_i,t + losses_t  (+ meter noise)

Attack library = the six Jokar functions (ids 1-6) plus two NOVEL attacks that
exist in field reports but not in the standard benchmark:
  7  seasonal-gated theft : under-report only on the customer's high-use days
                            (e.g. winter heating) — hides inside seasonality
  8  gradual ramp         : under-reporting that grows slowly over the year
                            (meter-tampering drift) — defeats change-point logic
These give an honest "unseen attack" test bed for the learned augmenter.

Output is a `FeederData` (drop-in for the whole PI-STGAT pipeline) plus a
`SimInfo` holding the hidden truth (true series, temperature, per-feeder loss
fractions) used for generator training pairs and diagnostics — never as model
input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ntl.data import ATTACKS
from ntl.data_graph import FeederData

ARCHETYPES = ("residential", "commercial", "industrial")
N_TARIFF = 3
# static feature order:
# [contracted_power, meter_age, region, tariff(3), n_alarms, archetype(3)] -> D=10


@dataclass
class SimInfo:
    temp: np.ndarray         # (T,)   shared temperature series (deg C)
    X_true: np.ndarray       # (n, T) hidden true consumption (generator pairs / diagnostics)
    archetype: np.ndarray    # (n,)   archetype index 0..2
    tl_feeder: np.ndarray    # (F,)   realized mean technical-loss fraction per feeder
    tl_eff: float            # global mean loss fraction (pass to the model's tl_frac)
    heads: np.ndarray        # (F, T) daily head readings (before head_mean noise)


# --------------------------------------------------------------------------- #
# Weather + demand
# --------------------------------------------------------------------------- #

def _temperature(T: int, rng) -> np.ndarray:
    """Annual sinusoid + AR(1) weather noise (deg C)."""
    d = np.arange(T)
    base = 12.0 + 9.0 * np.sin(2 * np.pi * (d - 110) / 365.0)
    eps = np.zeros(T)
    for t in range(1, T):
        eps[t] = 0.7 * eps[t - 1] + rng.normal(0, 1.8)
    return (base + eps).astype(np.float32)


_ARCH_P = (0.6, 0.3, 0.1)
_BASE_KWH = (9.0, 25.0, 60.0)            # typical daily kWh by archetype
_WEEKEND = ((1.05, 1.30), (0.40, 0.70), (0.70, 0.95))
_HEAT = ((0.020, 0.050), (0.000, 0.020), (0.000, 0.010))   # per deg C below 15
_COOL = ((0.000, 0.020), (0.020, 0.050), (0.000, 0.010))   # per deg C above 24
_CONTRACT = ((1.4, 2.4), (1.3, 2.0), (1.2, 1.8))


def _demand(arch: int, T: int, temp: np.ndarray, rng) -> np.ndarray:
    """One customer's true daily-kWh series."""
    level = float(np.exp(rng.normal(np.log(_BASE_KWH[arch]), 0.45)))
    dow = np.arange(T) % 7
    wk = np.where(dow >= 5, rng.uniform(*_WEEKEND[arch]), 1.0)
    hc = rng.uniform(*_HEAT[arch])
    cc = rng.uniform(*_COOL[arch])
    thermal = 1.0 + hc * np.clip(15.0 - temp, 0, None) + cc * np.clip(temp - 24.0, 0, None)
    noise = rng.lognormal(0.0, 0.10, size=T)
    x = level * wk * thermal * noise
    if arch == 0:                                   # residential vacancy gaps
        for _ in range(rng.integers(0, 2)):
            s = rng.integers(0, max(1, T - 12))
            d = rng.integers(3, 12)
            x[s:s + d] *= rng.uniform(0.0, 0.3)
    return np.clip(x, 0.0, None).astype(np.float32)


# --------------------------------------------------------------------------- #
# Attacks: six Jokar functions + two novel field-style attacks
# --------------------------------------------------------------------------- #

def _attack_seasonal(x: np.ndarray, rng) -> np.ndarray:
    """(7) Under-report only the customer's own high-consumption days."""
    thr = np.quantile(x, 0.6)
    a = rng.uniform(0.2, 0.6)
    y = x.copy()
    y[x >= thr] *= a
    return y


def _attack_ramp(x: np.ndarray, rng) -> np.ndarray:
    """(8) Slowly growing under-reporting (tampering drift)."""
    beta = rng.uniform(0.4, 0.85)
    return (x * (1.0 - np.linspace(0.0, beta, len(x), dtype=np.float32))).astype(np.float32)


def apply_attack(x: np.ndarray, attack_id: int, rng) -> np.ndarray:
    """attack_id 1..6 = Jokar h1..h6; 7 = seasonal-gated; 8 = ramp."""
    if 1 <= attack_id <= 6:
        return ATTACKS[attack_id - 1](x, rng).astype(np.float32)
    if attack_id == 7:
        return _attack_seasonal(x, rng)
    if attack_id == 8:
        return _attack_ramp(x, rng)
    raise ValueError(f"unknown attack id {attack_id}")


# --------------------------------------------------------------------------- #
# Network simulation
# --------------------------------------------------------------------------- #

def simulate_network(
    n_feeders: int = 150,
    k_range=(8, 32),
    T: int = 365,
    hot_frac: float = 0.25,
    hot_theft_p: float = 0.45,
    cold_theft_p: float = 0.03,
    attack_pool=(1, 2, 3, 4, 5, 6, 7, 8),
    tl_range=(0.04, 0.09),
    head_noise: float = 0.01,
    seed: int = 0,
    label_static: bool = True,
) -> tuple[FeederData, SimInfo]:
    """label_static=False redraws meter age and alarm counts independently of theft,
    removing the label signal the simulator otherwise places in those features. The
    redraws use their own generator, so the network itself is unchanged."""
    rng = np.random.default_rng(seed)
    rng_static = np.random.default_rng(seed + 1000)
    temp = _temperature(T, rng)
    attack_pool = tuple(attack_pool)

    Ys, Xs, ys, atks, Ss, regions, fids, stolens, archs = [], [], [], [], [], [], [], [], []
    head_means, tl_feeder, heads_all = [], [], []

    for f in range(n_feeders):
        K = int(rng.integers(k_range[0], k_range[1] + 1))
        hot = rng.random() < hot_frac
        theft_p = hot_theft_p if hot else cold_theft_p
        feeder_region = float(np.clip(rng.normal(0.6 if hot else 0.4, 0.12), 0, 1))

        Xf, Yf, dists = [], [], []
        for _ in range(K):
            arch = int(rng.choice(3, p=_ARCH_P))
            x = _demand(arch, T, temp, rng)
            is_thief = rng.random() < theft_p
            if is_thief:
                aid = int(rng.choice(attack_pool))
                y_series = apply_attack(x, aid, rng)
            else:
                aid, y_series = 0, x.copy()
            stolen = max(0.0, float(x.mean() - y_series.mean()))

            contracted = float(x.mean()) * rng.uniform(*_CONTRACT[arch])
            meter_age = float(np.clip(rng.normal(9 if is_thief else 7, 3), 0, 20))
            tariff = arch if rng.random() < 0.85 else int(rng.integers(0, N_TARIFF))
            tar = np.zeros(N_TARIFF, dtype=np.float32); tar[tariff] = 1.0
            n_alarms = float(rng.poisson(1.8 if is_thief else 0.8))
            if not label_static:
                meter_age = float(np.clip(rng_static.normal(7, 3), 0, 20))
                n_alarms = float(rng_static.poisson(0.8))
            reg = float(np.clip(feeder_region + rng.normal(0, 0.05), 0, 1))
            aoh = np.zeros(3, dtype=np.float32); aoh[arch] = 1.0
            S = np.concatenate([[contracted, meter_age, reg], tar, [n_alarms], aoh]).astype(np.float32)

            Xf.append(x); Yf.append(y_series); dists.append(rng.uniform(0.3, 1.0))
            Xs.append(x); Ys.append(y_series); ys.append(int(is_thief)); atks.append(aid)
            Ss.append(S); regions.append(reg); fids.append(f); stolens.append(stolen)
            archs.append(arch)

        # ---- physical losses on this feeder (quadratic in load) ----
        Xf = np.stack(Xf)                                  # (K, T) true
        dists = np.asarray(dists)[:, None]                 # (K, 1)
        line_raw = (dists * Xf ** 2).sum(0)                # I^2 R-style per-customer line loss
        tx_raw = (Xf.sum(0) ** 2) / max(1, len(Xf))        # transformer copper loss
        raw = line_raw + 0.3 * tx_raw                      # (T,) unscaled
        tl_target = rng.uniform(*tl_range)
        scale = tl_target * Xf.sum() / max(raw.sum(), 1e-9)
        losses = scale * raw                               # mean fraction == tl_target
        H = Xf.sum(0) + losses                             # (T,) daily head readings
        heads_all.append(H.astype(np.float32))
        tl_feeder.append(float(losses.sum() / max(Xf.sum(), 1e-9)))
        head_means.append(float(H.mean()) * (1.0 + rng.normal(0, head_noise)))

    X_true = np.stack(Xs).astype(np.float32)
    fd = FeederData(
        Y=np.stack(Ys).astype(np.float32),
        y=np.asarray(ys, dtype=np.int64),
        atk=np.asarray(atks, dtype=np.int64),
        S=np.stack(Ss).astype(np.float32),
        region=np.asarray(regions, dtype=np.float32),
        feeder_id=np.asarray(fids, dtype=np.int64),
        head_mean=np.asarray(head_means, dtype=np.float32),
        stolen=np.asarray(stolens, dtype=np.float32),
        n_feeders=n_feeders,
    )
    info = SimInfo(
        temp=temp,
        X_true=X_true,
        archetype=np.asarray(archs, dtype=np.int64),
        tl_feeder=np.asarray(tl_feeder, dtype=np.float32),
        tl_eff=float(np.average(tl_feeder)),
        heads=np.stack(heads_all),
    )
    return fd, info
