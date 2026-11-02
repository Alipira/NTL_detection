"""
Feeder-structured data for PI-STGAT.

The base model used isolated per-customer series. PI-STGAT needs the spatial and
physical context that base lacked:
  - a feeder graph: which meters sit on the same distribution transformer,
  - a feeder-head reading: total energy measured at the transformer,
  - static features: contracted power, meter age, tariff, region, alarms.

This generator emits all of that. Crucially it keeps the *true* consumption X
hidden from the model and exposes only the *reported* meter readings Y (which
thieves manipulate) plus the head reading H. The per-feeder gap
    rho_f = H_f - sum(reported Y) ≈ technical loss + total stolen
is the physical signal the conservation head learns from.

Theft is clustered on a subset of "hot" feeders (so the graph branch has
structure to exploit) and hot feeders correlate with a higher region index (so
the fairness check has something to find). Energy-reducing attacks (h1-h4)
move the head balance; pattern-only attacks (h5,h6) do not — which is exactly
why a temporal branch and a physics branch are complementary.

Real-data layout (documented in build_feeder_datasets): the same fields can be
populated from a utility export — a meter->transformer map, smart-meter series,
a contract/GIS table, and transformer head readings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ntl.data import ATTACKS, Normalizer, _honest_profile, impute

N_TARIFF = 3
# static feature order: [contracted_power, meter_age, region, tariff(3), n_alarms]


@dataclass
class FeederData:
    Y: np.ndarray            # (n, T) reported readings (model sees these)
    y: np.ndarray            # (n,)   labels {0,1}
    atk: np.ndarray          # (n,)   attack id (0 = honest)
    S: np.ndarray            # (n, D) static features
    region: np.ndarray       # (n,)   region index in [0,1] (fairness bucket)
    feeder_id: np.ndarray    # (n,)   transformer id
    head_mean: np.ndarray    # (F,)   per-feeder mean head reading (raw units)
    stolen: np.ndarray       # (n,)   true mean stolen energy (0 for honest); for metrics only
    n_feeders: int
    meter_id: Optional[np.ndarray] = None    # (n,) original meter ids (real data)
    feeder_key: Optional[np.ndarray] = None  # (F,) original transformer ids (real data)


def _attack(x, rng):
    idx = int(rng.integers(0, len(ATTACKS)))
    return ATTACKS[idx](x, rng), idx + 1


def generate_feeder_data(
    n_feeders: int = 180,
    k_range=(8, 36),
    T: int = 365,
    hot_frac: float = 0.25,
    hot_theft_p: float = 0.45,
    cold_theft_p: float = 0.03,
    tl_frac: float = 0.06,
    head_noise: float = 0.01,
    seed: int = 0,
) -> FeederData:
    rng = np.random.default_rng(seed)
    Ys, ys, atks, Ss, regions, fids, stolens = [], [], [], [], [], [], []
    head_means = []

    for f in range(n_feeders):
        K = int(rng.integers(k_range[0], k_range[1] + 1))
        hot = rng.random() < hot_frac
        theft_p = hot_theft_p if hot else cold_theft_p
        # poorer areas (higher region index) are modestly more hot -> fairness tension
        feeder_region = np.clip(rng.normal(0.6 if hot else 0.4, 0.12), 0, 1)

        true_sum = 0.0
        reported_sum = 0.0
        for _ in range(K):
            x = _honest_profile(T, rng)                  # hidden true consumption
            is_thief = rng.random() < theft_p
            if is_thief:
                y_series, aid = _attack(x, rng)
            else:
                y_series, aid = x.copy(), 0
            stolen = max(0.0, float(x.mean() - y_series.mean()))

            # static features
            contracted = float(x.mean()) * rng.uniform(1.3, 2.2)   # contract > usage (Terciyanli signal)
            meter_age = float(np.clip(rng.normal(9 if is_thief else 7, 3), 0, 20))
            tariff = int(rng.integers(0, N_TARIFF))
            tar = np.zeros(N_TARIFF, dtype=np.float32); tar[tariff] = 1.0
            n_alarms = float(rng.poisson(2.0 if is_thief else 0.8))
            reg = float(np.clip(feeder_region + rng.normal(0, 0.05), 0, 1))
            S = np.concatenate([[contracted, meter_age, reg], tar, [n_alarms]]).astype(np.float32)

            Ys.append(y_series); ys.append(int(is_thief)); atks.append(aid)
            Ss.append(S); regions.append(reg); fids.append(f); stolens.append(stolen)
            true_sum += float(x.mean())
            reported_sum += float(y_series.mean())

        # head reading = true consumption + technical loss (+ small meter noise)
        head = true_sum * (1.0 + tl_frac) * (1.0 + rng.normal(0, head_noise))
        head_means.append(head)

    return FeederData(
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


# --------------------------------------------------------------------------- #
# Dataset (one item = one feeder) + padding collate
# --------------------------------------------------------------------------- #

class FeederDataset(Dataset):
    def __init__(self, fd: FeederData, feeders, norm: Normalizer, stat_mean, stat_std,
                 unit: float, pu_known: Optional[np.ndarray] = None,
                 augmenter=None, aug_p: float = 0.0, seed: int = 0):
        """`augmenter` is a callable (x_raw, rng) -> (y_raw, attack_id): either
        ntl.data.apply_random_attack (fixed six functions) or a fitted
        ntl.generator.GenAugmenter (learned cVAE). Train-time only.

        Augmentation is physically coherent: the feeder head reading is left
        unchanged while the customer's reported series drops, so the
        conservation residual grows by exactly the amount "stolen" -- the same
        signature a real thief leaves. cust_mean / stolen / label / pu_known
        are all recomputed for the item, and augmented positives count as
        KNOWN positives under PU (we created them, so they are labeled)."""
        self.fd = fd
        self.feeders = list(feeders)
        self.norm = norm
        self.stat_mean = stat_mean
        self.stat_std = stat_std
        self.unit = unit
        self.pu_known = pu_known if pu_known is not None else fd.y  # default: fully labeled
        self.augmenter = augmenter
        self.aug_p = aug_p
        self.rng = np.random.default_rng(seed + 17)
        # group customer indices by feeder
        self.by_feeder = {f: np.where(fd.feeder_id == f)[0] for f in self.feeders}

    def __len__(self):
        return len(self.feeders)

    def _stat(self, S):
        return ((S - self.stat_mean) / self.stat_std).astype(np.float32)

    def __getitem__(self, i):
        f = self.feeders[i]
        idx = self.by_feeder[f]

        Yf = self.fd.Y[idx]
        lab = self.fd.y[idx].astype(np.float32)
        pu = self.pu_known[idx].astype(np.float32)
        atk = self.fd.atk[idx].copy()
        stolen_raw = self.fd.stolen[idx].astype(np.float32)

        if self.augmenter is not None and self.aug_p > 0:
            Yf = Yf.copy(); lab = lab.copy(); pu = pu.copy(); stolen_raw = stolen_raw.copy()
            for j in range(len(idx)):
                if pu[j] == 0 and self.rng.random() < self.aug_p:
                    new, aid = self.augmenter(Yf[j], self.rng)
                    stolen_raw[j] = max(0.0, float(Yf[j].mean() - new.mean()))
                    Yf[j] = new
                    lab[j] = 1.0
                    pu[j] = 1.0          # synthetic positives are known positives
                    atk[j] = aid

        series = self.norm.transform(Yf)                        # (K, T)
        return {
            "series": torch.from_numpy(series).unsqueeze(1),    # (K, 1, T)
            "static": torch.from_numpy(self._stat(self.fd.S[idx])),  # (K, D)
            "label": torch.from_numpy(lab),
            "pu_known": torch.from_numpy(pu),
            "region": torch.from_numpy(self.fd.region[idx]),
            "cust_mean": torch.tensor(Yf.mean(1) / self.unit, dtype=torch.float32),  # (K,)
            "head": torch.tensor(float(self.fd.head_mean[f]) / self.unit, dtype=torch.float32),
            "stolen": torch.tensor(stolen_raw / self.unit, dtype=torch.float32),     # (K,)
            "atk": torch.from_numpy(atk),
            "cidx": torch.from_numpy(idx.astype(np.int64)),
        }


def collate_feeders(batch):
    B = len(batch)
    Kmax = max(b["series"].shape[0] for b in batch)
    T = batch[0]["series"].shape[-1]
    D = batch[0]["static"].shape[-1]

    series = torch.zeros(B, Kmax, 1, T)
    static = torch.zeros(B, Kmax, D)
    label = torch.zeros(B, Kmax)
    pu_known = torch.zeros(B, Kmax)
    region = torch.zeros(B, Kmax)
    cust_mean = torch.zeros(B, Kmax)
    stolen = torch.zeros(B, Kmax)
    atk = torch.zeros(B, Kmax, dtype=torch.long)
    cidx = torch.full((B, Kmax), -1, dtype=torch.long)
    mask = torch.zeros(B, Kmax, dtype=torch.bool)
    head = torch.zeros(B)

    for b, item in enumerate(batch):
        K = item["series"].shape[0]
        series[b, :K] = item["series"]
        static[b, :K] = item["static"]
        label[b, :K] = item["label"]
        pu_known[b, :K] = item["pu_known"]
        region[b, :K] = item["region"]
        cust_mean[b, :K] = item["cust_mean"]
        stolen[b, :K] = item["stolen"]
        atk[b, :K] = item["atk"]
        cidx[b, :K] = item["cidx"]
        mask[b, :K] = True
        head[b] = item["head"]

    return {"series": series, "static": static, "label": label, "pu_known": pu_known,
            "region": region, "cust_mean": cust_mean, "stolen": stolen, "atk": atk,
            "cidx": cidx, "mask": mask, "head": head}


def split_feeders(n_feeders, fracs=(0.7, 0.15, 0.15), seed=0):
    rng = np.random.default_rng(seed)
    f = np.arange(n_feeders); rng.shuffle(f)
    n_tr = int(fracs[0] * n_feeders); n_va = int(fracs[1] * n_feeders)
    return f[:n_tr], f[n_tr:n_tr + n_va], f[n_tr + n_va:]


def pu_known_labels(fd: FeederData, pu_reveal: float, seed: int = 0) -> np.ndarray:
    """Labels a practitioner could actually see: only a pu_reveal fraction of the
    true thieves in the TRAIN and VAL feeders stay positive. TEST keeps full labels
    for evaluation, while model selection and calibration see no hidden truth."""
    tr_f, va_f, _ = split_feeders(fd.n_feeders, seed=seed)
    known = fd.y.copy().astype(np.float32)
    if pu_reveal < 1.0:
        rng = np.random.default_rng(seed + 1)
        for feeders in (tr_f, va_f):
            pos = np.where((fd.y == 1) & np.isin(fd.feeder_id, feeders))[0]
            hide = rng.choice(pos, size=int(round((1 - pu_reveal) * len(pos))), replace=False)
            known[hide] = 0.0  # becomes unlabeled
    return known


def build_feeder_datasets(fd: FeederData, batch_size=16, pu_reveal=1.0, seed=0,
                          augmenter=None, aug_p: float = 0.0):
    """Split by feeder (whole transformers held out), fit normalizers on train.

    pu_reveal < 1 simulates Positive-Unlabeled labels (see pu_known_labels).

    augmenter/aug_p apply on-the-fly synthetic-thief augmentation to the TRAIN
    split only (see FeederDataset).
    """
    tr_f, va_f, te_f = split_feeders(fd.n_feeders, seed=seed)
    tr_mask_cust = np.isin(fd.feeder_id, tr_f)

    Yc = impute(fd.Y)
    norm = Normalizer().fit(Yc[tr_mask_cust])
    Str = fd.S[tr_mask_cust]
    stat_mean = Str.mean(0); stat_std = Str.std(0) + 1e-6
    unit = float(fd.head_mean[tr_f].mean() + 1e-6)

    pu_known = pu_known_labels(fd, pu_reveal, seed)

    def mk(feeders, aug=None, ap=0.0):
        return FeederDataset(fd, feeders, norm, stat_mean, stat_std, unit, pu_known,
                             augmenter=aug, aug_p=ap, seed=seed)

    g = torch.Generator().manual_seed(seed)
    meta = {"d_static": fd.S.shape[1], "unit": unit, "seed": seed,
            "norm": norm, "stat_mean": stat_mean, "stat_std": stat_std,
            "train_feeders": tr_f, "train_cust_mask": tr_mask_cust}
    return (
        DataLoader(mk(tr_f, augmenter, aug_p), batch_size=batch_size, shuffle=True,
                   collate_fn=collate_feeders, generator=g),
        DataLoader(mk(va_f), batch_size=batch_size, shuffle=False, collate_fn=collate_feeders),
        DataLoader(mk(te_f), batch_size=batch_size, shuffle=False, collate_fn=collate_feeders),
        meta,
    )
