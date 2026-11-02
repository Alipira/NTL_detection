"""
Experiment: does augmentation help generalize to UNSEEN attack types?

Setup
  D_train : network sim where thieves use only attacks h1-h4 ("seen"), at LOW
            prevalence -- the label-scarce regime where augmentation matters.
  D_test  : an independent network where thieves use ALL EIGHT attacks,
            including h5/h6 (energy-preserving) and the novel h7 (seasonal-
            gated) and h8 (ramp). None of these manipulation styles, nor any
            customer, appears in training.
  Augmenter variants on the train split (temporal-only classifier so the
  comparison isolates the positive-sample distribution):
    none     -- only the scarce real positives
    jokar    -- fixed six-function augmentation (the literature default;
                hand-coded prior knowledge, includes h5/h6 the data lacks)
    learned  -- cVAE fitted ONLY on the train split's confirmed h1-h4 pairs

Usage (staged so each step fits a short run):
    python -m scripts.exp_augment prepare
    python -m scripts.exp_augment run --aug none
    python -m scripts.exp_augment run --aug jokar
    python -m scripts.exp_augment run --aug learned
Repeat with --seed N on both stages; each seed keeps its own cache.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch
from scipy.special import expit
from torch.utils.data import DataLoader

from ntl.data import apply_random_attack
from ntl.data_graph import FeederDataset, build_feeder_datasets, collate_feeders, split_feeders
from ntl.generator import GenAugmenter
from ntl.losses import FocalLoss
from ntl.metrics import best_f1_threshold, compute_metrics
from ntl.pistgat import build_pistgat
from ntl.simulator import simulate_network
from scripts.train_pistgat import fit_temperature, gather
from ntl.results import clean_metrics, log_run

CACHE = os.path.join("runs", "exp_augment")
ATTACK_NAMES = {1: "h1 scale", 2: "h2 zero-win", 3: "h3 per-step", 4: "h4 frac-mean",
                5: "h5 flat-mean", 6: "h6 reverse", 7: "h7 seasonal", 8: "h8 ramp"}


def cache_dir(seed: int) -> str:
    return os.path.join(CACHE, f"seed{seed}")


def prepare(seed: int = 0, cvae_epochs: int = 400):
    cache = cache_dir(seed)
    os.makedirs(cache, exist_ok=True)
    # label-scarce training network, "seen" attacks only
    fd_tr, info_tr = simulate_network(n_feeders=110, T=240, seed=seed,
                                      attack_pool=(1, 2, 3, 4),
                                      hot_theft_p=0.12, cold_theft_p=0.012)
    # independent test network, all 8 attacks at realistic prevalence
    fd_te, _ = simulate_network(n_feeders=60, T=240, seed=seed + 7,
                                attack_pool=(1, 2, 3, 4, 5, 6, 7, 8))
    with open(f"{cache}/data.pkl", "wb") as f:
        pickle.dump({"fd_tr": fd_tr, "X_true_tr": info_tr.X_true, "fd_te": fd_te}, f)

    # fit the cVAE on TRAIN-SPLIT confirmed pairs only (no val/test leakage)
    tr_f, _, _ = split_feeders(fd_tr.n_feeders, seed=seed)
    pair = np.isin(fd_tr.feeder_id, tr_f) & (fd_tr.y == 1)
    gen = GenAugmenter(seed=seed).fit(info_tr.X_true[pair], fd_tr.Y[pair],
                                      epochs=cvae_epochs, verbose=False)
    gen.save(f"{cache}/cvae.pt")
    print(f"[prepare] train: {len(fd_tr.y)} cust, theft={fd_tr.y.mean():.1%} "
          f"(attacks 1-4 only) | test: {len(fd_te.y)} cust, theft={fd_te.y.mean():.1%} "
          f"(attacks 1-8) | cVAE fit on {int(pair.sum())} pairs")


def run(aug: str, seed: int = 0, epochs: int = 8, aug_p: float = 0.25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = cache_dir(seed)
    if not os.path.exists(f"{cache}/data.pkl"):
        raise SystemExit(f"{cache}/data.pkl not found -- run: exp_augment.py prepare --seed {seed}")
    with open(f"{cache}/data.pkl", "rb") as f:
        d = pickle.load(f)
    fd_tr, fd_te = d["fd_tr"], d["fd_te"]

    augmenter = None
    if aug == "jokar":
        augmenter = apply_random_attack
    elif aug == "learned":
        augmenter = GenAugmenter(seed=seed).load(f"{cache}/cvae.pt")
    elif aug == "both":
        gen = GenAugmenter(seed=seed).load(f"{cache}/cvae.pt")

        def augmenter(x, rng):
            """Prior knowledge (fixed rules) + data adaptation (cVAE), 50/50."""
            return apply_random_attack(x, rng) if rng.random() < 0.5 else gen(x, rng)

    torch.manual_seed(seed); np.random.seed(seed)
    train_loader, val_loader, _, meta = build_feeder_datasets(
        fd_tr, batch_size=16, seed=seed, augmenter=augmenter,
        aug_p=aug_p if augmenter else 0.0)

    # temporal-only classifier: isolates the effect of the positive distribution
    model = build_pistgat(meta, d_model=64, use_graph=False, use_static=False,
                          use_physics=False).to(device)
    foc = FocalLoss(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            r = model(b); m = b["mask"]
            loss = foc(r["logit"][m], b["label"][m])
            opt.zero_grad(); loss.backward(); opt.step()
        dv = gather(model, val_loader, device)
        a = compute_metrics(dv["label"], dv["p"])["aupr"]
        if a > best:
            best, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # calibrate + threshold on validation, evaluate on the UNSEEN network
    dv = gather(model, val_loader, device)
    T = fit_temperature(dv["logit"], dv["label"])
    thr = best_f1_threshold(dv["label"], expit(dv["logit"] / T))

    te_ds = FeederDataset(fd_te, range(fd_te.n_feeders), meta["norm"],
                          meta["stat_mean"], meta["stat_std"], meta["unit"])
    te_loader = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_feeders)
    dt = gather(model, te_loader, device)
    dt["p"] = expit(dt["logit"] / T)
    mt = compute_metrics(dt["label"], dt["p"], threshold=thr)

    n = max(1, int(round(0.10 * len(dt["p"]))))
    top = set(np.argsort(-dt["p"])[:n].tolist())
    per_atk = {}
    for a in sorted(set(int(v) for v in np.unique(dt["atk"]) if v > 0)):
        idx = np.where(dt["atk"] == a)[0]
        per_atk[a] = sum(1 for i in idx if i in top) / len(idx)

    print(f"\n=== aug={aug} -> unseen-network test ===")
    print(f"AUPR={mt['aupr']:.3f}  ROC={mt['roc_auc']:.3f}  F1={mt['f1']:.3f}  "
          f"P@10%={mt['precision@10%']:.3f}  R@10%={mt['recall@10%']:.3f}")
    seen = " ".join(f"{ATTACK_NAMES[a].split()[0]}={per_atk[a]:.2f}" for a in (1, 2, 3, 4) if a in per_atk)
    uns = " ".join(f"{ATTACK_NAMES[a].split()[0]}={per_atk[a]:.2f}" for a in (5, 6, 7, 8) if a in per_atk)
    print(f"recall@10% seen   [{seen}]")
    print(f"recall@10% unseen [{uns}]")

    log_run({"script": "exp_augment", "model": "temporal-only",
             "config": f"aug={aug}", "dataset": "sim:h1-4->h1-8",
             "aug": aug, "aug_p": aug_p if aug != "none" else None,
             "n_feeders": fd_tr.n_feeders, "T": fd_tr.Y.shape[1],
             "epochs": epochs, "seed": seed, **clean_metrics(mt),
             **{f"r10_h{a}": v for a, v in per_atk.items()}})
    return mt, per_atk


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "run"])
    p.add_argument("--aug", choices=["none", "jokar", "learned", "both"], default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--aug-p", type=float, default=0.25)
    p.add_argument("--cvae-epochs", type=int, default=400)
    a = p.parse_args()
    if a.stage == "prepare":
        prepare(a.seed, a.cvae_epochs)
    else:
        run(a.aug, a.seed, a.epochs, a.aug_p)
