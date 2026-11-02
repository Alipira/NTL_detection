"""
Train + evaluate the base 1D-CNN theft detector.

Runs out of the box on synthetic data:
    python -m scripts.train
Train on real data (SGCC-style CSV: rows=customers, a `label` column):
    python -m scripts.train --data path/to/data.csv --label-col FLAG --id-col CONS_NO
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ntl.data import generate_synthetic, load_csv, make_loaders
from ntl.losses import FocalLoss, weighted_bce
from ntl.metrics import best_f1_threshold, compute_metrics, format_metrics
from ntl.results import clean_metrics, log_run
from ntl.model import build_base_model


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        logit = model(x.to(device))
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ----
    if args.data == "synthetic":
        X, y, _ = generate_synthetic(
            n_customers=args.n, T=args.T, theft_ratio=args.theft_ratio, seed=args.seed
        )
        print(f"[data] synthetic: {X.shape[0]} customers, T={X.shape[1]}, "
              f"theft={y.mean():.1%}")
    else:
        X, y = load_csv(args.data, label_col=args.label_col, id_col=args.id_col)
        print(f"[data] {args.data}: {X.shape[0]} customers, T={X.shape[1]}, "
              f"theft={y.mean():.1%}")

    train_loader, val_loader, test_loader, _, _ = make_loaders(
        X, y, batch_size=args.batch, augment=args.augment, aug_p=args.aug_p, seed=args.seed
    )

    # ---- model + loss ----
    model = build_base_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] base CNN, {n_params:,} params")

    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
        criterion = weighted_bce(pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---- train loop with best-val-AUPR checkpointing ----
    best_aupr, best_state, best_thr = -1.0, None, 0.5
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, yb in train_loader:
            x, yb = x.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(x), yb)
            loss.backward()
            optim.step()
            running += loss.item() * len(yb)
        train_loss = running / len(train_loader.dataset)

        val_scores, val_labels = predict(model, val_loader, device)
        val = compute_metrics(val_labels, val_scores)
        tag = ""
        if val["aupr"] > best_aupr:
            best_aupr = val["aupr"]
            best_thr = best_f1_threshold(val_labels, val_scores)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            tag = "  <- best"
        print(f"[epoch {epoch:02d}] loss={train_loss:.4f}  "
              f"val_aupr={val['aupr']:.4f}  val_roc={val['roc_auc']:.4f}{tag}")

    # ---- test with the val-selected operating point ----
    if best_state is not None:
        model.load_state_dict(best_state)
    test_scores, test_labels = predict(model, test_loader, device)
    test = compute_metrics(test_labels, test_scores, threshold=best_thr)

    print("\n=== TEST (threshold selected on validation) ===")
    print(format_metrics(test))

    os.makedirs(args.out, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "threshold": best_thr}, os.path.join(args.out, "base_cnn.pt"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(test, f, indent=2)
    print(f"\n[saved] {args.out}/base_cnn.pt  and  metrics.json")

    log_run({"script": "train", "model": "base-CNN", "config": "base-cnn",
             "dataset": "synthetic-percustomer" if args.data == "synthetic" else args.data,
             "n": X.shape[0], "T": X.shape[1], "epochs": args.epochs, "seed": args.seed,
             "aug": "jokar" if args.augment else "none", **clean_metrics(test)})
    return test


def build_argparser():
    p = argparse.ArgumentParser(description="Base 1D-CNN NTL detector")
    p.add_argument("--data", default="synthetic", help="'synthetic' or path to CSV")
    p.add_argument("--label-col", default="label")
    p.add_argument("--id-col", default=None)
    p.add_argument("--n", type=int, default=4000, help="synthetic: customers")
    p.add_argument("--T", type=int, default=365, help="synthetic: series length")
    p.add_argument("--theft-ratio", type=float, default=0.09)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=["focal", "bce"], default="focal")
    p.add_argument("--focal-alpha", type=float, default=0.75)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--augment", action="store_true", default=True,
                   help="on-the-fly Jokar augmentation (train only)")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--aug-p", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/base")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
