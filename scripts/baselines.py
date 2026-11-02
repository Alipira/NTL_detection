"""
Classical baseline: XGBoost on hand-engineered per-customer features.

This is the floor every NTL paper is expected to beat (it is the family that
won in Buzau et al., 2018). Reporting it next to the CNN shows the deep model
earns its complexity. Runs on the same synthetic data and the same metrics.

    python -m scripts.baselines                 # synthetic
    python -m scripts.baselines --data x.csv --label-col FLAG --id-col CONS_NO
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import stats
from xgboost import XGBClassifier

from ntl.data import generate_synthetic, impute, load_csv, stratified_split
from ntl.metrics import best_f1_threshold, compute_metrics, format_metrics
from ntl.results import clean_metrics, log_run


def engineer(X: np.ndarray) -> np.ndarray:
    """Summary statistics that capture theft signatures: low/flat/zero-heavy
    consumption, irregular variability, weak day-to-day correlation."""
    X = impute(X)
    feats = [
        X.mean(1), X.std(1), X.min(1), X.max(1), np.median(X, axis=1),
        (X == 0).mean(1),                                   # fraction of zero days
        stats.skew(X, axis=1), stats.kurtosis(X, axis=1),
        X.std(1) / (X.mean(1) + 1e-6),                      # coeff. of variation
    ]
    # lag-1 autocorrelation per row
    Xc = X - X.mean(1, keepdims=True)
    num = (Xc[:, 1:] * Xc[:, :-1]).sum(1)
    den = (Xc ** 2).sum(1) + 1e-6
    feats.append(num / den)
    return np.stack(feats, axis=1).astype(np.float32)


def run(args):
    if args.data == "synthetic":
        X, y, _ = generate_synthetic(n_customers=args.n, T=args.T,
                                     theft_ratio=args.theft_ratio, seed=args.seed)
    else:
        X, y = load_csv(args.data, label_col=args.label_col, id_col=args.id_col)
    print(f"[data] {X.shape[0]} customers, T={X.shape[1]}, theft={y.mean():.1%}")

    F = engineer(X)
    tr, va, te = stratified_split(y, seed=args.seed)
    pos_w = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())

    clf = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_w, eval_metric="aucpr",
        n_jobs=4, random_state=args.seed,
    )
    clf.fit(F[tr], y[tr])

    val_scores = clf.predict_proba(F[va])[:, 1]
    thr = best_f1_threshold(y[va], val_scores)
    test_scores = clf.predict_proba(F[te])[:, 1]
    m = compute_metrics(y[te], test_scores, threshold=thr)

    print("\n=== XGBoost baseline -- TEST ===")
    print(format_metrics(m))

    log_run({"script": "baselines", "model": "XGBoost", "config": "xgboost",
             "dataset": "synthetic-percustomer" if args.data == "synthetic" else args.data,
             "n": X.shape[0], "T": X.shape[1], "seed": args.seed, **clean_metrics(m)})
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="XGBoost NTL baseline")
    p.add_argument("--data", default="synthetic")
    p.add_argument("--label-col", default="label")
    p.add_argument("--id-col", default=None)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--T", type=int, default=365)
    p.add_argument("--theft-ratio", type=float, default=0.09)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())
