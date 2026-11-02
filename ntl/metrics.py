"""
Evaluation metrics for NTL detection.

Under a few-percent theft prevalence, accuracy and ROC-AUC are misleading, so
the suite leads with the metrics the literature argues for:
  - AUPR (average precision)        -- proper under heavy imbalance
  - precision@k / recall@k          -- a utility inspects a top-k list
  - Bayesian detection rate (BDR)   -- P(theft | flagged), the FPR-cost view
We still report ROC-AUC, F1, precision, recall for comparability with prior work.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold maximising F1 on the precision-recall curve."""
    p, r, thr = precision_recall_curve(y_true, y_score)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    if len(thr) == 0:
        return 0.5
    return float(thr[max(0, np.argmax(f1[:-1]))])


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    """Of the top k-fraction of customers by score, the fraction truly stealing."""
    n = max(1, int(round(k_frac * len(y_true))))
    top = np.argsort(-y_score)[:n]
    return float(np.mean(y_true[top]))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    """Of all thieves, the fraction caught within the top k-fraction inspected."""
    n = max(1, int(round(k_frac * len(y_true))))
    top = np.argsort(-y_score)[:n]
    pos = max(1, int(y_true.sum()))
    return float(y_true[top].sum() / pos)


def bayesian_detection_rate(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    """BDR = P(theft | flagged) = TP / (TP + FP) at the operating threshold."""
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
    k_fracs=(0.01, 0.05, 0.10, 0.20),
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if threshold is None:
        threshold = best_f1_threshold(y_true, y_score)
    pred = (y_score >= threshold).astype(int)

    out = {
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "aupr": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "bdr": bayesian_detection_rate(y_true, y_score, threshold),
    }
    for k in k_fracs:
        out[f"precision@{int(k*100)}%"] = precision_at_k(y_true, y_score, k)
        out[f"recall@{int(k*100)}%"] = recall_at_k(y_true, y_score, k)
    return out


def format_metrics(m: dict) -> str:
    order = ["aupr", "roc_auc", "f1", "precision", "recall", "bdr",
             "prevalence", "threshold"]
    lines = ["  metric           value"]
    for k in order:
        if k in m:
            lines.append(f"  {k:<15s}  {m[k]:.4f}")
    lines.append("  --- top-k inspection list ---")
    for k in [k for k in m if k.startswith("precision@")]:
        kk = k.split("@")[1]
        lines.append(f"  P@{kk:<5s} {m[k]:.4f}   R@{kk:<5s} {m['recall@'+kk]:.4f}")
    return "\n".join(lines)
