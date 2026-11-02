"""
Evaluate an inspection ranking against hidden ground truth.

    python -m scripts.eval_ranking --ranking inspect.csv --truth truth_labels.csv

Reports, on OUT-OF-SAMPLE meters (in_train_split == False):
  - AUPR/ROC against the FULL truth vs against confirmed-only labels --
    the gap quantifies how much PU evaluation understates the model,
  - hit rate and new catches (true thieves absent from labels.csv) in the
    top-k of the list, for both ranking keys (P and P x stolen),
  - estimated stolen energy actually sitting in the top-k.

On real data there is no truth file; this script is for simulator rehearsals
(make_fake_export.py) and, later, for scoring a finished inspection campaign
by passing the campaign outcomes as the truth file.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ntl.results import log_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--k", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    ap.add_argument("--tag", default="rehearsal", help="config label in results.csv")
    ap.add_argument("--seed", type=int, default=0, help="seed of the training run")
    a = ap.parse_args()

    rank = pd.read_csv(a.ranking)
    truth = pd.read_csv(a.truth).set_index("meter_id")
    rank["true"] = rank.meter_id.map(truth["true_label"]).fillna(0).astype(int)
    oo = rank[~rank.in_train_split].copy()
    if oo["true"].sum() == 0:
        raise SystemExit("no true thieves among out-of-sample meters -- "
                         "check meter_id values match between the two files")

    print(f"out-of-sample meters: {len(oo)} | true thieves: {int(oo['true'].sum())} "
          f"({oo['true'].mean():.1%}) | confirmed visible: {int(oo.confirmed_label.sum())}")
    print(f"AUPR vs confirmed-only (what PU evaluation sees): "
          f"{average_precision_score(oo.confirmed_label, oo.p_theft):.3f}")
    print(f"AUPR vs full ground truth                       : "
          f"{average_precision_score(oo['true'], oo.p_theft):.3f}")
    print(f"ROC-AUC vs full ground truth                    : "
          f"{roc_auc_score(oo['true'], oo.p_theft):.3f}")

    rec = {"script": "eval_ranking", "config": a.tag, "dataset": "fake-real", "seed": a.seed,
           "aupr_visible": float(average_precision_score(oo.confirmed_label, oo.p_theft)),
           "aupr_true": float(average_precision_score(oo["true"], oo.p_theft)),
           "roc_true": float(roc_auc_score(oo["true"], oo.p_theft)),
           "base_rate": float(oo["true"].mean())}
    srt_p = oo.sort_values("p_theft", ascending=False)
    for k in a.k:
        top = srt_p.head(max(1, int(round(k * len(srt_p)))))
        rec[f"hit_p_{int(k*100)}"] = float(top["true"].mean())
        rec[f"new_p_{int(k*100)}"] = int((top["true"].astype(bool)
                                          & (top.confirmed_label == 0)).sum())
    log_run(rec)

    for key in ("p_theft", "score_p_x_stolen"):
        srt = oo.sort_values(key, ascending=False)
        print(f"\nranked by {key}:")
        for k in a.k:
            top = srt.head(max(1, int(round(k * len(srt)))))
            new = int((top["true"].astype(bool) & (top.confirmed_label == 0)).sum())
            stolen = truth["stolen_kwh_day"].reindex(top.meter_id).fillna(0).sum()
            tot = truth["stolen_kwh_day"].reindex(oo.meter_id).fillna(0).sum()
            print(f"  top {k:>4.0%}: hit-rate {top['true'].mean():.0%} "
                  f"(base {oo['true'].mean():.0%}) | new catches {new} | "
                  f"stolen energy captured {stolen / max(tot, 1e-9):.0%}")


if __name__ == "__main__":
    main()
