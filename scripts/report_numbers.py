"""
Print every number the research report needs, straight from your own files.
No training, no GPU, runs in seconds.

    python -m scripts.report_numbers
    python -m scripts.report_numbers --csv results/results.csv --rh exports/rh

Part A reads results/results.csv  (ablation per seed, energy recovery, fairness,
        augmentation experiment, rehearsal + prior sweep rows).
Part B re-scores the rehearsal rankings on TEST feeders only. The ranking CSV marks
        val+test meters as out-of-sample, but the validation feeders were used for
        early stopping, so they are not a clean hold-out. Part B rebuilds the exact
        feeder split used in training (split_feeders, seed 0) and evaluates on the
        test feeders alone, with confidence intervals.
Paste the whole output back.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ntl.data_graph import split_feeders
from ntl.results import load_latest

ABL = ["temporal", "temporal+static", "temporal+static+graph", "full", "full+ssl+aug:both"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def part_a(csv):
    df = load_latest(csv)
    pd.set_option("display.width", 200)
    print("=" * 78)
    print("A1. ablation, per seed (network sim)")
    sub = df[(df.script == "train_pistgat") & (df.dataset == "sim") & df.config.isin(ABL)]
    cols = [c for c in ["config", "seed", "aupr", "roc_auc", "r_at_10", "prevalence",
                        "temperature", "n_feeders", "T", "epochs"] if c in sub.columns]
    print(sub[cols].sort_values(["config", "seed"]).to_string(index=False))

    print("\nA2. ablation mean / sd / n")
    g = sub.groupby("config")[["aupr", "roc_auc", "r_at_10"]]
    print(pd.concat([g.mean().add_suffix("_mean"), g.std().add_suffix("_sd"),
                     g.size().rename("n")], axis=1).reindex(ABL).round(4).to_string())

    print("\nA3. paired per-seed differences (same seed => same simulated network & split)")
    piv = sub.pivot_table(index="seed", columns="config", values="aupr")
    pairs = [("temporal+static", "temporal"), ("temporal+static+graph", "temporal+static"),
             ("full", "temporal+static+graph"), ("full", "temporal")]
    for a, b in pairs:
        if a in piv and b in piv:
            d = (piv[a] - piv[b]).dropna()
            print(f"  AUPR {a:<22s} - {b:<22s}: per-seed {np.round(d.values, 4)}  "
                  f"mean={d.mean():+.4f} sd={d.std():.4f}")

    print("\nA4. energy recovery + fairness (mean over seeds)")
    ex = [c for c in df.columns if c.startswith(("er_p_", "er_ps_", "prev_", "flag_", "rec_"))]
    print(sub.groupby("config")[ex].mean().reindex(ABL).round(3).to_string())

    print("\nA5. per-attack recall@10% (mean over seeds)")
    rc = sorted([c for c in df.columns if c.startswith("r10_h")])
    print(sub.groupby("config")[rc].mean().reindex(ABL).round(3).to_string())

    print("\nA6. base models")
    b = df[df.script.isin(["train", "baselines"])]
    print(b[[c for c in ["script", "config", "dataset", "seed", "aupr", "roc_auc", "prevalence",
                         "r_at_10"] if c in b.columns]].to_string(index=False))

    print("\nA7. augmentation experiment")
    e = df[df.script == "exp_augment"]
    print(e[[c for c in ["config", "seed", "aupr", "roc_auc", "r_at_10", "prevalence"] + rc
             if c in e.columns]].to_string(index=False))

    print("\nA8. rehearsal + prior sweep (eval_ranking rows)")
    r = df[df.script == "eval_ranking"]
    print(r[[c for c in r.columns if c in ("config", "seed", "timestamp", "aupr_visible", "aupr_true",
             "roc_true", "base_rate") or c.startswith(("hit_p_", "new_p_"))]].to_string(index=False))


def part_c(csv):
    df = load_latest(csv)
    sim = df[(df.script == "train_pistgat") & (df.dataset == "sim")]
    groups = [("A9. SSL under PU (test labels complete; val labels hidden too)",
               ["full+pu0.3", "full+ssl+pu0.3"]),
              ("A10. without label signal in meter age / alarms",
               ["temporal+static+nolabelstatic", "full+nolabelstatic"]),
              ("A11. position code", ["full", "full+pos"]),
              ("A12. region removed from inputs", ["full", "full+noregion"])]
    cols = ["aupr", "roc_auc", "r_at_10", "rec_lo", "rec_hi", "flag_lo", "flag_hi", "r10_h6", "r10_h8"]
    cols = [c for c in cols if c in sim.columns]
    for title, cfgs in groups:
        print(f"\n{title}")
        g = sim[sim.config.isin(cfgs)].groupby("config")[cols]
        print(pd.concat([g.mean().round(3), g.std().add_suffix("_sd").round(3)[["aupr_sd", "roc_auc_sd"]],
                         g.size().rename("n")], axis=1).reindex(cfgs).to_string())
    e = df[df.script == "exp_augment"]
    print("\nA13. augmentation experiment, mean / sd over seeds")
    g = e.groupby("config")[[c for c in ("aupr", "roc_auc", "r_at_10", "r10_h7", "r10_h8") if c in e.columns]]
    print(pd.concat([g.mean().round(3), g.std().add_suffix("_sd").round(3), g.size().rename("n")], axis=1).to_string())


def part_b(rh, seed, n_boot):
    truth = pd.read_csv(os.path.join(rh, "truth_labels.csv")).set_index("meter_id")
    files = sorted(glob.glob(os.path.join(rh, "inspect*.csv")))
    print("\n" + "=" * 78)
    print("B. rehearsal rankings re-scored on TEST feeders only (val feeders excluded)")
    print(f"truth file: {len(truth)} meters, {int(truth.true_label.sum())} thieves "
          f"({truth.true_label.mean():.4f}); attack counts "
          f"{truth.attack_id.value_counts().sort_index().to_dict()}")
    rng = np.random.default_rng(0)
    for fpath in files:
        rk = pd.read_csv(fpath)
        tx = np.array(sorted(rk.transformer.astype(str).unique()))
        _, va_f, te_f = split_feeders(len(tx), seed=seed)
        te_tx, va_tx = set(tx[te_f]), set(tx[va_f])
        rk["true"] = rk.meter_id.map(truth["true_label"]).fillna(0).astype(int)
        rk["stolen"] = rk.meter_id.map(truth["stolen_kwh_day"]).fillna(0.0)
        print(f"\n[{os.path.basename(fpath)}] ranked meters={len(rk)} thieves among them="
              f"{int(rk['true'].sum())} ({rk['true'].mean():.4f}) confirmed={int(rk.confirmed_label.sum())}")
        oos = rk[~rk.in_train_split]
        te = rk[rk.transformer.astype(str).isin(te_tx)]
        va = rk[rk.transformer.astype(str).isin(va_tx)]
        chk = len(oos) == len(te) + len(va)
        print(f"\n--- {os.path.basename(fpath)}   feeders={len(tx)} "
              f"test feeders={len(te_tx)} val feeders={len(va_tx)}  split-consistent={chk}")
        for name, d in (("out-of-sample (val+test)", oos), ("val only", va), ("TEST only", te)):
            if d["true"].nunique() < 2:
                print(f"  {name}: single class, skipped"); continue
            ap_t = average_precision_score(d["true"], d.p_theft)
            ap_v = (average_precision_score(d.confirmed_label, d.p_theft)
                    if d.confirmed_label.nunique() > 1 else float("nan"))
            roc = roc_auc_score(d["true"], d.p_theft)
            c = d.confirmed_label.sum() / max(1, d["true"].sum())
            print(f"  {name:<26s} n={len(d):4d} thieves={int(d['true'].sum()):3d} "
                  f"confirmed={int(d.confirmed_label.sum()):3d} (c={c:.2f}) base={d['true'].mean():.3f}")
            print(f"  {'':26s} AUPR_true={ap_t:.3f} AUPR_visible={ap_v:.3f} "
                  f"ratio={ap_t / ap_v if ap_v == ap_v and ap_v > 0 else float('nan'):.2f} "
                  f"(SCAR expectation ~1/c={1 / c if c > 0 else float('nan'):.2f})  ROC_true={roc:.3f}")
            srt = d.sort_values("p_theft", ascending=False)
            for k in (0.05, 0.10, 0.15):
                n = max(1, int(round(k * len(srt))))
                hits = int(srt.head(n)["true"].sum())
                lo, hi = wilson(hits, n)
                er = srt.head(n)["stolen"].sum() / max(d["stolen"].sum(), 1e-9)
                print(f"  {'':26s} top {int(k*100):2d}%: {hits}/{n} = {hits/n:.2f} "
                      f"[95% Wilson {lo:.2f}, {hi:.2f}]  stolen-energy captured={er:.2f}")
            if name == "TEST only" and n_boot > 0:
                fe = d.transformer.astype(str).values
                uf = np.unique(fe)
                aps, rocs = [], []
                for _ in range(n_boot):
                    pick = rng.choice(uf, len(uf), replace=True)
                    idx = np.concatenate([np.where(fe == u)[0] for u in pick])
                    yy, ss = d["true"].values[idx], d.p_theft.values[idx]
                    if yy.min() == yy.max():
                        continue
                    aps.append(average_precision_score(yy, ss)); rocs.append(roc_auc_score(yy, ss))
                print(f"  {'':26s} feeder-cluster bootstrap ({len(aps)} draws): "
                      f"AUPR_true 95% [{np.percentile(aps, 2.5):.3f}, {np.percentile(aps, 97.5):.3f}]  "
                      f"ROC_true 95% [{np.percentile(rocs, 2.5):.3f}, {np.percentile(rocs, 97.5):.3f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--rh", default="exports/rh")
    ap.add_argument("--seed", type=int, default=0, help="seed used for the rehearsal training")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--skip-table", action="store_true", help="only re-score the rehearsal")
    a = ap.parse_args()
    if a.skip_table:
        pass
    elif os.path.exists(a.csv):
        part_a(a.csv)
        part_c(a.csv)
    else:
        print(f"[skip A] {a.csv} not found")
    if os.path.exists(os.path.join(a.rh, "truth_labels.csv")):
        part_b(a.rh, a.seed, a.boot)
    else:
        print(f"[skip B] {a.rh}/truth_labels.csv not found")
