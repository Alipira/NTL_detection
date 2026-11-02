"""
Generate figures from the central results table.

    python -m scripts.plot_results                  # reads results/results.csv
    python -m scripts.plot_results --csv path.csv --out results/figures
    python -m scripts.plot_results --seed 3 --out figures/seed_3   # one seed only

Produces (skipping any figure whose rows are missing):
  fig1_model_comparison.png   base CNN vs XGBoost vs PI-STGAT (AUPR / ROC-AUC)
  fig2_ablation.png           PI-STGAT component ablation
  fig3_attack_recall.png      recall@10% per attack type, per ablation config
  fig4_energy_recovery.png    energy recovered @ top-k, rank by P vs P*stolen
  fig5_fairness.png           flag rate & recall by region group, per config
  fig6_augmentation.png       augmentation experiment, overall metrics
  fig7_aug_per_attack.png     augmentation experiment, seen vs unseen attacks
  fig8_rehearsal.png          real-data pipeline rehearsal against hidden truth
  fig8_prior_default.png      the same, with the PU prior left at the confirmed rate

If the same configuration appears several times, the largest, then most recent,
run wins (see ntl.results.load_latest).
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ntl.results import CSV_DEFAULT, load_latest

ABL_ORDER = ["temporal", "temporal+static", "temporal+static+graph", "full"]
ABL_NICE = {"temporal": "temporal\nonly", "temporal+static": "+ static",
            "temporal+static+graph": "+ graph", "full": "full\n(+physics)"}
REPORT_ORDER = ABL_ORDER + ["full+ssl+aug:both"]   # side experiments stay in report_numbers.py
AUG_ORDER = ["aug=none", "aug=jokar", "aug=learned", "aug=both"]
AUG_NICE = {"aug=none": "none", "aug=jokar": "fixed\n(Jokar)",
            "aug=learned": "learned\n(cVAE)", "aug=both": "both\n(50/50)"}
ATTACKS = [f"r10_h{i}" for i in range(1, 9)]
ATTACK_NICE = ["h1\nscale", "h2\nzero-win", "h3\nper-step", "h4\nfrac-mean",
               "h5\nflat-mean", "h6\nreverse", "h7\nseasonal", "h8\nramp"]


def _annotate(ax, bars, fmt="{:.3f}", fs=8):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=fs)


def _grouped_bars(ax, labels, series: dict, width=None, fmt="{:.3f}", errs=None):
    x = np.arange(len(labels))
    n = len(series)
    width = width or 0.8 / n
    for i, (name, vals) in enumerate(series.items()):
        yerr = errs.get(name) if errs else None
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=name,
                      yerr=yerr, capsize=3 if yerr is not None else 0)
        _annotate(ax, bars, fmt)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.12)


def _matrix(ax, M, row_labels, col_labels, title):
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "-" if not np.isfinite(v) else f"{v:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if (not np.isfinite(v) or 0.25 < v < 0.8) else "white")
    ax.set_title(title)
    return im


def _rows_stats(df, script, configs):
    """Mean / std / n across seeds for each config (latest run per seed)."""
    sub = df[(df.script == script) & (df.config.isin(list(configs)))]
    if not len(sub):
        e = sub.set_index("config")
        return e, e, pd.Series(dtype=int)
    g = sub.groupby("config")
    mean = g.mean(numeric_only=True)
    std = g.std(numeric_only=True).fillna(0.0)
    n = g.size()
    order = [c for c in configs if c in mean.index]
    return mean.loc[order], std.loc[order], n.loc[order]


def _rows(df, script, configs):
    mean, _, _ = _rows_stats(df, script, configs)
    return mean


def fig1_models(df, out):
    def pick(script, cfg, dataset):
        sub = df[(df.script == script) & (df.config == cfg)]
        sub = sub[sub.dataset.astype(str).map(os.path.basename) == dataset]
        return sub.mean(numeric_only=True) if len(sub) else None   # mean across seeds

    spec = [
        ("base CNN\n(easy per-cust.)",    pick("train", "base-cnn", "synthetic-percustomer")),
        ("XGBoost\n(easy per-cust.)",     pick("baselines", "xgboost", "synthetic-percustomer")),
        ("base CNN\n(network sim)",       pick("train", "base-cnn", "sim_flat.csv")),
        ("PI-STGAT full\n(network sim)",  pick("train_pistgat", "full", "sim")),
    ]
    rows = [(lbl, r) for lbl, r in spec if r is not None]
    if len(rows) < 2:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    _grouped_bars(ax, [lbl for lbl, _ in rows],
                  {"AUPR": [r.aupr for _, r in rows],
                   "ROC-AUC": [r.roc_auc for _, r in rows]})
    ax.set_title("Model comparison — left pair: easy generator; right pair: network simulator.\n"
                 "Like-for-like reads are within a dataset (AUPR's floor = each test set's prevalence).")
    fig.tight_layout(); p = os.path.join(out, "fig1_model_comparison.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig2_ablation(df, out):
    df = df[(df.script != "train_pistgat") | (df.dataset == "sim")]
    mean, std, n = _rows_stats(df, "train_pistgat", ABL_ORDER)
    sub = mean.dropna(subset=["aupr"]) if len(mean) else mean
    if len(sub) < 2:
        return None
    std = std.loc[sub.index]
    labels = [ABL_NICE[c] for c in sub.index]
    multi = int(n.max()) > 1
    errs = ({"AUPR": std.aupr.values, "ROC-AUC": std.roc_auc.values,
             "recall@10%": std.r_at_10.values} if multi else None)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    _grouped_bars(ax, labels, {"AUPR": sub.aupr.values, "ROC-AUC": sub.roc_auc.values,
                               "recall@10%": sub.r_at_10.values}, errs=errs)
    extra = f" -- mean ± std over {int(n.max())} seeds" if multi else ""
    ax.set_title("PI-STGAT ablation: each branch added in turn (network sim)" + extra)
    fig.tight_layout(); p = os.path.join(out, "fig2_ablation.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig3_attacks(df, out):
    df = df[(df.script != "train_pistgat") | (df.dataset == "sim")]
    sub = _rows(df, "train_pistgat", REPORT_ORDER)
    if not len(sub):
        return None
    cols = [c for c in ATTACKS if c in sub.columns]
    M = sub[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(sub) + 1.8))
    im = _matrix(ax, M, [ABL_NICE.get(c, c).replace("\n", " ") for c in sub.index],
                 [ATTACK_NICE[ATTACKS.index(c)] for c in cols],
                 "recall@10% by attack type ('-' = attack absent in test split)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); p = os.path.join(out, "fig3_attack_recall.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig4_energy(df, out):
    sub = df[(df.script == "train_pistgat") & (df.get("physics") == True)
             & (df.dataset == "sim") & df.config.isin(REPORT_ORDER)]  # noqa: E712
    if not len(sub):
        return None
    sub = sub.groupby("config").mean(numeric_only=True)            # mean across seeds
    ks = [5, 10, 20]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for lbl, r in sub.iterrows():
        ax.plot(ks, [r[f"er_p_{k}"] for k in ks], "-o", label=f"{lbl}: rank by P")
        ax.plot(ks, [r[f"er_ps_{k}"] for k in ks], "--s",
                label=f"{lbl}: rank by P×stolen")
    ax.set_xticks(ks); ax.set_xlabel("inspected top-k (%)")
    ax.set_ylabel("share of stolen energy recovered")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Cost-aware ranking via the physics head")
    fig.tight_layout(); p = os.path.join(out, "fig4_energy_recovery.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig5_fairness(df, out):
    df = df[(df.script != "train_pistgat") | (df.dataset == "sim")]
    sub = _rows(df, "train_pistgat", REPORT_ORDER).dropna(subset=["flag_lo"])
    if not len(sub):
        return None
    labels = [ABL_NICE.get(c, c).replace("\n", " ") for c in sub.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _grouped_bars(axes[0], labels, {"region<0.5": sub.flag_lo.values,
                                    "region≥0.5": sub.flag_hi.values}, fmt="{:.2f}")
    axes[0].set_title("flag rate by region group")
    # prevalence reference (theft really is unevenly distributed in the sim)
    axes[0].axhline(float(sub.prev_lo.iloc[0]), ls=":", c="C0", lw=1)
    axes[0].axhline(float(sub.prev_hi.iloc[0]), ls=":", c="C1", lw=1)
    axes[0].text(0.99, float(sub.prev_hi.iloc[0]) + .01, "true prevalence",
                 transform=axes[0].get_yaxis_transform(), ha="right", fontsize=8)
    _grouped_bars(axes[1], labels, {"region<0.5": sub.rec_lo.values,
                                    "region≥0.5": sub.rec_hi.values}, fmt="{:.2f}")
    axes[1].set_title("recall by region group")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Fairness check (dotted = each group's true theft prevalence)")
    fig.tight_layout(); p = os.path.join(out, "fig5_fairness.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig6_augmentation(df, out):
    sub = _rows(df, "exp_augment", AUG_ORDER).dropna(subset=["aupr"])
    if len(sub) < 2:
        return None
    labels = [AUG_NICE[c] for c in sub.index]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    _grouped_bars(ax, labels, {"AUPR": sub.aupr.values, "ROC-AUC": sub.roc_auc.values,
                               "recall@10%": sub.r_at_10.values})
    ax.set_title("Augmentation → unseen-network test\n"
                 "(train: h1–h4 only, few confirmed pairs; test: independent network, h1–h8)")
    fig.tight_layout(); p = os.path.join(out, "fig6_augmentation.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig7_aug_attacks(df, out):
    sub = _rows(df, "exp_augment", AUG_ORDER)
    if not len(sub):
        return None
    cols = [c for c in ATTACKS if c in sub.columns]
    M = sub[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(sub) + 2.0))
    im = _matrix(ax, M, [AUG_NICE[c].replace("\n", " ") for c in sub.index],
                 [ATTACK_NICE[ATTACKS.index(c)] for c in cols],
                 "recall@10% per attack — left of line: seen in training (h1–h4); right: unseen")
    ax.axvline(3.5, color="black", lw=2)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); p = os.path.join(out, "fig7_aug_per_attack.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def fig8_rehearsal(df, out, tag="rehearsal", name="fig8_rehearsal.png"):
    sub = df[(df.script == "eval_ranking") & (df.config == tag)]
    if not len(sub):
        return None
    r = sub.mean(numeric_only=True)                            # mean across seeds
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    vals = [r.aupr_visible, r.aupr_true, r.roc_true]
    b = axes[0].bar(["AUPR vs\nconfirmed-only", "AUPR vs\nfull truth", "ROC-AUC vs\nfull truth"],
                    vals, color=["#c44e52", "#4c72b0", "#4c72b0"])
    _annotate(axes[0], b)
    axes[0].set_ylim(0, 1.05); axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_title("PU evaluation bias\n(confirmed-only labels understate the model)")
    ks = [k for k in (5, 10, 15) if f"hit_p_{k}" in r.index and np.isfinite(r[f"hit_p_{k}"])]
    hv = [float(r[f"hit_p_{k}"]) for k in ks]
    b2 = axes[1].bar([f"top {k}%" for k in ks], hv, color="#55a868")
    _annotate(axes[1], b2, "{:.0%}")
    axes[1].axhline(float(r.base_rate), ls="--", c="k", lw=1)
    axes[1].text(0.99, float(r.base_rate) + 0.01, "base rate",
                 transform=axes[1].get_yaxis_transform(), ha="right", fontsize=8)
    axes[1].set_ylim(0, max(0.5, max(hv + [float(r.base_rate)]) * 1.35))
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_title("inspection hit-rate by P(theft)\n(out-of-sample meters, new catches incl.)")
    fig.suptitle("Rehearsal: real-data pipeline on simulator export with hidden truth")
    fig.tight_layout()
    p = os.path.join(out, name)
    fig.savefig(p, dpi=150); plt.close(fig); return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--out", default=os.path.join("results", "figures"))
    ap.add_argument("--seed", type=int, default=None, help="plot one seed only")
    args = ap.parse_args()

    df = load_latest(args.csv)
    if args.seed is not None:
        df = df[df.seed == args.seed]
    os.makedirs(args.out, exist_ok=True)
    print(f"[plot] {len(df)} unique runs from {args.csv}")
    sim = df[(df.script == "train_pistgat") & (df.dataset == "sim")]
    sizes = sim[["n_feeders", "T", "epochs"]].drop_duplicates() if len(sim) else sim
    if len(sizes) > 1:
        print(f"[plot][WARN] sim runs use {len(sizes)} different sizes "
              f"(n_feeders, T, epochs); bars may not be comparable")

    made = [f(df, args.out) for f in (fig1_models, fig2_ablation, fig3_attacks,
                                      fig4_energy, fig5_fairness,
                                      fig6_augmentation, fig7_aug_attacks,
                                      fig8_rehearsal)]
    made.append(fig8_rehearsal(df, args.out, "rehearsal-prior-default", "fig8_prior_default.png"))
    for p in made:
        print(f"  {'wrote ' + p if p else 'skipped (missing rows)'}")


if __name__ == "__main__":
    main()
