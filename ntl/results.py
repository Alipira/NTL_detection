"""
Central results table.

Every entry-point script (train.py, baselines.py, train_pistgat.py,
exp_augment.py) appends one row per run to ``results/results.csv`` via
``log_run``. The table accumulates across runs, so all ablations, the SSL/PU
comparisons, and the augmentation experiment live side by side. Plot it with::

    python -m scripts.plot_results

Conventions
- Columns are a union: rows simply leave blank what doesn't apply (e.g. the
  XGBoost row has no SSL column).
- ``config`` is a short human label ("temporal", "+graph", "full+ssl+aug", ...)
  used as the x-axis in plots; ``pistgat_label`` builds it from the flags.
- Re-running a config appends a new row. Per (script, config, dataset, seed, prior)
  the plot script keeps the largest run (n_feeders, T, n, epochs, ssl_epochs), then
  the most recent, so a small test run never replaces a full one. load_latest also
  drops the SSL epoch count from labels ("+ssl8" -> "+ssl"), so plots and tables
  find those runs at any setting.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

CSV_DEFAULT = os.path.join("results", "results.csv")

# nicer column names for the @k metrics coming out of compute_metrics()
_RENAME = {
    "precision@1%": "p_at_1", "recall@1%": "r_at_1",
    "precision@5%": "p_at_5", "recall@5%": "r_at_5",
    "precision@10%": "p_at_10", "recall@10%": "r_at_10",
    "precision@20%": "p_at_20", "recall@20%": "r_at_20",
}


def clean_metrics(m: dict) -> dict:
    return {_RENAME.get(k, k): v for k, v in m.items()}


def pistgat_label(args) -> str:
    """Short config label from train_pistgat args, e.g. 'full+ssl+aug:both'."""
    on = [not args.no_static, not args.no_graph, not args.no_physics]
    if not any(on):
        base = "temporal"
    elif all(on):
        base = "full"
    else:
        base = "temporal" + ("+static" if on[0] else "") + ("+graph" if on[1] else "") \
               + ("+physics" if on[2] else "")
    if args.ssl_epochs > 0:
        base += f"+ssl{args.ssl_epochs}"
    if args.aug != "none":
        base += f"+aug:{args.aug}"
    if args.pu:
        base += f"+pu{args.pu_reveal:g}"
    if args.pos_enc:
        base += "+pos"
    if args.no_label_static:
        base += "+nolabelstatic"
    if args.no_region_feature:
        base += "+noregion"
    return base


def log_run(record: dict, csv_path: str = CSV_DEFAULT) -> str:
    """Append one run to the central CSV (created on first use)."""
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new = pd.DataFrame([row]).dropna(axis=1, how="all")   # avoids pandas concat FutureWarning
    if os.path.exists(csv_path):
        df = pd.concat([pd.read_csv(csv_path), new], ignore_index=True)
    else:
        df = new
    df.to_csv(csv_path, index=False)
    print(f"[results] appended to {csv_path}  ({len(df)} rows total)")
    return csv_path


def load_latest(csv_path: str = CSV_DEFAULT) -> pd.DataFrame:
    """Read the table, keeping the largest, then most recent, row per configuration."""
    df = pd.read_csv(csv_path)
    keys = [c for c in ("script", "config", "dataset", "seed", "prior") if c in df.columns]
    df["config"] = df["config"].astype(str).str.replace(r"\+ssl\d+", "+ssl", regex=True)
    size = [c for c in ("n_feeders", "T", "n", "epochs", "ssl_epochs") if c in df.columns]
    rank = df[size].fillna(-1).assign(timestamp=df["timestamp"])
    order = rank.sort_values(size + ["timestamp"]).index
    return (df.loc[order]
              .drop_duplicates(subset=keys, keep="last")
              .reset_index(drop=True))
