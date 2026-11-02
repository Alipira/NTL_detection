"""
Full-dress rehearsal: turn the network simulator into a MESSY utility export.

Purpose: test the ENTIRE real-data pipeline (loader, diagnostics, PU training,
calibration, ranking export, evaluation) on files with the same schema and the
same defects as a real export -- but with ground truth still known, so every
claim can be checked. This is the difference between "the code runs" and "the
code is trustworthy on imperfect data".

Injected defects (all configurable):
  --missing-frac   meters with random multi-day gaps (NaN spans)
  --blank-frac     share of those missing readings exported as rows with a blank
                   kwh, as real exports do; the rest are absent rows
  --late-frac      meters that join mid-window (no history before join date)
  --dead-frac      near-empty meters       -> must be dropped by coverage filter
  --unmapped-frac  meters absent from topology.csv -> must be dropped + counted
  --wrong-topo     meters assigned to the WRONG transformer -> should surface as
                   negative-residual feeder warnings in the loader diagnostics
  --unmetered-kwh  constant unmetered load (street lighting) added to each head
                   -> inflates residuals; the tl auto-estimate partially absorbs
                   it, which is exactly the bias to understand before real runs

Writes to --out-dir:
  meters.csv  topology.csv  heads.csv  labels.csv  static.csv     (the "export")
  truth_labels.csv   meter_id, true_label, attack_id, stolen_kwh_day  (hidden truth)

Then:
  python -m scripts.train_pistgat --data real --meters .../meters.csv \\
      --topology .../topology.csv --heads .../heads.csv --labels .../labels.csv \\
      --static-file .../static.csv --region-col district \\
      --save-ranking .../inspect.csv
  python -m scripts.eval_ranking --ranking .../inspect.csv --truth .../truth_labels.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ntl.simulator import simulate_network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="exports/rehearsal")
    ap.add_argument("--n-feeders", type=int, default=60)
    ap.add_argument("--T", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reveal", type=float, default=0.3,
                    help="fraction of true thieves confirmed in labels.csv")
    ap.add_argument("--missing-frac", type=float, default=0.10)
    ap.add_argument("--blank-frac", type=float, default=0.3)
    ap.add_argument("--late-frac", type=float, default=0.05)
    ap.add_argument("--dead-frac", type=float, default=0.01)
    ap.add_argument("--unmapped-frac", type=float, default=0.01)
    ap.add_argument("--wrong-topo", type=float, default=0.01)
    ap.add_argument("--unmetered-kwh", type=float, default=0.0,
                    help="constant unmetered feeder load added to every head (kWh/day)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    fd, info = simulate_network(n_feeders=a.n_feeders, T=a.T, seed=a.seed)
    n, T = fd.Y.shape
    dates = pd.date_range("2025-01-01", periods=T, freq="D")
    mid = np.array([f"M{i:05d}" for i in range(n)])
    tx_of = np.array([f"TX{f:03d}" for f in fd.feeder_id])
    tx_all = [f"TX{f:03d}" for f in range(fd.n_feeders)]

    # ---- meter series with realistic gaps ----
    Y = fd.Y.astype(float).copy()
    miss = rng.choice(n, int(a.missing_frac * n), replace=False)
    for i in miss:
        s = rng.integers(0, T - 21); Y[i, s:s + rng.integers(3, 21)] = np.nan
    late = rng.choice(np.setdiff1d(np.arange(n), miss), int(a.late_frac * n), replace=False)
    for i in late:
        Y[i, :rng.integers(int(0.05 * T), int(0.30 * T))] = np.nan
    dead = rng.choice(np.setdiff1d(np.arange(n), np.concatenate([miss, late])),
                      max(1, int(a.dead_frac * n)), replace=False)
    for i in dead:
        Y[i, rng.integers(0, T // 10):] = np.nan          # ~dies early -> coverage drop
    long = pd.DataFrame({"meter_id": np.repeat(mid, T), "date": np.tile(dates, n),
                         "kwh": Y.reshape(-1)})
    long = long[long.kwh.notna() | (rng.random(len(long)) < a.blank_frac)]
    long.to_csv(f"{a.out_dir}/meters.csv", index=False)

    # ---- topology with realistic errors ----
    topo = pd.DataFrame({"meter_id": mid, "transformer_id": tx_of})
    unmapped = rng.choice(n, max(1, int(a.unmapped_frac * n)), replace=False)
    wrong = rng.choice(np.setdiff1d(np.arange(n), unmapped),
                       max(1, int(a.wrong_topo * n)), replace=False)
    for i in wrong:
        topo.loc[i, "transformer_id"] = rng.choice([t for t in tx_all if t != tx_of[i]])
    topo = topo.drop(index=list(unmapped))
    topo.to_csv(f"{a.out_dir}/topology.csv", index=False)

    # ---- head readings (+ optional unmetered load, + measurement noise) ----
    H = info.heads + a.unmetered_kwh
    H = H * (1.0 + rng.normal(0, 0.01, size=H.shape))
    heads = pd.DataFrame({"transformer_id": np.repeat(tx_all, T),
                          "date": np.tile(dates, fd.n_feeders),
                          "kwh": H.reshape(-1)})
    heads.to_csv(f"{a.out_dir}/heads.csv", index=False)

    # ---- PU labels + static ----
    pos = np.where(fd.y == 1)[0]
    conf = rng.choice(pos, int(round(a.reveal * len(pos))), replace=False)
    pd.DataFrame({"meter_id": mid[conf], "label": 1}).to_csv(
        f"{a.out_dir}/labels.csv", index=False)
    pd.DataFrame({"meter_id": mid, "contracted_kw": fd.S[:, 0],
                  "meter_age": fd.S[:, 1],
                  "district": np.where(fd.region < 0.5, "north", "south"),
                  "tariff": np.argmax(fd.S[:, 3:6], 1)}).to_csv(
        f"{a.out_dir}/static.csv", index=False)

    # ---- hidden truth for evaluation ----
    pd.DataFrame({"meter_id": mid, "true_label": fd.y, "attack_id": fd.atk,
                  "stolen_kwh_day": fd.stolen}).to_csv(
        f"{a.out_dir}/truth_labels.csv", index=False)

    print(f"[export] {a.out_dir}/: {n} meters x {T} days, {fd.n_feeders} transformers")
    print(f"[export] true thieves={len(pos)} ({fd.y.mean():.1%}); "
          f"confirmed in labels.csv={len(conf)} ({a.reveal:.0%} reveal)")
    print(f"[export] defects: {len(miss)} gap meters, {len(late)} late joiners, "
          f"{len(dead)} dead, {len(unmapped)} unmapped, {len(wrong)} wrong-topology, "
          f"unmetered={a.unmetered_kwh} kWh/day per feeder")
    print(f"[next]  python -m scripts.train_pistgat --data real --meters {a.out_dir}/meters.csv "
          f"--topology {a.out_dir}/topology.csv --heads {a.out_dir}/heads.csv "
          f"--labels {a.out_dir}/labels.csv --static-file {a.out_dir}/static.csv "
          f"--region-col district --days {T} --save-ranking {a.out_dir}/inspect.csv")
    print(f"[next]  python -m scripts.eval_ranking --ranking {a.out_dir}/inspect.csv "
          f"--truth {a.out_dir}/truth_labels.csv")


if __name__ == "__main__":
    main()
