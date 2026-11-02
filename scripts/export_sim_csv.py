"""
Export the network simulator to the base model's per-customer CSV format.

Why this exists: fig1 compares models across DIFFERENT datasets (the base CNN
runs on the easy per-customer generator; PI-STGAT runs on the harder network
simulator), so its AUPR bars are context, not a like-for-like claim. This
script fills the missing cell -- "base CNN on simulator data" -- by flattening
the simulator's reported series into the CSV shape train.py already accepts
(one customer per row + a `label` column, same as SGCC).

    python -m scripts.export_sim_csv                      # -> sim_flat.csv
    python -m scripts.train --data sim_flat.csv           # base CNN on sim data

The run auto-logs to results/results.csv with dataset="sim_flat.csv", so it
appears as its own row and never overwrites the easy-generator row.

Note what gets lost in the flattening: feeder membership, head readings, and
static features. That is the point -- this is exactly the information the base
model cannot use and PI-STGAT can.
"""
from __future__ import annotations

import argparse

import pandas as pd

from ntl.simulator import simulate_network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim_flat.csv")
    ap.add_argument("--n-feeders", type=int, default=150)
    ap.add_argument("--T", type=int, default=365)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fd, info = simulate_network(n_feeders=args.n_feeders, T=args.T, seed=args.seed)
    df = pd.DataFrame(fd.Y)
    df.insert(0, "label", fd.y)
    df.to_csv(args.out, index=False)
    print(f"[export] {len(fd.y)} customers x T={args.T}, theft={fd.y.mean():.1%}, "
          f"effective tl={info.tl_eff:.3f} -> {args.out}")
    print(f"[next]   python -m scripts.train --data {args.out}")


if __name__ == "__main__":
    main()
