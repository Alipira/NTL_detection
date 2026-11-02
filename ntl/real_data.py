"""
Real-data ingestion for the PI-STGAT pipeline.

Turns a utility export into the same `FeederData` the simulator produces, so
everything downstream (training, SSL, PU, calibration, plots) runs unchanged.

Expected files (CSV):

  meters.csv     long:  meter_id, date, kwh          (sub-daily rows are summed
                 wide:  meter_id, <date columns...>   to daily automatically)
  topology.csv         meter_id, transformer_id
  heads.csv      long:  transformer_id, date, kwh
                 mean:  transformer_id, head_mean     (daily-kWh average)
  labels.csv     opt.:  meter_id, label   (1 = confirmed theft from inspection;
                        everyone absent is *unlabeled*, NOT honest -> PU)
  static.csv     opt.:  meter_id, <numeric / categorical columns...>
                        (categoricals are one-hot encoded; pass --region-col to
                        name the column used for the fairness check)

What the loader does, in order:
  1. align all meters on a shared daily calendar (last --days days by default),
  2. drop meters below --min-coverage non-missing share, impute the rest,
  3. join topology; drop meters without a transformer and feeders smaller than
     --min-feeder-size or without a head reading. The head still carries the load
     of meters dropped for coverage, so their observed mean is subtracted from it;
     load that cannot be placed on a feeder (unmapped meters) or measured (mapped
     meters absent from meters.csv) is reported as a warning, since it stays in
     the residual and looks like theft,
  4. compute per-feeder balance diagnostics:
         resid_frac_f = (head_f - sum(reported_i)) / head_f
     and FLAG feeders where it is negative beyond noise (meters sum to more
     than the head -> almost certainly topology / clock / CT errors, the most
     common real-world failure) or implausibly large,
  5. estimate technical loss if not given: tl_auto = the 25th percentile of
     resid_frac across healthy feeders (assumes >=75% of feeders carry little
     theft, so the low quartile ~ pure technical loss) -- override with
     --tl-frac when an engineering value is known,
  6. build static features and region index (the region column is kept out of
     the model inputs: it is the fairness check variable), and return FeederData
     plus a report dict.

Ground truth that real data does NOT have is left at safe defaults: atk = 0
(no per-attack table) and stolen = 0 (energy-recovery@k against true stolen is
not computable; the model's own stolen-energy estimate s_i takes its place in
the exported inspection ranking).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ntl.data import impute
from ntl.data_graph import FeederData


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #

def _read_meters(path: str, days: int, start=None, end=None):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "date" in cols and "kwh" in cols:                      # long format
        df = df.rename(columns={cols["meter_id"]: "meter_id",
                                cols["date"]: "date", cols["kwh"]: "kwh"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        # min_count: a blank reading is missing data, not zero consumption
        daily = df.groupby(["meter_id", "date"], as_index=False)["kwh"].sum(min_count=1)
        wide = daily.pivot(index="meter_id", columns="date", values="kwh")
    else:                                                     # wide format
        df = df.rename(columns={cols["meter_id"]: "meter_id"})
        wide = df.set_index("meter_id")
        wide.columns = pd.to_datetime(wide.columns).normalize()
        wide = wide.apply(pd.to_numeric, errors="coerce")
    wide = wide.sort_index(axis=1)
    if start or end:
        wide = wide.loc[:, slice(pd.to_datetime(start) if start else None,
                                 pd.to_datetime(end) if end else None)]
    if days and wide.shape[1] > days:
        wide = wide.iloc[:, -days:]
    return wide                                               # (meters x dates)


def _read_heads(path: str, dates) -> pd.Series:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(columns={cols["transformer_id"]: "transformer_id"})
    if "date" in cols and "kwh" in cols:                      # long: average over window
        df = df.rename(columns={cols["date"]: "date", cols["kwh"]: "kwh"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df[df["date"].isin(dates)]
        daily = df.groupby(["transformer_id", "date"])["kwh"].sum(min_count=1)
        return daily.groupby("transformer_id").mean()
    mcol = cols.get("head_mean") or cols.get("kwh")
    return df.set_index("transformer_id")[mcol].astype(float)


def _build_static(path, meter_index, region_col):
    if path is None:
        S = np.zeros((len(meter_index), 1), dtype=np.float32)
        return S, np.zeros(len(meter_index), dtype=np.float32), ["(none)"]
    df = pd.read_csv(path)
    df = df.rename(columns={ {c.lower(): c for c in df.columns}["meter_id"]: "meter_id"})
    df = df.set_index("meter_id").reindex(meter_index)
    region = np.zeros(len(meter_index), dtype=np.float32)
    if region_col and region_col in df.columns:
        r = pd.to_numeric(df[region_col], errors="coerce")
        if r.isna().all():                                    # categorical region -> rank codes
            r = df[region_col].astype("category").cat.codes.astype(float)
        r = r.fillna(r.median())
        region = ((r.rank(pct=True)).to_numpy(dtype=np.float32))   # -> [0,1]
        df = df.drop(columns=[region_col])                    # fairness variable, not an input
    num = df.select_dtypes(include=[np.number]).fillna(0.0)
    cat = df.select_dtypes(exclude=[np.number])
    parts = [num.to_numpy(dtype=np.float32)] if len(num.columns) else []
    names = list(num.columns)
    for c in cat.columns:
        oh = pd.get_dummies(df[c].astype(str), prefix=c)
        parts.append(oh.to_numpy(dtype=np.float32)); names += list(oh.columns)
    S = np.concatenate(parts, axis=1) if parts else np.zeros((len(meter_index), 1), np.float32)
    return np.nan_to_num(S), region, names


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #

def load_real(meters: str, topology: str, heads: str, labels: str | None = None,
              static: str | None = None, region_col: str | None = None,
              days: int = 365, start=None, end=None,
              min_coverage: float = 0.8, min_feeder_size: int = 3,
              tl_frac: float | None = None, verbose: bool = True):
    log = print if verbose else (lambda *a, **k: None)

    # 1-2: series on a shared calendar, coverage filter, impute
    wide = _read_meters(meters, days=days, start=start, end=end)
    cov = wide.notna().mean(axis=1)
    dropped_cov = int((cov < min_coverage).sum())
    observed_load = wide.mean(axis=1).fillna(0.0)             # kWh/day, before any drop
    low_cov = wide.index[cov < min_coverage]
    wide = wide[cov >= min_coverage]
    Y = impute(wide.to_numpy(dtype=np.float32))
    log(f"[real] meters: {len(wide)} kept on {wide.shape[1]} days "
        f"({wide.columns[0].date()}..{wide.columns[-1].date()}); "
        f"{dropped_cov} dropped for coverage<{min_coverage:.0%}")

    # 3: topology join + head join
    topo = pd.read_csv(topology)
    topo = topo.rename(columns={ {c.lower(): c for c in topo.columns}["meter_id"]: "meter_id",
                                 {c.lower(): c for c in topo.columns}["transformer_id"]: "transformer_id"})
    topo = topo.drop_duplicates("meter_id").set_index("meter_id")["transformer_id"]
    tx = topo.reindex(wide.index)
    no_topo = int(tx.isna().sum())
    keep = ~tx.isna()
    wide, Y, tx = wide[keep], Y[keep.to_numpy()], tx[keep]

    head = _read_heads(heads, set(wide.columns))
    # load of coverage-dropped meters is still in the head: take it out
    low_tx = topo.reindex(low_cov).dropna()
    if len(low_tx):
        removed = observed_load[low_tx.index].groupby(low_tx.values).sum()
        head = head - removed.reindex(head.index).fillna(0.0)
    # load that cannot be taken out stays in the residual
    unmapped_load = float(observed_load[topo.reindex(observed_load.index).isna()].sum())
    absent = topo[~topo.index.isin(observed_load.index)]
    if unmapped_load > 0:
        log(f"[real][WARN] {no_topo} unmapped meters carry {unmapped_load:.1f} kWh/day "
            f"({unmapped_load / max(float(observed_load.sum()), 1e-9):.1%} of metered load); "
            f"it inflates residuals on unknown feeders and can read as theft -- fix the topology")
    if len(absent):
        log(f"[real][WARN] {len(absent)} meters in topology.csv are absent from meters.csv, "
            f"on {absent.nunique()} feeders; their load inflates those residuals")
    groups, kept_tx, drop_small, drop_nohead = {}, [], 0, 0
    for t, idx in pd.Series(range(len(tx)), index=tx.values).groupby(level=0):
        if len(idx) < min_feeder_size:
            drop_small += 1; continue
        if t not in head.index or not np.isfinite(head.loc[t]) or head.loc[t] <= 0:
            drop_nohead += 1; continue
        groups[t] = idx.to_numpy(); kept_tx.append(t)
    log(f"[real] topology: {no_topo} meters without transformer dropped; "
        f"{drop_small} feeders < {min_feeder_size} meters and "
        f"{drop_nohead} without head reading dropped -> {len(kept_tx)} feeders")
    if not kept_tx:
        raise SystemExit("no usable feeders after filtering -- check id columns match "
                         "between meters.csv, topology.csv and heads.csv")

    order = np.concatenate([groups[t] for t in kept_tx])
    Y = Y[order]
    meter_id = wide.index.to_numpy()[order]
    feeder_id = np.concatenate([[f] * len(groups[t]) for f, t in enumerate(kept_tx)])
    head_mean = np.array([float(head.loc[t]) for t in kept_tx], dtype=np.float32)

    # 4: balance diagnostics
    rep_sum = np.array([Y[feeder_id == f].mean(1).sum() for f in range(len(kept_tx))])
    resid_frac = (head_mean - rep_sum) / np.maximum(head_mean, 1e-9)
    bad_neg = np.where(resid_frac < -0.02)[0]
    bad_big = np.where(resid_frac > 0.40)[0]
    log(f"[real] feeder balance residual (head - sum(meters))/head: "
        f"median={np.median(resid_frac):.3f}, IQR=[{np.percentile(resid_frac,25):.3f}, "
        f"{np.percentile(resid_frac,75):.3f}]")
    if len(bad_neg):
        log(f"[real][WARN] {len(bad_neg)} feeders have NEGATIVE residual < -2% "
            f"(meters sum above head) -> suspected topology/CT/clock errors, e.g. "
            f"{[kept_tx[i] for i in bad_neg[:5]]}")
    if len(bad_big):
        log(f"[real][WARN] {len(bad_big)} feeders have residual > 40% -> check unmetered "
            f"loads (street lighting etc.) or head metering, e.g. {[kept_tx[i] for i in bad_big[:5]]}")

    # 5: technical-loss estimate
    healthy = resid_frac[(resid_frac >= 0) & (resid_frac <= 0.40)]
    tl_auto = float(np.percentile(healthy, 25)) if len(healthy) else 0.06
    tl_used = tl_frac if tl_frac is not None else tl_auto
    log(f"[real] technical loss: auto estimate (25th pct of healthy residuals) = "
        f"{tl_auto:.3f}; using tl_frac = {tl_used:.3f}")

    # labels (PU): confirmed thieves = 1, everyone else unlabeled (0)
    y = np.zeros(len(meter_id), dtype=np.int64)
    if labels:
        lab = pd.read_csv(labels)
        lab = lab.rename(columns={ {c.lower(): c for c in lab.columns}["meter_id"]: "meter_id"})
        lcol = [c for c in lab.columns if c.lower() == "label"][0]
        pos = set(lab.loc[lab[lcol] == 1, "meter_id"])
        y = np.array([1 if m in pos else 0 for m in meter_id], dtype=np.int64)
        log(f"[real] labels: {int(y.sum())} confirmed thieves "
            f"({y.mean():.2%} of kept meters) -- remaining meters are UNLABELED, "
            f"train with --pu (real labels are positive-unlabeled by nature)")

    S, region, snames = _build_static(static, pd.Index(meter_id), region_col)
    log(f"[real] static features: {S.shape[1]} columns "
        f"({', '.join(snames[:6])}{'...' if len(snames) > 6 else ''})")

    fd = FeederData(Y=Y.astype(np.float32), y=y, atk=np.zeros_like(y),
                    S=S.astype(np.float32), region=region,
                    feeder_id=feeder_id.astype(np.int64), head_mean=head_mean,
                    stolen=np.zeros(len(y), dtype=np.float32),
                    n_feeders=len(kept_tx),
                    meter_id=meter_id, feeder_key=np.asarray(kept_tx))
    report = {"tl_auto": tl_auto, "tl_used": tl_used,
              "resid_frac": resid_frac, "feeder_key": kept_tx,
              "confirmed_rate": float(y.mean()),
              "n_flag_neg": int(len(bad_neg)), "n_flag_big": int(len(bad_big)),
              "unmapped_load": unmapped_load, "n_absent": int(len(absent))}
    return fd, report
