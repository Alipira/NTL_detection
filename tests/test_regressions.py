"""Regression tests for data and results bugs: python -m pytest tests"""
import numpy as np
import pandas as pd
import torch

from ntl.data_graph import build_feeder_datasets, pu_known_labels, split_feeders
from ntl.pistgat import PISTGAT, load_pistgat
from ntl.real_data import _read_meters, load_real
from ntl.results import load_latest
from ntl.simulator import simulate_network


def _long(path, rows):
    pd.DataFrame(rows, columns=["meter_id", "date", "kwh"]).to_csv(path, index=False)


def test_blank_reading_is_missing_not_zero(tmp_path):
    p = tmp_path / "m.csv"
    _long(p, [("A", "2025-01-01", 5.0), ("A", "2025-01-02", None), ("A", "2025-01-03", 5.0)])
    w = _read_meters(str(p), 365)
    assert np.isnan(w.iloc[0, 1]) and w.notna().mean(axis=1).iloc[0] < 1


def _feeder(tmp_path, extra, topo_extra=()):
    days = [str(d.date()) for d in pd.date_range("2025-01-01", periods=10)]
    rows = [(m, d, 10.0) for m in "ABC" for d in days] + extra(days)
    _long(tmp_path / "m.csv", rows)
    pd.DataFrame({"meter_id": list("ABC") + list(topo_extra),
                  "transformer_id": "T"}).to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"transformer_id": ["T"], "head_mean": [60.0]}).to_csv(tmp_path / "h.csv", index=False)
    return load_real(str(tmp_path / "m.csv"), str(tmp_path / "t.csv"), str(tmp_path / "h.csv"), verbose=False)


def test_coverage_dropped_load_is_removed_from_head(tmp_path):
    _, rep = _feeder(tmp_path, lambda d: [("D", x, 30.0) for x in d[:2]], topo_extra=["D"])
    assert abs(rep["resid_frac"][0]) < 1e-6


def test_unplaceable_load_is_reported(tmp_path):
    _, rep = _feeder(tmp_path, lambda d: [("U", x, 30.0) for x in d], topo_extra=["ABSENT"])
    assert rep["unmapped_load"] == 30.0 and rep["n_absent"] == 1


def test_region_column_is_not_an_input(tmp_path):
    days = [str(d.date()) for d in pd.date_range("2025-01-01", periods=10)]
    _long(tmp_path / "m.csv", [(m, d, 10.0) for m in "ABC" for d in days])
    pd.DataFrame({"meter_id": list("ABC"), "transformer_id": "T"}).to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"transformer_id": ["T"], "head_mean": [33.0]}).to_csv(tmp_path / "h.csv", index=False)
    pd.DataFrame({"meter_id": list("ABC"), "age": [1, 2, 3], "district": ["n", "s", "n"]}).to_csv(tmp_path / "s.csv", index=False)
    fd, _ = load_real(str(tmp_path / "m.csv"), str(tmp_path / "t.csv"), str(tmp_path / "h.csv"),
                      static=str(tmp_path / "s.csv"), region_col="district", verbose=False)
    assert fd.S.shape[1] == 1 and fd.region.max() > fd.region.min()


def test_padding_does_not_change_predictions_or_batchnorm():
    fd, _ = simulate_network(n_feeders=12, T=60, seed=0)
    batch = next(iter(build_feeder_datasets(fd, batch_size=12, seed=0)[0]))
    assert not batch["mask"].all()
    wider = dict(batch)
    for k in ("series", "static", "label", "pu_known", "region", "cust_mean", "stolen", "atk", "cidx", "mask"):
        pad = list(batch[k].shape); pad[1] = 5
        wider[k] = torch.cat([batch[k], torch.zeros(pad, dtype=batch[k].dtype)], 1)

    model = PISTGAT(d_static=fd.S.shape[1])
    first_bn = model.temporal.conv[1]
    model.train()
    model(batch)
    stats = first_bn.running_mean.clone()
    first_bn.reset_running_stats()
    model(wider)
    assert torch.allclose(first_bn.running_mean, stats)

    model.eval()
    a = model(batch)["logit"][batch["mask"]]
    b = model(wider)["logit"][wider["mask"]]
    assert torch.allclose(a, b, atol=1e-5)


def test_pu_hides_train_and_val_only():
    fd, _ = simulate_network(n_feeders=30, T=30, seed=0)
    known = pu_known_labels(fd, 0.3, 0)
    tr, va, te = split_feeders(30, seed=0)
    for fs, full in ((tr, False), (va, False), (te, True)):
        m = np.isin(fd.feeder_id, fs) & (fd.y == 1)
        assert (known[m].sum() == m.sum()) == full


def test_hidden_thief_can_be_augmented_known_positive_is_not():
    fd, _ = simulate_network(n_feeders=20, T=30, seed=0)
    calls = []

    def aug(x, rng):
        calls.append(1)
        return x * 0.5, 1
    ds = build_feeder_datasets(fd, pu_reveal=0.3, augmenter=aug, aug_p=1.0)[0].dataset
    for i in range(len(ds)):
        it = ds[i]
        assert it["pu_known"].min() == 1          # everything unlabeled was converted
    known = pu_known_labels(fd, 0.3, 0)
    tr = np.isin(fd.feeder_id, ds.feeders)
    assert len(calls) == int((known[tr] == 0).sum())


def test_no_label_static_keeps_the_same_network():
    a, _ = simulate_network(n_feeders=10, T=30, seed=0)
    b, _ = simulate_network(n_feeders=10, T=30, seed=0, label_static=False)
    assert (a.y == b.y).all() and (a.Y == b.Y).all() and (a.S[:, 1] != b.S[:, 1]).any()


def test_small_run_does_not_replace_full_run(tmp_path):
    p = tmp_path / "r.csv"
    base = {"script": "train_pistgat", "config": "full", "dataset": "sim", "seed": 0}
    pd.DataFrame([{**base, "timestamp": "2026-01-01", "n_feeders": 150, "T": 365, "epochs": 20, "aupr": .9},
                  {**base, "timestamp": "2026-01-02", "n_feeders": 30, "T": 120, "epochs": 1, "aupr": .1},
                  {**base, "timestamp": "2026-01-03", "n_feeders": 150, "T": 365, "epochs": 20, "aupr": .8}]).to_csv(p, index=False)
    assert load_latest(str(p)).aupr.tolist() == [.8]


def test_ssl_epoch_count_does_not_hide_runs_from_plots(tmp_path):
    p = tmp_path / "r.csv"
    pd.DataFrame([{"timestamp": "2026-01-01", "script": "train_pistgat", "config": "full+ssl50+aug:both",
                   "dataset": "sim", "seed": 0, "ssl_epochs": 50}]).to_csv(p, index=False)
    assert load_latest(str(p)).config.tolist() == ["full+ssl+aug:both"]


def test_checkpoint_round_trip(tmp_path):
    fd, _ = simulate_network(n_feeders=6, T=30, seed=0)
    model = PISTGAT(d_static=fd.S.shape[1], pos_enc=True)
    args = {"d_static": fd.S.shape[1], "d_model": 64, "tl_frac": .06, "use_graph": True,
            "use_static": True, "use_physics": True, "pos_enc": True}
    p = tmp_path / "c.pt"
    torch.save({"state_dict": model.state_dict(), "model_args": args, "norm_mean": 1.0}, p)
    loaded, ck = load_pistgat(str(p))
    assert ck["norm_mean"] == 1.0 and loaded.temporal.pos_enc
