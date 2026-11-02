"""
Train + evaluate PI-STGAT.

Out of the box on the network simulator (physical losses, weather-driven
archetypes, 8 attack types):
    python -m scripts.train_pistgat
Full design (SSL pretraining + learned generative augmentation):
    python -m scripts.train_pistgat --ssl-epochs 8 --aug learned --aug-p 0.15
Ablations / variants:
    python -m scripts.train_pistgat --no-physics            # drop a component
    python -m scripts.train_pistgat --aug jokar             # fixed six-function aug
    python -m scripts.train_pistgat --pu --pu-reveal 0.3    # Positive-Unlabeled labels
    python -m scripts.train_pistgat --data simple           # old simple generator

Real data: --data real ingests a utility export (see ntl/real_data.py); rehearse
it first with make_fake_export.py. --aug learned on real data is not wired yet: it
needs confirmed-inspection pairs (post-inspection corrected consumption = the true
series).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.special import expit

from ntl.data import apply_random_attack
from ntl.data_graph import build_feeder_datasets, generate_feeder_data, pu_known_labels, split_feeders
from ntl.generator import GenAugmenter
from ntl.losses import FocalLoss
from ntl.metrics import best_f1_threshold, compute_metrics, format_metrics
from ntl.pistgat import build_pistgat
from ntl.pu import nnpu_loss
from ntl.results import clean_metrics, log_run, pistgat_label
from ntl.simulator import simulate_network
from ntl.ssl_pretrain import pretrain_temporal


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def gather(model, loader, device):
    """Flatten predictions over all real customers in a loader."""
    model.eval()
    keys = ["label", "region", "stolen", "atk", "cidx", "pu_known"]
    out = {k: [] for k in keys + ["p", "s", "logit"]}
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        res = model(batch)
        m = batch["mask"]
        out["logit"].append(res["logit"][m].cpu().numpy())
        out["p"].append(torch.sigmoid(res["logit"][m]).cpu().numpy())
        out["s"].append(res["s"][m].cpu().numpy())
        for k in keys:
            out[k].append(batch[k][m].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def fit_temperature(logits, labels):
    """1-parameter calibration: scale logits to minimise validation BCE."""
    t = torch.ones(1, requires_grad=True)
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=50)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(z / t.clamp(min=0.5, max=10.0), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.clamp(min=0.5, max=10.0).item())


def energy_recovery_at_k(stolen_true, score, k_frac):
    """Share of total truly-stolen energy recovered by inspecting the top-k by score."""
    n = max(1, int(round(k_frac * len(score))))
    top = np.argsort(-score)[:n]
    tot = stolen_true.sum() + 1e-9
    return float(stolen_true[top].sum() / tot)


def extras_dict(d, threshold):
    """Same quantities report_extras prints, as flat columns for results.csv."""
    p, s, stolen, y, region, atk = d["p"], d["s"], d["stolen"], d["label"], d["region"], d["atk"]
    out = {}
    for k in (0.05, 0.10, 0.20):
        kk = int(k * 100)
        out[f"er_p_{kk}"] = energy_recovery_at_k(stolen, p, k)
        out[f"er_ps_{kk}"] = energy_recovery_at_k(stolen, p * s, k)
    pred = (p >= threshold).astype(int)
    for tag, m in (("lo", region < 0.5), ("hi", region >= 0.5)):
        if m.sum():
            out[f"prev_{tag}"] = float(y[m].mean())
            out[f"flag_{tag}"] = float(pred[m].mean())
            out[f"rec_{tag}"] = float((pred[m] & (y[m] == 1)).sum() / max(1, int(y[m].sum())))
    n = max(1, int(round(0.10 * len(p))))
    top = set(np.argsort(-p)[:n].tolist())
    for a in sorted(set(int(v) for v in np.unique(atk) if v > 0)):
        idx = np.where(atk == a)[0]
        out[f"r10_h{a}"] = float(sum(1 for i in idx if i in top) / len(idx))
    return out


def report_extras(d, threshold):
    s = d["s"]; p = d["p"]; stolen = d["stolen"]; y = d["label"]; region = d["region"]; atk = d["atk"]
    cost_aware = p * s                      # expected recovered energy (physics-informed)
    if stolen.sum() <= 0:
        lines = ["  --- energy recovered @ top-k: ground-truth stolen energy unknown on "
                 "real data ---",
                 f"  model's own estimate: top-10% flagged meters carry "
                 f"{float(np.sort(cost_aware)[::-1][:max(1,int(0.1*len(p)))].sum()):.2f} "
                 f"(unit-scaled) expected recoverable energy; see --save-ranking"]
    else:
        lines = ["  --- energy recovered @ top-k inspections ---"]
    if stolen.sum() > 0:
        for k in (0.05, 0.10, 0.20):
            by_p = energy_recovery_at_k(stolen, p, k)
            by_ps = energy_recovery_at_k(stolen, cost_aware, k)
            lines.append(f"  k={int(k*100):>2d}%   by P(theft)={by_p:.3f}   by P*stolen (cost-aware)={by_ps:.3f}")

    lines.append("  --- fairness: flag rate / recall by region ---")
    pred = (p >= threshold).astype(int)
    for name, m in [("region<0.5", region < 0.5), ("region>=0.5", region >= 0.5)]:
        if m.sum() == 0:
            continue
        prev = y[m].mean()
        flag = pred[m].mean()
        rec = (pred[m] & (y[m] == 1)).sum() / max(1, int(y[m].sum()))
        lines.append(f"  {name:<11s}  prevalence={prev:.3f}  flag_rate={flag:.3f}  recall={rec:.3f}")

    lines.append("  --- recall@10% by attack type (h1-h4 reduce energy; h5,h6 shift pattern; h7,h8 novel) ---")
    n = max(1, int(round(0.10 * len(p))))
    top = set(np.argsort(-p)[:n].tolist())
    names = {1: "h1 scale", 2: "h2 zero-window", 3: "h3 per-step", 4: "h4 frac-mean",
             5: "h5 flat-mean", 6: "h6 reverse", 7: "h7 seasonal", 8: "h8 ramp",
             9: "gen (cVAE)"}
    for a in sorted(set(int(v) for v in np.unique(atk) if v > 0)):
        idx = np.where(atk == a)[0]
        if len(idx) == 0:
            continue
        caught = sum(1 for i in idx if i in top) / len(idx)
        lines.append(f"  {names.get(a, f'atk{a}'):<14s}  n={len(idx):>3d}  recall@10%={caught:.3f}")
    return "\n".join(lines)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data: network simulator (default) or the old simple generator ----
    if args.data == "sim":
        fd, info = simulate_network(n_feeders=args.n_feeders, T=args.T, seed=args.seed,
                                    label_static=not args.no_label_static)
        tl_frac = info.tl_eff if args.tl_frac is None else args.tl_frac
        print(f"[data] network sim: {len(fd.y)} customers / {fd.n_feeders} feeders, "
              f"theft={fd.y.mean():.1%}, effective tl={info.tl_eff:.3f} (model uses {tl_frac:.3f})")
    elif args.data == "real":
        from ntl.real_data import load_real
        fd, rep = load_real(meters=args.meters, topology=args.topology, heads=args.heads,
                            labels=args.labels, static=args.static_file,
                            region_col=args.region_col, days=args.days,
                            min_coverage=args.min_coverage,
                            min_feeder_size=args.min_feeder_size, tl_frac=args.tl_frac)
        info, tl_frac = None, rep["tl_used"]
        if fd.y.sum() == 0:
            raise SystemExit("real mode needs labels.csv with at least some confirmed "
                             "thieves to train on (inspection outcomes).")
        if not args.pu:
            args.pu = True
            print("[real] labels are positive-unlabeled by nature -> enabling --pu")
        args.pu_reveal = 1.0   # labels.csv already IS the revealed set; hide nothing
        if args.prior is None:
            print(f"[real][WARN] PU prior defaulting to the CONFIRMED rate "
                  f"{rep['confirmed_rate']:.3f}; the true theft prevalence is higher -- "
                  f"set --prior from an audit estimate when available.")
        if args.aug in ("learned", "both"):
            raise SystemExit("--aug learned/both on real data needs post-inspection "
                             "corrected-consumption pairs (not wired yet); use --aug "
                             "jokar or none.")
        print(f"[data] real: {len(fd.y)} meters / {fd.n_feeders} feeders, "
              f"confirmed={fd.y.mean():.2%}, tl_frac={tl_frac:.3f}")
    else:
        fd, info = generate_feeder_data(n_feeders=args.n_feeders, T=args.T,
                                        tl_frac=args.tl_frac or 0.06, seed=args.seed), None
        tl_frac = args.tl_frac or 0.06
        print(f"[data] simple gen: {len(fd.y)} customers / {fd.n_feeders} feeders, "
              f"theft={fd.y.mean():.1%}, tl_frac={tl_frac}")
    prior = float(fd.y.mean()) if args.prior is None else args.prior
    if args.no_region_feature:
        if args.data == "real":
            print("[real] the region column is always excluded from the inputs")
        else:
            fd.S[:, 2] = 0.0   # region index stays available for the fairness check

    # ---- augmenter: none | fixed six functions | learned cVAE | both ----
    augmenter, aug_p = None, 0.0
    if args.aug == "jokar":
        augmenter, aug_p = apply_random_attack, args.aug_p
        print(f"[aug] fixed six-function (Jokar) augmentation, p={aug_p}")
    elif args.aug in ("learned", "both"):
        if info is None:
            raise SystemExit("--aug learned/both needs --data sim (true/reported pairs come "
                             "from the simulator; on real data, fit GenAugmenter on "
                             "confirmed-inspection pairs).")
        tr_f, _, _ = split_feeders(fd.n_feeders, seed=args.seed)   # same split as loaders
        known = pu_known_labels(fd, args.pu_reveal if args.pu else 1.0, args.seed)
        pair = np.isin(fd.feeder_id, tr_f) & (known == 1)          # confirmed train thieves only
        print(f"[aug] fitting cVAE augmenter on {int(pair.sum())} confirmed train pairs ...")
        gen = GenAugmenter(seed=args.seed).fit(
            info.X_true[pair], fd.Y[pair], epochs=args.cvae_epochs, verbose=True)
        if args.aug == "learned":
            augmenter = gen
        else:
            def augmenter(x, rng):
                """Prior knowledge (fixed rules) + data adaptation (cVAE), 50/50."""
                return apply_random_attack(x, rng) if rng.random() < 0.5 else gen(x, rng)
            print("[aug] combined mode: 50% fixed rules / 50% cVAE samples")
        aug_p = args.aug_p

    train_loader, val_loader, test_loader, meta = build_feeder_datasets(
        fd, batch_size=args.batch, pu_reveal=args.pu_reveal if args.pu else 1.0,
        seed=args.seed, augmenter=augmenter, aug_p=aug_p)

    model = build_pistgat(meta, d_model=args.d_model, tl_frac=tl_frac,
                          use_graph=not args.no_graph, use_static=not args.no_static,
                          use_physics=not args.no_physics, pos_enc=args.pos_enc).to(device)
    cfg = (f"graph={not args.no_graph} static={not args.no_static} "
           f"physics={not args.no_physics} pu={args.pu} aug={args.aug} ssl={args.ssl_epochs}")
    print(f"[model] PI-STGAT  ({cfg})  params={sum(p.numel() for p in model.parameters()):,}")

    # ---- self-supervised pretraining of the temporal encoder (train split only) ----
    if args.ssl_epochs > 0:
        series = meta["norm"].transform(fd.Y[meta["train_cust_mask"]])
        print(f"[ssl] masked-reconstruction pretraining on {series.shape[0]} unlabeled "
              f"train series ...")
        state = pretrain_temporal(series, d_model=args.d_model,
                                  epochs=args.ssl_epochs, seed=args.seed,
                                  pos_enc=args.pos_enc)
        model.temporal.load_state_dict(state)
        print("[ssl] encoder weights transferred into PI-STGAT temporal branch")

    focal = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # under PU, selection and calibration see only the labels a practitioner would have
    val_key = "pu_known" if args.pu else "label"
    best_aupr, best_state, best_thr = -1.0, None, 0.5
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0; nb = 0
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            res = model(batch)
            m = batch["mask"]
            if args.pu:
                cls = nnpu_loss(res["logit"], batch["pu_known"], m, prior)
            else:
                cls = focal(res["logit"][m], batch["label"][m])
            cons = model.conservation_loss(res["s"], batch)
            loss = cls + args.lambda_cons * cons
            optim.zero_grad(); loss.backward(); optim.step()
            running += loss.item(); nb += 1

        d = gather(model, val_loader, device)
        val = compute_metrics(d[val_key], d["p"])
        tag = ""
        if val["aupr"] > best_aupr:
            best_aupr = val["aupr"]
            best_thr = best_f1_threshold(d[val_key], d["p"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            tag = "  <- best"
        print(f"[epoch {epoch:02d}] loss={running/max(1,nb):.4f}  "
              f"val_aupr={val['aupr']:.4f}  val_roc={val['roc_auc']:.4f}{tag}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # calibrate probability on validation, pick threshold on the SAME calibrated
    # scale, then evaluate test on calibrated probabilities (consistent operating point)
    dv = gather(model, val_loader, device)
    temp = fit_temperature(dv["logit"], dv[val_key])
    thr = best_f1_threshold(dv[val_key], expit(dv["logit"] / temp))
    dt = gather(model, test_loader, device)
    dt["p"] = expit(dt["logit"] / temp)

    test = compute_metrics(dt["label"], dt["p"], threshold=thr)
    print(f"\n=== TEST  ({cfg}) ===")
    print(format_metrics(test))
    print(report_extras(dt, thr))
    print(f"  [calibration] temperature={temp:.3f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "threshold": thr, "temperature": temp,
                "model_args": {"d_static": meta["d_static"], "d_model": args.d_model,
                               "tl_frac": float(tl_frac), "use_graph": not args.no_graph,
                               "use_static": not args.no_static,
                               "use_physics": not args.no_physics, "pos_enc": args.pos_enc},
                "norm_mean": meta["norm"].mean, "norm_std": meta["norm"].std,
                "stat_mean": meta["stat_mean"].tolist(), "stat_std": meta["stat_std"].tolist(),
                "unit": meta["unit"]},
               os.path.join(args.out, "pistgat.pt"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(test, f, indent=2)
    print(f"\n[saved] {args.out}/pistgat.pt and metrics.json")

    if args.save_ranking:
        import pandas as pd
        from torch.utils.data import DataLoader
        from ntl.data_graph import FeederDataset, collate_feeders
        ds_all = FeederDataset(fd, range(fd.n_feeders), meta["norm"], meta["stat_mean"],
                               meta["stat_std"], meta["unit"])
        da = gather(model, DataLoader(ds_all, batch_size=args.batch, shuffle=False,
                                      collate_fn=collate_feeders), device)
        ci = da["cidx"].astype(int)
        p_all = expit(da["logit"] / temp)
        s_kwh = da["s"] * meta["unit"]                      # back to kWh/day
        in_train = np.isin(fd.feeder_id[ci], meta["train_feeders"])
        rank = pd.DataFrame({
            "meter_id": (fd.meter_id[ci] if fd.meter_id is not None else ci),
            "transformer": (fd.feeder_key[fd.feeder_id[ci]] if fd.feeder_key is not None
                            else fd.feeder_id[ci]),
            "p_theft": p_all, "stolen_kwh_day_est": s_kwh,
            "score_p_x_stolen": p_all * s_kwh,
            "confirmed_label": da["label"].astype(int), "in_train_split": in_train,
        }).sort_values("p_theft", ascending=False)
        rank.to_csv(args.save_ranking, index=False)
        print(f"[saved] inspection ranking ({len(rank)} meters) -> {args.save_ranking}")
        print("        sorted by P(theft); switch to score_p_x_stolen for cost-aware")
        print("        ranking once the classifier is validated (it degenerates to")
        print("        'largest consumers' when probabilities are still flat).")
        print("        rank out-of-sample candidates first: in_train_split == False")

    log_run({"script": "train_pistgat", "model": "PI-STGAT", "config": pistgat_label(args),
             "dataset": args.data, "graph": not args.no_graph, "static": not args.no_static,
             "physics": not args.no_physics, "pu": args.pu,
             "pu_reveal": args.pu_reveal if args.pu else None,
             "ssl_epochs": args.ssl_epochs, "aug": args.aug,
             "aug_p": args.aug_p if args.aug != "none" else None,
             "n_feeders": fd.n_feeders, "T": fd.Y.shape[1], "epochs": args.epochs,
             "seed": args.seed, "temperature": temp, "prior": args.prior,
             "label_static": not args.no_label_static, "pos_enc": args.pos_enc,
             "region_feature": not args.no_region_feature and args.data != "real",
             **clean_metrics(test), **extras_dict(dt, thr)})
    return test


def build_argparser():
    p = argparse.ArgumentParser(description="PI-STGAT NTL detector")
    p.add_argument("--data", choices=["sim", "simple", "real"], default="sim",
                   help="'sim' = network simulator (default); 'simple' = old generator; "
                        "'real' = utility export (see --meters/--topology/--heads)")
    p.add_argument("--meters", help="real: meters CSV (long: meter_id,date,kwh | wide)")
    p.add_argument("--topology", help="real: meter_id,transformer_id CSV")
    p.add_argument("--heads", help="real: transformer head readings CSV")
    p.add_argument("--labels", help="real: meter_id,label CSV (1 = confirmed theft)")
    p.add_argument("--static-file", help="real: per-meter static features CSV")
    p.add_argument("--region-col", help="real: column in static CSV for fairness check")
    p.add_argument("--days", type=int, default=365, help="real: calendar window length")
    p.add_argument("--min-coverage", type=float, default=0.8)
    p.add_argument("--min-feeder-size", type=int, default=3)
    p.add_argument("--save-ranking", help="write a ranked inspection list CSV")
    p.add_argument("--n-feeders", type=int, default=150)
    p.add_argument("--T", type=int, default=365)
    p.add_argument("--tl-frac", type=float, default=None,
                   help="technical-loss fraction for the conservation head "
                        "(default: simulator's effective value, or 0.06 for 'simple')")
    p.add_argument("--ssl-epochs", type=int, default=0,
                   help="masked-reconstruction pretraining epochs (0 = off)")
    p.add_argument("--aug", choices=["none", "jokar", "learned", "both"], default="none",
                   help="on-the-fly synthetic-thief augmentation of the train split")
    p.add_argument("--aug-p", type=float, default=0.15,
                   help="per-honest-customer augmentation probability")
    p.add_argument("--cvae-epochs", type=int, default=400)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16, help="feeders per batch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--lambda-cons", type=float, default=0.5, help="weight on conservation loss")
    p.add_argument("--focal-alpha", type=float, default=0.75)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--pu", action="store_true", help="use Positive-Unlabeled loss")
    p.add_argument("--pu-reveal", type=float, default=0.3, help="fraction of true thieves revealed")
    p.add_argument("--prior", type=float, default=None, help="PU class prior (default: true prevalence)")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--no-static", action="store_true")
    p.add_argument("--no-physics", action="store_true")
    p.add_argument("--pos-enc", action="store_true",
                   help="add a position code to the temporal tokens")
    p.add_argument("--no-label-static", action="store_true",
                   help="sim: draw meter age and alarms independently of theft")
    p.add_argument("--no-region-feature", action="store_true",
                   help="remove the region index from the model inputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/pistgat")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
