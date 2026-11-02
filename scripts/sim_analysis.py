"""Simulator-only analysis (no training): figures + statistics for the report.
Runs the project's own simulator; --seed, --n-feeders, --T and --out are optional.
torch is stubbed because the simulator path is pure numpy."""
import sys, types, json
from sklearn.metrics import roc_auc_score
try:
    import torch  # real torch present: no stub needed
    _STUB = False
except ImportError:
    _STUB = True
for name in (["torch", "torch.utils", "torch.utils.data", "torch.nn", "torch.nn.functional"] if _STUB else []):
    m = types.ModuleType(name); sys.modules[name] = m
if _STUB:
    sys.modules["torch"].Tensor = type("Tensor", (), {})
    sys.modules["torch.utils.data"].Dataset = object
    sys.modules["torch.utils.data"].DataLoader = object
import argparse, os
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-feeders", type=int, default=150)
ap.add_argument("--T", type=int, default=365)
ap.add_argument("--out", default="figs")
opt = ap.parse_args()
os.makedirs(opt.out, exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ntl.simulator import simulate_network, _temperature, _demand, apply_attack

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
out = {}


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(opt.out, f"{name}.{ext}"), dpi=150)
    plt.close(fig)


fd, info = simulate_network(n_feeders=opt.n_feeders, T=opt.T, seed=opt.seed)
n, T = fd.Y.shape
out["n_customers"] = int(n); out["T"] = int(T); out["n_feeders"] = int(fd.n_feeders)
out["prevalence"] = float(fd.y.mean())
out["tl_eff"] = float(info.tl_eff)
out["tl_feeder_min"] = float(info.tl_feeder.min()); out["tl_feeder_max"] = float(info.tl_feeder.max())
out["attack_counts"] = {int(a): int((fd.atk == a).sum()) for a in range(1, 9)}

# ---- Fig S1: temperature + one customer per archetype ----
fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
ax[0].plot(info.temp, lw=0.8, color="#444")
ax[0].axhline(15, ls=":", c="tab:blue", lw=0.8); ax[0].axhline(24, ls=":", c="tab:red", lw=0.8)
ax[0].set_ylabel("temperature (°C)")
names = ["residential", "commercial", "industrial"]
for a in range(3):
    i = int(np.where(info.archetype == a)[0][0])
    ax[1].plot(info.X_true[i], lw=0.7, label=names[a])
ax[1].set_yscale("log"); ax[1].set_ylabel("true daily kWh"); ax[1].set_xlabel("day")
ax[1].legend(frameon=False, ncol=3, fontsize=8)
save(fig, "s1_weather_demand")

# ---- Fig S2: the eight attacks applied to one honest residential series ----
i0 = int(np.where((info.archetype == 0) & (fd.y == 0))[0][3])
x = info.X_true[i0]
titles = {1: "h1 scale", 2: "h2 zero window", 3: "h3 per-step factor", 4: "h4 fraction of mean",
          5: "h5 flat mean", 6: "h6 reversal", 7: "h7 seasonal-gated (new)", 8: "h8 ramp (new)"}
fig, axs = plt.subplots(4, 2, figsize=(7.2, 6.4), sharex=True, sharey=True)
stolen_frac = {}
for k, axx in zip(range(1, 9), axs.ravel()):
    y = apply_attack(x, k, np.random.default_rng(11 + k))
    axx.plot(x, lw=0.6, color="#999", label="true $x_t$")
    axx.plot(y, lw=0.6, color="tab:red", label="reported $y_t$")
    stolen_frac[k] = float((x.mean() - y.mean()) / x.mean())
    axx.set_title(f"{titles[k]}   ($\\bar x-\\bar y)/\\bar x$={abs(stolen_frac[k]) if abs(stolen_frac[k])<1e-6 else stolen_frac[k]:.2f}", fontsize=8)
axs[0, 0].legend(frameon=False, fontsize=7)
for axx in axs[-1]: axx.set_xlabel("day")
save(fig, "s2_attacks")
out["example_stolen_frac"] = stolen_frac

# ---- per-feeder energy balance ----
F = fd.n_feeders
ybar = fd.Y.mean(1); xbar = info.X_true.mean(1); sig = xbar - ybar
Hbar_true = info.heads.mean(1)                    # before head noise
Hbar = fd.head_mean                               # with 1% head noise (what model sees)
tau = info.tl_eff
r = np.array([Hbar[f] * (1 - tau) - ybar[fd.feeder_id == f].sum() for f in range(F)])
S = np.array([sig[fd.feeder_id == f].sum() for f in range(F)])
X = np.array([xbar[fd.feeder_id == f].sum() for f in range(F)])
K = np.array([(fd.feeder_id == f).sum() for f in range(F)])
nthief = np.array([fd.y[fd.feeder_id == f].sum() for f in range(F)])
err = r - S
out["corr_residual_vs_stolen"] = float(np.corrcoef(r, S)[0, 1])
out["err_over_feederload_mean"] = float(np.mean(err / X)); out["err_over_feederload_sd"] = float(np.std(err / X))
out["stolen_over_feederload_mean_thieffeeders"] = float(np.mean((S / X)[nthief > 0]))
out["frac_feeders_with_thief"] = float(np.mean(nthief > 0))
# decomposition of the error: -tau^2 X  +  (tau_f - tau) X  + head noise
out["tau_sq"] = float(tau ** 2)
# attack-wise mean stolen fraction across simulated thieves
for a in range(1, 9):
    m = fd.atk == a
    if m.any(): out[f"stolen_frac_h{a}"] = float(np.mean(sig[m] / xbar[m]))

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].scatter(S / X, r / X, s=8, c=np.where(nthief > 0, "tab:red", "tab:blue"), alpha=0.7)
lim = [min((S / X).min(), (r / X).min()) - 0.01, max((S / X).max(), (r / X).max()) + 0.01]
ax[0].plot(lim, lim, "k:", lw=0.8)
ax[0].set_xlabel(r"true stolen share $\sum_i\sigma_i / \sum_i\bar x_i$")
ax[0].set_ylabel(r"residual share $r_f / \sum_i\bar x_i$")
ax[0].set_title(f"{F} feeders, corr = {out['corr_residual_vs_stolen']:.3f}", fontsize=8)
ax[1].hist((err / X), bins=30, color="#777")
ax[1].set_xlabel(r"error share $(r_f-\sum_i\sigma_i)/\sum_i\bar x_i$"); ax[1].set_ylabel("feeders")
ax[1].set_title("residual error (loss-model + head noise)", fontsize=8)
save(fig, "s3_balance")

# ---- technical-loss fraction varies with load (one feeder) ----
f0 = 0
idx = fd.feeder_id == f0
Ht = info.heads[f0]; Xt = info.X_true[idx].sum(0)
loss_frac_t = (Ht - Xt) / Xt
out["feeder0_lossfrac_min"] = float(loss_frac_t.min()); out["feeder0_lossfrac_max"] = float(loss_frac_t.max())
fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.4))
ax.plot(loss_frac_t, lw=0.8, color="#444")
ax.set_xlabel("day"); ax.set_ylabel(r"$\ell_{f,t}/\sum_i x_{i,t}$")
ax.set_title("feeder 0: technical-loss fraction varies with load (quadratic loss model)", fontsize=8)
save(fig, "s4_lossfrac")

# ---- how much label signal the simulator places in single static features ----
S_ = fd.S
contracted, age, alarms = S_[:, 0], S_[:, 1], S_[:, 6]
out["auc_meter_age_alone"] = float(roc_auc_score(fd.y, age))
out["auc_alarms_alone"] = float(roc_auc_score(fd.y, alarms))
out["auc_contract_over_reported_alone"] = float(roc_auc_score(fd.y, contracted / np.maximum(ybar, 1e-6)))
out["auc_region_alone"] = float(roc_auc_score(fd.y, fd.region))
# reported mean alone (lower = more suspicious)
out["auc_neg_reported_mean_alone"] = float(roc_auc_score(fd.y, -ybar))

json.dump(out, open(os.path.join(opt.out, "sim_stats.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
