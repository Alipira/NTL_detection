"""Rehearsal figures across all seeds of a run_pipeline.sh output folder.

    python -m scripts.plot_rehearsal_seeds outputs/gpu_run report/figs

Reads only files the pipeline wrote:
  logs/seed_N/abl_*.log, full_ssl_aug.log   validation curves of the simulator runs (seed 0)
  logs/seed_N/rh_<prior>.log                validation curves of the rehearsal trainings
  results/rehearsal_seed_N.txt              test-feeder re-scoring by report_numbers.py

Writes
  s5_curves.pdf/png      left: simulator validation AUPR, seed 0; right: rehearsal
                         validation ROC-AUC with the confirmed-rate prior, every seed
  s6_prior_sweep.pdf/png fraction of epochs with validation ROC-AUC < 0.5, and test-feeder
                         AUPR / ROC-AUC against the full truth, per prior and seed
and prints the numbers behind them.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
RUN = sys.argv[1] if len(sys.argv) > 1 else "outputs/gpu_run"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RUN, "figures", "all_seeds")
os.makedirs(OUT, exist_ok=True)
SEEDS = sorted(int(d.split("_")[1]) for d in os.listdir(os.path.join(RUN, "logs"))
               if d.startswith("seed_"))
PRIORS = [("prior_default", r"$\rho$"), ("p0.06", "0.06"), ("p0.10", "0.10"),
          ("p0.13", "0.13"), ("p0.18", "0.18")]
EPOCH = re.compile(r"\[epoch (\d+)\].*val_aupr=([\d.]+)\s+val_roc=([\d.]+)(.*best)?")


def curve(path):
    """Epoch rows of one training log: (epoch, val_aupr, val_roc, is_best)."""
    rows = []
    for ln in open(path):
        m = EPOCH.search(ln)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), bool(m.group(4))))
    n_ep = max(r[0] for r in rows)
    return rows[:n_ep]          # the trainer reprints the selected epoch after the loop


def test_only(path):
    """{inspect file: (AUPR_true, ROC_true)} for TEST-only feeders."""
    out = {}
    for blk in re.split(r"\n--- ", open(path).read())[1:]:
        m = re.search(r"TEST only.*?AUPR_true=([\d.]+).*?ROC_true=([\d.]+)", blk, re.S)
        if m:
            out[blk.split()[0]] = (float(m.group(1)), float(m.group(2)))
    return out


# ---- s5: validation curves ---------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
names = [("abl_temporal", "temporal only"), ("abl_static", "+ static"), ("abl_graph", "+ graph"),
         ("abl_full", "full"), ("full_ssl_aug", "full + SSL + aug(both)")]
s0 = SEEDS[0]
for f, lab in names:
    c = curve(os.path.join(RUN, "logs", f"seed_{s0}", f"{f}.log"))
    ax[0].plot([r[0] for r in c], [r[1] for r in c], lw=0.9, label=lab)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation AUPR"); ax[0].set_ylim(0.2, 1.0)
ax[0].set_title(f"network simulator, seed {s0} (full labels)", fontsize=8)
ax[0].legend(frameon=False, fontsize=7, loc="lower right")
for s in SEEDS:
    c = curve(os.path.join(RUN, "logs", f"seed_{s}", "rh_prior_default.log"))
    line, = ax[1].plot([r[0] for r in c], [r[2] for r in c], lw=0.8, label=f"seed {s}")
    best = [r for r in c if r[3]][-1]
    ax[1].plot(best[0], best[2], "o", ms=4, color=line.get_color())
ax[1].axhline(0.5, ls=":", c="k", lw=0.8)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation ROC-AUC (confirmed labels)")
ax[1].set_ylim(0.2, 1.0)
ax[1].set_title("rehearsal, prior = confirmed rate (dot: selected epoch)", fontsize=8)
ax[1].legend(frameon=False, fontsize=7, ncol=3, loc="lower center")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"s5_curves.{ext}"), dpi=150)
plt.close(fig)

# ---- s6: prior sweep ---------------------------------------------------------------
frac, ap, roc = {}, {}, {}
for key, _ in PRIORS:
    frac[key], ap[key], roc[key] = [], [], []
    for s in SEEDS:
        c = curve(os.path.join(RUN, "logs", f"seed_{s}", f"rh_{key}.log"))
        frac[key].append(np.mean([r[2] < 0.5 for r in c]))
        t = test_only(os.path.join(RUN, "results", f"rehearsal_seed_{s}.txt"))[f"inspect_{key}.csv"]
        ap[key].append(t[0]); roc[key].append(t[1])

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
x = np.arange(len(PRIORS))
for j, (data, ylab) in enumerate([(frac, "share of epochs with\nvalidation ROC-AUC < 0.5"),
                                  (ap, "test-feeder AUPR (full truth)"),
                                  (roc, "test-feeder ROC-AUC (full truth)")]):
    for i, (key, _) in enumerate(PRIORS):
        v = np.array(data[key])
        ax[j].scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=9, c="#888")
        ax[j].plot([i - 0.25, i + 0.25], [v.mean()] * 2, c="k", lw=1.5)
    ax[j].set_xticks(x); ax[j].set_xticklabels([p[1] for p in PRIORS], fontsize=7)
    ax[j].set_ylabel(ylab, fontsize=8); ax[j].set_xlabel(r"PU prior $\hat\pi$ ($\rho$: confirmed rate)", fontsize=7)
ax[0].set_ylim(-0.03, 0.8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"s6_prior_sweep.{ext}"), dpi=150)
plt.close(fig)

print("prior            frac_val_roc<0.5        test AUPR_true          test ROC_true")
for key, _ in PRIORS:
    print(f"{key:14s} " + "  ".join(f"{np.mean(d[key]):.3f}±{np.std(d[key], ddof=1):.3f}"
                                    for d in (frac, ap, roc))
          + "   per seed frac " + str(np.round(frac[key], 2)))
