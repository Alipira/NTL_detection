"""Validation curves parsed verbatim from the author's training logs.
run_pipeline.sh writes both logs for each seed."""
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
import sys, os
# usage: python -m scripts.plot_val_curves <simulator_runs.log> <rehearsal_run.log> [out_dir]
A, B = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "figs"
os.makedirs(OUT, exist_ok=True)

def blocks(path):
    txt = open(path).read().split("\n"); cur = None; out = {}
    for ln in txt:
        m = re.search(r"\[model\] PI-STGAT\s+\((.*?)\)", ln)
        if m: cur = m.group(1); out[cur] = []
        m = re.search(r"\[epoch (\d+)\].*val_aupr=([\d.]+)\s+val_roc=([\d.]+)", ln)
        if m and cur: out[cur].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out

sim = blocks(A); reh = blocks(B)
prior = re.search(r"CONFIRMED rate ([\d.]+)", open(B).read())
lab = {"graph=False static=False physics=False": "temporal only",
       "graph=False static=True physics=False": "+ static",
       "graph=True static=True physics=False": "+ graph",
       "graph=True static=True physics=True pu=False aug=none": "full",
       "aug=both ssl=8": "full + SSL + aug(both)"}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for k, v in sim.items():
    name = next((n for key, n in lab.items() if key in k), k)
    if "aug=both" in k: name = "full + SSL + aug(both)"
    ax[0].plot([e for e, _, _ in v], [a for _, a, _ in v], marker=".", lw=1, ms=3, label=name)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation AUPR"); ax[0].set_ylim(0.2, 1.0)
ax[0].set_title("network simulator, seed 0 (full labels)", fontsize=8); ax[0].legend(frameon=False, fontsize=7)
(k, v), = reh.items()
ax[1].plot([e for e, _, _ in v], [r for _, _, r in v], marker=".", c="tab:red", lw=1, label="val ROC-AUC")
ax[1].plot([e for e, _, _ in v], [a for _, a, _ in v], marker=".", c="tab:blue", lw=1, label="val AUPR")
best = max(v, key=lambda r: r[1])[0]      # the trainer selects the first epoch with the top val AUPR
ax[1].axhline(0.5, ls=":", c="k", lw=0.8); ax[1].axvline(best, ls="--", c="#888", lw=0.8)
ax[1].text(best + 0.3, 0.92, f"selected epoch ({best})", fontsize=7, color="#555")
ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1.0)
ax[1].set_title(rf"rehearsal, PU prior $\hat\pi$={float(prior.group(1)):.3f} (confirmed rate)", fontsize=8)
ax[1].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"s5_curves.{ext}"), dpi=150)
for k, v in sim.items(): print(k, len(v), max(a for _, a, _ in v))
