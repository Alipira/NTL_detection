# NTL detection — base model → PI-STGAT (complete design)

**Research report:** [PI-STGAT — Physics-Informed Spatio-Temporal Graph Attention for Non-Technical Loss Detection under Positive–Unlabelled Supervision](report/PI-STGAT_research_report.pdf) (Ali Pira, independent research report, not peer reviewed).

A runnable research codebase for **non-technical-loss / electricity-theft
detection** in low-voltage networks, built in additive stages that share one
data pipeline and metric suite:

1. **Base** — 1-D CNN over per-customer series (+ XGBoost classical floor).
2. **PI-STGAT** — temporal (dilated TCN + self-attention) + feeder
   graph-attention + static branches, fused into a theft-risk head and an
   **energy-conservation head** tied to the transformer balance.
3. **Completion of the design (this stage)** —
   a physically grounded **network simulator**, **self-supervised
   masked-reconstruction pretraining**, and a **learned generative augmenter**
   (cVAE) alongside the fixed six-function attacks.

Everything runs out of the box on CPU with no external dataset.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

To use an environment that already has these packages, skip the install and pass
its interpreter to the pipeline: `./run_pipeline.sh --python /path/to/env/bin/python ...`

## Run everything

`run_pipeline.sh` runs every experiment for every seed and saves the figures per
seed. Check the flow first (a couple of minutes), then start the long run:

```bash
./run_pipeline.sh --smoke
./run_pipeline.sh --seeds "0 1 2 3 4" --epochs 200 --base-epochs 150 \
    --exp-epochs 100 --ssl-epochs 50 --cvae-epochs 2000 --out outputs/gpu_run
```

Re-running the same command resumes: finished steps are skipped and failed ones
retried. Add seeds later by re-running with a longer `--seeds`. `--help` lists all
options. Results land in the `--out` folder:

```
figures/seed_N/      fig1-fig8 and s1-s5 for one seed, written as each seed finishes
figures/all_seeds/   fig1-fig8, mean over the finished seeds
results/             results.csv, report_numbers.txt, rehearsal_seed_N.txt
logs/                one log per step
runs/  exports/      checkpoints and rehearsal data
```

XGBoost, data generation and the cVAE stay on the CPU; the neural models use the GPU.

## Run one piece

Run these from the project root:

```bash
# base
python -m scripts.train                     # 1-D CNN
python -m scripts.baselines                 # XGBoost floor

# PI-STGAT on the network simulator (default data source)
python -m scripts.train_pistgat

# the complete design: SSL pretraining + combined augmentation
python -m scripts.train_pistgat --ssl-epochs 8 --aug both --aug-p 0.15

# label-scarce reality: Positive-Unlabeled labels + SSL
python -m scripts.train_pistgat --pu --pu-reveal 0.3 --ssl-epochs 8

# augmentation-generalization experiment (staged)
python -m scripts.exp_augment prepare --seed 0
python -m scripts.exp_augment run --aug none --seed 0     # then: jokar | learned | both

# regression tests (needs pytest)
python -m pytest tests
```

## The network simulator (`ntl/simulator.py`)

Simulates the LV network end to end so every component has the structure it
needs, with the truth hidden from the model:

- **Weather-driven demand**: shared temperature series; three customer
  archetypes (residential / commercial / industrial) with distinct weekly
  patterns and heating/cooling response.
- **Physical losses**: per-feeder line loss quadratic in each customer's load
  (I²R-style) + transformer copper loss, calibrated to a per-feeder loss
  fraction in 4–9%. The technical-loss *fraction varies with load*, so the
  conservation head must cope with an imperfect flat-tl assumption — as on a
  real feeder. Sanity check: per-feeder conservation residual vs true stolen
  energy correlates **0.94–0.96** (seeds 0–2).
- **Head metering** with noise; static features incl. contracted power, meter
  age, tariff, region index, alarms, archetype.
- **Attack library**: the six Jokar functions **plus two novel field-style
  attacks** — `h7` seasonal-gated (steal only on high-use days) and `h8`
  gradual ramp (tampering drift) — giving an honest "unseen attack" test bed.
- Returns `FeederData` (drop-in for the whole pipeline) + `SimInfo` with the
  hidden truth (true series for generator pairs, per-feeder tl, temperature).

## Self-supervised pretraining (`ntl/ssl_pretrain.py`)

Masked **span** reconstruction (hide random multi-day windows, reconstruct
them) on unlabeled training series; the encoder then initializes PI-STGAT's
temporal branch. A model that can fill in a customer's missing week has
internalized their seasonality and rhythm — the structure theft violates.

**Measured effect with scarce labels (PU, 30% of thieves revealed, sim data, 3
seeds; hidden labels stay hidden on validation too):**

| setting              | test AUPR     | test ROC      |
|----------------------|:-------------:|:-------------:|
| from scratch         | 0.535 ± 0.068 | 0.815 ± 0.024 |
| + SSL (8 epochs)     | 0.525 ± 0.127 | 0.779 ± 0.015 |

These runs show no benefit from SSL. An earlier single-seed gain, measured with
full validation labels, was not reproduced.

## Learned generative augmenter (`ntl/generator.py`)

A **conditional VAE over the multiplicative theft mask** `m = reported/true ∈
(0,1)`: generated thieves are `m̂ ⊙ honest`, hence physically valid
under-reporters *by construction* (unlike SMOTE in raw kWh space). The honest
profile conditions the decoder; latent `z` carries attack style (β tuned to
0.01 for sample diversity). cVAE chosen over full TimeGAN because we condition
on the honest profile and have (true, reported) pairs — the conditional-VAE
setting — and it trains stably in seconds on CPU. Pairs come from the
simulator here; on real data, from confirmed inspections where post-inspection
corrected consumption provides the true series (standard back-billing
practice). Stated limitation: masks in (0,1) cannot represent
energy-preserving attacks (h6 reversal) — those remain the temporal/graph
branches' job.

**Generalization experiment** (`exp_augment.py`): train on a network whose
thieves use only h1–h4 at low prevalence (38–50 confirmed pairs), test on an
independent network using all eight attacks. Mean ± s.d. over 3 seeds:

| augmentation | AUPR | ROC | R@10% | recall@10% on novel h7 |
|---|:---:|:---:|:---:|:---:|
| none           | 0.599 ± 0.050 | 0.752 ± 0.028 | 0.477 | 0.46 ± 0.18 |
| fixed (Jokar)  | **0.683 ± 0.045** | 0.804 ± 0.007 | **0.567** | 0.46 ± 0.05 |
| learned (cVAE) | 0.557 ± 0.039 | 0.765 ± 0.016 | 0.435 | 0.36 ± 0.14 |
| both (50/50)   | 0.667 ± 0.020 | **0.829 ± 0.011** | 0.511 | 0.47 ± 0.13 |

Reading: the combination has the best ROC on all three seeds and fixed rules the
best AUPR on two of three. No variant is reliably better on the novel h7 attack,
and the cVAE alone has lower AUPR than no augmentation on every seed. Most of the
fixed rules' gain comes from h5, which they hand-code. Augmentation is **physically
coherent**: converting an honest customer leaves the head reading fixed, so the
feeder's conservation residual grows by exactly the stolen amount -- the same
signature a real thief leaves.

## Full-design end-to-end (sim, 150 feeders, T=365, 20 epochs, 3 seeds, CPU)

`--ssl-epochs 8 --aug both` → test AUPR 0.871 ± 0.054, ROC 0.964 ± 0.017. The full
model without SSL or augmentation gets AUPR 0.881 ± 0.027, ROC 0.965 ± 0.005, so
this combination adds no measurable overall gain. Per-attack recall@10%: h2–h4
caught at 0.92–1.00, h1 0.75, h7 0.85, h5 0.63, h8 0.51; h6 reversal is the hard
tail at 0.06. Test prevalence varies from 5.7% to 16.9% across seeds, so report
spreads, not one seed. The simulator's static features carry label signal by
design; `--no-label-static` and `--no-region-feature` measure how much. Numbers
are directional, not benchmarks.

## Results table & plots

Every entry point (`train`, `baselines`, `train_pistgat`, `exp_augment`,
`eval_ranking`) automatically appends one row per run to
**`results/results.csv`** — config flags, all metrics, energy-recovery@k,
fairness numbers, and per-attack recall as columns. Re-running a config adds a
new row; plotting keeps the largest, then most recent, run per configuration and
seed, so a small test run never replaces a full one.

```bash
python -m scripts.plot_results        # -> results/figures/fig1..fig8 .png
```

Figures: model comparison (CNN/XGBoost/PI-STGAT), component ablation,
per-attack recall matrix, cost-aware energy-recovery curves, fairness by
region, and the augmentation experiment (overall + seen-vs-unseen attack
matrix), and the rehearsal.

The report source, figures and PDF are in `report/`. Results behind the report are
in `outputs/reference_cpu/`.

A caution on **fig1**: it compares across *different datasets and
prevalences*, so its AUPR bars are context, not a like-for-like claim. The
base CNN runs on the easy homogeneous generator (six attacks, train-time
augmentation matched to the test attacks, test prevalence 9.2%), while
PI-STGAT runs on the harder network simulator (heterogeneous archetypes,
weather, eight attacks, test prevalence 5.7% on seed 0 -- and AUPR's random baseline
*equals* prevalence, so the same ranking skill scores lower at lower
prevalence). On prevalence-free ROC-AUC and on recall@10%, PI-STGAT wins even
across datasets; the equal-footing architecture comparison is **fig2** (same
sim data, 3 seeds: temporal-only 0.814 -> +static 0.896 -> full 0.881 AUPR). To fill the missing cell --
base CNN *on* simulator data -- run `python -m scripts.export_sim_csv` then
`python -m scripts.train --data sim_flat.csv`; the row logs under its own dataset name. The results behind the report are in `outputs/reference_cpu/`; replot them with
`python -m scripts.plot_results --csv outputs/reference_cpu/results.csv --out my_figures`.

## Files

```
run_pipeline.sh      every experiment and figure, for every seed
requirements.txt
ntl/                 library code
  data.py            base synthetic generator, six Jokar attacks, preprocessing, CSV hook
  model.py           base 1-D CNN encoder + head
  losses.py          focal loss, weighted BCE
  metrics.py         AUPR, ROC-AUC, F1, precision@k, recall@k, BDR
  data_graph.py      feeder datasets/loaders; physically coherent on-the-fly augmentation
  pistgat.py         PI-STGAT; conservation loss; checkpoint loading
  pu.py              non-negative Positive-Unlabeled loss
  simulator.py       whole-network simulator -> FeederData + SimInfo
  ssl_pretrain.py    masked-span reconstruction pretraining
  generator.py       cVAE theft-mask augmenter
  real_data.py       real utility-export loader + data-quality diagnostics
  results.py         central results table
scripts/             entry points, run as python -m scripts.NAME
  train.py           base CNN trainer
  baselines.py       XGBoost floor
  train_pistgat.py   PI-STGAT trainer (sim, simple or real data)
  exp_augment.py     seen->unseen attack generalization experiment
  make_fake_export.py simulator -> messy utility export + hidden truth (rehearsal)
  eval_ranking.py    score an inspection ranking against truth / campaign outcomes
  export_sim_csv.py  flatten the simulator to per-customer CSV
  plot_results.py    fig1-fig8 from results.csv (all seeds, or --seed N)
  plot_val_curves.py s5 validation curves from training logs
  sim_analysis.py    s1-s4 simulator figures and statistics
  report_numbers.py  every number the research report uses
tests/               regression tests
report/              report.tex, figs/ and the PDF
outputs/             reference_cpu/ holds the results behind the report;
                     gpu_run/ the 5-seed GPU run (results/ and all_seeds figures)
```

## Using real data

**Rehearse first.** Before touching the real export, run the whole real-data
pipeline on a simulator-generated export that has the same schema and the same
defects (gaps, dead meters, late joiners, unmapped meters, wrong topology,
unmetered loads) -- with ground truth kept aside so every step is checkable:

```bash
python -m scripts.make_fake_export --out-dir exports/rh --unmetered-kwh 1.5
python -m scripts.train_pistgat --data real --meters exports/rh/meters.csv \
    --topology exports/rh/topology.csv --heads exports/rh/heads.csv \
    --labels exports/rh/labels.csv --static-file exports/rh/static.csv \
    --region-col district --save-ranking exports/rh/inspect.csv
python -m scripts.eval_ranking --ranking exports/rh/inspect.csv --truth exports/rh/truth_labels.csv
```

Verified behavior: the loader treats blank readings as missing, drops dead/low-coverage and
unmapped meters with counts, removes coverage-dropped load from the head, warns
about load it cannot place (it stays in the residuals and can read as theft), flags wrong-topology feeders via negative balance residuals, absorbs
unmetered load into the tl estimate (a documented bias), and the evaluator
quantifies how much PU evaluation understates the model (true AUPR was ~3.2x
the confirmed-only AUPR out of sample at prior 0.13, one export and seed). Once this loop looks sane, swap in the
real files -- same commands, minus eval_ranking (no truth file exists; score a
finished inspection campaign with it instead).

`--data real` ingests a utility export directly (`ntl/real_data.py`):

```
meters.csv     meter_id, date, kwh        (long; sub-daily summed to daily)
               or wide: meter_id + date columns
topology.csv   meter_id, transformer_id
heads.csv      transformer_id, date, kwh   or   transformer_id, head_mean
labels.csv     meter_id, label             (1 = confirmed theft; optional)
static.csv     meter_id + any columns      (optional; categoricals one-hot)
```

```bash
python -m scripts.train_pistgat --data real \
  --meters meters.csv --topology topology.csv --heads heads.csv \
  --labels labels.csv --static-file static.csv --region-col district \
  --days 365 --ssl-epochs 8 --epochs 20 --save-ranking inspect.csv
```

The loader aligns everything on a shared daily calendar, drops low-coverage
meters and unmapped/headless feeders (reporting counts), prints per-feeder
balance diagnostics — **negative residuals flag topology/CT/clock errors**,
the most common real-data failure — and auto-estimates the technical-loss
fraction (25th percentile of healthy residuals; override with `--tl-frac`).
The region column is used for the fairness check only, not as a model input.
PU mode is enabled automatically: confirmed cases are positives, everyone
else is *unlabeled*, never assumed honest.

Honest caveats for real runs: metrics computed against confirmed-only labels
are **lower bounds** (unconfirmed thieves count as false positives); set `--prior`
from an audit estimate or sweep it (the confirmed rate badly underestimates it,
and in rehearsal it flagged no meters at all);
labels are not yet time-windowed (a thief is marked for the whole window,
including post-regularization periods); inspected-clean negatives are treated
as unlabeled; unmetered loads (street lighting) inflate residuals. The real
deliverable is `--save-ranking`: a per-meter CSV with P(confirmed theft), calibrated on
confirmed labels and so lower than the true theft probability,
estimated stolen kWh/day, and the cost-aware score, sorted detection-first.
For `--aug learned/both`, post-inspection corrected-consumption pairs are
needed (not wired yet). SSL needs only unlabeled series — run it first.

## Design-to-evidence summary

| design claim | evidence in this repo |
|---|---|
| static features help (partly by simulator design) | ablation, 3 seeds: temporal 0.814 → +static 0.896 AUPR; 0.877 with label signal removed from age and alarms |
| feeder attention and physics head | no measurable ranking gain at this scale: +graph 0.892, full 0.881 AUPR |
| physics head grounds & ranks | residual↔stolen corr 0.94–0.96; stolen-energy estimates drive cost-aware ranking |
| PU is the honest framing | prior sweep and rehearsal in `outputs/reference_cpu/report_numbers.txt`; the confirmed-rate prior flags nobody |
| SSL pays off when labels are scarce | **not supported**: PU test AUPR 0.535 → 0.525 over 3 seeds |
| combined augmentation | best ROC on all 3 seeds; no reliable gain on novel h7 (table above) |
