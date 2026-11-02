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
  energy correlates **0.94–0.98** (seeds 0–4).
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

**Measured effect with scarce labels (PU, 30% of thieves revealed, full model,
simulator, 5 seeds; hidden labels stay hidden on training and validation):**

| setting              | test AUPR     | test ROC-AUC |
|----------------------|:-------------:|:------------:|
| from scratch         | 0.657 ± 0.125 | 0.857        |
| + SSL (50 epochs)    | 0.732 ± 0.067 | 0.894        |

The mean favours SSL, but AUPR was higher on three seeds and lower on two; five
seeds do not settle it.

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

**Generalization experiment** (`exp_augment.py`): train a temporal-only model on a
network whose thieves use only h1–h4 at low prevalence (38–50 confirmed pairs), test
on an independent network using all eight attacks. Mean over 5 seeds:

| augmentation   | AUPR  | ROC-AUC | recall@10% on unseen h7 | recall@10% on unseen h8 |
|----------------|:-----:|:-------:|:-----:|:-----:|
| none           | 0.634 | 0.778   | 0.52  | 0.16  |
| fixed (Jokar)  | 0.764 | 0.857   | 0.53  | 0.09  |
| learned (cVAE) | 0.685 | 0.828   | 0.62  | 0.34  |
| both           | **0.815** | **0.913** | 0.60  | 0.18  |

Reading: against no augmentation, AUPR is higher on 5/5 seeds for both and for the
fixed functions, and on 4/5 for the cVAE; both beats fixed alone on 3/5. The cVAE
raises recall on the unseen ramp h8 on every seed, because a multiplicative
under-reporting mask is inside its support; it leaves the energy-preserving h5/h6
unchanged, which are outside it. The fixed functions contain h5 and h6 by hand, so
those are not unseen for that variant, and they trade h1 recall for h5. h7 differences
are within seed noise. Augmentation is **physically coherent**: converting an honest
customer leaves the head reading fixed, so the feeder's conservation residual grows
by exactly the stolen amount -- the same signature a real thief leaves.

## Reported run (simulator, 150 feeders, T=365, 200 epochs, 5 seeds, one RTX 4080)

Produced by the `run_pipeline.sh` command above; everything is in `outputs/gpu_run/`.
Mean ± s.d. over 5 seeds:

| configuration              | AUPR          | ROC-AUC       |
|----------------------------|:-------------:|:-------------:|
| temporal only              | 0.901 ± 0.037 | 0.950 ± 0.026 |
| + static                   | 0.940 ± 0.029 | 0.977 ± 0.014 |
| + feeder attention         | 0.948 ± 0.019 | 0.980 ± 0.011 |
| + physics (full)           | 0.935 ± 0.028 | 0.977 ± 0.010 |
| full + SSL + augmentation  | 0.969 ± 0.009 | 0.984 ± 0.007 |
| full + positional code     | 0.971 ± 0.007 | 0.988 ± 0.007 |

Paired per seed: static attributes help on 5/5 seeds (still 5/5 with age and alarms
redrawn label-free); feeder attention 3/5, not separable from noise; the physics
branch lowers AUPR on 4/5 (−0.013), consistent with the conservation term fixing
feeder totals rather than within-feeder ranking; SSL + augmentation and the
positional code each raise AUPR on 5/5. Per-attack recall@10%: h2–h4 0.87–1.00,
h7 0.81–0.94, h8 0.58–0.76, h5 0.60–0.84, h1 0.35 → 0.53–0.56 once attention or
physics is added; h6 reversal is the hard tail at 0.28–0.42 (0.48 with the
positional code). Test prevalence varies from 0.057 to 0.216 across seeds, so
report spreads, not one seed.

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

The report source, figures and PDF are in `report/`. The run behind the report is
`outputs/gpu_run/` (results table, printed numbers, rehearsal summaries and the
all-seed figures; checkpoints and logs are regenerated by the pipeline). Replot with
`python -m scripts.plot_results --csv outputs/gpu_run/results/results.csv --out my_figures`.

A caution on **fig1**: it is not a controlled comparison. On the simulator the
per-customer CNN reaches AUPR 0.950 ± 0.021 and full PI-STGAT 0.935 ± 0.028, but the
CNN splits customers (meters of one feeder fall into both training and test) while
PI-STGAT holds out whole feeders, and the CNN trains with h1–h6 augmentation that
matches six of the eight test attacks. No architectural advantage is established;
a same-split, same-augmentation comparison is the next step.

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
outputs/gpu_run/     the 5-seed run behind the report (results/, figures/all_seeds/)
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
quantifies how much PU evaluation understates the model (over 5 exports x 5 priors
on test feeders, true AUPR averaged 3.5x the confirmed-only AUPR; predicted 3.7x,
range 1.2-8.3). Once this loop looks sane, swap in the
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
and in rehearsal it made training unstable, with validation ROC-AUC below 0.5 in
22% of epochs on average);
labels are not yet time-windowed (a thief is marked for the whole window,
including post-regularization periods); inspected-clean negatives are treated
as unlabeled; unmetered loads (street lighting) inflate residuals. The real
deliverable is `--save-ranking`: a per-meter CSV with P(confirmed theft), calibrated on
confirmed labels and so lower than the true theft probability,
estimated stolen kWh/day, and the cost-aware score, sorted detection-first.
For `--aug learned/both`, post-inspection corrected-consumption pairs are
needed (not wired yet). SSL needs only unlabeled series — run it first.

## Design-to-evidence summary

| design claim | evidence in this repo (5 seeds) |
|---|---|
| static features help (partly by simulator design) | temporal 0.901 → +static 0.940 AUPR, 5/5 seeds; still 5/5 with age and alarms label-free |
| feeder attention | not separable from noise: 0.948, higher on 3/5 seeds |
| physics head | no discrimination gain (0.935, lower on 4/5); its value is the stolen-energy estimate: top 10% captures 0.80 of stolen kWh with cost-aware ranking |
| conservation identity | residual↔stolen correlation 0.94–0.98 over five networks |
| PU is the honest framing | the confirmed-rate prior makes training unstable; confirmed-only AUPR understates true AUPR ~3.5x (`outputs/gpu_run/results/`) |
| deployment rehearsal | top 5% of each test ranking holds 76–93% thieves vs a 16% base rate, five defective exports |
| SSL pays off when labels are scarce | not settled: PU AUPR 0.657 → 0.732, higher on 3/5 seeds |
| combined augmentation | best AUPR/ROC on the unseen-attack test; cVAE helps the unseen h8 on 5/5 seeds |
| advantage over a per-customer CNN | **not established**: CNN 0.950 vs full 0.935 under different splits |
