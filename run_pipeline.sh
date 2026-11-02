#!/usr/bin/env bash
# Run every experiment end to end, for every seed, and save the figures per seed.
#
#   ./run_pipeline.sh --smoke                      # quick check of the whole flow
#   ./run_pipeline.sh --seeds "0 1 2 3 4" --epochs 200 --base-epochs 150 \
#       --exp-epochs 100 --ssl-epochs 50 --cvae-epochs 2000 --out outputs/gpu_run
#
# Re-running the same command resumes: finished steps are skipped and failed ones
# retried. Add seeds to an existing run with the same settings and a longer --seeds.
#
# Output, under --out:
#   figures/seed_N/    fig1-fig8 and s1-s5 for one seed, written as each seed finishes
#   figures/all_seeds/ fig1-fig8, mean over the finished seeds
#   results/           results.csv, report_numbers.txt
#   logs/              one log per step; runs/ checkpoints; exports/ rehearsal data
#
# Options (defaults are the settings used in the report):
#   --seeds "0 1 2"      seeds to run                     (default "0 1 2 3 4")
#   --epochs N           PI-STGAT epochs                  (20)
#   --base-epochs N      1-D CNN epochs                   (15)
#   --exp-epochs N       augmentation experiment epochs   (8)
#   --ssl-epochs N       masked pretraining epochs        (8)
#   --cvae-epochs N      cVAE epochs                      (400)
#   --n-feeders N, --T N simulator size                   (150, 365)
#   --out DIR            output folder                    (outputs/run)
#   --python PATH        interpreter (default: .venv/bin/python if present, else python)
#   --allow-cpu          run even when no GPU is visible
#   --smoke              tiny sizes, 2 seeds, outputs/smoke, for checking the flow
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="0 1 2 3 4"; EPOCHS=20; BASE_EPOCHS=15; EXP_EPOCHS=8; SSL_EPOCHS=8; CVAE_EPOCHS=400
N_FEEDERS=150; T=365; BASE_N=4000; RH_FEEDERS=60; RH_T=240; BOOT=2000
OUT="outputs/run"; PY=""; ALLOW_CPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --base-epochs) BASE_EPOCHS="$2"; shift 2 ;;
    --exp-epochs) EXP_EPOCHS="$2"; shift 2 ;;
    --ssl-epochs) SSL_EPOCHS="$2"; shift 2 ;;
    --cvae-epochs) CVAE_EPOCHS="$2"; shift 2 ;;
    --n-feeders) N_FEEDERS="$2"; shift 2 ;;
    --T) T="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --allow-cpu) ALLOW_CPU=1; shift ;;
    --smoke) SEEDS="0 1"; EPOCHS=1; BASE_EPOCHS=1; EXP_EPOCHS=1; SSL_EPOCHS=1; CVAE_EPOCHS=20
             N_FEEDERS=30; T=90; BASE_N=400; RH_FEEDERS=30; RH_T=90; BOOT=20; OUT="outputs/smoke"; shift ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
done

if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; else PY="python"; fi
fi
[[ "$OUT" = /* ]] || OUT="$ROOT/$OUT"
mkdir -p "$OUT" && cd "$OUT" || exit 1
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg

# ---- preflight --------------------------------------------------------------
if ! "$PY" -c "import torch, xgboost, sklearn, pandas, scipy, matplotlib, pytest" 2>/dev/null; then
  echo "[error] '$PY' is missing required packages. To set them up:"
  echo "    cd $ROOT && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if "$PY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  DEVICE="$("$PY" -c "import torch; print(torch.cuda.get_device_name(0))")"
elif [[ $ALLOW_CPU = 1 ]]; then
  DEVICE="CPU"
else
  echo "[error] PyTorch cannot see a GPU. Check nvidia-smi and the torch build, or pass --allow-cpu."
  exit 1
fi

# Results from different settings must not share a folder; extra seeds are fine.
CONFIG="epochs=$EPOCHS base=$BASE_EPOCHS exp=$EXP_EPOCHS ssl=$SSL_EPOCHS cvae=$CVAE_EPOCHS n_feeders=$N_FEEDERS T=$T n=$BASE_N rh=$RH_FEEDERS/$RH_T"
if [[ -f config.txt && "$(cat config.txt)" != "$CONFIG" ]]; then
  echo "[error] $OUT was started with different settings:"
  echo "    existing: $(cat config.txt)"
  echo "    now:      $CONFIG"
  echo "Pass a new --out folder."
  exit 1
fi
echo "$CONFIG" > config.txt
mkdir -p logs .done results figures

# ---- step runner ------------------------------------------------------------
STEPS_PER_SEED=31
N_SEEDS=$(wc -w <<< "$SEEDS")
TOTAL=$((N_SEEDS * STEPS_PER_SEED))
DONE=0; RAN_SECONDS=0; RAN=0; FAILED=()
START=$SECONDS

human() { printf '%dh%02dm%02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

step() {   # step NAME command...: skip when finished; log to logs/NAME.log
  local name="$1"; shift
  DONE=$((DONE + 1))
  if [[ -f ".done/${name//\//__}" ]]; then
    echo "[$DONE/$TOTAL] $name: already done"
    return 0
  fi
  mkdir -p "logs/$(dirname "$name")"
  local t0=$SECONDS
  if "$@" > "logs/$name.log" 2>&1; then
    touch ".done/${name//\//__}"
    local took=$((SECONDS - t0))
    RAN=$((RAN + 1)); RAN_SECONDS=$((RAN_SECONDS + took))
    echo "[$DONE/$TOTAL] $name: $(human $took), about $(human $((RAN_SECONDS / RAN * (TOTAL - DONE)))) left"
  else
    FAILED+=("$name")
    echo "[$DONE/$TOTAL] $name: FAILED, see $OUT/logs/$name.log"
  fi
}

report() {   # figures and tables are rebuilt every time, never skipped
  local name="$1"; shift
  mkdir -p "logs/$(dirname "$name")"
  "$@" > "logs/$name.log" 2>&1 || { FAILED+=("$name"); echo "  $name: FAILED, see $OUT/logs/$name.log"; }
}

run()  { "$PY" -m "scripts.$1" "${@:2}"; }

echo "[pipeline] device: $DEVICE"
echo "[pipeline] seeds: $SEEDS | $CONFIG"
echo "[pipeline] output: $OUT"

if ! "$PY" -m pytest -q -p no:cacheprovider "$ROOT/tests" > logs/tests.log 2>&1; then
  echo "[error] regression tests failed, see $OUT/logs/tests.log"
  exit 1
fi
echo "[pipeline] regression tests passed"

# ---- one seed ---------------------------------------------------------------
run_seed() {
  local s=$1 d="seed_$s" fig="figures/seed_$s" sim="data/seed_$s/sim_flat.csv" rh="exports/seed_$s"
  mkdir -p "data/seed_$s" "$fig" "runs/$d"
  local pi=(train_pistgat --n-feeders "$N_FEEDERS" --T "$T" --epochs "$EPOCHS" --seed "$s")

  # base models
  step "$d/base_cnn"     run train --n "$BASE_N" --T "$T" --epochs "$BASE_EPOCHS" --seed "$s" --out "runs/$d/base_cnn"
  step "$d/xgboost"      run baselines --n "$BASE_N" --T "$T" --seed "$s"
  step "$d/export_sim"   run export_sim_csv --n-feeders "$N_FEEDERS" --T "$T" --seed "$s" --out "$sim"
  step "$d/base_cnn_sim" run train --data "$sim" --epochs "$BASE_EPOCHS" --seed "$s" --out "runs/$d/base_cnn_sim"

  # ablation
  step "$d/abl_temporal" run "${pi[@]}" --no-graph --no-static --no-physics --out "runs/$d/abl_temporal"
  step "$d/abl_static"   run "${pi[@]}" --no-graph --no-physics --out "runs/$d/abl_static"
  step "$d/abl_graph"    run "${pi[@]}" --no-physics --out "runs/$d/abl_graph"
  step "$d/abl_full"     run "${pi[@]}" --out "runs/$d/abl_full"

  # full design and side experiments
  step "$d/full_ssl_aug" run "${pi[@]}" --ssl-epochs "$SSL_EPOCHS" --aug both --aug-p 0.15 --cvae-epochs "$CVAE_EPOCHS" --out "runs/$d/full_ssl_aug"
  step "$d/pu"           run "${pi[@]}" --pu --pu-reveal 0.3 --out "runs/$d/pu"
  step "$d/pu_ssl"       run "${pi[@]}" --pu --pu-reveal 0.3 --ssl-epochs "$SSL_EPOCHS" --out "runs/$d/pu_ssl"
  step "$d/pos_enc"      run "${pi[@]}" --pos-enc --out "runs/$d/pos_enc"
  step "$d/no_region"    run "${pi[@]}" --no-region-feature --out "runs/$d/no_region"
  step "$d/nolabel_static" run "${pi[@]}" --no-label-static --no-graph --no-physics --out "runs/$d/nolabel_static"
  step "$d/nolabel_full" run "${pi[@]}" --no-label-static --out "runs/$d/nolabel_full"

  # unseen-attack augmentation experiment
  step "$d/aug_prepare"  run exp_augment prepare --seed "$s" --cvae-epochs "$CVAE_EPOCHS"
  local a
  for a in none jokar learned both; do
    step "$d/aug_$a"     run exp_augment run --aug "$a" --seed "$s" --epochs "$EXP_EPOCHS"
  done

  # deployment rehearsal: this seed's export, default prior, then the prior sweep
  step "$d/rh_export"    run make_fake_export --out-dir "$rh" --n-feeders "$RH_FEEDERS" --T "$RH_T" --seed "$s" --unmetered-kwh 1.5
  local real=(train_pistgat --data real --meters "$rh/meters.csv" --topology "$rh/topology.csv"
              --heads "$rh/heads.csv" --labels "$rh/labels.csv" --static-file "$rh/static.csv"
              --region-col district --days "$RH_T" --epochs "$EPOCHS" --ssl-epochs "$SSL_EPOCHS" --seed "$s")
  step "$d/rh_prior_default" run "${real[@]}" --out "runs/$d/rh_prior_default" --save-ranking "$rh/inspect_prior_default.csv"
  step "$d/rh_prior_default_eval" run eval_ranking --ranking "$rh/inspect_prior_default.csv" --truth "$rh/truth_labels.csv" --tag rehearsal-prior-default --seed "$s"
  local p tag
  for p in 0.06 0.10 0.13 0.18; do
    tag="rehearsal-p$p"; [[ $p = 0.13 ]] && tag="rehearsal"
    step "$d/rh_p$p"      run "${real[@]}" --prior "$p" --out "runs/$d/rh_p$p" --save-ranking "$rh/inspect_p$p.csv"
    step "$d/rh_p${p}_eval" run eval_ranking --ranking "$rh/inspect_p$p.csv" --truth "$rh/truth_labels.csv" --tag "$tag" --seed "$s"
  done

  # figures and numbers for this seed
  report "$d/sim_analysis" run sim_analysis --seed "$s" --n-feeders "$N_FEEDERS" --T "$T" --out "$fig"
  ( cd "logs/$d" && cat abl_temporal.log abl_static.log abl_graph.log abl_full.log full_ssl_aug.log > simulator_runs.log 2>/dev/null )
  report "$d/plot_results" run plot_results --seed "$s" --out "$fig"
  report "$d/plot_val_curves" run plot_val_curves "logs/$d/simulator_runs.log" "logs/$d/rh_prior_default.log" "$fig"
  report "$d/report_numbers" run report_numbers --skip-table --rh "$rh" --seed "$s" --boot "$BOOT"
  cp "logs/$d/report_numbers.log" "results/rehearsal_seed_$s.txt" 2>/dev/null

  # all finished seeds so far, so a stopped run still leaves usable figures
  report all_seeds_plot run plot_results --out figures/all_seeds
  report all_seeds_numbers run report_numbers --rh none
  cp logs/all_seeds_numbers.log results/report_numbers.txt 2>/dev/null
  echo "[pipeline] seed $s finished; figures in $OUT/$fig"
}

for s in $SEEDS; do run_seed "$s"; done
report rehearsal_seeds run plot_rehearsal_seeds . figures/all_seeds

echo
echo "[pipeline] finished in $(human $((SECONDS - START)))"
echo "  per-seed figures: $OUT/figures/seed_*/"
echo "  mean over seeds:  $OUT/figures/all_seeds/"
echo "  tables:           $OUT/results/"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[pipeline] ${#FAILED[@]} step(s) failed; re-run the same command to retry them:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
