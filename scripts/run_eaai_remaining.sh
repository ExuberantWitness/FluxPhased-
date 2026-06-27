#!/usr/bin/env bash
# Run WP3.1 + WP4 + WP2 mini-comparison sequentially (WP3.2 already done).
# Use after WP3.2 robustness eval completes.

set -euo pipefail

CKPT_DIR="${1:-checkpoints/wp1_gate_seed42}"
NUM_ENVS="${2:-4}"
N_GAMES="${N_EVAL_GAMES:-50}"
MAX_STEPS="${MAX_STEPS:-500}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/fluxphased/bin/python}"

cd "$(dirname "$0")/.."

echo "============================================================"
echo "EAAI Remaining Evals (WP3.1 + WP4 + WP2)"
echo "============================================================"
echo "Checkpoint dir : $CKPT_DIR"
echo "Num envs       : $NUM_ENVS"
echo "Start time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo
echo "[1/3] WP3.1 CRLB achieved RMSE / ratio..."
$PYTHON -u scripts/wp3_crlb_achieved.py \
    --checkpoint-dir "$CKPT_DIR" \
    --n-episodes 20 --max-steps 300 --num-envs "$NUM_ENVS" \
    --output-json logs/wp3_crlb_achieved.json \
    --output-log logs/wp3_crlb_achieved.log \
    --output-fig figures/wp3_crlb_achieved.pdf \
    2>&1 | tee logs/wp3_crlb_achieved_run.log
echo "[1/3] done: $(date '+%H:%M:%S')"

echo
echo "[2/3] WP4 generalization matrix..."
$PYTHON -u scripts/wp4_generalization_eval.py \
    --checkpoint-dir "$CKPT_DIR" \
    --test-configs configs/wp4_dynamics_static.yaml \
                    configs/wp4_dynamics_maneuver.yaml \
                    configs/wp4_geom_tight_baseline.yaml \
                    configs/wp4_geom_wide_baseline.yaml \
                    configs/wp4_ew_jam.yaml \
                    configs/wp4_ew_exposure.yaml \
    --n-eval-games "$N_GAMES" --max-steps "$MAX_STEPS" --num-envs "$NUM_ENVS" \
    --output-json logs/wp4_generalization.json \
    --output-log logs/wp4_generalization.log \
    --output-fig figures/wp4_generalization.pdf \
    2>&1 | tee logs/wp4_generalization_run.log
echo "[2/3] done: $(date '+%H:%M:%S')"

echo
echo "[3/3] WP2 mini comparison (FluxLeague vs Classical MPC)..."
$PYTHON -u scripts/wp2_main_comparison.py \
    --entries "fluxleague_red=ckpt:$CKPT_DIR" \
              "mpc=rule:classical_mpc" \
    --config configs/wp1_gate.yaml \
    --n-eval-games "$N_GAMES" --max-steps "$MAX_STEPS" --num-envs "$NUM_ENVS" \
    --output-json logs/wp2_main_comparison.json \
    --output-log logs/wp2_main_comparison.log \
    --output-fig figures/wp2_winrate_heatmap.pdf \
    2>&1 | tee logs/wp2_main_comparison_run.log
echo "[3/3] done: $(date '+%H:%M:%S')"

echo
echo "============================================================"
echo "All evals complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Generate consolidated summary
$PYTHON scripts/eaai_results_summary.py
