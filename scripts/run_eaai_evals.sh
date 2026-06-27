#!/usr/bin/env bash
# =============================================================================
# EAAI Evaluation Orchestrator — runs all post-training evaluation experiments
# =============================================================================
#
# Runs sequentially after WP1 training produces main_team*_gen*.pt checkpoints.
# Each experiment produces a JSON in logs/ that the figure pipeline consumes.
#
# Resource note: shares GPU with WP1 training if it's still running. num-envs=4
# keeps memory below 35 GB so it fits alongside the ~60 GB training allocation.
# Once WP1 finishes, re-run with --num-envs 12 for full-parallel final numbers.
#
# Usage:
#     bash scripts/run_eaai_evals.sh [CHECKPOINT_DIR] [NUM_ENVS]
#
# Default: CHECKPOINT_DIR=checkpoints/wp1_gate_seed42  NUM_ENVS=4
# =============================================================================

set -euo pipefail

CKPT_DIR="${1:-checkpoints/wp1_gate_seed42}"
NUM_ENVS="${2:-4}"
N_GAMES="${N_EVAL_GAMES:-50}"
MAX_STEPS="${MAX_STEPS:-500}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/fluxphased/bin/python}"

cd "$(dirname "$0")/.."

echo "============================================================"
echo "EAAI Evaluation Orchestrator"
echo "============================================================"
echo "Checkpoint dir : $CKPT_DIR"
echo "Num envs       : $NUM_ENVS"
echo "Games per cell : $N_GAMES"
echo "Max steps      : $MAX_STEPS"
echo "Python         : $PYTHON"
echo "Start time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Sanity: ensure checkpoint exists
if [ ! -f "$CKPT_DIR/main_team0_gen0.pt" ]; then
    echo "ERROR: no main_team0_gen*.pt in $CKPT_DIR" >&2
    exit 2
fi

# --- WP3.2 Robustness sweep (5 damage cells + baseline) ---------------------
echo
echo "============================================================"
echo "[1/4] WP3.2 robustness sweep (5 damage cells + baseline)"
echo "============================================================"
$PYTHON -u scripts/wp3_robustness_eval.py \
    --checkpoint-dir "$CKPT_DIR" \
    --baseline-config configs/wp1_gate.yaml \
    --n-eval-games "$N_GAMES" --max-steps "$MAX_STEPS" \
    --num-envs "$NUM_ENVS" \
    --output-json logs/wp3_robustness_eval.json \
    --output-log logs/wp3_robustness_eval.log \
    2>&1 | tee logs/wp3_robustness_eval_run.log
echo "[1/4] done: $(date '+%H:%M:%S')"

# --- WP3.1 Achieved RMSE / CRLB ratio ---------------------------------------
echo
echo "============================================================"
echo "[2/4] WP3.1 achieved-RMSE / CRLB ratio (Fig C)"
echo "============================================================"
$PYTHON -u scripts/wp3_crlb_achieved.py \
    --checkpoint-dir "$CKPT_DIR" \
    --baseline-config configs/wp1_gate.yaml \
    --n-episodes 20 --max-steps 300 \
    --num-envs "$NUM_ENVS" \
    --output-json logs/wp3_crlb_achieved.json \
    --output-log logs/wp3_crlb_achieved.log \
    --output-fig figures/wp3_crlb_achieved.pdf \
    2>&1 | tee logs/wp3_crlb_achieved_run.log
echo "[2/4] done: $(date '+%H:%M:%S')"

# --- WP4 Generalization matrix (6 OOD conditions + baseline) ----------------
echo
echo "============================================================"
echo "[3/4] WP4 generalization matrix (6 OOD conditions)"
echo "============================================================"
$PYTHON -u scripts/wp4_generalization_eval.py \
    --checkpoint-dir "$CKPT_DIR" \
    --train-config configs/wp1_gate.yaml \
    --test-configs configs/wp4_dynamics_static.yaml \
                    configs/wp4_dynamics_maneuver.yaml \
                    configs/wp4_geom_tight_baseline.yaml \
                    configs/wp4_geom_wide_baseline.yaml \
                    configs/wp4_ew_jam.yaml \
                    configs/wp4_ew_exposure.yaml \
    --n-eval-games "$N_GAMES" --max-steps "$MAX_STEPS" \
    --num-envs "$NUM_ENVS" \
    --output-json logs/wp4_generalization.json \
    --output-log logs/wp4_generalization.log \
    --output-fig figures/wp4_generalization.pdf \
    2>&1 | tee logs/wp4_generalization_run.log
echo "[3/4] done: $(date '+%H:%M:%S')"

# --- WP2 Main comparison (FluxLeague vs Classical MPC) ----------------------
echo
echo "============================================================"
echo "[4/4] WP2 main comparison: FluxLeague-full vs Classical MPC"
echo "============================================================"
$PYTHON -u scripts/wp2_main_comparison.py \
    --entries "fluxleague_red=ckpt:$CKPT_DIR" \
              "mpc=rule:classical_mpc" \
    --config configs/wp1_gate.yaml \
    --n-eval-games "$N_GAMES" --max-steps "$MAX_STEPS" \
    --num-envs "$NUM_ENVS" \
    --output-json logs/wp2_main_comparison.json \
    --output-log logs/wp2_main_comparison.log \
    --output-fig figures/wp2_winrate_heatmap.pdf \
    2>&1 | tee logs/wp2_main_comparison_run.log
echo "[4/4] done: $(date '+%H:%M:%S')"

# --- Summary ----------------------------------------------------------------
echo
echo "============================================================"
echo "EAAI Evaluation Suite — Complete"
echo "============================================================"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo
echo "Results JSONs:"
ls -la logs/wp3_robustness_eval.json \
       logs/wp3_crlb_achieved.json \
       logs/wp4_generalization.json \
       logs/wp2_main_comparison.json 2>/dev/null
echo
echo "Figures:"
ls -la figures/wp3_crlb_achieved.pdf \
       figures/wp4_generalization.pdf \
       figures/wp2_winrate_heatmap.pdf 2>/dev/null
echo
echo "Next: feed JSONs to paper figure pipeline (TBD)."
