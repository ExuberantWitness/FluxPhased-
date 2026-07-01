#!/bin/bash
# monitor_run.sh — hourly monitor + finalizer for an APP-style experiment run.
#
# WHAT:
#   - If training is still running: refresh metrics.json + AGENTS.md + figures
#     from the live log so the experiments/ directory is always current.
#   - If training just finished: copy the live log into the run directory,
#     write final metadata, run compare_runs.py, regenerate everything, and
#     print "DONE" so the caller (cron) knows to commit + push.
#
# USAGE:
#   bash scripts/experiments/monitor_run.sh experiments/phase1.5_mappo_seed42 \
#        logs/phase1.5_mappo.log configs/laser_25x25_mappo.yaml
#
# EXIT CODES:
#   0  — refreshed, still running
#   1  — ref error (bad paths, missing pid)
#   99 — training finished AND finalization succeeded (caller: commit + push)
set -euo pipefail

RUN_DIR="${1:?usage: monitor_run.sh RUN_DIR LIVE_LOG CONFIG}"
LIVE_LOG="${2:?}"
CONFIG="${3:?}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fluxphased

# Refresh metrics + figures from the live log (training-running or final).
if [ -f "$LIVE_LOG" ]; then
    python scripts/experiments/parse_log_to_metrics.py \
        --log "$LIVE_LOG" \
        --out "$RUN_DIR/metrics.json" \
        --run-id "$(basename "$RUN_DIR")" || echo "[monitor] metrics parse failed (training may be mid-line)"
fi

# Regenerate figures + AGENTS.md so the directory is browsable mid-run.
if [ -f "$RUN_DIR/metrics.json" ]; then
    python scripts/experiments/make_figures.py \
        --run "$(basename "$RUN_DIR"):$RUN_DIR/metrics.json" \
        --out-dir "$RUN_DIR/figures" 2>/dev/null || echo "[monitor] figures skipped (incomplete data)"
fi
python scripts/experiments/write_agents_md.py \
    --run-dir "$RUN_DIR" \
    --baseline-dir experiments/phase1_pfsp_seed42

# Detect whether training is still running via /tmp/train_laser.pid
PID_FILE=/tmp/train_laser.pid
PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    TRAINING_RUNNING=1
else
    TRAINING_RUNNING=0
fi

# Capture / recover start time. The monitor is invoked repeatedly, so the FIRST
# call (when training just started) records the start time to a file. Later
# calls reuse it. On finalization (training done), we read it back.
START_FILE="$RUN_DIR/.start_iso"
if [ ! -f "$START_FILE" ]; then
    # First monitor call — record "now" as the start. If training is already
    # gone by the time we get here (race), use mtime of the live log as a fallback.
    if [ "$TRAINING_RUNNING" = "1" ]; then
        date -Iseconds > "$START_FILE"
    else
        stat -c %y "$LIVE_LOG" 2>/dev/null | xargs -I{} date -d '{}' -Iseconds > "$START_FILE" || date -Iseconds > "$START_FILE"
    fi
fi
START_ISO="$(cat "$START_FILE")"

if [ "$TRAINING_RUNNING" = "1" ]; then
    echo "[monitor] PID=$PID still running — refreshed metrics/figures/AGENTS.md"
    # Quick progress summary
    if [ -f "$RUN_DIR/metrics.json" ]; then
        python -c "
import json
m = json.load(open('$RUN_DIR/metrics.json'))
n = m['iter_count']
f = m.get('final') or {}
print(f'[monitor] iter={n}/20  kr={f.get(\"kr_m\",\"n/a\")}m  cum_red={f.get(\"cum_red\",\"n/a\")}  aim_res={f.get(\"aim_res_m\",\"n/a\")}m')
"
    fi
    exit 0
fi

# Training finished — finalize.
echo "[monitor] training PID=$PID no longer running — finalizing"
cp -f "$LIVE_LOG" "$RUN_DIR/train.log"

# Re-parse the final log (now complete) + regenerate figures
python scripts/experiments/parse_log_to_metrics.py \
    --log "$RUN_DIR/train.log" \
    --out "$RUN_DIR/metrics.json" \
    --run-id "$(basename "$RUN_DIR")"
python scripts/experiments/make_figures.py \
    --run "$(basename "$RUN_DIR"):$RUN_DIR/metrics.json" \
    --out-dir "$RUN_DIR/figures"

# Update metadata end time + wall-clock
python scripts/experiments/write_metadata.py \
    --run-dir "$RUN_DIR" \
    --config "$CONFIG" \
    --seed 42 \
    --start "$START_ISO" \
    --end   "$(date -Iseconds)" \
    --reproduce "bash scripts/run_train.sh $CONFIG $LIVE_LOG" \
    --notes "Phase 1.5 MAPPO baseline. CTDE team critic + PFSP off. Compared vs PfspFix reference."

# Generate comparison.md (only if candidate has final values)
python scripts/experiments/compare_runs.py \
    --baseline experiments/phase1_pfsp_seed42/metrics.json \
    --candidate "$RUN_DIR/metrics.json" \
    --baseline-name PfspFix \
    --candidate-name MAPPO \
    --out "$RUN_DIR/comparison.md" || echo "[monitor] compare_runs.py failed"

# Regenerate per-run AGENTS.md with final values
python scripts/experiments/write_agents_md.py \
    --run-dir "$RUN_DIR" \
    --baseline-dir experiments/phase1_pfsp_seed42

# Regenerate top-level index
python scripts/experiments/write_agents_md.py --index

echo "[monitor] DONE — all artifacts up to date, ready to commit"
exit 99
