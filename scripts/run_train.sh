#!/bin/bash
# scripts/run_train.sh — standard training launcher
# ----------------------------------------------------------------------------
# Why this exists: python stdout is block-buffered (4KB) when piped to a file,
# so a long rollout phase can look "hung" (no log writes for ~hour) while the
# process is actually computing at 100% CPU. This launcher:
#   1. PYTHONUNBUFFERED=1 + python -u  → byte-level flush, no block buffering
#   2. -X faulthandler                  → traceback dump on SIGSEGV/SIGABRT
#   3. prints PID + py-spy hint         → on suspected stall, run py-spy dump
#                                         (NON-invasive) instead of SIGUSR1
#                                         (which terminates by default).
#
# Usage:
#   scripts/run_train.sh [config.yaml] [log_path]
#   defaults: configs/laser_25x25_pro6000_stable.yaml, logs/phase1_pfsp.log
#
# Watch live:   tail -f <log_path>
# Stall check:  /home/ubuntu/miniconda3/envs/fluxphased/bin/py-spy dump --pid <PID>
# ----------------------------------------------------------------------------
set -e

CONFIG="${1:-configs/laser_25x25_pro6000_stable.yaml}"
LOG="${2:-logs/phase1_pfsp.log}"

# Always operate from repo root regardless of where script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

mkdir -p logs

# Activate conda env (fluxphased) — required for torch + warp + py-spy
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fluxphased

# Unbuffered stdout/stderr — root fix for the "fake hang" we hit on 2026-06-30
export PYTHONUNBUFFERED=1

# Launch training in background
nohup python -X faulthandler -u -m algo._shared.train_laser --config "$CONFIG" > "$LOG" 2>&1 &
PID=$!
disown

# Persist PID for watchdog / py-spy
echo "$PID" > /tmp/train_laser.pid

echo "Training started"
echo "  PID:     $PID"
echo "  config:  $CONFIG"
echo "  log:     $LOG"
echo ""
echo "Live tail:      tail -f $LOG"
echo "Stall check:    /home/ubuntu/miniconda3/envs/fluxphased/bin/py-spy dump --pid $PID"
echo "  (py-spy is NON-invasive — never use kill -USR1, it terminates by default)"
