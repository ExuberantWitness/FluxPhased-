#!/bin/bash
# Ablation runner: 4 variants × 2h each, sequential.
# Auto-kills at 2h, logs metrics, picks winner by kill_rate trend.
#
# Usage: bash run_ablation.sh [variant_name]
#   no arg → run all 4 sequentially
#   v1|v2|v3|v4 → run single variant

set -u
cd /home/ubuntu/CODE/FluxPhased-

# WandB offline (avoid fake-key network failures)
export WANDB_MODE=offline
export WANDB_SILENT=true
export TOKENIZERS_PARALLELISM=false

PYTHON=/home/ubuntu/miniconda3/envs/fluxphased/bin/python
DURATION=7200  # 2 hours in seconds
LOG_DIR=logs/ablation_f1f8
mkdir -p "$LOG_DIR"

run_variant() {
    local NAME=$1
    local CFG=configs/ablation_f1f8/${NAME}.yaml
    local LOG="$LOG_DIR/${NAME}_$(date +%Y%m%d_%H%M%S).log"
    local PIDFILE="$LOG_DIR/${NAME}.pid"

    if [ ! -f "$CFG" ]; then
        echo "[!] Config $CFG not found, skipping $NAME"
        return 1
    fi

    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] Starting $NAME (2h budget)"
    echo "  Config: $CFG"
    echo "  Log:    $LOG"
    echo "=============================================================="

    # Start training in background
    $PYTHON -m training.train --config "$CFG" --device cuda > "$LOG" 2>&1 &
    local PID=$!
    echo "$PID" > "$PIDFILE"
    echo "[i] Training PID: $PID"

    # Timer loop: check every 60s if 2h elapsed or process died
    local START=$(date +%s)
    while true; do
        sleep 60
        local ELAPSED=$(($(date +%s) - START))

        # Check if process is still alive
        if ! kill -0 $PID 2>/dev/null; then
            echo "[$(date +%H:%M:%S)] $NAME process exited early after ${ELAPSED}s"
            break
        fi

        # Check if 2h budget exhausted
        if [ $ELAPSED -ge $DURATION ]; then
            echo "[$(date +%H:%M:%S)] $NAME reached 2h budget, killing PID $PID"
            kill -TERM $PID 2>/dev/null
            sleep 10
            kill -KILL $PID 2>/dev/null
            break
        fi

        # Progress heartbeat
        local KILL_RATE=$(grep -oP 'kill_rate=\K[\d.]+' "$LOG" 2>/dev/null | tail -1)
        local POLICY_LOSS=$(grep -oP 'pl=\K[-\d.]+' "$LOG" 2>/dev/null | tail -1)
        local KR=$(grep -oP 'kill_radius anneal: \K[\d.]+' "$LOG" 2>/dev/null | tail -1)
        echo "[$(date +%H:%M:%S)] $NAME ${ELAPSED}s/${DURATION}s  kr=${KR:-hold}  kill_rate=${KILL_RATE:-NA}  pl=${POLICY_LOSS:-NA}"
    done

    # Wait for clean shutdown
    wait $PID 2>/dev/null
    rm -f "$PIDFILE"

    echo "[$(date +%H:%M:%S)] $NAME complete. Final 30 log lines:"
    tail -30 "$LOG"
    echo ""
}

# Main
mkdir -p logs/ablation_f1f8

if [ $# -eq 1 ]; then
    run_variant "$1"
else
    for V in v4_control v1_conservative v2_aggressive v3_scaling; do
        run_variant "$V"
    done
fi

echo "=============================================================="
echo "[$(date +%H:%M:%S)] All variants complete."
echo "Analyze with: $PYTHON scripts/analyze_ablation.py"
echo "=============================================================="
