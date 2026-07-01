#!/bin/bash
# Master script: run 4 v5 variants sequentially, 2h each, early-kill if no
# dart-board convergence after 30 min. Output: logs/diag/v5_master_<DATE>.log
#
# Early-kill heuristic (checked every 10 min after first 30 min):
#   - Extract last 5 [dart] lines from current log
#   - If min_dist_change avg over last 5 ≥ -1.0m AND no kills → KILL (no potential)
#   - If min_dist_change ≤ -10m sustained → KEEP RUNNING
#   - Otherwise run to 2h cap
set -u

cd /home/ubuntu/CODE/FluxPhased-
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fluxphased

DATE=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/diag/v5_master_${DATE}.log"
RESULTS_DIR="/tmp/v5_results"
mkdir -p "$RESULTS_DIR"
echo "$MASTER_LOG" > /tmp/v5_master_logpath
echo "Master log: $MASTER_LOG"
echo "Results dir: $RESULTS_DIR"

VARIANTS=(v5a_big_residual v5b_no_residual v5c_anchor_noise v5d_combo)
MAX_RUNTIME_SEC=7200        # 2h
EARLY_CHECK_START_SEC=1800  # 30 min
CHECK_INTERVAL_SEC=600      # 10 min
KILL_THRESHOLD_M=-1.0       # avg change ≥ -1.0m → kill (no convergence)
PASS_THRESHOLD_M=-10.0      # avg change ≤ -10m → clearly learning

for variant in "${VARIANTS[@]}"; do
    echo "" | tee -a "$MASTER_LOG"
    echo "==========================================================" | tee -a "$MASTER_LOG"
    echo "[$(date +%H:%M:%S)] Starting $variant" | tee -a "$MASTER_LOG"
    echo "==========================================================" | tee -a "$MASTER_LOG"

    # Clear checkpoints so each variant starts fresh
    rm -rf "checkpoints/laser_pro6000_league_${variant}/"

    VAR_LOG="logs/diag/${variant}_${DATE}.log"
    echo "  log: $VAR_LOG" | tee -a "$MASTER_LOG"

    # Launch training
    python -m algo._shared.train --config "configs/ablation_f1f8/${variant}.yaml" > "$VAR_LOG" 2>&1 &
    PID=$!
    echo "$PID" > "/tmp/${variant}.pid"
    echo "  PID: $PID" | tee -a "$MASTER_LOG"

    START=$(date +%s)
    KILLED_EARLY=false

    # Monitor loop
    while true; do
        sleep "$CHECK_INTERVAL_SEC"
        NOW=$(date +%s)
        ELAPSED=$((NOW - START))

        # Check if process still alive
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "[$(date +%H:%M:%S)] $variant: process exited on its own after ${ELAPSED}s" | tee -a "$MASTER_LOG"
            break
        fi

        # Hit 2h cap
        if [ $ELAPSED -ge $MAX_RUNTIME_SEC ]; then
            echo "[$(date +%H:%M:%S)] $variant: reached 2h cap, killing" | tee -a "$MASTER_LOG"
            kill -TERM $PID 2>/dev/null
            sleep 5
            kill -KILL $PID 2>/dev/null
            break
        fi

        # After 30 min, start checking convergence
        if [ $ELAPSED -ge $EARLY_CHECK_START_SEC ]; then
            # Extract last 5 [dart] lines and parse min_dist_change values
            CHANGES=$(grep "\[dart\]" "$VAR_LOG" | tail -5 | sed -E 's/.*change=([-+]?[0-9.]+)m.*/\1/' | head -5)
            if [ -z "$CHANGES" ]; then
                echo "[$(date +%H:%M:%S)] $variant: ${ELAPSED}s elapsed, no [dart] logs yet" | tee -a "$MASTER_LOG"
                continue
            fi

            # Compute average change
            AVG=$(echo "$CHANGES" | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n; else print "NA"}')
            N_DART=$(grep -c "\[dart\]" "$VAR_LOG")
            N_KILL_EP=$(grep "ep [0-9]*/" "$VAR_LOG" | grep -c "kill")

            echo "[$(date +%H:%M:%S)] $variant: ${ELAPSED}s, ${N_DART} dart-samples, avg_change=${AVG}m, kills=${N_KILL_EP}" | tee -a "$MASTER_LOG"

            # Early-kill if clearly no potential: avg change ≥ -1.0m AND no kills
            IS_BAD=$(echo "$AVG" | awk -v t="$KILL_THRESHOLD_M" '{if ($1 + 0 >= t + 0) print 1; else print 0}')
            if [ "$IS_BAD" = "1" ] && [ "$N_KILL_EP" -eq 0 ]; then
                echo "[$(date +%H:%M:%S)] $variant: EARLY KILL — no convergence (avg=${AVG}m, no kills)" | tee -a "$MASTER_LOG"
                kill -TERM $PID 2>/dev/null
                sleep 5
                kill -KILL $PID 2>/dev/null
                KILLED_EARLY=true
                break
            fi

            # Early-terminate if clearly winning (don't waste 2h)
            IS_GOOD=$(echo "$AVG" | awk -v t="$PASS_THRESHOLD_M" '{if ($1 + 0 <= t + 0) print 1; else print 0}')
            if [ "$IS_GOOD" = "1" ]; then
                echo "[$(date +%H:%M:%S)] $variant: EARLY PASS — strong convergence (avg=${AVG}m), but continuing to 2h for full data" | tee -a "$MASTER_LOG"
                # Don't kill — let it run for full data, just note the pass
            fi
        fi
    done

    # Final summary for this variant
    FINAL_DARTS=$(grep "\[dart\]" "$VAR_LOG" | tail -10)
    N_KILL_EP=$(grep "ep [0-9]*/" "$VAR_LOG" | grep -c "kill")
    N_TIMEOUT_EP=$(grep "ep [0-9]*/" "$VAR_LOG" | grep -c "timeout")

    echo "" >> "$RESULTS_DIR/${variant}.txt"
    echo "=== $variant ($(date)) ===" >> "$RESULTS_DIR/${variant}.txt"
    echo "early_killed: $KILLED_EARLY" >> "$RESULTS_DIR/${variant}.txt"
    echo "kill_episodes: $N_KILL_EP" >> "$RESULTS_DIR/${variant}.txt"
    echo "timeout_episodes: $N_TIMEOUT_EP" >> "$RESULTS_DIR/${variant}.txt"
    echo "" >> "$RESULTS_DIR/${variant}.txt"
    echo "last 10 dart metrics:" >> "$RESULTS_DIR/${variant}.txt"
    echo "$FINAL_DARTS" >> "$RESULTS_DIR/${variant}.txt"

    # GPU settle
    sleep 10
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$MASTER_LOG"
done

echo "" | tee -a "$MASTER_LOG"
echo "==========================================================" | tee -a "$MASTER_LOG"
echo "[$(date +%H:%M:%S)] All 4 variants complete. Results in $RESULTS_DIR/" | tee -a "$MASTER_LOG"
echo "==========================================================" | tee -a "$MASTER_LOG"
