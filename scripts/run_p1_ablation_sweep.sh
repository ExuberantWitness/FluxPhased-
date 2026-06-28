#!/bin/bash
# P1 Ablation Sweep: B → A → C → D, each 3 iter (~45 min), serial.
# 修改建议 §6 P1 protocol — fast trial-and-error before large-scale.
set -e
cd /home/ubuntu/CODE/FluxPhased-
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate fluxphased

LOG_DIR=logs/diag
CFG_DIR=configs/ablation_f1f8
SWEEP_LOG=$LOG_DIR/p1_ablation_sweep_$(date +%Y%m%d_%H%M%S).log
echo "Sweep start: $(date)" | tee "$SWEEP_LOG"

for tag in B_f1f2 A_baseline C_f1only D_f2only; do
    case $tag in
        B_f1f2)     cfg=v3_p1_f1f2_alpha07;;
        A_baseline) cfg=v3_p1_A_baseline;;
        C_f1only)   cfg=v3_p1_C_f1only;;
        D_f2only)   cfg=v3_p1_D_f2only;;
    esac
    ckpt=checkpoints/laser_pro6000_league_v3_p1_${cfg#v3_p1_}
    # Clear stale state so all variants start from identical init
    rm -rf "$ckpt"
    log=$LOG_DIR/p1_${tag}_$(date +%Y%m%d_%H%M%S).log
    echo "" | tee -a "$SWEEP_LOG"
    echo "=== [$tag] config=$cfg log=$log ===" | tee -a "$SWEEP_LOG"
    echo "Start: $(date)" | tee -a "$SWEEP_LOG"
    python -m training.train --config $CFG_DIR/$cfg.yaml > "$log" 2>&1
    rc=$?
    echo "End: $(date) rc=$rc" | tee -a "$SWEEP_LOG"
    # Extract kill_rate trajectory
    echo "[$tag] kill_rate trajectory:" | tee -a "$SWEEP_LOG"
    grep -E "Iteration [0-9]+ complete|alpha=|kill_rate" "$log" | \
        grep -E "kill_rate|alpha=" | tail -6 | tee -a "$SWEEP_LOG"
done

echo "" | tee -a "$SWEEP_LOG"
echo "Sweep complete: $(date)" | tee -a "$SWEEP_LOG"
