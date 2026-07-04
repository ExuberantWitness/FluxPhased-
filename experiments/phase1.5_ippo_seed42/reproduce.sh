#!/bin/bash
# Reproduces run: phase1.5_ippo_seed42
# Commit: af0d4c20fd2a14eed05f3e8d39f28ad6f43dd1d6  dirty: True  seed: 42
set -euo pipefail

# 1. Verify cwd
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
echo "[reproduce] cwd: $(pwd)"

# 2. Verify commit (warn if mismatch — don't fail; user may be intentionally re-running)
CURRENT_COMMIT=$(git rev-parse HEAD)
META_COMMIT="af0d4c20fd2a14eed05f3e8d39f28ad6f43dd1d6"
if [ "$CURRENT_COMMIT" != "$META_COMMIT" ]; then
    echo "[reproduce] WARN: HEAD ($CURRENT_COMMIT) != recorded commit ($META_COMMIT)"
    echo "[reproduce] Reproduction may differ. To use the recorded commit:"
    echo "    git checkout $META_COMMIT"
fi

# 3. Activate conda env
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fluxphased

# 4. Re-run training (new pure-Python entry)
echo "[reproduce] running: python main.py --config algo/ippo/code/config.yaml"
python main.py --config algo/ippo/code/config.yaml

# 5. Re-parse metrics + regenerate figures
echo "[reproduce] regenerating metrics + figures"
RUN_DIR="$REPO_DIR/experiments/phase1.5_ippo_seed42"
python scripts/experiments/parse_log_to_metrics.py \
    --log "$RUN_DIR/train.log" \
    --out "$RUN_DIR/metrics.json" \
    --run-id phase1.5_ippo_seed42
python scripts/experiments/make_figures.py \
    --run phase1.5_ippo_seed42:"$RUN_DIR/metrics.json" \
    --out-dir "$RUN_DIR/figures"

echo "[reproduce] done.  See $RUN_DIR/AGENTS.md"
