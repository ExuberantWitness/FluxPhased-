"""Amendment 02 multi-seed 平均性能曲线 (3 seeds: 20260729 / 20260730 / 20260801).

左 panel:train rollout_drop(SMA-20)
右 panel:val macro_drop
叠加:3 个 seed 个别曲线 + 平均曲线(实线粗) + ±1 std 带
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = [20260729, 20260730, 20260801]
AMEND02_DIRS = {sd: HERE / f"s1_ppo_output_amend02_seed{sd}" for sd in SEEDS}
BASELINE_GOOD = HERE / "s1_ppo_output_seed20260729"     # baseline seed=20260729
BASELINE_BAD = HERE / "s1_ppo_output_anneal0.3_coef1e-3"  # baseline seed=20260730
OUT_PNG = HERE / "amend02_multiseed_performance.png"


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def sma(xs, w=20):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


# Load Amend02 runs
amend_tr = {}
amend_val = {}
for sd in SEEDS:
    tr = load(AMEND02_DIRS[sd] / "train_metrics.jsonl")
    va = load(AMEND02_DIRS[sd] / "val_metrics.jsonl")
    amend_tr[sd] = ([r["iteration"] for r in tr], sma([r["rollout_drop"] for r in tr], 20))
    amend_val[sd] = ([r["iter"] for r in va], [r["val_macro_drop"] for r in va])

# Compute mean/std across seeds at each iter (all 3 runs have same iter grid 9, 19, ... 999)
tr_iters = amend_tr[SEEDS[0]][0]
tr_mat = np.array([amend_tr[sd][1] for sd in SEEDS])  # shape [3, 1000]
tr_mean = tr_mat.mean(axis=0)
tr_std = tr_mat.std(axis=0)

val_iters = amend_val[SEEDS[0]][0]
val_mat = np.array([amend_val[sd][1] for sd in SEEDS])  # shape [3, 100]
val_mean = val_mat.mean(axis=0)
val_std = val_mat.std(axis=0)

# Reference: baseline seed=20260729 (known broke out, 0.2113 final)
base_good_val = load(BASELINE_GOOD / "val_metrics.jsonl")
bg_vit = [r["iter"] for r in base_good_val]
bg_vdrop = [r["val_macro_drop"] for r in base_good_val]

# Reference: baseline seed=20260730 (known stuck, 0.0929 final)
base_bad_val = load(BASELINE_BAD / "val_metrics.jsonl")
bb_vit = [r["iter"] for r in base_bad_val]
bb_vdrop = [r["val_macro_drop"] for r in base_bad_val]

# Reference lines
SCRATCH = 0.0149
BASELINE_RR = 0.1653
WITNESS = 0.2680
LITE_SAT = 0.2628

SEED_COLORS = {20260729: "#1f77b4", 20260730: "#ff7f0e", 20260801: "#2ca02c"}
MEAN_COLOR = "#cc0000"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))

# ============ LEFT: train rollout_drop SMA-20 ============
for sd in SEEDS:
    ax1.plot(amend_tr[sd][0], amend_tr[sd][1],
             color=SEED_COLORS[sd], lw=1.2, alpha=0.7,
             label=f"Amend02 seed={sd}")
ax1.plot(tr_iters, tr_mean,
         color=MEAN_COLOR, lw=3.0,
         label=f"3-seed mean (n=3)")
ax1.fill_between(tr_iters, tr_mean - tr_std, tr_mean + tr_std,
                 color=MEAN_COLOR, alpha=0.15, label="±1 std")

ax1.axhline(0.2113, color="#888888", ls=":", lw=1.0, alpha=0.7)
ax1.text(700, 0.215, "baseline seed=20260729\nfinal = 0.2113",
         fontsize=8.5, color="#555")

ax1.set_xlabel("PPO iteration", fontsize=12)
ax1.set_ylabel("train rollout drop_ratio (SMA-20)\n[= mean episode reward proxy]",
               fontsize=11)
ax1.set_title("Train performance (rollout mean)", fontsize=12)
ax1.set_xlim(-10, 1020)
ax1.set_ylim(-0.01, 0.30)
ax1.grid(alpha=0.3)
ax1.legend(loc="lower right", fontsize=9)

# ============ RIGHT: val macro_drop ============
for sd in SEEDS:
    ax2.plot(amend_val[sd][0], amend_val[sd][1],
             color=SEED_COLORS[sd], marker="o", ms=5, lw=1.5, alpha=0.75,
             label=f"Amend02 seed={sd} (final={amend_val[sd][1][-1]:.4f})")
ax2.plot(val_iters, val_mean,
         color=MEAN_COLOR, lw=3.2,
         label=f"3-seed mean (final={val_mean[-1]:.4f})")
ax2.fill_between(val_iters, val_mean - val_std, val_mean + val_std,
                 color=MEAN_COLOR, alpha=0.15, label="±1 std")

for y, c, lab in [
    (SCRATCH, "#444444", f"scratch = {SCRATCH:.4f}"),
    (BASELINE_RR, "#2ca02c", f"round-robin = {BASELINE_RR:.4f}"),
    (WITNESS, "#d62728", f"witness ref = {WITNESS:.4f}"),
]:
    ax2.axhline(y, color=c, ls="--", lw=1.0, alpha=0.55)
    ax2.text(5, y + 0.003, lab, fontsize=8.5, color=c)

ax2.text(720, 0.0945, "seed=20260730 stuck\n@0.092 plateau",
         fontsize=9, color="#ff7f0e", style="italic")
ax2.text(720, 0.235, "seed=20260801 best\n@0.237",
         fontsize=9, color="#2ca02c", style="italic")

finals = [amend_val[sd][1][-1] for sd in SEEDS]
summary = (
    "Amend02 multi-seed summary (n=3):\n"
    f"  finals: {[f'{f:.4f}' for f in finals]}\n"
    f"  mean final:  {sum(finals)/3:.4f}\n"
    f"  std final:   {(sum((f - sum(finals)/3)**2 for f in finals)/3)**0.5:.4f}\n"
    f"  range:       [{min(finals):.4f}, {max(finals):.4f}]\n"
    f"  break-out:   2/3 seeds (20260729, 20260801)\n"
    f"  stuck:       1/3 seeds (20260730)\n\n"
    f"vs baseline seed=20260729: 0.2113\n"
    f"vs baseline seed=20260730: 0.0929 (stuck)\n"
    f"vs lite R2 ext:           0.2628"
)
ax2.text(0.015, 0.97, summary, transform=ax2.transAxes,
         fontsize=9, ha="left", va="top", family="monospace",
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888", alpha=0.95))

ax2.set_xlabel("PPO iteration", fontsize=12)
ax2.set_ylabel("val macro_drop (mean episode reward, 64 seeds)", fontsize=11)
ax2.set_title("Validation performance (held-out 64 seeds)", fontsize=12)
ax2.set_xlim(-10, 1020)
ax2.set_ylim(-0.02, 0.30)
ax2.grid(alpha=0.3)
ax2.legend(loc="lower right", fontsize=9)

fig.suptitle(
    "S1 PPO Amendment 02 multi-seed: 3 seeds × (entropy 5e-3, anneal 0.5, target_kl 0.02)\n"
    "Verdict: 2/3 broke out (mean 0.214), 1/3 stuck (seed 20260730 = known-bad) — exploration is NOT the bottleneck",
    fontsize=12.5, y=1.00,
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"finals: {['%.4f' % f for f in finals]}")
print(f"mean:   {sum(finals)/3:.4f}")
print(f"std:    {(sum((f - sum(finals)/3)**2 for f in finals)/3)**0.5:.4f}")
