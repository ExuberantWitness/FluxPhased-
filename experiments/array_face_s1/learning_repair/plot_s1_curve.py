"""S1 learning curve plot — train rollout_drop + val macro_drop vs PPO iter.

Comparison anchors:
  - lite scratch_init ≈ 0.0149 (random init, no learning)
  - lite best baseline (round_robin) ≈ 0.1653
  - lite witness ref ≈ 0.2680 (NOT deployable; upper-bound estimate)
  - lite R2 final (200 iter) ≈ 0.1991
  - lite R2 extended final (3000 iter) ≈ 0.29 (saturation)

If S1 PPO reaches similar saturation as lite, the radar array factor did not
break learnability. If S1 saturates higher, AF created exploitable structure
PPO can use. If lower, AF noise/distortion hurt PPO.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "s1_ppo_output"
OUT_PNG = OUT_DIR / "s1_learning_curve.png"

train_rows = [json.loads(l) for l in (OUT_DIR / "train_metrics.jsonl").read_text().splitlines()]
val_rows = [json.loads(l) for l in (OUT_DIR / "val_metrics.jsonl").read_text().splitlines()]

iters = [r["iteration"] for r in train_rows]
rollout_drop = [r["rollout_drop"] for r in train_rows]
val_iters = [r["iter"] for r in val_rows]
val_drops = [r["val_macro_drop"] for r in val_rows]

SCRATCH_INIT = 0.0149
BEST_BASELINE = 0.1653
WITNESS_REF = 0.2680
LITE_R2_FINAL = 0.1991
LITE_R2_SAT = 0.29

window = 20
sma = [sum(rollout_drop[max(0, i - window): i + 1]) /
       (i + 1 - max(0, i - window)) for i in range(len(rollout_drop))]

peak = max(val_rows, key=lambda r: r["val_macro_drop"]) if val_rows else None

fig, ax = plt.subplots(figsize=(13, 7.5))

# Train per-iter (noisy)
ax.plot(iters, rollout_drop, color="#aaaaaa", lw=0.7, alpha=0.6,
        label="train rollout_drop (per-iter)")
# SMA
ax.plot(iters, sma, color="C0", lw=2.0, label=f"train SMA-{window}")
# Val
ax.plot(val_iters, val_drops, color="C1", marker="o", ms=5, lw=2.0,
        label="val macro_drop (64 fresh seeds, every 10 iter)")

# Reference lines
for y, c, lab in [
    (SCRATCH_INIT, "#444444", f"scratch_init = {SCRATCH_INIT:.4f}"),
    (BEST_BASELINE, "#2ca02c", f"lite best baseline (round_robin) = {BEST_BASELINE:.4f}"),
    (LITE_R2_FINAL, "#ff7f0e", f"lite R2 final (200 iter) = {LITE_R2_FINAL:.4f}"),
    (WITNESS_REF, "#d62728", f"lite witness ref = {WITNESS_REF:.4f}"),
    (LITE_R2_SAT, "#9467bd", f"lite R2 saturation (3000 iter) = {LITE_R2_SAT:.4f}"),
]:
    ax.axhline(y, color=c, ls="--", lw=1.2)
    ax.text(5, y + 0.003, lab, fontsize=8.5, color=c)

# Annotations
if val_iters:
    final_v = val_drops[-1]
    ax.annotate(f"final val = {final_v:.4f}\n(iter {val_iters[-1]})",
                xy=(val_iters[-1], final_v),
                xytext=(val_iters[-1] - 200, final_v + 0.03),
                fontsize=9, color="C1",
                arrowprops=dict(arrowstyle="->", color="C1", lw=1.2))
if peak is not None and (not val_iters or peak["iter"] != val_iters[-1]):
    ax.annotate(f"peak = {peak['val_macro_drop']:.4f}\n(iter {peak['iter']})",
                xy=(peak["iter"], peak["val_macro_drop"]),
                xytext=(peak["iter"] - 200, peak["val_macro_drop"] + 0.04),
                fontsize=9, color="C1", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="C1", lw=1.2))

ax.set_xlabel("PPO iteration", fontsize=12)
ax.set_ylabel("mission drop ratio  (approx normalized episode return)", fontsize=12)
ax.set_title(
    "S1 PPO learning curve -- Array-Face phase 1: radar 1D ULA + AF (jammer scalar)\n"
    "profile=mdp_sanity_v1, lr=3e-5, target_kl=0.01, 16 envs x 64 steps, 1000 iter (1.024M trans)\n"
    "Exploratory only (over prereg 0.5M cap); NOT for gate re-judgment",
    fontsize=10.5,
)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

# Summary box
summary_lines = ["S1 EXPLORATORY (no gate re-judgment):"]
if val_iters:
    summary_lines.append(f"  iter 0 (scratch):  val = {val_drops[0]:.4f}")
    if len(val_iters) > 10:
        summary_lines.append(f"  iter 100:          val = {val_drops[10]:.4f}")
    if len(val_iters) > 50:
        summary_lines.append(f"  iter 500:          val = {val_drops[50]:.4f}")
    summary_lines.append(f"  iter {val_iters[-1]} (final): val = {val_drops[-1]:.4f}")
    if peak:
        summary_lines.append(f"  peak val = {peak['val_macro_drop']:.4f} at iter {peak['iter']}")
    delta_lite_sat = val_drops[-1] - LITE_R2_SAT
    summary_lines.append(f"  delta vs lite saturation (0.29): {delta_lite_sat*100:+.2f}pp")
summary = "\n".join(summary_lines)

ax.text(0.015, 0.97, summary, transform=ax.transAxes,
        fontsize=9, ha="left", va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888888", alpha=0.95))

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
