"""S1 mean episode reward vs PPO iteration.

Episode return = n_drops + potential-based shaping; shaping telescopes to 0
over the episode, so mean_episode_return ≈ n_drops. drop_ratio =
n_drops / n_eligible_missions, which is the canonical normalized performance
metric used by lite / R2 / witness comparisons.

Plots:
  - train mean reward (per-iter, gray dots) + SMA-20 (blue)
  - val mean reward (every 10 iter, 64 fresh seeds, orange)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "s1_ppo_output"
OUT_PNG = OUT_DIR / "s1_mean_reward.png"

train = [json.loads(l) for l in (OUT_DIR / "train_metrics.jsonl").read_text().splitlines()]
val = [json.loads(l) for l in (OUT_DIR / "val_metrics.jsonl").read_text().splitlines()]

train_it = [r["iteration"] for r in train]
train_r = [r["rollout_drop"] for r in train]
val_it = [r["iter"] for r in val]
val_r = [r["val_macro_drop"] for r in val]

win = 20
train_sma = [sum(train_r[max(0, i - win): i + 1]) /
             (i + 1 - max(0, i - win)) for i in range(len(train_r))]

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(train_it, train_r, color="#aaaaaa", lw=0.6, alpha=0.5,
        label=f"train mean reward (per-iter, {len(train_it)} pts)")
ax.plot(train_it, train_sma, color="#1f77b4", lw=2.2,
        label=f"train SMA-{win}")
ax.plot(val_it, val_r, color="#ff7f0e", marker="o", ms=6, lw=2.2,
        label="val mean reward (64 fresh seeds, every 10 iter)")

ax.axhline(val_r[0], color="#888", ls=":", lw=1.0, alpha=0.6)
ax.text(5, val_r[0] + 0.003, f"scratch (iter 0) = {val_r[0]:.4f}",
        fontsize=9, color="#555")

ax.annotate(f"final val = {val_r[-1]:.4f}\n(iter {val_it[-1]})",
            xy=(val_it[-1], val_r[-1]),
            xytext=(val_it[-1] - 250, val_r[-1] + 0.03),
            fontsize=10, color="#ff7f0e", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.3))

peak_idx = max(range(len(val_r)), key=lambda i: val_r[i])
if peak_idx != len(val_r) - 1:
    ax.annotate(f"peak = {val_r[peak_idx]:.4f}\n(iter {val_it[peak_idx]})",
                xy=(val_it[peak_idx], val_r[peak_idx]),
                xytext=(val_it[peak_idx] - 200, val_r[peak_idx] + 0.035),
                fontsize=9.5, color="#ff7f0e",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.1))

ax.set_xlabel("PPO iteration", fontsize=12)
ax.set_ylabel("mean episode reward  (= mission drop ratio, normalized)",
              fontsize=12)
ax.set_title(
    "S1 PPO mean reward vs training iteration\n"
    "Array-Face phase 1: radar 1D ULA + AF (jammer scalar), profile=mdp_sanity_v1\n"
    "lr=3e-5, target_kl=0.01, 16 envs x 64 steps, 1000 iter (1.024M transitions)",
    fontsize=11,
)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=10)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"  train: {len(train_it)} iters, final SMA = {train_sma[-1]:.4f}")
print(f"  val:   {len(val_it)} evals,  final = {val_r[-1]:.4f}, peak = {max(val_r):.4f} @ iter {val_it[peak_idx]}")
