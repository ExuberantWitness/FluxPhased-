"""R2 性能(= mission drop ratio = 归一化 return)随训练 iter 的曲线。

drop ratio 是 jammer objective 的直接度量;在这个 env 里 per-step reward =
drop 计数 + potential-based shaping,后者在完整 episode 上 telescopes to 0,
所以 episode 总 reward ≈ drop 计数。drop ratio = drop 计数 / n_eligible,
= 归一化后的 per-episode return。

train rollout_drop = 每 iter 16-env × 64-step rollout 上的平均 drop ratio
                      (训练场景,policy 在线 sample)
val macro_drop      = 64 个 fresh validation scenarios 上的平均 drop ratio
                      (每 10 iter 评估一次,policy 仍 sample)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CURVES_DIR = HERE / "r2_gate3_output" / "curves_lr3e-05_kl0.01"
OUT_PNG = HERE / "r2_gate3_output" / "r2_learning_curve.png"

train_rows = [json.loads(line) for line in (CURVES_DIR / "train_metrics.jsonl").read_text().splitlines()]
val_rows = [json.loads(line) for line in (CURVES_DIR / "val_metrics.jsonl").read_text().splitlines()]

iters = [r["iteration"] for r in train_rows]
rollout_drop = [r["rollout_drop"] for r in train_rows]
val_iters = [r["iter"] for r in val_rows]
val_drops = [r["val_macro_drop"] for r in val_rows]

# baseline / reference 数值取自 R2_GATE3_RESULT.json + FINAL_VERDICT.md
SCRATCH_INIT = 0.0149
RANDOM_UNTRAINED = 0.0123
TIME_ONLY = 0.0143
SHUFFLED_OBS = 0.0453
BEST_BASELINE = 0.1653          # budgeted_round_robin = greedy_radar_follower
WITNESS_REF = 0.2680            # Gate 1 reference, NOT a deployable bound

fig, ax = plt.subplots(figsize=(11, 6.5))

# train curve (per iter, noisy)
ax.plot(iters, rollout_drop, color="#888888", lw=1.0, alpha=0.7,
        label="train rollout_drop (per-iter, online sample on train scenarios)")

# train SMA to highlight trend
window = 10
if len(rollout_drop) >= window:
    sma = [sum(rollout_drop[max(0, i - window): i + 1]) /
           (i + 1 - max(0, i - window)) for i in range(len(rollout_drop))]
    ax.plot(iters, sma, color="C0", lw=2.2, label=f"train SMA-{window}")

# validation curve (every 10 iters, 64 fresh seeds)
ax.plot(val_iters, val_drops, color="C1", marker="o", ms=7, lw=2.0,
        label="val macro_drop (every 10 iters, 64 fresh seeds, sample)")

# 关键参考线
ax.axhline(SCRATCH_INIT, color="#444444", ls=":", lw=1.2)
ax.text(2, SCRATCH_INIT + 0.005, f"scratch_init = {SCRATCH_INIT:.4f}",
        fontsize=9, color="#444444")

ax.axhline(RANDOM_UNTRAINED, color="#9467bd", ls=":", lw=1.0)
ax.text(2, RANDOM_UNTRAINED - 0.012, f"random_untrained = {RANDOM_UNTRAINED:.4f}",
        fontsize=8, color="#9467bd")

ax.axhline(BEST_BASELINE, color="#2ca02c", ls="--", lw=1.6)
ax.text(2, BEST_BASELINE + 0.005,
        f"best non-witness baseline (round_robin) = {BEST_BASELINE:.4f}",
        fontsize=9, color="#2ca02c")

ax.axhline(WITNESS_REF, color="#d62728", ls="--", lw=1.6)
ax.text(2, WITNESS_REF + 0.005, f"witness ref (Gate 1, NOT deployable) = {WITNESS_REF:.4f}",
        fontsize=9, color="#d62728")

# Gate-3 阈值线: 5pp above baseline + 80% headroom recovery
gate3_point = BEST_BASELINE + 0.05
ax.axhline(gate3_point, color="#ff7f0e", ls="-.", lw=1.2)
ax.text(150, gate3_point + 0.005,
        f"Gate-3 point threshold = baseline + 5pp = {gate3_point:.4f}",
        fontsize=9, color="#ff7f0e")

gate3_headroom = BEST_BASELINE + 0.80 * (WITNESS_REF - BEST_BASELINE)
ax.axhline(gate3_headroom, color="#8c564b", ls="-.", lw=1.2)
ax.text(150, gate3_headroom + 0.005,
        f"Gate-3 80%-headroom threshold = {gate3_headroom:.4f}",
        fontsize=9, color="#8c564b")

# 标注最终 val 点
final_val = val_drops[-1]
ax.annotate(f"final val = {final_val:.4f}\n(iter {val_iters[-1]})",
            xy=(val_iters[-1], final_val),
            xytext=(val_iters[-1] - 50, final_val + 0.04),
            fontsize=10, color="C1", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="C1", lw=1.2))

ax.set_xlabel("PPO iteration", fontsize=12)
ax.set_ylabel("mission drop ratio  (approx normalized episode return)", fontsize=12)
ax.set_title(
    "R2 scratch masked PPO learning curve -- performance (jam mission drop) vs training iter\n"
    "profile=mdp_sanity_v1, lr=3e-5, target_kl=0.01, 16 envs x 64 steps, 200 iters (204.8k transitions)",
    fontsize=11,
)
ax.set_xlim(-3, 205)
ax.set_ylim(-0.02, 0.32)
ax.grid(alpha=0.3)
ax.legend(loc="center right", fontsize=9)

# 用文本框总结 gate-3 结论
summary = (
    "Gate-3 verdict: 7/9 PASS, 2/9 FAIL → BLOCKED_LEARNING_CONTRIBUTION\n"
    f"  • trained vs scratch_init:     LCB95 = +0.165  (PASS: > 0)\n"
    f"  • trained vs best baseline:    LCB95 = +0.015  (PASS: > 0)\n"
    f"  • point improvement:           +3.08 pp       (FAIL: < 5 pp)\n"
    f"  • headroom recovered:          30.0%          (FAIL: < 80%)"
)
ax.text(0.98, 0.02, summary, transform=ax.transAxes,
        fontsize=8.5, ha="right", va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888888", alpha=0.92))

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
