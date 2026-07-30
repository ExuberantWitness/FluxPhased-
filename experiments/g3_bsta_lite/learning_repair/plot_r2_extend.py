"""R2 探索性续训(0-1999 iter)完整学习曲线 — 对比 R2 原始段(0-199)与扩展段(200-1999)。

注意:这是探索性扩展, NOT for gate re-judgment.
原 R2 判定 BLOCKED_LEARNING_CONTRIBUTION 保持不变 (因超出 prereg 500k transition cap).
本图仅用于观察 PPO 的学习上限 / 饱和点.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ORIG_DIR = HERE / "r2_gate3_output" / "curves_lr3e-05_kl0.01"
EXT_DIR = HERE / "r2_gate3_output" / "curves_lr3e-05_kl0.01_extended"
OUT_PNG = HERE / "r2_gate3_output" / "r2_extended_learning_curve.png"

orig_train = [json.loads(l) for l in (ORIG_DIR / "train_metrics.jsonl").read_text().splitlines()]
ext_train = [json.loads(l) for l in (EXT_DIR / "train_metrics.jsonl").read_text().splitlines()]
orig_val = [json.loads(l) for l in (ORIG_DIR / "val_metrics.jsonl").read_text().splitlines()]
ext_val = [json.loads(l) for l in (EXT_DIR / "val_metrics.jsonl").read_text().splitlines()]

# train: concat (drop SMA across boundary to keep clean)
all_train = orig_train + ext_train
iters = [r["iteration"] for r in all_train]
rollout_drop = [r["rollout_drop"] for r in all_train]
window = 20
sma = [sum(rollout_drop[max(0, i - window): i + 1]) /
       (i + 1 - max(0, i - window)) for i in range(len(rollout_drop))]

# val: concat (drop the duplicate iter 199 — orig ends at 199, ext starts at 209)
all_val = orig_val + ext_val
val_iters = [r["iter"] for r in all_val]
val_drops = [r["val_macro_drop"] for r in all_val]

SCRATCH_INIT = 0.0149
BEST_BASELINE = 0.1653
WITNESS_REF = 0.2680
GATE3_POINT = BEST_BASELINE + 0.05    # 0.2153
GATE3_HEADROOM_80 = BEST_BASELINE + 0.80 * (WITNESS_REF - BEST_BASELINE)  # 0.2475

# 关键里程碑
iter_first_pass_point = next((r["iter"] for r in all_val if r["val_macro_drop"] >= GATE3_POINT), None)
iter_first_pass_witness = next((r["iter"] for r in all_val if r["val_macro_drop"] >= WITNESS_REF), None)
iter_first_pass_headroom = next((r["iter"] for r in all_val if r["val_macro_drop"] >= GATE3_HEADROOM_80), None)
peak = max(all_val, key=lambda r: r["val_macro_drop"])

fig, ax = plt.subplots(figsize=(13, 7.5))

# phase 背景色
ax.axvspan(0, 199, color="#eeeeee", alpha=0.6)
ax.axvspan(199, 1999, color="#fff7e6", alpha=0.6)
ax.axvspan(1999, 2999, color="#ffe4cc", alpha=0.5)
ax.text(100, 0.305, "R2 original budget\n(0-199 iter, 204.8k trans)",
        ha="center", fontsize=9, color="#555555")
ax.text(700, 0.305, "EXPLORATORY extension round 1\n(200-1999 iter, +1.84M trans)",
        ha="center", fontsize=9, color="#aa6600")
ax.text(2500, 0.305, "round 2 (SATURATED)\n(2000-2999 iter, +1.02M trans)",
        ha="center", fontsize=9, color="#994400")

# train + SMA
ax.plot(iters, rollout_drop, color="#aaaaaa", lw=0.7, alpha=0.6,
        label="train rollout_drop (per-iter)")
ax.plot(iters, sma, color="C0", lw=2.0, label=f"train SMA-{window}")

# val
ax.plot(val_iters, val_drops, color="C1", marker="o", ms=5, lw=2.0,
        label="val macro_drop (64 fresh seeds, every 10 iter)")

# 参考线
for y, c, lab in [
    (SCRATCH_INIT, "#444444", f"scratch_init = {SCRATCH_INIT:.4f}"),
    (BEST_BASELINE, "#2ca02c", f"best baseline (round_robin) = {BEST_BASELINE:.4f}"),
    (WITNESS_REF, "#d62728", f"witness ref (privileged info, NOT deployable) = {WITNESS_REF:.4f}"),
    (GATE3_POINT, "#ff7f0e", f"Gate-3 point threshold = baseline + 5pp = {GATE3_POINT:.4f}"),
    (GATE3_HEADROOM_80, "#8c564b", f"Gate-3 80%-headroom threshold = {GATE3_HEADROOM_80:.4f}"),
]:
    ax.axhline(y, color=c, ls="--", lw=1.2)
    ax.text(5, y + 0.003, lab, fontsize=8.5, color=c)

# milestone 标注
if iter_first_pass_point:
    ax.axvline(iter_first_pass_point, color="#ff7f0e", ls=":", lw=1, alpha=0.7)
    ax.annotate(f"first crosses\nGate-3 5pp threshold\n(iter {iter_first_pass_point})",
                xy=(iter_first_pass_point, GATE3_POINT),
                xytext=(iter_first_pass_point + 100, GATE3_POINT - 0.06),
                fontsize=8.5, color="#ff7f0e",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1))
if iter_first_pass_headroom:
    ax.axvline(iter_first_pass_headroom, color="#8c564b", ls=":", lw=1, alpha=0.7)
if iter_first_pass_witness:
    ax.axvline(iter_first_pass_witness, color="#d62728", ls=":", lw=1, alpha=0.7)
    ax.annotate(f"first crosses\nwitness ref\n(iter {iter_first_pass_witness})",
                xy=(iter_first_pass_witness, WITNESS_REF),
                xytext=(iter_first_pass_witness + 80, WITNESS_REF + 0.015),
                fontsize=8.5, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))

# peak / final 标注
ax.annotate(f"peak = {peak['val_macro_drop']:.4f}\n(iter {peak['iter']})",
            xy=(peak["iter"], peak["val_macro_drop"]),
            xytext=(peak["iter"] - 350, peak["val_macro_drop"] + 0.02),
            fontsize=9, color="C1", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="C1", lw=1.2))
ax.annotate(f"final = {val_drops[-1]:.4f}\n(iter {val_iters[-1]})",
            xy=(val_iters[-1], val_drops[-1]),
            xytext=(val_iters[-1] - 350, val_drops[-1] - 0.05),
            fontsize=9, color="C1",
            arrowprops=dict(arrowstyle="->", color="C1", lw=1.2))

# R2 原终点
ax.axvline(199, color="#888888", ls="-", lw=0.8, alpha=0.6)
ax.text(199, 0.005, "R2 stop\n(iter 199)",
        ha="center", fontsize=8, color="#555555")
# 续训 r1 终点
ax.axvline(1999, color="#aa6600", ls="-", lw=0.8, alpha=0.6)
ax.text(1999, 0.005, "r1 stop\n(iter 1999)",
        ha="center", fontsize=8, color="#aa6600")

ax.set_xlabel("PPO iteration", fontsize=12)
ax.set_ylabel("mission drop ratio  (approx normalized episode return)", fontsize=12)
ax.set_title(
    "R2 EXPLORATORY extension (0-2999 iter) -- scratch masked PPO, lr=3e-5, target_kl=0.01\n"
    "DOES NOT re-judge R2 (over prereg 0.5M transition cap by 6.1x); curve saturates at ~0.29 after iter 1500",
    fontsize=10.5,
)
ax.set_xlim(-5, 3010)
ax.set_ylim(-0.02, 0.33)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

# 总结框
summary = (
    "EXPLORATORY (no gate re-judgment):\n"
    f"  iter 199  (R2 stop):       val = 0.1991  (+3.08pp / 30.0%  -> 2/9 FAIL)\n"
    f"  iter {iter_first_pass_point} (first 5pp PASS):      val >= {GATE3_POINT:.4f}\n"
    f"  iter {iter_first_pass_headroom} (first 80% headroom): val >= {GATE3_HEADROOM_80:.4f}\n"
    f"  iter {iter_first_pass_witness} (first BEATS witness): val >= {WITNESS_REF:.4f}\n"
    f"  iter {peak['iter']} (peak):          val = {peak['val_macro_drop']:.4f}\n"
    f"  iter 1999 (r1 stop):       val = 0.2818  (+11.65pp / 113.4%)\n"
    f"  iter 2999 (r2 stop):       val = {val_drops[-1]:.4f}  (+{(val_drops[-1]-BEST_BASELINE)*100:.2f}pp / {(val_drops[-1]-BEST_BASELINE)/(WITNESS_REF-BEST_BASELINE)*100:.1f}%)\n"
    f"  saturation: r1->r2 (+1000 iter, +1.02M trans) only +{(val_drops[-1]-0.2818)*100:.2f}pp"
)
ax.text(0.015, 0.97, summary, transform=ax.transAxes,
        fontsize=9, ha="left", va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888888", alpha=0.95))

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
