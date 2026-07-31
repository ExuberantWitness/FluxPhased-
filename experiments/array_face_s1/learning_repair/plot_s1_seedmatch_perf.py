"""S1 seed-match 性能曲线:train rollout_drop + val macro_drop vs PPO iter.

对比:
  - S1 seed=20260730 (原 baseline,卡 0.092 局部最优)
  - S1 seed=20260729 (新 seed-match,破壳到 0.21)
  - lite R2 ext seed=20260729 (lite env 参考,饱和 0.2628)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S1_BAD = HERE / "s1_ppo_output_anneal0.3_coef1e-3"            # seed 20260730
S1_GOOD = HERE / "s1_ppo_output_seed20260729"                  # seed 20260729 (new)
LITE_R2 = (HERE.parents[1] / "g3_bsta_lite/learning_repair/r2_gate3_output"
           / "curves_lr3e-05_kl0.01_extended")
OUT_PNG = S1_GOOD / "s1_seedmatch_performance.png"


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def sma(xs, w=20):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


s1_bad_tr = load(S1_BAD / "train_metrics.jsonl")
s1_bad_val = load(S1_BAD / "val_metrics.jsonl")
s1_good_tr = load(S1_GOOD / "train_metrics.jsonl")
s1_good_val = load(S1_GOOD / "val_metrics.jsonl")
lite_tr = load(LITE_R2 / "train_metrics.jsonl")
lite_val = load(LITE_R2 / "val_metrics.jsonl")

# S1 baseline original (seed 20260730) — only has 0..999 iter (1000 total)
s1_bad_tr_it = [r["iteration"] for r in s1_bad_tr]
s1_bad_tr_drop = [r["rollout_drop"] for r in s1_bad_tr]
s1_bad_val_it = [r["iter"] for r in s1_bad_val]
s1_bad_val_drop = [r["val_macro_drop"] for r in s1_bad_val]

# S1 seed-match (seed 20260729) — 0..999
s1_good_tr_it = [r["iteration"] for r in s1_good_tr]
s1_good_tr_drop = [r["rollout_drop"] for r in s1_good_tr]
s1_good_val_it = [r["iter"] for r in s1_good_val]
s1_good_val_drop = [r["val_macro_drop"] for r in s1_good_val]

# Lite R2 ext (seed 20260729) — 0..2999
lite_tr_it = [r["iteration"] for r in lite_tr]
lite_tr_drop = [r["rollout_drop"] for r in lite_tr]
lite_val_it = [r["iter"] for r in lite_val]
lite_val_drop = [r["val_macro_drop"] for r in lite_val]

SCRATCH = 0.0149
BASELINE = 0.1653
WITNESS = 0.2680
LITE_SAT = 0.2628
S1_GOOD_FINAL = s1_good_val_drop[-1]
S1_BAD_FINAL = s1_bad_val_drop[-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ============ LEFT: train rollout_drop (mean reward per episode proxy) ============
ax1.plot(s1_bad_tr_it, sma(s1_bad_tr_drop, 20),
         color="#888888", lw=1.8, alpha=0.85,
         label=f"S1 seed=20260730 (stuck) — SMA20")
ax1.plot(s1_good_tr_it, sma(s1_good_tr_drop, 20),
         color="#cc0000", lw=2.2,
         label=f"S1 seed=20260729 (broke out) — SMA20")
ax1.plot(lite_tr_it[:1000], sma(lite_tr_drop[:1000], 20),
         color="#2ca02c", lw=2.0, ls="--",
         label=f"lite R2 ext seed=20260729 — SMA20 (iter 0-999)")

# raw dots for S1 good (faint)
ax1.scatter(s1_good_tr_it, s1_good_tr_drop,
            color="#cc0000", s=4, alpha=0.15)

ax1.axvline(229, color="#cc0000", ls=":", lw=1.2, alpha=0.7)
ax1.text(234, 0.18, "S1 seed=20260729\nbreak-out (iter 229)",
         fontsize=8.5, color="#cc0000")

ax1.set_xlabel("PPO iteration", fontsize=12)
ax1.set_ylabel("train rollout drop_ratio (SMA-20)\n[= mean episode reward proxy]",
               fontsize=11)
ax1.set_title("Train performance (rollout mean)", fontsize=12)
ax1.set_xlim(-10, 1020)
ax1.set_ylim(-0.01, 0.30)
ax1.grid(alpha=0.3)
ax1.legend(loc="lower right", fontsize=9.5)

# ============ RIGHT: validation macro_drop ============
ax2.plot(s1_bad_val_it, s1_bad_val_drop,
         color="#888888", marker="s", ms=6, lw=1.8,
         label=f"S1 seed=20260730 (stuck) — final={S1_BAD_FINAL:.4f}")
ax2.plot(s1_good_val_it, s1_good_val_drop,
         color="#cc0000", marker="o", ms=7, lw=2.2,
         label=f"S1 seed=20260729 (broke out) — final={S1_GOOD_FINAL:.4f}")
ax2.plot(lite_val_it[:100], lite_val_drop[:100],
         color="#2ca02c", marker="^", ms=6, lw=1.8, ls="--",
         label=f"lite R2 ext seed=20260729 (first 1000 iter)")

for y, c, lab in [
    (SCRATCH, "#444444", f"scratch_init = {SCRATCH:.4f}"),
    (BASELINE, "#2ca02c", f"round-robin baseline = {BASELINE:.4f}"),
    (WITNESS, "#d62728", f"witness ref = {WITNESS:.4f}"),
]:
    ax2.axhline(y, color=c, ls="--", lw=1.0, alpha=0.6)
    ax2.text(5, y + 0.003, lab, fontsize=8.5, color=c)

ax2.axhspan(0.205, 0.220, color="#cc0000", alpha=0.10)
ax2.text(700, 0.225, "S1 seed=20260729\nsaturation ~0.21",
         fontsize=9, color="#cc0000", style="italic")

ax2.annotate(f"S1 seed=20260730 stuck\nat 0.092 plateau",
             xy=(800, S1_BAD_FINAL),
             xytext=(550, 0.04),
             fontsize=9.5, color="#444444",
             arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0))

ax2.set_xlabel("PPO iteration", fontsize=12)
ax2.set_ylabel("val macro_drop (mean episode reward, 64 seeds)", fontsize=11)
ax2.set_title("Validation performance (held-out 64 seeds)", fontsize=12)
ax2.set_xlim(-10, 1020)
ax2.set_ylim(-0.02, 0.30)
ax2.grid(alpha=0.3)
ax2.legend(loc="lower right", fontsize=9.5)

fig.suptitle(
    "S1 PPO seed-matched comparison: S1 env failure was seed luck, not env difficulty\n"
    "S1 seed=20260729 broke out at iter 229, saturated at 0.2113 (lite R2 ext: 0.2628, ~5pp gap)",
    fontsize=12.5, y=1.00,
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"  S1 seed=20260730 final: {S1_BAD_FINAL:.4f} (stuck)")
print(f"  S1 seed=20260729 final: {S1_GOOD_FINAL:.4f} (broke out)")
print(f"  gap vs lite R2 ext:     {LITE_SAT - S1_GOOD_FINAL:+.4f}pp")
