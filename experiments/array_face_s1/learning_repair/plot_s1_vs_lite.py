"""S1 vs lite R2 extended performance comparison.

Reads:
  - S1 train+val metrics: experiments/array_face_s1/learning_repair/s1_ppo_output/
  - lite R2 extended train+val: experiments/g3_bsta_lite/learning_repair/r2_gate3_output/curves_lr3e-05_kl0.01_extended/

Plots on a single axis (mean episode reward / mission drop ratio):
  - S1 train SMA-20
  - S1 val macro_drop (every 10 iter, 64 fresh seeds)
  - lite R2 ext train SMA-20
  - lite R2 ext val macro_drop (every 10 iter, 64 fresh seeds)

Reward note: episode return = n_drops + potential shaping, and shaping
telescopes to 0 over the episode, so mean_episode_return ~= n_drops.
Both S1 and lite have identical n_services / arrival_rate / horizon, so
drop_ratio is a directly comparable normalized performance metric.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
S1_DIR = HERE / "s1_ppo_output"
LITE_DIR = REPO / "experiments/g3_bsta_lite/learning_repair/r2_gate3_output/curves_lr3e-05_kl0.01_extended"
OUT_PNG = S1_DIR / "s1_vs_lite_performance.png"


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def sma(xs, win=20):
    return [sum(xs[max(0, i - win): i + 1]) /
            (i + 1 - max(0, i - win)) for i in range(len(xs))]


s1_train = load(S1_DIR / "train_metrics.jsonl")
s1_val = load(S1_DIR / "val_metrics.jsonl")
lite_train = load(LITE_DIR / "train_metrics.jsonl")
lite_val = load(LITE_DIR / "val_metrics.jsonl")

s1_train_it = [r["iteration"] for r in s1_train]
s1_train_drop = [r["rollout_drop"] for r in s1_train]
s1_train_sma = sma(s1_train_drop)
s1_val_it = [r["iter"] for r in s1_val]
s1_val_drop = [r["val_macro_drop"] for r in s1_val]

lite_train_it = [r["iteration"] for r in lite_train]
lite_train_drop = [r["rollout_drop"] for r in lite_train]
lite_train_sma = sma(lite_train_drop)
lite_val_it = [r["iter"] for r in lite_val]
lite_val_drop = [r["val_macro_drop"] for r in lite_val]

SCRATCH_INIT = 0.0149
BEST_BASELINE = 0.1653
WITNESS_REF = 0.2680
LITE_R2_SAT = 0.29

s1_peak = max(s1_val, key=lambda r: r["val_macro_drop"]) if s1_val else None
lite_peak = max(lite_val, key=lambda r: r["val_macro_drop"]) if lite_val else None

# iter where lite first broke out of the "first 16 jam" local min (>0.10)
lite_breakout = next((r["iter"] for r in lite_val if r["val_macro_drop"] > 0.12), None)

fig, ax = plt.subplots(figsize=(14, 8))

# === lite R2 ext (background, full 3000 iter) ===
ax.plot(lite_train_it, lite_train_drop, color="#aaaaff", lw=0.6, alpha=0.4)
ax.plot(lite_train_it, lite_train_sma, color="#6666cc", lw=1.8, alpha=0.7,
        label="lite R2 ext train SMA-20")
ax.plot(lite_val_it, lite_val_drop, color="#000099", marker="s", ms=5, lw=1.8,
        label="lite R2 ext val macro_drop (64 fresh seeds / 10 iter)")

# === S1 (foreground, 1000 iter) ===
ax.plot(s1_train_it, s1_train_drop, color="#ffaaaa", lw=0.6, alpha=0.4)
ax.plot(s1_train_it, s1_train_sma, color="#cc6666", lw=2.0,
        label="S1 train SMA-20")
ax.plot(s1_val_it, s1_val_drop, color="#cc0000", marker="o", ms=6, lw=2.2,
        label="S1 val macro_drop (64 fresh seeds / 10 iter)")

# === reference horizontal lines ===
for y, c, lab in [
    (SCRATCH_INIT, "#444444", f"scratch_init = {SCRATCH_INIT:.4f}"),
    (BEST_BASELINE, "#2ca02c", f"best baseline (round_robin) = {BEST_BASELINE:.4f}"),
    (WITNESS_REF, "#d62728", f"witness ref = {WITNESS_REF:.4f}"),
    (LITE_R2_SAT, "#9467bd", f"lite R2 saturation (iter ~1500+) = {LITE_R2_SAT:.4f}"),
]:
    ax.axhline(y, color=c, ls="--", lw=1.2, alpha=0.7)
    ax.text(5, y + 0.003, lab, fontsize=8.5, color=c)

# === S1 endpoint annotation ===
if s1_val_it:
    s1_final = s1_val_drop[-1]
    ax.annotate(f"S1 final = {s1_final:.4f}\n(iter {s1_val_it[-1]})",
                xy=(s1_val_it[-1], s1_final),
                xytext=(s1_val_it[-1] - 50, s1_final + 0.06),
                fontsize=9.5, color="#cc0000", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.2))
if s1_peak and s1_peak["iter"] != s1_val_it[-1]:
    ax.annotate(f"S1 peak = {s1_peak['val_macro_drop']:.4f}\n(iter {s1_peak['iter']})",
                xy=(s1_peak["iter"], s1_peak["val_macro_drop"]),
                xytext=(s1_peak["iter"] + 50, s1_peak["val_macro_drop"] + 0.04),
                fontsize=9, color="#cc0000",
                arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.0))

# === lite endpoint annotation ===
if lite_val_it:
    lite_final = lite_val_drop[-1]
    ax.annotate(f"lite final = {lite_final:.4f}\n(iter {lite_val_it[-1]})",
                xy=(lite_val_it[-1], lite_final),
                xytext=(lite_val_it[-1] - 350, lite_final - 0.06),
                fontsize=9.5, color="#000099", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#000099", lw=1.2))

# === lite breakout iter (where it escaped the local min) ===
if lite_breakout:
    ax.axvline(lite_breakout, color="#000099", ls=":", lw=1.5, alpha=0.7)
    ax.text(lite_breakout + 20, 0.005,
            f"lite broke out of\n'first 16 jam' local min\n(iter {lite_breakout})",
            fontsize=8.5, color="#000099")

# === S1 budget marker ===
ax.axvline(999, color="#cc0000", ls="-", lw=1.0, alpha=0.6)
ax.text(999, 0.305, "S1 stop\n(iter 999)",
        ha="center", fontsize=8.5, color="#cc0000")

# === phase shading ===
ax.axvspan(0, 999, color="#ffeeee", alpha=0.25)
ax.axvspan(999, 2999, color="#eeeeff", alpha=0.25)
ax.text(500, 0.32, "S1 budget\n(1.024M trans)",
        ha="center", fontsize=9, color="#cc0000")
ax.text(2000, 0.32, "lite-only continued\n(+1.79M trans)",
        ha="center", fontsize=9, color="#000099")

ax.set_xlabel("PPO iteration", fontsize=12)
ax.set_ylabel("mission drop ratio  (~ mean episode return, normalized)", fontsize=12)
ax.set_title(
    "S1 vs lite R2 extended -- mean reward (mission drop ratio) comparison\n"
    "S1: profile=mdp_sanity_v1, lr=3e-5, kl=0.01, 16 envs x 64 steps, 1000 iter (1.024M trans)\n"
    "lite: same hparams, 3000 iter (3.07M trans); both exploratory (over prereg 0.5M cap)",
    fontsize=10.5,
)
ax.set_xlim(-10, 3010)
ax.set_ylim(-0.02, 0.34)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9.5)

# summary box
s1_final_v = s1_val_drop[-1] if s1_val_it else 0.0
lite_at_s1_budget = next((r["val_macro_drop"] for r in lite_val if r["iter"] >= 999), None)
summary_lines = [
    "PERFORMANCE COMPARISON (mean episode return ~ drop ratio):",
    f"  S1      iter 999 (budget stop): val = {s1_final_v:.4f}",
    f"  lite    iter 999 (same budget): val = {lite_at_s1_budget:.4f}" if lite_at_s1_budget else "",
    f"  lite    iter 199 (R2 stop):     val = 0.1991",
    f"  lite    iter {lite_val_it[-1]} (ext stop):    val = {lite_val_drop[-1]:.4f}  (saturated)",
    "",
    f"  gap at same 1.024M budget: S1 - lite = {(s1_final_v - lite_at_s1_budget)*100:+.2f}pp" if lite_at_s1_budget else "",
    f"  S1 final vs lite saturation (0.29): {(s1_final_v - LITE_R2_SAT)*100:+.2f}pp",
    "",
    "Interpretation:",
    "  S1 stuck at 'first 16 jam' local min (drop=0.094)",
    "  lite also stuck there iter 49-129, broke out at iter ~150",
    "  S1 needs more iter OR entropy floor to break out",
]
summary = "\n".join(s for s in summary_lines if s or True)

ax.text(0.015, 0.97, summary, transform=ax.transAxes,
        fontsize=9, ha="left", va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888888", alpha=0.95))

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"  S1 final val = {s1_final_v:.4f} at iter {s1_val_it[-1]}")
print(f"  lite val at same iter (999) = {lite_at_s1_budget:.4f}" if lite_at_s1_budget else "")
print(f"  lite final val (iter {lite_val_it[-1]}) = {lite_val_drop[-1]:.4f}")
