"""Amendment 01 (entropy_coef=5e-3, anneal_frac=1.0) vs baseline comparison.

Reads:
  - baseline: s1_ppo_output_anneal0.3_coef1e-3/{train,val}_metrics.jsonl
  - Amend01 (partial, killed at iter 489): s1_ppo_output/val_metrics_amend01_partial.jsonl

Plots val_macro_drop vs PPO iteration for both, with reference lines.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE / "s1_ppo_output_anneal0.3_coef1e-3"
AMEND_DIR = HERE / "s1_ppo_output"
OUT_PNG = AMEND_DIR / "amend01_vs_baseline.png"


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


base_val = load(BASE_DIR / "val_metrics.jsonl")
amend_val = load(AMEND_DIR / "val_metrics_amend01_partial.jsonl")

base_it = [r["iter"] for r in base_val]
base_d = [r["val_macro_drop"] for r in base_val]
am_it = [r["iter"] for r in amend_val]
am_d = [r["val_macro_drop"] for r in amend_val]

SCRATCH = 0.0149
BASELINE = 0.1653
WITNESS = 0.2680
LITE_SAT = 0.29

fig, ax = plt.subplots(figsize=(12, 7.5))

ax.plot(base_it, base_d, color="#888888", marker="s", ms=6, lw=2.0,
        label="baseline (anneal=0.3, coef=1e-3) -- 1000 iter complete")
ax.plot(am_it, am_d, color="#cc0000", marker="o", ms=7, lw=2.2,
        label="Amend01 (anneal=1.0, coef=5e-3) -- killed at iter 489 (plateau confirmed)")

for y, c, lab in [
    (SCRATCH, "#444444", f"scratch_init = {SCRATCH:.4f}"),
    (BASELINE, "#2ca02c", f"best baseline (round_robin) = {BASELINE:.4f}"),
    (WITNESS, "#d62728", f"witness ref = {WITNESS:.4f}"),
]:
    ax.axhline(y, color=c, ls="--", lw=1.2, alpha=0.7)
    ax.text(5, y + 0.003, lab, fontsize=8.5, color=c)

ax.axhline(0.0929, color="#cc0000", ls=":", lw=1.5, alpha=0.6)
ax.text(700, 0.0929 + 0.003, "shared plateau = 0.0929",
        fontsize=9, color="#cc0000", style="italic")

ax.annotate(f"baseline final = {base_d[-1]:.4f}\n(iter {base_it[-1]})",
            xy=(base_it[-1], base_d[-1]),
            xytext=(base_it[-1] - 250, base_d[-1] - 0.04),
            fontsize=9.5, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))

ax.annotate(f"Amend01 final = {am_d[-1]:.4f}\n(iter {am_it[-1]}, killed)",
            xy=(am_it[-1], am_d[-1]),
            xytext=(am_it[-1] - 250, am_d[-1] + 0.05),
            fontsize=9.5, color="#cc0000", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.2))

ax.set_xlabel("PPO iteration", fontsize=12)
ax.set_ylabel("val macro_drop  (mean episode reward, normalized)",
              fontsize=12)
ax.set_title(
    "S1 PPO Amendment 01 vs baseline -- both stuck at val=0.0929 plateau\n"
    "Amend01 hyperparams: entropy_coef=5e-3 (5x), entropy_anneal_frac=1.0 (never)\n"
    "Result: entropy stayed ~5-8x higher than baseline BUT val curve identical -> not an exploration-strength issue",
    fontsize=10.5,
)
ax.set_xlim(-10, 1020)
ax.set_ylim(-0.02, 0.30)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=10)

summary = (
    "VERDICT: Amendment 01 FAILS\n"
    f"  baseline iter 999: val = {base_d[-1]:.4f}\n"
    f"  Amend01 iter 489 (killed): val = {am_d[-1]:.4f}\n"
    f"  Amend01 max in 489 iter:    val = {max(am_d):.4f}\n"
    f"  delta: {am_d[-1] - base_d[-1]:+.4f}pp (zero improvement)\n\n"
    "Root cause update:\n"
    "  5e-3 entropy_coef keeps entropy ~5-8x higher than baseline\n"
    "  BUT val curve is identical -> local min is genuinely hard,\n"
    "  not a 'too-weak exploration bonus' problem.\n\n"
    "Next levers to try:\n"
    "  - entropy_coef = 1e-2 or 3e-2 (much stronger)\n"
    "  - actor_lr 3e-5 -> 1e-4 (3x larger steps)\n"
    "  - warm-start from lite iter 2999 (already drop=0.2734 on S1 env)\n"
    "  - n_envs 16 -> 32 (more diverse samples per iter)"
)
ax.text(0.015, 0.97, summary, transform=ax.transAxes,
        fontsize=9, ha="left", va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888", alpha=0.95))

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"  baseline: {len(base_it)} val points, final = {base_d[-1]:.4f}")
print(f"  Amend01:  {len(am_it)} val points, final = {am_d[-1]:.4f}, max = {max(am_d):.4f}")
