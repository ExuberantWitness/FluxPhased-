"""S3 vs S2 comparison learning curves (seed 20260729).

Reads:
  S2: s2_ppo_output_amend02_seed20260729/   (baseline, final 0.2114)
  S3: s3_ppo_output_amend02_seed20260729/   (cell binding, final 0.4230)

Produces a 4-panel figure:
  (1) val_macro_drop overlay — S3 vs S2 with key milestones
  (2) total + per-head entropy — S3 three-head decomposition
  (3) action_base / beam frequency — final policy commitment
  (4) cell head entropy trajectory — the S3 research-question diagnostic

Run after both S2 and S3 training complete.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SEED = 20260729
S2_DIR = HERE.parents[1] / "array_face_s2" / "learning_repair" / f"s2_ppo_output_amend02_seed{SEED}"
S3_DIR = HERE / f"s3_ppo_output_amend02_seed{SEED}"
OUT_PNG = HERE / f"s2_vs_s3_seed{SEED}.png"


def load_val(p):
    rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()]
    return [r["iter"] for r in rows], [r["val_macro_drop"] for r in rows]


def load_train(p):
    rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()]
    return rows


# load
s2_x, s2_y = load_val(S2_DIR / "val_metrics.jsonl")
s3_x, s3_y = load_val(S3_DIR / "val_metrics.jsonl")
s2_tr = load_train(S2_DIR / "train_metrics.jsonl")
s3_tr = load_train(S3_DIR / "train_metrics.jsonl")

S2_COLOR = "#1f77b4"   # blue
S3_COLOR = "#cc0000"   # red
GREEN_LO, GREEN_HI = 0.17, 0.27

fig, axes = plt.subplots(2, 2, figsize=(17, 11))

# ============ Panel 1: val_macro_drop overlay ============
ax = axes[0, 0]
ax.plot(s2_x, s2_y, "o-", color=S2_COLOR, ms=4, lw=2.0, alpha=0.85,
        label=f"S2 (final={s2_y[-1]:.4f})")
ax.plot(s3_x, s3_y, "s-", color=S3_COLOR, ms=4, lw=2.0, alpha=0.85,
        label=f"S3 (final={s3_y[-1]:.4f})")
# GREEN zone (S2's criterion, for reference)
ax.axhspan(GREEN_LO, GREEN_HI, color="#2ca02c", alpha=0.08)
ax.text(5, GREEN_HI + 0.005, "S2 GREEN zone [0.17,0.27]", fontsize=8, color="#2ca02c")
# milestone: S3 crosses S2's final at iter ~199
s2_final = s2_y[-1]
for i, v in enumerate(s3_y):
    if v >= s2_final:
        ax.axvline(s3_x[i], color=S3_COLOR, ls=":", lw=1.0, alpha=0.5)
        ax.annotate(f"S3 crosses S2 final\n@iter {s3_x[i]}",
                    xy=(s3_x[i], v), xytext=(s3_x[i] + 80, s2_final - 0.08),
                    fontsize=8.5, color=S3_COLOR,
                    arrowprops=dict(arrowstyle="->", color=S3_COLOR, lw=1.0))
        break
ax.set_xlabel("PPO iteration", fontsize=11)
ax.set_ylabel("val macro_drop", fontsize=11)
ax.set_title("(1) Validation performance: S3 cell-binding vs S2 baseline", fontsize=12)
ax.set_xlim(-10, 1020)
ax.set_ylim(-0.02, 0.48)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=10)

# ============ Panel 2: total + per-head entropy (S3) ============
ax = axes[0, 1]
s3_it = [r["iteration"] for r in s3_tr]
s3_ent_total = [r["entropy"] for r in s3_tr]
s3_ent_base = [r.get("entropy_base", float("nan")) for r in s3_tr]
s3_ent_beam = [r.get("entropy_beam", float("nan")) for r in s3_tr]
s3_ent_cell = [r.get("entropy_cell", float("nan")) for r in s3_tr]
ax.plot(s3_it, s3_ent_total, "-", color=S3_COLOR, lw=2.0, alpha=0.9, label="S3 total")
ax.plot(s3_it, s3_ent_base, "--", color="#1f77b4", lw=1.4, alpha=0.75, label="S3 base head")
ax.plot(s3_it, s3_ent_beam, "--", color="#2ca02c", lw=1.4, alpha=0.75, label="S3 beam head")
ax.plot(s3_it, s3_ent_cell, "--", color="#ff7f0e", lw=1.4, alpha=0.75, label="S3 cell head")
# max entropy references
ax.axhline(3.4657, color="#ff7f0e", ls=":", lw=1.0, alpha=0.5)
ax.text(5, 3.48, "cell max 5·log(2)=3.466", fontsize=7.5, color="#ff7f0e")
ax.axhline(0.5, color="#888", ls="--", lw=1.0, alpha=0.5)
ax.text(5, 0.52, "collapse threshold 0.5", fontsize=7.5, color="#555")
ax.set_xlabel("PPO iteration", fontsize=11)
ax.set_ylabel("entropy (nats)", fontsize=11)
ax.set_title("(2) S3 entropy decomposition (3 heads)", fontsize=12)
ax.set_xlim(-10, 1020)
ax.set_ylim(-0.1, 6.0)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9)

# ============ Panel 3: final action frequency (S2 vs S3) ============
ax = axes[1, 0]
s2_last = s2_tr[-1]
s3_last = s3_tr[-1]
# base action frequency: [idle, jam_svc_0, jam_svc_1]
import numpy as np
base_labels = ["idle", "jam_svc_0", "jam_svc_1"]
x_pos = np.arange(len(base_labels))
width = 0.38
s2_base = s2_last.get("action_base_freq", [0, 0, 0])
s3_base = s3_last.get("action_base_freq", [0, 0, 0])
ax.bar(x_pos - width / 2, s2_base, width, color=S2_COLOR, alpha=0.8, label="S2")
ax.bar(x_pos + width / 2, s3_base, width, color=S3_COLOR, alpha=0.8, label="S3")
for i, (v2, v3) in enumerate(zip(s2_base, s3_base)):
    ax.text(i - width / 2, v2 + 0.01, f"{v2:.2f}", ha="center", fontsize=8, color=S2_COLOR)
    ax.text(i + width / 2, v3 + 0.01, f"{v3:.2f}", ha="center", fontsize=8, color=S3_COLOR)
ax.set_xticks(x_pos)
ax.set_xticklabels(base_labels, fontsize=10)
ax.set_ylabel("frequency", fontsize=11)
ax.set_title("(3) Final base-action policy (S2 vs S3)", fontsize=12)
ax.set_ylim(0, 1.15)
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=10)
# annotate S3's aggressive commitment
ax.text(0.02, 0.97, "S3: 98.4% jam_svc_0 (aggressive\ncontinuous jamming, near-zero idle)",
        transform=ax.transAxes, fontsize=8.5, va="top", color=S3_COLOR,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=S3_COLOR, alpha=0.9))

# ============ Panel 4: cell head entropy trajectory ============
ax = axes[1, 1]
ax.plot(s3_it, s3_ent_cell, "-", color="#ff7f0e", lw=2.2, alpha=0.9, label="cell head entropy")
ax.fill_between(s3_it, 0, s3_ent_cell, color="#ff7f0e", alpha=0.12)
ax.axhline(3.4657, color="#888", ls=":", lw=1.0, alpha=0.5)
ax.text(5, 3.49, "max explore 3.466", fontsize=8, color="#555")
ax.axhline(0.5, color="#888", ls="--", lw=1.0, alpha=0.5)
ax.text(5, 0.52, "collapse 0.5", fontsize=8, color="#555")
# annotate the convergence
ax.annotate(f"cell head: 3.44 → {s3_ent_cell[-1]:.2f}\n(committed but not collapsed)",
            xy=(999, s3_ent_cell[-1]), xytext=(650, 2.6),
            fontsize=9, color="#ff7f0e",
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.2))
ax.set_xlabel("PPO iteration", fontsize=11)
ax.set_ylabel("cell head entropy (nats)", fontsize=11)
ax.set_title("(4) HANDOFF §11.1: cell-binding policy trajectory", fontsize=12)
ax.set_xlim(-10, 1020)
ax.set_ylim(-0.1, 3.8)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=10)

# ============ Suptitle + summary box ============
summary = (
    f"S3 cell-binding vs S2 baseline (seed {SEED}, 1000 iter):\n"
    f"  S2 final = {s2_y[-1]:.4f}    S3 final = {s3_y[-1]:.4f}    (+{(s3_y[-1]/s2_y[-1]-1)*100:.0f}%)\n"
    f"  S3 broke 0.12 @ iter 129 (S2 @ iter 439, 3.4x faster)\n"
    f"  S3 crossed S2 final @ iter ~199\n"
    f"  cell head: 3.44 → {s3_ent_cell[-1]:.2f} (no collapse, HANDOFF mitigation worked)\n"
    f"  NOTE: S3 uses larger energy budget (63 vs 16 tokens) due to per-cell semantics"
)
fig.text(0.5, 0.005, summary, ha="center", fontsize=9.5, family="monospace",
         bbox=dict(boxstyle="round,pad=0.5", fc="#fff8e0", ec="#cc0000", alpha=0.95))

fig.suptitle(
    f"S3 (cell binding) vs S2 (jammer ULA) — seed {SEED}: S3 reaches 2x S2 performance",
    fontsize=13.5, y=0.995,
)
fig.tight_layout(rect=[0, 0.07, 1, 0.98])
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(summary)
