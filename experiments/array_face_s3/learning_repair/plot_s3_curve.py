"""S3 PPO learning curve + S2 comparison + three-head entropy decomposition.

Reads:
  experiments/array_face_s3/learning_repair/s3_ppo_output_amend02_seed20260729/  (S3, final 0.4230)
  experiments/array_face_s2/learning_repair/s2_ppo_output_amend02_seed20260729/  (S2, final 0.2114)

Produces a 4-panel figure:
  (1) val_macro_drop: S3 vs S2 (S3 reaches 2x S2, breaks 0.12 at iter 129 vs S2's 439)
  (2) S3 three-head entropy decomposition (base / beam / cell)
  (3) S3 train rollout_drop (SMA-20)
  (4) final policy diagnostic: action freqs (base / beam)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
S3_VAL = HERE / "s3_ppo_output_amend02_seed20260729" / "val_metrics.jsonl"
S3_TR = HERE / "s3_ppo_output_amend02_seed20260729" / "train_metrics.jsonl"
S2_VAL = HERE.parents[1] / "array_face_s2" / "learning_repair" / "s2_ppo_output_amend02_seed20260729" / "val_metrics.jsonl"
S2_TR = HERE.parents[1] / "array_face_s2" / "learning_repair" / "s2_ppo_output_amend02_seed20260729" / "train_metrics.jsonl"
OUT_PNG = HERE / "s3_learning_curve.png"


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()]


def sma(xs, w=20):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


s3_va = load(S3_VAL)
s3_tr = load(S3_TR)
s2_va = load(S2_VAL)
s2_tr = load(S2_TR)

S3_COLOR = "#cc0000"
S2_COLOR = "#1f77b4"

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# ============ Panel 1 (top-left): val_macro_drop S3 vs S2 ============
ax = axes[0, 0]
s3_it = [r["iter"] for r in s3_va]
s3_vd = [r["val_macro_drop"] for r in s3_va]
s2_it = [r["iter"] for r in s2_va]
s2_vd = [r["val_macro_drop"] for r in s2_va]
ax.plot(s2_it, s2_vd, "o-", color=S2_COLOR, ms=4, lw=1.8, alpha=0.8,
        label=f"S2 amend02 (final={s2_vd[-1]:.4f})")
ax.plot(s3_it, s3_vd, "s-", color=S3_COLOR, ms=4, lw=2.0, alpha=0.85,
        label=f"S3 cell-binding (final={s3_vd[-1]:.4f})")
# S2 GREEN zone for reference
ax.axhspan(0.17, 0.27, color="#2ca02c", alpha=0.08)
ax.text(5, 0.273, "S2 GREEN [0.17,0.27]", fontsize=8.5, color="#2ca02c")
ax.axhline(0.2114, color=S2_COLOR, ls=":", lw=1.0, alpha=0.6)
ax.text(500, 0.22, "S2 final 0.2114", fontsize=8, color=S2_COLOR)
ax.axhline(0.4230, color=S3_COLOR, ls=":", lw=1.0, alpha=0.6)
ax.text(500, 0.43, "S3 final 0.4230", fontsize=8, color=S3_COLOR)
# breakout annotations
ax.annotate("S3 breaks 0.12\n@ iter 129", xy=(129, 0.124), xytext=(250, 0.18),
            fontsize=8.5, color=S3_COLOR,
            arrowprops=dict(arrowstyle="->", color=S3_COLOR, lw=1.0))
ax.annotate("S2 breaks 0.12\n@ iter 439", xy=(439, 0.1255), xytext=(560, 0.07),
            fontsize=8.5, color=S2_COLOR,
            arrowprops=dict(arrowstyle="->", color=S2_COLOR, lw=1.0))
ax.set_xlabel("PPO iteration", fontsize=11)
ax.set_ylabel("val macro_drop", fontsize=11)
ax.set_title("S3 vs S2: validation performance (seed 20260729)", fontsize=12)
ax.set_xlim(-10, 1020)
ax.set_ylim(-0.02, 0.48)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9.5)

# ============ Panel 2 (top-right): S3 three-head entropy decomposition ============
ax = axes[0, 1]
s3_tit = [r["iteration"] for r in s3_tr]
s3_e_base = [r.get("entropy_base", float("nan")) for r in s3_tr]
s3_e_beam = [r.get("entropy_beam", float("nan")) for r in s3_tr]
s3_e_cell = [r.get("entropy_cell", float("nan")) for r in s3_tr]
s3_e_total = [r["entropy"] for r in s3_tr]
ax.plot(s3_tit, s3_e_total, "-", color="black", lw=2.0, alpha=0.7, label="total")
ax.plot(s3_tit, s3_e_base, "-", color="#1f77b4", lw=1.5, alpha=0.8, label="base (Cat3)")
ax.plot(s3_tit, s3_e_beam, "-", color="#ff7f0e", lw=1.5, alpha=0.8, label="beam (Cat5)")
ax.plot(s3_tit, s3_e_cell, "-", color="#2ca02c", lw=1.5, alpha=0.8, label="cell (Bern5)")
ax.axhline(3.4657, color="#2ca02c", ls=":", lw=1.0, alpha=0.5)
ax.text(5, 3.49, "cell max log(2)·5=3.466", fontsize=8, color="#2ca02c")
ax.axhline(0.5, color="#888", ls="--", lw=1.0, alpha=0.5)
ax.text(5, 0.53, "collapse threshold 0.5", fontsize=8, color="#555")
ax.set_xlabel("PPO iteration", fontsize=11)
ax.set_ylabel("entropy (nats)", fontsize=11)
ax.set_title("S3 three-head entropy decomposition", fontsize=12)
ax.set_xlim(-10, 1020)
ax.set_ylim(-0.1, 5.8)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9.5)

# ============ Panel 3 (bottom-left): train rollout_drop (SMA-20) ============
ax = axes[1, 0]
s2_tit = [r["iteration"] for r in s2_tr]
ax.plot(s2_tit, sma([r["rollout_drop"] for r in s2_tr], 20),
        "-", color=S2_COLOR, lw=1.8, alpha=0.8, label="S2 (SMA-20)")
ax.plot(s3_tit, sma([r["rollout_drop"] for r in s3_tr], 20),
        "-", color=S3_COLOR, lw=2.0, alpha=0.85, label="S3 (SMA-20)")
ax.set_xlabel("PPO iteration", fontsize=11)
ax.set_ylabel("train rollout drop_ratio (SMA-20)", fontsize=11)
ax.set_title("Train reward proxy (rollout mean drop)", fontsize=12)
ax.set_xlim(-10, 1020)
ax.set_ylim(-0.01, 0.45)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9.5)

# ============ Panel 4 (bottom-right): final policy action distributions ============
ax = axes[1, 1]
s3_last = s3_tr[-1]
s2_last = s2_tr[-1]
import numpy as np
# base action comparison
x_base = np.arange(3)
w = 0.35
base_labels = ["idle", "jam_svc0", "jam_svc1"]
ax.bar(x_base - w/2, s2_last.get("action_base_freq", [0]*3), w,
       color=S2_COLOR, alpha=0.8, label="S2 base")
ax.bar(x_base + w/2, s3_last.get("action_base_freq", [0]*3), w,
       color=S3_COLOR, alpha=0.8, label="S3 base")
ax.set_xticks(x_base)
ax.set_xticklabels(base_labels)
ax.set_ylabel("frequency", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("Final base-action policy (S2 vs S3)", fontsize=12)
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=9.5)
# annotate S3's dominant action
s3_base = s3_last.get("action_base_freq", [0]*3)
dominant = max(range(3), key=lambda i: s3_base[i])
ax.text(dominant + w/2, s3_base[dominant] + 0.03, f"{s3_base[dominant]:.1%}",
        ha="center", fontsize=10, color=S3_COLOR, fontweight="bold")

fig.suptitle(
    "S3 PPO (cell binding) vs S2: val_drop 0.4230 vs 0.2114 (+100%), "
    "breaks 0.12 at iter 129 vs 439 (3.4x faster)\n"
    "HANDOFF §11.1 answer: PPO learns focused full-power jamming (98% jam_svc0 + broadside), "
    "not dynamic cell switching",
    fontsize=12, y=1.00,
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"S3 final val_drop: {s3_vd[-1]:.4f}  (S2: {s2_vd[-1]:.4f}, +{(s3_vd[-1]/s2_vd[-1]-1)*100:.0f}%)")
