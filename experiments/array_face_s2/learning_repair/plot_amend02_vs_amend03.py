"""Plot amend03 vs amend02 comparison for seed 20260729.

Reads:
  s2_ppo_output_amend02_seed20260729/val_metrics.jsonl  (baseline, final 0.2114)
  s2_ppo_output_amend03_seed20260729/val_metrics.jsonl   (per-head entropy + norm)

Produces a side-by-side: val curve overlay + entropy decomposition.
Run after amend03 training completes.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SEED = 20260729
AMEND02 = HERE / f"s2_ppo_output_amend02_seed{SEED}" / "val_metrics.jsonl"
AMEND03 = HERE / f"s2_ppo_output_amend03_seed{SEED}" / "val_metrics.jsonl"
AMEND02_TR = HERE / f"s2_ppo_output_amend02_seed{SEED}" / "train_metrics.jsonl"
AMEND03_TR = HERE / f"s2_ppo_output_amend03_seed{SEED}" / "train_metrics.jsonl"
OUT_PNG = HERE / f"amend02_vs_amend03_seed{SEED}.png"


def load_val(p):
    rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()]
    return [r["iter"] for r in rows], [r["val_macro_drop"] for r in rows]


def load_entropy(p):
    rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()]
    return (
        [r["iteration"] for r in rows],
        [r.get("entropy", float("nan")) for r in rows],
        [r.get("entropy_base", float("nan")) for r in rows],
        [r.get("entropy_beam", float("nan")) for r in rows],
    )


fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

# --- Panel 1: val_macro_drop ---
ax = axes[0]
if AMEND02.exists():
    x, y = load_val(AMEND02)
    ax.plot(x, y, "o-", color="#1f77b4", ms=4, lw=1.8, alpha=0.8,
            label=f"amend02 (final={y[-1]:.4f})")
if AMEND03.exists():
    x, y = load_val(AMEND03)
    ax.plot(x, y, "s-", color="#cc0000", ms=4, lw=1.8, alpha=0.8,
            label=f"amend03 (final={y[-1]:.4f})")
ax.axhspan(0.17, 0.27, color="#2ca02c", alpha=0.08)
ax.text(5, 0.273, "GREEN [0.17,0.27]", fontsize=8.5, color="#2ca02c")
ax.axhline(0.2114, color="#888", ls=":", lw=1.0)
ax.text(5, 0.215, "amend02 plateau 0.2114", fontsize=8, color="#555")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("val macro_drop")
ax.set_title(f"Validation performance (seed {SEED})")
ax.set_xlim(-10, 1020)
ax.set_ylim(-0.02, 0.30)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=10)

# --- Panel 2: total entropy ---
ax = axes[1]
for path, color, lab in [(AMEND02_TR, "#1f77b4", "amend02"), (AMEND03_TR, "#cc0000", "amend03")]:
    if path.exists():
        it, ent, _, _ = load_entropy(path)
        ax.plot(it, ent, "-", color=color, lw=1.5, alpha=0.8, label=f"{lab} total")
ax.axhline(0.5, color="#888", ls="--", lw=1.0)
ax.text(5, 0.52, "collapse threshold 0.5", fontsize=8, color="#555")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("joint entropy (nats)")
ax.set_title("Total policy entropy")
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

# --- Panel 3: entropy decomposition (base vs beam) ---
ax = axes[2]
for path, color, lab in [(AMEND02_TR, "#1f77b4", "amend02"), (AMEND03_TR, "#cc0000", "amend03")]:
    if path.exists():
        it, _, eb, em = load_entropy(path)
        ax.plot(it, em, "-", color=color, lw=1.5, alpha=0.7, label=f"{lab} beam")
        ax.plot(it, eb, "--", color=color, lw=1.5, alpha=0.7, label=f"{lab} base")
ax.axhline(1.6094, color="#888", ls=":", lw=1.0)
ax.text(5, 1.62, "max log(5)=1.609", fontsize=8, color="#555")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("per-head entropy (nats)")
ax.set_title("Entropy decomposition (beam vs base)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, ncol=2)

verdict = ""
if AMEND03.exists():
    _, y03 = load_val(AMEND03)
    final03 = y03[-1]
    if final03 > 0.22:
        verdict = f"amend03 final={final03:.4f} > 0.22 → PLATEAU BROKEN (exploration was the bottleneck)"
    elif final03 > 0.215:
        verdict = f"amend03 final={final03:.4f} marginally above amend02 → weak improvement"
    else:
        verdict = f"amend03 final={final03:.4f} ≈ amend02 0.2114 → plateau is task/physics ceiling"
    fig.text(0.5, 0.01, verdict, ha="center", fontsize=11, color="#cc0000",
             style="italic")

fig.suptitle(
    f"Amend03 vs Amend02 (seed {SEED}): per-head beam anneal (0.5→0.3) + return norm + log-ratio clamp",
    fontsize=12.5, y=1.00,
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
if AMEND03.exists():
    print(verdict)
