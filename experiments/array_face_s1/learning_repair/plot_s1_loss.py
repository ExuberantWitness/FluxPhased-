"""S1 PPO loss / diagnostic curves.

8-subplot panel:
  - policy_loss
  - value_loss
  - entropy (+ entropy_coef anneal)
  - actor_grad_norm
  - explained_variance (critic fit quality)
  - kl_mean_post + kl_max_post (per-minibatch KL rollback trigger)
  - clip_frac_mean (PPO clip fraction)
  - adv_std (advantage magnitude)

Reads train_metrics.jsonl (1000 rows) and writes s1_loss_curves.png.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "s1_ppo_output"
OUT_PNG = OUT_DIR / "s1_loss_curves.png"

rows = [json.loads(l) for l in (OUT_DIR / "train_metrics.jsonl").read_text().splitlines()]
iters = [r["iteration"] for r in rows]
policy_loss = [r["policy_loss"] for r in rows]
value_loss = [r["value_loss"] for r in rows]
entropy = [r["entropy"] for r in rows]
entropy_coef = [r["entropy_coef"] for r in rows]
actor_grad_norm = [r["actor_grad_norm"] for r in rows]
explained_variance = [r["explained_variance"] for r in rows]
kl_mean_post = [r["kl_mean_post"] for r in rows]
kl_max_post = [r["kl_max_post"] for r in rows]
clip_frac_mean = [r["clip_frac_mean"] for r in rows]
adv_std = [r["adv_std"] for r in rows]

# Detect KL rollback events
kl_rollback_iters = [r["iteration"] for r in rows if r.get("kl_rollback")]

fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
fig.suptitle(
    "S1 PPO loss & diagnostic curves -- Array-Face phase 1\n"
    "profile=mdp_sanity_v1, lr=3e-5, target_kl=0.01, 16 envs x 64 steps, 1000 iter\n"
    f"KL rollbacks triggered at iters: {kl_rollback_iters if kl_rollback_iters else 'NONE'}",
    fontsize=12,
)

# 1. policy_loss
ax = axes[0, 0]
ax.plot(iters, policy_loss, color="C0", lw=1.0, alpha=0.7, label="per-iter")
win = 20
sma = [sum(policy_loss[max(0, i - win): i + 1]) /
       (i + 1 - max(0, i - win)) for i in range(len(policy_loss))]
ax.plot(iters, sma, color="C1", lw=2.0, label=f"SMA-{win}")
ax.axhline(0, color="#888888", ls="--", lw=0.8)
ax.set_ylabel("policy_loss (surrogate)", fontsize=10)
ax.set_title("Policy loss (PPO clipped surrogate)", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
ax.set_yscale("symlog", linthresh=1e-4)

# 2. value_loss
ax = axes[0, 1]
ax.plot(iters, value_loss, color="C2", lw=1.0, alpha=0.7)
sma_v = [sum(value_loss[max(0, i - win): i + 1]) /
         (i + 1 - max(0, i - win)) for i in range(len(value_loss))]
ax.plot(iters, sma_v, color="C3", lw=2.0, label=f"SMA-{win}")
ax.set_ylabel("value_loss (MSE)", fontsize=10)
ax.set_title("Value loss (critic)", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)

# 3. entropy + entropy_coef
ax = axes[1, 0]
ax.plot(iters, entropy, color="C4", lw=1.2, label="actor entropy")
ax.plot(iters, entropy_coef, color="C5", lw=1.2, ls="--", label="entropy_coef (annealed)")
ax.set_ylabel("entropy / coef", fontsize=10)
ax.set_title("Actor entropy + annealing schedule", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
ax.text(0.02, 0.05,
        f"init ent = {entropy[0]:.4f}\nfinal ent = {entropy[-1]:.4f}",
        transform=ax.transAxes, fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", alpha=0.9))

# 4. actor_grad_norm
ax = axes[1, 1]
ax.plot(iters, actor_grad_norm, color="C6", lw=1.0, alpha=0.7)
sma_g = [sum(actor_grad_norm[max(0, i - win): i + 1]) /
         (i + 1 - max(0, i - win)) for i in range(len(actor_grad_norm))]
ax.plot(iters, sma_g, color="C7", lw=2.0, label=f"SMA-{win}")
ax.axhline(0.5, color="#888", ls="--", lw=0.8, label="grad_clip = 0.5")
ax.set_ylabel("actor L2 grad norm", fontsize=10)
ax.set_title("Actor gradient norm (pre-clip)", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
ax.set_yscale("log")

# 5. explained_variance
ax = axes[2, 0]
ax.plot(iters, explained_variance, color="C8", lw=1.2)
ax.axhline(1.0, color="#2ca02c", ls="--", lw=0.8, label="perfect (1.0)")
ax.axhline(0.0, color="#888", ls="--", lw=0.8)
ax.set_ylabel("explained variance", fontsize=10)
ax.set_title("Critic explained variance (returns)", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
ax.set_ylim(-0.05, 1.05)
ax.text(0.02, 0.05,
        f"init EV = {explained_variance[0]:.3f}\nfinal EV = {explained_variance[-1]:.3f}",
        transform=ax.transAxes, fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", alpha=0.9))

# 6. KL divergence (mean + max post-update)
ax = axes[2, 1]
ax.plot(iters, kl_mean_post, color="C9", lw=1.2, label="kl_mean_post")
ax.plot(iters, kl_max_post, color="C3", lw=1.2, label="kl_max_post")
ax.axhline(0.01, color="#d62728", ls="--", lw=1.0, label="target_kl = 0.01")
ax.set_ylabel("KL post-update", fontsize=10)
ax.set_title("Per-minibatch KL (trigger rollback if > target_kl)", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
ax.set_yscale("log")

# 7. clip_frac_mean
ax = axes[3, 0]
ax.plot(iters, clip_frac_mean, color="C10", lw=1.2)
ax.axhline(0.2, color="#888", ls="--", lw=0.8, label="clip ratio = 0.2 (healthy ~0.1-0.3)")
ax.set_ylabel("clip_frac_mean", fontsize=10)
ax.set_title("PPO clip fraction (per-minibatch avg)", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)

# 8. adv_std
ax = axes[3, 1]
ax.plot(iters, adv_std, color="C11", lw=1.2)
sma_a = [sum(adv_std[max(0, i - win): i + 1]) /
         (i + 1 - max(0, i - win)) for i in range(len(adv_std))]
ax.plot(iters, sma_a, color="k", lw=2.0, label=f"SMA-{win}")
ax.set_ylabel("adv_std", fontsize=10)
ax.set_title("Advantage std (signal magnitude)", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8)

for ax in axes[-1]:
    ax.set_xlabel("PPO iteration", fontsize=10)

fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"  total KL rollbacks: {len(kl_rollback_iters)}")
print(f"  final policy_loss={policy_loss[-1]:.2e}  value_loss={value_loss[-1]:.4f}  "
      f"entropy={entropy[-1]:.4f}  EV={explained_variance[-1]:.3f}  "
      f"kl_max_post={kl_max_post[-1]:.2e}  clip_frac={clip_frac_mean[-1]:.3f}")
