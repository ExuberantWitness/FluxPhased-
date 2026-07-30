"""读 R2 训练 metrics 并绘制曲线。"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CURVES_DIR = HERE / "r2_gate3_output" / "curves_lr3e-05_kl0.01"
OUT_PNG = HERE / "r2_gate3_output" / "r2_curves.png"

train_rows = [json.loads(line) for line in (CURVES_DIR / "train_metrics.jsonl").read_text().splitlines()]
val_rows = [json.loads(line) for line in (CURVES_DIR / "val_metrics.jsonl").read_text().splitlines()]

iters = [r["iteration"] for r in train_rows]
rollout_drop = [r["rollout_drop"] for r in train_rows]
policy_loss = [r["policy_loss"] for r in train_rows]
value_loss = [r["value_loss"] for r in train_rows]
kl_max = [r["kl_max_post"] for r in train_rows]
kl_mean = [r["kl_mean_post"] for r in train_rows]
entropy = [r["entropy"] for r in train_rows]
adv_std = [r["adv_std"] for r in train_rows]
explained_var = [r["explained_variance"] for r in train_rows]
entropy_coef = [r["entropy_coef"] for r in train_rows]
act_freq = [r["action_freq"] for r in train_rows]
cum_trans = [r["cumulative_transitions"] for r in train_rows]

val_iters = [r["iter"] for r in val_rows]
val_drops = [r["val_macro_drop"] for r in val_rows]

fig, axes = plt.subplots(3, 3, figsize=(16, 10))

# 1. rollout + val drop
ax = axes[0, 0]
ax.plot(iters, rollout_drop, label="train rollout_drop", color="C0", lw=1.0)
ax.plot(val_iters, val_drops, label="val macro_drop (64 seeds)", color="C1", marker="o", ms=4)
ax.axhline(0.1653, color="C2", ls="--", lw=1, label="best baseline (round_robin) = 0.1653")
ax.axhline(0.2680, color="C3", ls="--", lw=1, label="witness ref = 0.2680")
ax.axhline(0.0149, color="C4", ls=":", lw=1, label="scratch_init = 0.0149")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("mission drop ratio")
ax.set_title("Drop ratio: train (rollout) vs validation")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

# 2. policy_loss
ax = axes[0, 1]
ax.plot(iters, policy_loss, color="C0")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("policy loss (clipped surr)")
ax.set_title("Actor policy loss")
ax.grid(alpha=0.3)

# 3. value_loss
ax = axes[0, 2]
ax.plot(iters, value_loss, color="C1")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("value loss (MSE)")
ax.set_title("Critic value loss")
ax.grid(alpha=0.3)

# 4. KL post-minibatch (mean + max)
ax = axes[1, 0]
ax.plot(iters, kl_mean, label="mean post-minibatch KL", color="C0")
ax.plot(iters, kl_max, label="max post-minibatch KL", color="C3")
ax.axhline(0.01, color="C2", ls="--", lw=1, label="target_kl = 0.01")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("KL(old || new)")
ax.set_title("Per-minibatch KL divergence (with rollback at target_kl)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 5. entropy
ax = axes[1, 1]
ax.plot(iters, entropy, color="C0", label="actor entropy")
ax.plot(iters, entropy_coef, color="C3", label="entropy_coef (annealed 1e-3 -> 0 over first 30%)")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("nats")
ax.set_title("Actor entropy + entropy coefficient")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 6. advantage std
ax = axes[1, 2]
ax.plot(iters, adv_std, color="C0")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("adv_std (post-normalize)")
ax.set_title("Advantage std (GAE)")
ax.grid(alpha=0.3)

# 7. action frequency (3 actions: idle, jam svc 0, jam svc 1)
ax = axes[2, 0]
act_freq_arr = list(zip(*act_freq))  # transposed: each list is one action across iters
ax.plot(iters, act_freq_arr[0], label="idle", color="C0")
ax.plot(iters, act_freq_arr[1], label="jam svc 0", color="C1")
ax.plot(iters, act_freq_arr[2], label="jam svc 1", color="C2")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("action frequency")
ax.set_title("Action frequency on rollouts (16-env × 64-step)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 8. cumulative transitions
ax = axes[2, 1]
ax.plot(iters, cum_trans, color="C0")
ax.axhline(500_000, color="C3", ls="--", lw=1, label="0.5M cap")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("cumulative transitions")
ax.set_title("Compute budget used (<= 0.5M)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 9. explained variance
ax = axes[2, 2]
ax.plot(iters, explained_var, color="C0")
ax.axhline(0.0, color="black", lw=0.5)
ax.set_xlabel("PPO iteration")
ax.set_ylabel("1 - Var(res) / Var(ret)")
ax.set_title("Critic explained variance")
ax.grid(alpha=0.3)

fig.suptitle(
    f"R2 scratch PPO on mdp_sanity_v1 (lr=3e-5, target_kl=0.01, n_envs=16, horizon=64) — "
    f"{len(train_rows)} outer iters, {cum_trans[-1]} transitions, 0 KL rollbacks",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
