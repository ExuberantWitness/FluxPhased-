"""Tiny smoke test for R2 trainer pipeline."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from env.gpu.g3_bsta_lite import EnvConfig, PROFILE_MDP_SANITY, N_ACTIONS
from experiments.g3_bsta_lite.learning_repair.trainer import (
    R2PPOConfig, R2PPOTrainer, evaluate_actor, MaskedCategoricalActor,
)

cfg = R2PPOConfig(
    profile=PROFILE_MDP_SANITY,
    iterations=2,
    n_envs=4, horizon=16,
    epochs_per_iteration=2, minibatch_size=8,
    actor_lr=1e-4, critic_lr=1e-3,
    target_kl=0.02, seed=1, train_seed=1, device="cpu",
)
env_cfg = EnvConfig(
    n_envs=4, horizon=16, profile=PROFILE_MDP_SANITY, obs_delay_steps=0,
    device="cpu",
)
trainer = R2PPOTrainer(
    cfg=cfg, env_cfg=env_cfg,
    train_seeds=[21001101, 21001102, 21001103],
    manifest_path=HERE / "manifests" / "ppo_train.json",
    out_dir=HERE / "_smoke_out",
)
print("trainer init ok")
print("pristine init snapshot ok:", trainer.save_pristine_init().exists())

# Run pristine eval on 2 seeds.
pe = evaluate_actor(
    trainer.actor, env_cfg=env_cfg, scenario_seeds=[21002101, 21002102],
    n_action_reps=2, sample=True, device="cpu", action_seed=11,
)
print("pristine eval macro:", pe["macro_mean_drop"])

# Run 2 training iterations.
for it in range(2):
    m = trainer.train_iteration()
    print(f"iter {it}: drop={m['rollout_drop']:.3f} kl_post_max={m['kl_max_post']:.4f} rollback={m['kl_rollback']}")

# Re-eval.
pe2 = evaluate_actor(
    trainer.actor, env_cfg=env_cfg, scenario_seeds=[21002101, 21002102],
    n_action_reps=2, sample=True, device="cpu", action_seed=11,
)
print("post-train macro:", pe2["macro_mean_drop"])
print("cumulative_transitions:", trainer.cumulative_transitions)
