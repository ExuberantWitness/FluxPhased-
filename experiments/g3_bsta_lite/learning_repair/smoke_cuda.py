"""Smoke-test on GPU."""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from env.gpu.g3_bsta_lite import EnvConfig, PROFILE_MDP_SANITY, N_ACTIONS
from experiments.g3_bsta_lite.learning_repair.trainer import (
    R2PPOConfig, R2PPOTrainer, evaluate_actor,
)

device = "cuda"
cfg = R2PPOConfig(
    profile=PROFILE_MDP_SANITY,
    iterations=30,
    n_envs=16, horizon=64,
    epochs_per_iteration=4, minibatch_size=256,
    actor_lr=1e-4, critic_lr=1e-3,
    target_kl=0.02, seed=1, train_seed=1, device=device,
)
env_cfg = EnvConfig(
    n_envs=16, horizon=64, profile=PROFILE_MDP_SANITY, obs_delay_steps=0,
    device=device,
)
t0 = time.time()
trainer = R2PPOTrainer(
    cfg=cfg, env_cfg=env_cfg,
    train_seeds=[21001101 + i for i in range(8)],
    manifest_path=HERE / "manifests" / "ppo_train.json",
    out_dir=HERE / "_smoke_cuda_out",
)
print(f"setup: {time.time()-t0:.2f}s")

t0 = time.time()
trainer.save_pristine_init()
pe = evaluate_actor(
    trainer.actor, env_cfg=env_cfg, scenario_seeds=[21002101 + i for i in range(8)],
    n_action_reps=2, sample=True, device=device, action_seed=11,
)
print(f"pristine eval (8 seeds × 2 reps): {time.time()-t0:.2f}s  macro={pe['macro_mean_drop']:.4f}")

t0 = time.time()
for it in range(30):
    m = trainer.train_iteration()
    if it % 5 == 0 or it == 29:
        print(f"  iter {it}: drop={m['rollout_drop']:.3f} kl_post_max={m['kl_max_post']:.4f} "
              f"rollback={m['kl_rollback']} cum={trainer.cumulative_transitions} "
              f"act_freq={[round(x,2) for x in m['action_freq']]}")
print(f"30 iters: {time.time()-t0:.2f}s")

t0 = time.time()
pe2 = evaluate_actor(
    trainer.actor, env_cfg=env_cfg, scenario_seeds=[21002101 + i for i in range(8)],
    n_action_reps=4, sample=True, device=device, action_seed=11,
)
print(f"post-train eval: {time.time()-t0:.2f}s  macro={pe2['macro_mean_drop']:.4f}")
