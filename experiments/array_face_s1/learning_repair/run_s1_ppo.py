"""S1 driver: 1000-iter PPO run + per-iter validation curves.

Loads ppo_train (64 seeds) and checkpoint_validation (64 seeds) manifests,
trains S1PPOTrainer for 1000 iterations on mdp_sanity_v1 profile with
lr=3e-5, target_kl=0.01, and writes per-iter train metrics + per-10-iter
validation macro_drop curves to disk.

This run is exploratory (1.024M transitions > 0.5M lite prereg cap) and
DOES NOT re-judge any lite verdict. It only charts the S1 learning curve
for comparison against the lite R2 curve (which saturated at ~0.29).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s1 import EnvConfig, RadarULAConfig, PROFILE_ARRAY_FACE_S1
from env.gpu.g3_bsta_lite.observation import PROFILE_MDP_SANITY
from experiments.array_face_s1.learning_repair.trainer import (
    S1PPOConfig, S1PPOTrainer, evaluate_actor,
)


MANIFEST_DIR = HERE.parent / "manifests"
OUT_DIR = HERE / "s1_ppo_output"
N_ITERATIONS = 1000
VAL_EVERY = 10
VAL_REPS_INTERMEDIATE = 2


def load_seeds(name: str) -> list[int]:
    with open(MANIFEST_DIR / f"{name}.json") as f:
        m = json.load(f)
    return [int(e["seed"]) for e in m["entries"]]


def env_cfg_for(profile: str, device: str) -> EnvConfig:
    return EnvConfig(
        n_envs=16, horizon=64, n_services=2,
        dt=1.0, P_jam_W=50.0,
        active_budget_steps=16, duty_budget=0.25,
        arrival_rate_per_service=0.15, baseline_snr_db=22.0,
        mission_tau_window=6, detects_required=1,
        profile=profile, obs_delay_steps=1,
        potential_coef=0.05, gamma=0.99,
        device=device, seed=20260730,
    )


def main():
    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")
    print(f"S1 PPO: {len(train_seeds)} train seeds, {len(validation_seeds)} val seeds")
    print(f"manifest_dir: {MANIFEST_DIR}")
    print(f"out_dir: {OUT_DIR}")

    cfg = S1PPOConfig(
        profile=PROFILE_MDP_SANITY,
        iterations=N_ITERATIONS,
        n_envs=16, horizon=64,
        actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.01,
        seed=20260730, train_seed=20260730,
        device=device,
    )
    env_cfg = env_cfg_for(PROFILE_MDP_SANITY, device)
    physics = default_debug_physics_config(P_jam_W=50.0)
    radar = RadarULAConfig()
    trainer = S1PPOTrainer(
        cfg=cfg, env_cfg=env_cfg,
        physics=physics, radar=radar,
        train_seeds=train_seeds,
        manifest_path=MANIFEST_DIR / "ppo_train.json",
        out_dir=OUT_DIR,
    )
    trainer.save_pristine_init()

    rows: list[dict] = []
    val_rows: list[dict] = []
    t0 = time.time()
    for it in range(N_ITERATIONS):
        m = trainer.train_iteration()
        rows.append(m)
        if (it + 1) % VAL_EVERY == 0 or it == N_ITERATIONS - 1:
            ve = evaluate_actor(
                trainer.actor, env_cfg=env_cfg,
                physics=physics, radar=radar,
                scenario_seeds=validation_seeds,
                n_action_reps=VAL_REPS_INTERMEDIATE,
                sample=True, device=device, action_seed=4242,
            )
            val_rows.append({
                "iter": trainer.iteration,
                "val_macro_drop": ve["macro_mean_drop"],
                "elapsed_s": time.time() - t0,
            })
            print(f"  iter {trainer.iteration:4d}  rollout_drop={m['rollout_drop']:.4f}  "
                  f"val_drop={ve['macro_mean_drop']:.4f}  "
                  f"policy_loss={m['policy_loss']:.4f}  value_loss={m['value_loss']:.4f}  "
                  f"kl_max_post={m['kl_max_post']:.5f}  entropy={m['entropy']:.4f}  "
                  f"trans={trainer.cumulative_transitions}  "
                  f"elapsed={time.time()-t0:.0f}s")

    trainer.save_last_iter(trainer.iteration)

    with open(OUT_DIR / "train_metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT_DIR / "val_metrics.jsonl", "w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT_DIR}/train_metrics.jsonl ({len(rows)} rows)")
    print(f"wrote {OUT_DIR}/val_metrics.jsonl ({len(val_rows)} rows)")
    print(f"elapsed: {time.time() - t0:.1f}s  transitions: {trainer.cumulative_transitions}")


if __name__ == "__main__":
    main()
