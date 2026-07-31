"""S1 PPO Amendment 02: stronger exploration across 3 seeds.

Amendment 02 vs baseline:
  entropy_coef_init:  1e-3  -> 5e-3   (5x; stronger action-distribution flattening)
  entropy_anneal_frac: 0.3  -> 0.5    (between baseline 0.3 and Amend01 1.0)
  target_kl:          0.01 -> 0.02   (2x; let escape trajectories form)

Hypothesis: Amend01 failed because KL rollback fired before escape could develop.
            target_kl=0.02 is the key new lever.

Seeds: 20260729 (known broke-out w/ baseline), 20260730 (known stuck w/ baseline),
       20260801 (fresh). The 20260730 result is the key test of "exploration as fix".

Usage: python run_s1_ppo_explore_v2.py --seed 20260729
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.g3_bsta_lite.observation import PROFILE_MDP_SANITY
from env.gpu.array_face_s1 import EnvConfig, RadarULAConfig, PROFILE_ARRAY_FACE_S1
from experiments.array_face_s1.learning_repair.trainer import (
    S1PPOConfig, S1PPOTrainer, evaluate_actor,
)


MANIFEST_DIR = HERE.parent / "manifests"
N_ITERATIONS = 1000
VAL_EVERY = 10
VAL_REPS_INTERMEDIATE = 2


def load_seeds(name: str) -> list[int]:
    with open(MANIFEST_DIR / f"{name}.json") as f:
        m = json.load(f)
    return [int(e["seed"]) for e in m["entries"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True,
                        help="training seed (20260729 / 20260730 / 20260801 etc)")
    args = parser.parse_args()
    seed = int(args.seed)

    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")
    print(f"S1 PPO AMENDMENT 02 (stronger exploration)  seed={seed}")
    print(f"  entropy_coef=5e-3, anneal_frac=0.5, target_kl=0.02")
    print(f"  train_seeds={len(train_seeds)}  val_seeds={len(validation_seeds)}")

    cfg = S1PPOConfig(
        profile=PROFILE_MDP_SANITY,
        iterations=N_ITERATIONS,
        n_envs=16, horizon=64,
        actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02,                # <-- Amendment 02: 2x baseline
        entropy_coef_init=5e-3,         # <-- Amendment 02: 5x baseline
        entropy_anneal_frac=0.5,        # <-- Amendment 02: between 0.3 and 1.0
        seed=seed, train_seed=seed,
        device=device,
    )
    env_cfg = EnvConfig(
        n_envs=16, horizon=64, n_services=2,
        dt=1.0, P_jam_W=50.0,
        active_budget_steps=16, duty_budget=0.25,
        arrival_rate_per_service=0.15, baseline_snr_db=22.0,
        mission_tau_window=6, detects_required=1,
        profile=PROFILE_MDP_SANITY, obs_delay_steps=1,
        potential_coef=0.05, gamma=0.99,
        device=device, seed=seed,
    )
    physics = default_debug_physics_config(P_jam_W=50.0)
    radar = RadarULAConfig()

    out_dir = HERE / f"s1_ppo_output_amend02_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir={out_dir}")

    trainer = S1PPOTrainer(
        cfg=cfg, env_cfg=env_cfg,
        physics=physics, radar=radar,
        train_seeds=train_seeds,
        manifest_path=MANIFEST_DIR / "ppo_train.json",
        out_dir=out_dir,
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
                  f"entropy={m['entropy']:.4f}  "
                  f"trans={trainer.cumulative_transitions}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    trainer.save_last_iter(trainer.iteration)
    with open(out_dir / "train_metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "val_metrics.jsonl", "w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_dir}/train_metrics.jsonl ({len(rows)} rows)")
    print(f"wrote {out_dir}/val_metrics.jsonl ({len(val_rows)} rows)")
    print(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
