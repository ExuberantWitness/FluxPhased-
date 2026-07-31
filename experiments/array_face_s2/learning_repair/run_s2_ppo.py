"""S2 PPO multi-seed runner. Amendment 02 config (entropy 5e-3, anneal 0.5, target_kl 0.02).

Usage: python run_s2_ppo.py --seed 20260729
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
from env.gpu.array_face_s2 import EnvConfig, RadarULAConfig, JammerULAConfig
from experiments.array_face_s2.learning_repair.trainer import (
    S2PPOConfig, S2PPOTrainer, evaluate_actor,
)


# Reuse S1 manifest (seeds are env-agnostic)
MANIFEST_DIR = HERE.parents[1] / "array_face_s1" / "manifests"
N_ITERATIONS = 1000
VAL_EVERY = 10
VAL_REPS_INTERMEDIATE = 2


def load_seeds(name: str) -> list[int]:
    with open(MANIFEST_DIR / f"{name}.json") as f:
        m = json.load(f)
    return [int(e["seed"]) for e in m["entries"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = int(args.seed)

    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")
    print(f"S2 PPO AMENDMENT 02 (MultiDiscrete)  seed={seed}")
    print(f"  entropy_coef=5e-3, anneal_frac=0.5, target_kl=0.02")
    print(f"  train_seeds={len(train_seeds)}  val_seeds={len(validation_seeds)}")

    cfg = S2PPOConfig(
        profile=PROFILE_MDP_SANITY,
        iterations=N_ITERATIONS,
        n_envs=16, horizon=64,
        actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02,
        entropy_coef_init=5e-3,
        entropy_anneal_frac=0.5,
        seed=seed, train_seed=seed,
        device=device,
    )
    env_cfg = EnvConfig(
        n_envs=16, horizon=64, n_services=2,
        dt=1.0, P_jam_W=10.0,            # S2: 5 cells × 2.0 W
        active_budget_steps=16, duty_budget=0.25,
        arrival_rate_per_service=0.15, baseline_snr_db=22.0,
        mission_tau_window=6, detects_required=1,
        profile=PROFILE_MDP_SANITY, obs_delay_steps=1,
        potential_coef=0.05, gamma=0.99,
        device=device, seed=seed,
    )
    physics = default_debug_physics_config(P_jam_W=10.0)
    radar = RadarULAConfig()
    jammer = JammerULAConfig()

    out_dir = HERE / f"s2_ppo_output_amend02_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir={out_dir}")

    trainer = S2PPOTrainer(
        cfg=cfg, env_cfg=env_cfg,
        physics=physics, radar=radar, jammer=jammer,
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
                physics=physics, radar=radar, jammer=jammer,
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
                  f"entropy={m['entropy']:.4f} (b={m['entropy_base']:.3f}+m={m['entropy_beam']:.3f})  "
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
