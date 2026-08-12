"""S3 PPO driver — three-head (base + beam + cell) on ArrayFaceS3VecEnv.

First S3 training run. Cell head gets higher entropy coef + slower anneal
(HANDOFF §11.1 mitigation against Bernoulli(5) all-zero collapse).

Usage: python run_s3_ppo.py --seed 20260729 [--resume]
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
from env.gpu.array_face_s3 import EnvConfig, RadarULAConfig, JammerULAConfig, N_CELLS
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s3.learning_repair.trainer_s3 import (
    S3PPOTrainer, evaluate_actor_s3,
)


MANIFEST_DIR = HERE.parents[1] / "array_face_s1" / "manifests"
N_ITERATIONS = 1000
VAL_EVERY = 10
VAL_REPS_INTERMEDIATE = 2
CHECKPOINT_EVERY = 50


def load_seeds(name: str) -> list[int]:
    with open(MANIFEST_DIR / f"{name}.json") as f:
        m = json.load(f)
    return [int(e["seed"]) for e in m["entries"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seed = int(args.seed)

    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")
    print(f"S3 PPO (cell binding)  seed={seed}")
    print(f"  train_seeds={len(train_seeds)}  val_seeds={len(validation_seeds)}")

    cfg = S2PPOConfigV2(
        profile=PROFILE_MDP_SANITY, iterations=N_ITERATIONS,
        n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02,
        # Per-head entropy: cell head gets higher coef (anti-collapse) + slower
        # anneal. base/beam stay at amend02 values for S2 comparability.
        per_head_entropy=True,
        entropy_coef_per_head={"base": 5e-3, "beam": 5e-3, "cell": 1e-2},
        entropy_anneal_frac_per_head={"base": 0.5, "beam": 0.5, "cell": 0.5},
        seed=seed, train_seed=seed, device=device,
    )
    env_cfg = EnvConfig(
        n_envs=16, horizon=64, n_services=2,
        dt=1.0, P_jam_W=2.0,             # S3 per-cell
        active_budget_steps=63, duty_budget=1.0,  # generous budget for per-cell semantics
        arrival_rate_per_service=0.15, baseline_snr_db=22.0,
        mission_tau_window=6, detects_required=1,
        profile=PROFILE_MDP_SANITY, obs_delay_steps=1,
        potential_coef=0.05, gamma=0.99,
        device=device, seed=seed,
    )
    physics = default_debug_physics_config(P_jam_W=2.0)
    radar = RadarULAConfig()
    jammer = JammerULAConfig()

    head_specs = [
        HeadSpec("base", "categorical", 3),
        HeadSpec("beam", "categorical", 5),
        HeadSpec("cell", "bernoulli", N_CELLS),
    ]
    print(f"  heads: {[(s.name, s.kind, s.n_actions) for s in head_specs]}")

    out_dir = HERE / f"s3_ppo_output_amend02_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir={out_dir}")

    trainer = S3PPOTrainer(
        cfg=cfg, env_cfg=env_cfg,
        physics=physics, radar=radar, jammer=jammer,
        train_seeds=train_seeds,
        manifest_path=MANIFEST_DIR / "ppo_train.json",
        out_dir=out_dir,
        head_specs=head_specs,
    )

    resume_from = 0
    if args.resume:
        latest_ckpt = out_dir / "checkpoint_latest.pt"
        if latest_ckpt.exists():
            restored = trainer.load_checkpoint(latest_ckpt)
            resume_from = restored + 1
            print(f"  RESUMED from iter {restored}, continuing at {resume_from}")
        else:
            print(f"  --resume but no checkpoint; starting fresh")
            trainer.save_pristine_init()
    else:
        trainer.save_pristine_init()

    train_log = open(out_dir / "train_metrics.jsonl", "a", encoding="utf-8")
    val_log = open(out_dir / "val_metrics.jsonl", "a", encoding="utf-8")

    t0 = time.time()
    n_done = 0
    for it in range(resume_from, N_ITERATIONS):
        m = trainer.train_iteration()
        train_log.write(json.dumps(m) + "\n")
        train_log.flush()
        n_done += 1
        if (it + 1) % VAL_EVERY == 0 or it == N_ITERATIONS - 1:
            ve = evaluate_actor_s3(
                trainer.actor, env_cfg=env_cfg,
                physics=physics, radar=radar, jammer=jammer,
                scenario_seeds=validation_seeds,
                n_action_reps=VAL_REPS_INTERMEDIATE,
                sample=True, device=device, action_seed=4242,
            )
            val_row = {
                "iter": trainer.iteration,
                "val_macro_drop": ve["macro_mean_drop"],
                "elapsed_s": time.time() - t0,
            }
            val_log.write(json.dumps(val_row) + "\n")
            val_log.flush()
            # cell diagnostics
            cell_freq = [round(x, 3) for x in (m.get("action_cell_freq") or [])]
            print(f"  iter {trainer.iteration:4d}  rollout_drop={m['rollout_drop']:.4f}  "
                  f"val_drop={ve['macro_mean_drop']:.4f}  "
                  f"entropy={m['entropy']:.4f} "
                  f"(b={m.get('entropy_base',0):.3f}+m={m.get('entropy_beam',0):.3f}+c={m.get('entropy_cell',0):.3f})  "
                  f"trans={trainer.cumulative_transitions}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        if (it + 1) % CHECKPOINT_EVERY == 0:
            trainer.save_periodic(trainer.iteration)
            print(f"  [checkpoint] saved iter {trainer.iteration}", flush=True)

    train_log.close()
    val_log.close()
    trainer.save_last_iter(trainer.iteration)
    print(f"\nwrote {out_dir}/train_metrics.jsonl")
    print(f"wrote {out_dir}/val_metrics.jsonl")
    print(f"this session ran {n_done} iters; elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
