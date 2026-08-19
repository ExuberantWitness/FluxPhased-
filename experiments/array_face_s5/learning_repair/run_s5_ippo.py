"""S5 IPPO driver — two cooperative jammers, parameter-shared actor + central critic.

First S5 training run (HANDOFF §11.3). Research question: can 2 jammers learn
division of labor (e.g. staggered energy budgets / complementary beams) against
the radar's 2D sweep, beating the S4 single-jammer saturation point
(0.0943 ± 0.0158, 71% of a 0.132 oracle ceiling)?

Config notes (S4 lessons carried over, REPORT.md §6):
  - NO beam shaping (falsified in S4)
  - beam entropy anneal_frac = 0.9 (extended exploration window)
  - cell sparse-init bias -3.0 (energy-minimization bootstrap)
  - use_privileged_critic = True (the central critic IS the CTDE core)
  - stop guardrails per HANDOFF §11.6: watch for divergence / reward collapse

Usage: python run_s5_ippo.py --seed 20260729 [--resume] [--iterations N]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # .../experiments/array_face_s5/learning_repair -> repo root
sys.path.insert(0, str(REPO))

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.g3_bsta_lite.observation import PROFILE_MDP_SANITY
from env.gpu.array_face_s5 import (
    EnvConfig, UPAConfig, N_CELLS_S5, N_BEAM_DIRS_S5,
)
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s5.learning_repair.trainer_s5 import (
    S5IPPOTrainer, evaluate_actor_s5,
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
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--shared-budget", action="store_true",
                        help="ONE common 63-token pool for both jammers "
                             "(the commons-dilemma variant: time-division "
                             "must EMERGE to reach the ceiling)")
    args = parser.parse_args()
    seed = int(args.seed)
    n_iterations = int(args.iterations)

    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")
    print(f"S5 IPPO (2-jammer cooperative)  seed={seed}")
    print(f"  train_seeds={len(train_seeds)}  val_seeds={len(validation_seeds)}")
    print(f"  iterations={n_iterations}")

    cfg = S2PPOConfigV2(
        profile=PROFILE_MDP_SANITY, iterations=n_iterations,
        n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02,
        per_head_entropy=True,
        entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3},
        entropy_anneal_frac_per_head={"cell": 0.7, "beam": 0.9},  # S4 expD lesson
        use_privileged_critic=True,   # central critic (CTDE) — required in S5
        privileged_value_coef=0.5,
        distill_coef=0.1,
        seed=seed, train_seed=seed, device=device,
    )
    env_cfg = EnvConfig(
        n_envs=16, horizon=64, n_services=2,
        dt=1.0, P_jam_W=2.0,               # per jammer (each has its own PA)
        active_budget_steps=63,             # per jammer
        duty_budget=1.0,
        arrival_rate_per_service=0.15, baseline_snr_db=22.0,
        mission_tau_window=6, detects_required=1,
        profile=PROFILE_MDP_SANITY, obs_delay_steps=1,
        potential_coef=0.05, gamma=0.99,
        shared_budget=args.shared_budget,
        device=device, seed=seed,
    )
    physics = default_debug_physics_config(P_jam_W=2.0)
    radar = UPAConfig()
    jammer = UPAConfig()

    head_specs = [
        HeadSpec("cell", "bernoulli", N_CELLS_S5, bernoulli_logit_bias=-3.0),
        HeadSpec("beam", "categorical", N_BEAM_DIRS_S5),
    ]
    print(f"  heads: {[(s.name, s.kind, s.n_actions) for s in head_specs]}")

    out_dir = HERE / (f"s5_shared_output_seed{seed}" if args.shared_budget
                      else f"s5_ippo_output_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir={out_dir}")

    trainer = S5IPPOTrainer(
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
    for it in range(resume_from, n_iterations):
        m = trainer.train_iteration()
        train_log.write(json.dumps(m) + "\n")
        train_log.flush()
        n_done += 1
        if (it + 1) % VAL_EVERY == 0 or it == n_iterations - 1:
            ve = evaluate_actor_s5(
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
            cell_freq = m.get("action_cell_freq") or []
            print(f"  iter {trainer.iteration:4d}  rollout_drop={m['rollout_drop']:.4f}  "
                  f"val_drop={ve['macro_mean_drop']:.4f}  "
                  f"entropy={m['entropy']:.4f} "
                  f"(c={m.get('entropy_cell',0):.3f}+b={m.get('entropy_beam',0):.3f})  "
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
