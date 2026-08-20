"""S6 self-play driver — 1 jammer vs 2 learning radars (HANDOFF §11.4).

Research question: arms race or Nash stagnation? Tracked via three views
every VAL_EVERY iters:
  h2h      — learned jammer vs learned radars (the game itself)
  jam_only — learned jammer vs canonical scripted sweep (jammer vs old world)
  rad_only — learned radars vs idle jammer (radar competence floor)

Stop guardrails (HANDOFF §11.6): divergence / reward collapse / entropy lock
→ halt and discuss.

Usage: python run_s6_selfplay.py --seed 20260729 [--resume] [--iterations N]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s6 import (
    EnvConfig, UPAConfig, N_CELLS_S6, N_BEAM_DIRS_S6,
)
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s6.learning_repair.trainer_s6 import (
    S6SelfPlayTrainer, evaluate_s6,
)

MANIFEST_DIR = HERE.parents[1] / "array_face_s1" / "manifests"
N_ITERATIONS = 1000
VAL_EVERY = 10
CHECKPOINT_EVERY = 50


def load_seeds(name: str) -> list[int]:
    with open(MANIFEST_DIR / f"{name}.json") as f:
        return [int(e["seed"]) for e in json.load(f)["entries"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    args = parser.parse_args()
    seed = int(args.seed)
    n_iterations = int(args.iterations)

    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")
    print(f"S6 self-play (1 jammer vs 2 radars)  seed={seed}  iters={n_iterations}")

    cfg = S2PPOConfigV2(
        profile="array_face_s6_v1", iterations=n_iterations,
        n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02,
        per_head_entropy=True,
        entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3, "svc": 1e-2},
        entropy_anneal_frac_per_head={"cell": 0.7, "beam": 0.9, "svc": 0.5},
        seed=seed, train_seed=seed, device=device,
    )
    # S6b rebalance lives in the EnvConfig defaults (baseline_snr_db=12,
    # P_jam_W=0.1) — do NOT override them here. The seed-20260729 run trained
    # before this comment existed and accidentally carried the stale S3-S5
    # overrides (baseline_snr_db=22, P_jam_W=2.0); see REPORT.md "Regime note".
    env_cfg = EnvConfig(
        n_envs=16, horizon=64, n_services=2,
        dt=1.0, active_budget_steps=63, duty_budget=1.0,
        arrival_rate_per_service=0.15,
        mission_tau_window=6, detects_required=1,
        potential_coef=0.05, gamma=0.99,
        device=device, seed=seed,
    )
    physics = default_debug_physics_config(P_jam_W=0.1)  # S6b rebalance

    jammer_specs = [
        HeadSpec("cell", "bernoulli", N_CELLS_S6, bernoulli_logit_bias=-3.0),
        HeadSpec("beam", "categorical", N_BEAM_DIRS_S6),
    ]
    radar_specs = [
        HeadSpec("beam", "categorical", N_BEAM_DIRS_S6),
        HeadSpec("svc", "categorical", 2),
    ]

    out_dir = HERE / f"s6_selfplay_output_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir={out_dir}")

    trainer = S6SelfPlayTrainer(
        cfg=cfg, env_cfg=env_cfg, physics=physics,
        radar=UPAConfig(), jammer=UPAConfig(),
        train_seeds=train_seeds,
        manifest_path=MANIFEST_DIR / "ppo_train.json",
        out_dir=out_dir,
        jammer_specs=jammer_specs, radar_specs=radar_specs,
    )

    resume_from = 0
    ckpt = out_dir / "selfplay_latest.pt"
    if args.resume and ckpt.exists():
        resume_from = trainer.load_selfplay(ckpt) + 1
        print(f"  RESUMED from iter {resume_from - 1}")

    train_log = open(out_dir / "train_metrics.jsonl", "a", encoding="utf-8")
    val_log = open(out_dir / "val_metrics.jsonl", "a", encoding="utf-8")

    t0 = time.time()
    for it in range(resume_from, n_iterations):
        m = trainer.train_iteration()
        train_log.write(json.dumps(m) + "\n")
        train_log.flush()
        if (it + 1) % VAL_EVERY == 0 or it == n_iterations - 1:
            is_final = (it == n_iterations - 1)
            views = evaluate_s6(
                trainer.jam_actor, trainer.rad_actor,
                env_cfg=env_cfg, physics=physics,
                radar=UPAConfig(), jammer=UPAConfig(),
                scenario_seeds=validation_seeds if is_final else validation_seeds[:16],
                n_action_reps=2 if is_final else 1,
                device=device, action_seed=4242,
            )
            row = {"iter": trainer.iteration,
                   "h2h_drop": views["h2h"]["mean_drop"],
                   "h2h_success": views["h2h"]["mean_success"],
                   "jam_vs_sweep_drop": views["jam_only"]["mean_drop"],
                   "rad_vs_idle_success": views["rad_only"]["mean_success"],
                   "elapsed_s": time.time() - t0}
            val_log.write(json.dumps(row) + "\n")
            val_log.flush()
            print(f"  iter {trainer.iteration:4d}  h2h_drop={row['h2h_drop']:.4f}  "
                  f"jam_vs_sweep={row['jam_vs_sweep_drop']:.4f}  "
                  f"rad_vs_idle_succ={row['rad_vs_idle_success']:.4f}  "
                  f"j_ent={m['jammer_entropy']:.2f} r_ent={m['radar_entropy']:.2f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        if (it + 1) % CHECKPOINT_EVERY == 0:
            trainer.save_selfplay(ckpt)
            print(f"  [checkpoint] iter {trainer.iteration}", flush=True)

    train_log.close()
    val_log.close()
    trainer.save_selfplay(ckpt)
    print(f"done; wrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
