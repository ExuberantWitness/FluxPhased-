"""S2 PPO v2 driver — amend03 config (per-head entropy + numerical hardening).

Amend03 vs amend02 hypothesis:
  amend02's 3 seeds all deterministically plateaued at val_drop ~0.211
  (std=0.0004), and beam-head entropy stayed near log(5)≈1.61 for the entire
  run (never converged). This suggests beam exploration is over-sustained.
  Amend03 gives the beam head a faster entropy anneal (0.5 → 0.3) so it
  commits to a beam-selection policy earlier, plus return normalization and
  log-ratio clamping for numerical stability.

Single-seed pilot (20260729) to test whether the plateau can be broken.
If final > 0.22 → plateau was exploration-induced (amend03 helps).
If final ≈ 0.211 → plateau is a task/physics ceiling (amend03 neutral).

Usage: python run_s2_ppo_v2.py --seed 20260729 [--amend03|--amend02eq]
  --amend02eq : default config (bit-exact amend02 via v2 path, control)
  --amend03   : per-head entropy (beam anneal 0.3) + return norm + clamp (default)
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
from experiments.array_face_s2.learning_repair.trainer_v2 import (
    S2PPOConfigV2, S2PPOTrainerV2, evaluate_actor_v2,
)


MANIFEST_DIR = HERE.parents[1] / "array_face_s1" / "manifests"
N_ITERATIONS = 1000
VAL_EVERY = 10
VAL_REPS_INTERMEDIATE = 2


def load_seeds(name: str) -> list[int]:
    with open(MANIFEST_DIR / f"{name}.json") as f:
        m = json.load(f)
    return [int(e["seed"]) for e in m["entries"]]


def build_config(mode: str, seed: int, device: str) -> S2PPOConfigV2:
    """amend02eq = control (bit-exact amend02). amend03 = experimental."""
    common = dict(
        profile=PROFILE_MDP_SANITY, iterations=N_ITERATIONS,
        n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02, seed=seed, train_seed=seed, device=device,
    )
    if mode == "amend02eq":
        # bit-exact amend02 via v2 path (all v2 features off)
        return S2PPOConfigV2(
            entropy_coef_init=5e-3, entropy_anneal_frac=0.5,
            per_head_entropy=False, normalize_returns=False, log_ratio_clamp=0.0,
            **common,
        )
    elif mode == "amend03":
        # beam head anneals faster (0.3 vs base 0.5); base unchanged.
        # return normalization stabilizes value targets; log-ratio clamp
        # guards against rare large logp swings.
        return S2PPOConfigV2(
            entropy_coef_init=5e-3, entropy_anneal_frac=0.5,
            per_head_entropy=True,
            entropy_coef_per_head={"base": 5e-3, "beam": 5e-3},
            entropy_anneal_frac_per_head={"base": 0.5, "beam": 0.3},
            normalize_returns=True, return_norm_clip=10.0,
            log_ratio_clamp=20.0,
            **common,
        )
    elif mode == "amend04":
        # B1: privileged critic + distillation. Asymmetric value head that
        # sees privileged state (pending + health + beam azimuths = 14 dims)
        # during training; obs-only head distilled toward it for deployment.
        # Other settings stay at amend02 defaults (per_head_entropy=False,
        # no return norm, no clamp) to isolate the B1 effect.
        return S2PPOConfigV2(
            entropy_coef_init=5e-3, entropy_anneal_frac=0.5,
            per_head_entropy=False, normalize_returns=False, log_ratio_clamp=0.0,
            use_privileged_critic=True,
            privileged_value_coef=0.5, distill_coef=0.1,
            **common,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["amend03", "amend02eq", "amend04"], default="amend03")
    parser.add_argument("--resume", action="store_true",
                        help="resume from checkpoint_latest.pt if it exists")
    args = parser.parse_args()
    seed = int(args.seed)
    mode = args.mode

    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")
    print(f"S2 PPO v2 ({mode})  seed={seed}")
    print(f"  train_seeds={len(train_seeds)}  val_seeds={len(validation_seeds)}")

    cfg = build_config(mode, seed, device)
    print(f"  per_head_entropy={cfg.per_head_entropy}  normalize_returns={cfg.normalize_returns}")
    if cfg.per_head_entropy:
        print(f"  entropy_coef_per_head={cfg.entropy_coef_per_head}")
        print(f"  entropy_anneal_frac_per_head={cfg.entropy_anneal_frac_per_head}")
    print(f"  log_ratio_clamp={cfg.log_ratio_clamp}  return_norm_clip={cfg.return_norm_clip}")

    env_cfg = EnvConfig(
        n_envs=16, horizon=64, n_services=2,
        dt=1.0, P_jam_W=2.0,             # S2 per-cell (plan §2.2, post-fix)
        active_budget_steps=16, duty_budget=0.25,
        arrival_rate_per_service=0.15, baseline_snr_db=22.0,
        mission_tau_window=6, detects_required=1,
        profile=PROFILE_MDP_SANITY, obs_delay_steps=1,
        potential_coef=0.05, gamma=0.99,
        device=device, seed=seed,
    )
    physics = default_debug_physics_config(P_jam_W=2.0)
    radar = RadarULAConfig()
    jammer = JammerULAConfig()

    out_dir = HERE / f"s2_ppo_output_{mode}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir={out_dir}")

    trainer = S2PPOTrainerV2(
        cfg=cfg, env_cfg=env_cfg,
        physics=physics, radar=radar, jammer=jammer,
        train_seeds=train_seeds,
        manifest_path=MANIFEST_DIR / "ppo_train.json",
        out_dir=out_dir,
    )

    # resume support: if --resume and a checkpoint exists, restore full state
    # (weights, optimizer, iteration counter, RNG) and continue from there.
    resume_from = 0  # iteration index to start at (0 = fresh)
    if args.resume:
        latest_ckpt = out_dir / "checkpoint_latest.pt"
        if latest_ckpt.exists():
            restored = trainer.load_checkpoint(latest_ckpt)
            resume_from = restored + 1
            print(f"  RESUMED from checkpoint: iteration {restored}, continuing at {resume_from}")
        else:
            print(f"  --resume given but no checkpoint at {latest_ckpt}; starting fresh")
            trainer.save_pristine_init()
    else:
        trainer.save_pristine_init()

    # incremental metrics files (append mode so resume preserves prior rows)
    train_log = open(out_dir / "train_metrics.jsonl", "a", encoding="utf-8")
    val_log = open(out_dir / "val_metrics.jsonl", "a", encoding="utf-8")

    CHECKPOINT_EVERY = 50  # save full-state checkpoint every 50 iters
    t0 = time.time()
    n_done = 0
    for it in range(resume_from, N_ITERATIONS):
        m = trainer.train_iteration()
        train_log.write(json.dumps(m) + "\n")
        train_log.flush()
        n_done += 1
        if (it + 1) % VAL_EVERY == 0 or it == N_ITERATIONS - 1:
            ve = evaluate_actor_v2(
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
            print(f"  iter {trainer.iteration:4d}  rollout_drop={m['rollout_drop']:.4f}  "
                  f"val_drop={ve['macro_mean_drop']:.4f}  "
                  f"entropy={m['entropy']:.4f} (b={m['entropy_base']:.3f}+m={m['entropy_beam']:.3f})  "
                  f"trans={trainer.cumulative_transitions}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        # periodic full-state checkpoint (survives interrupt; resume continues exactly)
        if (it + 1) % CHECKPOINT_EVERY == 0:
            trainer.save_periodic(trainer.iteration)
            print(f"  [checkpoint] saved iter {trainer.iteration}", flush=True)

    train_log.close()
    val_log.close()
    trainer.save_last_iter(trainer.iteration)
    # count rows in the metrics files (accounts for resume appending)
    n_train_rows = sum(1 for _ in open(out_dir / "train_metrics.jsonl", encoding="utf-8"))
    n_val_rows = sum(1 for _ in open(out_dir / "val_metrics.jsonl", encoding="utf-8"))
    print(f"\nwrote {out_dir}/train_metrics.jsonl ({n_train_rows} rows total)")
    print(f"wrote {out_dir}/val_metrics.jsonl ({n_val_rows} rows total)")
    print(f"this session ran {n_done} iters; total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
