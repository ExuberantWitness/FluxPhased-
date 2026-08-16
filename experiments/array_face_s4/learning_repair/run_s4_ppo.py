"""S4 PPO driver — two-head (cell Bernoulli(25) + beam Categorical(25)) on 2D UPA.

First S4 training run (HANDOFF §11.2): no base head (idle = all-zero cells),
no service head (jammer always jams the radar's current service). The research
question is whether PPO can learn a useful policy in the 2D beam space
(25 directions = 5 az × 5 el) while managing 25 cells under a 63-token budget.

Entropy policy (HANDOFF §11.1 mitigation, scaled for the 25-dim Bernoulli):
  - cell head: entropy_coef = 2e-2 (2× S3's 1e-2) + slower anneal (0.7) —
    25 independent cells are far more prone to all-zero collapse than 5
  - beam head: 5e-3, anneal 0.5 (same as S3's categorical heads)

Usage: python run_s4_ppo.py --seed 20260729 [--resume]
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
from env.gpu.array_face_s4 import (
    EnvConfig, UPAConfig, N_CELLS_S4, N_BEAM_DIRS_S4,
)
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s4.learning_repair.trainer_s4 import (
    S4PPOTrainer, evaluate_actor_s4,
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
    # Ablation knobs (default = v2 config: average shaping @ 0.01 + beam trunk)
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--shaping-mode", type=str, default="average",
                        choices=["average", "tx_only"])
    parser.add_argument("--shaping-coef", type=float, default=0.01)
    parser.add_argument("--no-beam-trunk", action="store_true")
    parser.add_argument("--beam-anneal-frac", type=float, default=0.5,
                        help="beam head entropy anneal fraction of total iters")
    parser.add_argument("--beam-entropy-coef", type=float, default=5e-3,
                        help="beam head entropy coefficient")
    parser.add_argument("--outdir-tag", type=str, default="")
    args = parser.parse_args()
    seed = int(args.seed)
    n_iterations = int(args.iterations)

    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")
    print(f"S4 PPO (2D UPA cell binding)  seed={seed}")
    print(f"  train_seeds={len(train_seeds)}  val_seeds={len(validation_seeds)}")
    print(f"  shaping_mode={args.shaping_mode}  shaping_coef={args.shaping_coef}  "
          f"beam_trunk={not args.no_beam_trunk}  iterations={n_iterations}")

    cfg = S2PPOConfigV2(
        profile=PROFILE_MDP_SANITY, iterations=n_iterations,
        n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02,
        per_head_entropy=True,
        entropy_coef_per_head={"cell": 2e-2, "beam": args.beam_entropy_coef},
        entropy_anneal_frac_per_head={"cell": 0.7, "beam": args.beam_anneal_frac},
        seed=seed, train_seed=seed, device=device,
    )
    env_cfg = EnvConfig(
        n_envs=16, horizon=64, n_services=2,
        dt=1.0, P_jam_W=2.0,
        active_budget_steps=63, duty_budget=1.0,
        arrival_rate_per_service=0.15, baseline_snr_db=22.0,
        mission_tau_window=6, detects_required=1,
        profile=PROFILE_MDP_SANITY, obs_delay_steps=1,
        potential_coef=0.05,
        beam_shaping_coef=args.shaping_coef,       # Solution 1: beam alignment shaping
        beam_shaping_mode=args.shaping_mode,
        gamma=0.99,
        device=device, seed=seed,
    )
    physics = default_debug_physics_config(P_jam_W=2.0)
    radar = UPAConfig()
    jammer = UPAConfig()

    head_specs = [
        # Sparse-init bias -3.0: sigmoid(-3)≈0.047 → ~1 cell on per step at
        # init, so the 63-token budget lasts ~53 jamming steps instead of ~5.
        # Without this the uniform-init policy burns 12 cells/step, exhausts
        # the budget in ~5 steps, and gets no reward signal to learn from
        # (the S4 25-dim exploration failure; see run.log of the first attempt).
        HeadSpec("cell", "bernoulli", N_CELLS_S4, bernoulli_logit_bias=-3.0),
        HeadSpec("beam", "categorical", N_BEAM_DIRS_S4),
    ]
    print(f"  heads: {[(s.name, s.kind, s.n_actions) for s in head_specs]}")

    beam_trunk_heads = () if args.no_beam_trunk else ("beam",)  # Solution 2: separate trunk
    tag = f"_{args.outdir_tag}" if args.outdir_tag else ""
    out_dir = HERE / f"s4_ppo_output_seed{seed}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir={out_dir}")

    trainer = S4PPOTrainer(
        cfg=cfg, env_cfg=env_cfg,
        physics=physics, radar=radar, jammer=jammer,
        train_seeds=train_seeds,
        manifest_path=MANIFEST_DIR / "ppo_train.json",
        out_dir=out_dir,
        head_specs=head_specs,
        beam_trunk_heads=beam_trunk_heads,
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
            ve = evaluate_actor_s4(
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
            mean_cells = sum(cell_freq) if cell_freq else 0.0
            print(f"  iter {trainer.iteration:4d}  rollout_drop={m['rollout_drop']:.4f}  "
                  f"val_drop={ve['macro_mean_drop']:.4f}  "
                  f"entropy={m['entropy']:.4f} "
                  f"(c={m.get('entropy_cell',0):.3f}+b={m.get('entropy_beam',0):.3f})  "
                  f"mean_cells={mean_cells:.1f}  "
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
