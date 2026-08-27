"""S7 self-play driver — 2 jammers vs 2 radars, full two-team MAPPO (HANDOFF §11.5).

Research question: does the second jammer break S6's defense-dominant
equilibrium? Tracked via four views every VAL_EVERY iters:
  h2h      — 2 learned jammers vs 2 learned radars (the game itself)
  jam_only — 2 learned jammers vs canonical scripted sweep (raw jammer power)
  rad_only — 2 learned radars vs idle jammers (radar competence floor)
  j1_only  — jammer 0 only vs learned radars (S7-env 1v2 control; isolates the
             second jammer's marginal power at equal per-jammer leverage)

Stop guardrails (HANDOFF §11.6): divergence / reward collapse / entropy lock
→ halt and discuss.

Usage: python run_s7_selfplay.py --seed 20260801 [--resume] [--iterations N]
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
from env.gpu.array_face_s7 import (
    EnvConfig, UPAConfig, N_CELLS_S7, N_BEAM_DIRS_S7,
)
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s7.learning_repair.trainer_s7 import (
    S7SelfPlayTrainer, evaluate_s7,
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
    parser.add_argument("--out-dir", type=str, default=None,
                        help="override output dir (default: s7_selfplay_output_seed{seed})")
    parser.add_argument("--anneal-done", action="store_true",
                        help="continuation runs: pin all per-head entropy coefficients at "
                             "coef_min from the resume point on (anneal window ends exactly "
                             "at the loaded iteration). Without this, a longer --iterations "
                             "would re-ramp the coefficients mid-run and confound the read.")
    parser.add_argument("--jammer-az", type=str, default=None,
                        help="ablation: comma-separated jammer site azimuths (deg), "
                             "e.g. '+60,+60' for co-located jammers")
    parser.add_argument("--radar-az", type=str, default=None,
                        help="ablation: comma-separated radar site azimuths (deg)")
    parser.add_argument("--val-every", type=int, default=VAL_EVERY,
                        help="intermediate validation cadence; final validation always runs")
    parser.add_argument("--singleton-mix", type=float, default=0.0,
                        help="R5 opponent mixing: fraction of training iterations on which "
                             "the radars face the singleton (jammer 1 idle); e.g. 0.25/0.5/0.75")
    args = parser.parse_args()
    seed = int(args.seed)
    n_iterations = int(args.iterations)
    val_every = max(1, int(args.val_every))

    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")
    print(f"S7 self-play (2 jammers vs 2 radars)  seed={seed}  iters={n_iterations}")

    out_dir = Path(args.out_dir) if args.out_dir else HERE / f"s7_selfplay_output_seed{seed}"

    anneal_fracs = {"cell": 0.7, "beam": 0.9, "svc": 0.5}
    if args.anneal_done:
        probe = out_dir / "selfplay_latest.pt"
        done_iter = int(torch.load(probe, map_location="cpu")["iteration"]) + 1
        f = min(0.99, done_iter / float(n_iterations))
        anneal_fracs = {"cell": f, "beam": f, "svc": f}
        print(f"  [anneal-done] anneal window ends at iter {int(f * n_iterations)} "
              f"(= resume {done_iter}); coefficients at coef_min throughout the continuation")

    cfg = S2PPOConfigV2(
        profile="array_face_s7_v1", iterations=n_iterations,
        n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02,
        per_head_entropy=True,
        entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3, "svc": 1e-2},
        entropy_anneal_frac_per_head=anneal_fracs,
        use_privileged_critic=True,   # central critic (CTDE) — required in S7
        privileged_value_coef=0.5,
        distill_coef=0.1,
        seed=seed, train_seed=seed, device=device,
    )
    # Validated S6b regime lives in the EnvConfig defaults (baseline_snr_db=12,
    # P_jam_W=0.1); the team budget (63 steps) is split 32/31 across jammers.
    env_cfg = EnvConfig(
        n_envs=16, horizon=64, n_services=2,
        dt=1.0, active_budget_steps=63, duty_budget=1.0,
        arrival_rate_per_service=0.15,
        mission_tau_window=6, detects_required=1,
        potential_coef=0.05, gamma=0.99,
        jammer_az_deg=tuple(float(x) for x in args.jammer_az.split(",")) if args.jammer_az else None,
        radar_az_deg=tuple(float(x) for x in args.radar_az.split(",")) if args.radar_az else None,
        device=device, seed=seed,
    )
    physics = default_debug_physics_config(P_jam_W=0.1)  # S6b-validated regime

    jammer_specs = [
        HeadSpec("cell", "bernoulli", N_CELLS_S7, bernoulli_logit_bias=-3.0),
        HeadSpec("beam", "categorical", N_BEAM_DIRS_S7),
    ]
    radar_specs = [
        HeadSpec("beam", "categorical", N_BEAM_DIRS_S7),
        HeadSpec("svc", "categorical", 2),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir={out_dir}")

    trainer = S7SelfPlayTrainer(
        cfg=cfg, env_cfg=env_cfg, physics=physics,
        radar=UPAConfig(), jammer=UPAConfig(),
        train_seeds=train_seeds,
        manifest_path=MANIFEST_DIR / "ppo_train.json",
        out_dir=out_dir,
        jammer_specs=jammer_specs, radar_specs=radar_specs,
        singleton_mix_frac=args.singleton_mix,
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
        if (it + 1) % val_every == 0 or it == n_iterations - 1:
            is_final = (it == n_iterations - 1)
            views = evaluate_s7(
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
                   "j1_only_drop": views["j1_only"]["mean_drop"],
                   "elapsed_s": time.time() - t0}
            val_log.write(json.dumps(row) + "\n")
            val_log.flush()
            print(f"  iter {trainer.iteration:4d}  h2h_drop={row['h2h_drop']:.4f}  "
                  f"jam_vs_sweep={row['jam_vs_sweep_drop']:.4f}  "
                  f"rad_vs_idle_succ={row['rad_vs_idle_success']:.4f}  "
                  f"j1_only={row['j1_only_drop']:.4f}  "
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
