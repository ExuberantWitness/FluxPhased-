"""IPPO control driver for the S7 game — same protocol as run_s7_selfplay.py.

Trains 4 independent PPO learners (no parameter sharing, no CTDE central
critic) as the algorithm control for the MAPPO main results. Output layout,
metrics schema, validation cadence, and checkpoint cadence mirror the MAPPO
driver so downstream analysis is drop-in.

Usage: python _run_s7_ippo.py --seed 20260901 [--resume] [--iterations 2000]
"""
import sys, json, time, argparse
sys.path.insert(0, '.')
from pathlib import Path

import torch

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s7 import EnvConfig, UPAConfig, N_CELLS_S7, N_BEAM_DIRS_S7
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s7.learning_repair.trainer_s7_ippo import (
    S7IPPOTrainer, evaluate_s7_ippo,
)

HERE = Path(__file__).resolve().parent
MANIFEST_DIR = Path('experiments/array_face_s1/manifests')
N_ITERATIONS = 2000
VAL_EVERY = 50
CHECKPOINT_EVERY = 25


def load_seeds(name):
    with open(MANIFEST_DIR / f'{name}.json') as f:
        return [int(e["seed"]) for e in json.load(f)["entries"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--anneal-done", action="store_true")
    parser.add_argument("--val-every", type=int, default=VAL_EVERY)
    args = parser.parse_args()
    seed = int(args.seed)
    n_iterations = int(args.iterations)
    val_every = max(1, int(args.val_every))

    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")
    print(f"S7 IPPO control (independent learners)  seed={seed}  iters={n_iterations}",
          flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else \
        HERE / f"s7_ippo_output_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir={out_dir}", flush=True)

    anneal_fracs = {"cell": 0.7, "beam": 0.9, "svc": 0.5}
    if args.anneal_done:
        probe = out_dir / "selfplay_latest.pt"
        done_iter = int(torch.load(probe, map_location="cpu")["iteration"]) + 1
        f = min(0.99, done_iter / float(n_iterations))
        anneal_fracs = {"cell": f, "beam": f, "svc": f}

    cfg = S2PPOConfigV2(
        profile="array_face_s7_ippo", iterations=n_iterations,
        n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02, per_head_entropy=True,
        entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3, "svc": 1e-2},
        entropy_anneal_frac_per_head=anneal_fracs,
        # kept True only because the parent constructor requires it; the
        # privileged critics are never used by S7IPPOTrainer
        use_privileged_critic=True, privileged_value_coef=0.5, distill_coef=0.1,
        seed=seed, train_seed=seed, device=device,
    )
    env_cfg = EnvConfig(
        n_envs=16, horizon=64, n_services=2,
        dt=1.0, active_budget_steps=63, duty_budget=1.0,
        arrival_rate_per_service=0.15,
        mission_tau_window=6, detects_required=1,
        potential_coef=0.05, gamma=0.99,
        device=device, seed=seed,
    )
    physics = default_debug_physics_config(P_jam_W=0.1)

    jammer_specs = [HeadSpec("cell", "bernoulli", N_CELLS_S7, bernoulli_logit_bias=-3.0),
                    HeadSpec("beam", "categorical", N_BEAM_DIRS_S7)]
    radar_specs = [HeadSpec("beam", "categorical", N_BEAM_DIRS_S7),
                   HeadSpec("svc", "categorical", 2)]

    trainer = S7IPPOTrainer(
        cfg=cfg, env_cfg=env_cfg, physics=physics,
        radar=UPAConfig(), jammer=UPAConfig(),
        train_seeds=train_seeds, manifest_path=MANIFEST_DIR / 'ppo_train.json',
        out_dir=out_dir, jammer_specs=jammer_specs, radar_specs=radar_specs,
    )

    resume_from = 0
    ckpt = out_dir / "selfplay_latest.pt"
    if args.resume and ckpt.exists():
        resume_from = trainer.load_selfplay(ckpt) + 1
        print(f"  RESUMED from iter {resume_from - 1}", flush=True)

    train_log = open(out_dir / "train_metrics.jsonl", "a", encoding="utf-8")
    val_log = open(out_dir / "val_metrics.jsonl", "a", encoding="utf-8")

    t0 = time.time()
    for it in range(resume_from, n_iterations):
        m = trainer.train_iteration()
        train_log.write(json.dumps(m) + "\n")
        train_log.flush()
        if (it + 1) % val_every == 0 or it == n_iterations - 1:
            is_final = (it == n_iterations - 1)
            views = evaluate_s7_ippo(
                trainer.jam_actors, trainer.rad_actors,
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
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        if (it + 1) % CHECKPOINT_EVERY == 0:
            trainer.save_selfplay(ckpt)

    train_log.close()
    val_log.close()
    trainer.save_selfplay(ckpt)
    print(f"done; wrote metrics to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
