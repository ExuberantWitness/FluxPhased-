"""Greedy-stare counter-adaptation control — do jammers co-trained against a
mission-stare radar learn to punish it?

The evaluation-only baseline showed the self-play jammer teams cannot punish
a greedy mission-stare radar (drop 0.0889, invariant). Reviewer question: is
that a property of the game physics or of the training opponent distribution?
This control trains the jammer team AGAINST the fixed greedy radar (radar side
scripted, radar update skipped), then evaluates:

  greedy_vs_this_jam   — the direct question: greedy radar vs the
                         counter-adapted jammer team (baseline: 0.0889)
  sweep_vs_this_jam    — jammer raw power vs scripted sweeps (jvs view)
  this_jam_vs_radars   — cross-eval vs the standard pair-trained radar team

Usage: python _run_s7_greedy_counter.py --seed 20260921 [--resume]
       [--iterations 2000] [--val-every 50]
"""
import sys, json, time, argparse
sys.path.insert(0, '.')
from pathlib import Path

import torch

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s7 import (
    EnvConfig, UPAConfig, N_CELLS_S7, N_BEAM_DIRS_S7, N_JAMMERS, N_RADARS,
)
from experiments.array_face_s2.learning_repair.actor_heads import (
    HeadSpec, sample_multihead,
)
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s7.learning_repair.trainer_s7 import S7SelfPlayTrainer

HERE = Path('experiments/array_face_s7/learning_repair')
MANIFEST_DIR = Path('experiments/array_face_s1/manifests')
VAL_EVERY = 50
CHECKPOINT_EVERY = 25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--val-every", type=int, default=VAL_EVERY)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    seed = int(args.seed)
    n_iterations = int(args.iterations)
    val_every = max(1, int(args.val_every))
    device = "cuda"

    with open(MANIFEST_DIR / 'ppo_train.json') as f:
        train_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]
    with open(MANIFEST_DIR / 'checkpoint_validation.json') as f:
        validation_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]

    out_dir = Path(args.out_dir) if args.out_dir else \
        HERE / f"s7_greedycounter_output_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"S7 greedy-stare counter-adaptation  seed={seed}  iters={n_iterations}",
          flush=True)
    print(f"  out_dir={out_dir}", flush=True)

    anneal_fracs = {"cell": 0.7, "beam": 0.9, "svc": 0.5}
    if args.resume and (out_dir / "selfplay_latest.pt").exists():
        done_iter = int(torch.load(out_dir / "selfplay_latest.pt",
                                   map_location="cpu")["iteration"]) + 1
        f = min(0.99, done_iter / float(n_iterations))
        anneal_fracs = {"cell": f, "beam": f, "svc": f}

    cfg = S2PPOConfigV2(
        profile="array_face_s7_v1", iterations=n_iterations,
        n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02, per_head_entropy=True,
        entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3, "svc": 1e-2},
        entropy_anneal_frac_per_head=anneal_fracs,
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

    trainer = S7SelfPlayTrainer(
        cfg=cfg, env_cfg=env_cfg, physics=physics,
        radar=UPAConfig(), jammer=UPAConfig(),
        train_seeds=train_seeds, manifest_path=MANIFEST_DIR / 'ppo_train.json',
        out_dir=out_dir, jammer_specs=jammer_specs, radar_specs=radar_specs,
        radar_scripted="greedy",
    )
    ckpt = out_dir / "selfplay_latest.pt"
    resume_from = 0
    if args.resume and ckpt.exists():
        resume_from = trainer.load_selfplay(ckpt) + 1
        print(f"  RESUMED from iter {resume_from - 1}", flush=True)

    def greedy_view(seeds, action_seed):
        """Greedy radar vs the current jammer team (the question metric)."""
        from env.gpu.array_face_s7 import ArrayFaceS7VecEnv
        from dataclasses import replace
        eval_cfg = replace(env_cfg, n_envs=1)
        E, K, R = 1, N_JAMMERS, N_RADARS
        drops = []
        for sd in seeds:
            env = ArrayFaceS7VecEnv(eval_cfg, physics=physics,
                                    radar=UPAConfig(), jammer=UPAConfig())
            env.reset(seed=sd)
            gen = torch.Generator(device=device).manual_seed(action_seed + sd)
            for t in range(env_cfg.horizon):
                obs_j, obs_r = env._build_observation()
                mask_cell, mask_beam = env._compute_masks()
                cells, beams = [], []
                for k in range(K):
                    with torch.no_grad():
                        a_j, _ = sample_multihead(
                            trainer.jam_actor, obs_j[:, k],
                            {"cell": mask_cell[:, k], "beam": mask_beam[:, k]}, gen)
                    cells.append(a_j["cell"]); beams.append(a_j["beam"])
                jcell = torch.stack(cells, dim=1)
                jbeam = torch.stack(beams, dim=1)
                rb_ = torch.zeros(E, R, dtype=torch.int64, device=device)
                rs_ = torch.zeros(E, R, dtype=torch.int64, device=device)
                greedy_beam, greedy_svc = trainer._greedy_radar_actions(obs_r)
                rb_[:, :] = greedy_beam
                rs_[:, :] = greedy_svc
                env.step(jcell, jbeam, rb_, rs_)
            drops.append(float(env.drop_ratio()[0]))
        return sum(drops) / len(drops)

    train_log = open(out_dir / "train_metrics.jsonl", "a", encoding="utf-8")
    val_log = open(out_dir / "val_metrics.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    for it in range(resume_from, n_iterations):
        m = trainer.train_iteration()
        train_log.write(json.dumps(m) + "\n")
        train_log.flush()
        if (it + 1) % val_every == 0 or it == n_iterations - 1:
            is_final = (it == n_iterations - 1)
            seeds = validation_seeds if is_final else validation_seeds[:16]
            row = {"iter": trainer.iteration,
                   "greedy_vs_jam_drop": greedy_view(seeds, 4242),
                   "elapsed_s": time.time() - t0}
            val_log.write(json.dumps(row) + "\n")
            val_log.flush()
            print(f"  iter {trainer.iteration:4d}  greedy_vs_jam="
                  f"{row['greedy_vs_jam_drop']:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        if (it + 1) % CHECKPOINT_EVERY == 0:
            trainer.save_selfplay(ckpt)
    train_log.close()
    val_log.close()
    trainer.save_selfplay(ckpt)
    print(f"done; wrote metrics to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
