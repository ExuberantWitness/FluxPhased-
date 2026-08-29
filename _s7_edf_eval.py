"""Deadline-informed (EDF) oracle heuristic baseline vs trained jammer teams.

Both radar heads point at the (service, azimuth) of the pending incomplete
mission with the earliest deadline, read from the env tracker. Unlike the
observation-based greedy baseline, this heuristic accesses deadline state that
is NOT in the radar observation, so it is an oracle-style upper-bound
reference, not a deployable policy. Protocol matches final evaluation:
64 validation scenarios x 3 action seeds, appended into each checkpoint's
baseline_eval.json.

Usage: python _s7_edf_eval.py [--device cuda]
"""
import sys, json, time, argparse
sys.path.insert(0, '.')
import torch
from pathlib import Path
from dataclasses import replace
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s7 import (
    EnvConfig, UPAConfig, N_CELLS_S7, N_BEAM_DIRS_S7, N_JAMMERS, N_RADARS,
    ArrayFaceS7VecEnv,
)
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s7.learning_repair.trainer_s7 import (
    S7SelfPlayTrainer, sample_multihead,
)

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda")
args = p.parse_args()
device = args.device

CHECKPOINTS = [
    ('s7_continue2_output_seed20260801', 20260801),
    ('s7_seed02_cont_output_seed20260802', 20260802),
    ('s7_seed03_cont_output_seed20260803', 20260803),
]


def build_trainer(out_dir: Path, seed: int):
    cfg = S2PPOConfigV2(
        profile="array_face_s7_v1", iterations=1,
        n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02, per_head_entropy=True,
        entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3, "svc": 1e-2},
        entropy_anneal_frac_per_head={"cell": 0.7, "beam": 0.9, "svc": 0.5},
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
    jammer_specs = [HeadSpec("cell", "bernoulli", N_CELLS_S7, bernoulli_logit_bias=-3.0),
                    HeadSpec("beam", "categorical", N_BEAM_DIRS_S7)]
    radar_specs = [HeadSpec("beam", "categorical", N_BEAM_DIRS_S7),
                   HeadSpec("svc", "categorical", 2)]
    manifest_dir = Path('experiments/array_face_s1/manifests')
    with open(manifest_dir / 'ppo_train.json') as f:
        train_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]
    trainer = S7SelfPlayTrainer(
        cfg=cfg, env_cfg=env_cfg,
        physics=default_debug_physics_config(P_jam_W=0.1),
        radar=UPAConfig(), jammer=UPAConfig(),
        train_seeds=train_seeds, manifest_path=manifest_dir / 'ppo_train.json',
        out_dir=out_dir, jammer_specs=jammer_specs, radar_specs=radar_specs,
    )
    stored = trainer.load_selfplay(out_dir / "selfplay_latest.pt")
    print(f"loaded {out_dir.name} (stored iter={stored})", flush=True)
    return trainer, env_cfg


def run_edf(trainer, env_cfg, physics, val_seeds, action_seed):
    """EDF oracle radar vs trained jammer team; returns mean drop."""
    jam_actor = trainer.jam_actor
    eval_cfg = replace(env_cfg, n_envs=1)
    E, K, R = 1, N_JAMMERS, N_RADARS
    drops = []
    for sd in val_seeds:
        env = ArrayFaceS7VecEnv(eval_cfg, physics=physics,
                                radar=UPAConfig(), jammer=UPAConfig())
        env.reset(seed=sd)
        gen = torch.Generator(device=device).manual_seed(action_seed + sd)
        for t in range(env_cfg.horizon):
            obs_j, _ = env._build_observation()
            mask_cell, mask_beam = env._compute_masks()
            # trained jammer team acts
            cells, beams = [], []
            for k in range(K):
                with torch.no_grad():
                    a_j, _ = sample_multihead(
                        jam_actor, obs_j[:, k],
                        {"cell": mask_cell[:, k], "beam": mask_beam[:, k]}, gen)
                cells.append(a_j["cell"]); beams.append(a_j["beam"])
            jcell = torch.stack(cells, dim=1)
            jbeam = torch.stack(beams, dim=1)
            # EDF oracle: earliest-deadline incomplete pending mission
            # tracker item layout: [svc, az, arr, dl, detects]
            pend = [(m[3], m[2], m[0], m[1]) for m in env.tracker.pending[0]
                    if m[4] < env.tracker.detects_required]
            if pend:
                _, _, s_idx, a_idx = min(pend)
            else:
                s_idx, a_idx = 0, 0
            rb_ = torch.full((E, R), a_idx + 10, dtype=torch.int64, device=device)
            rs_ = torch.full((E, R), s_idx, dtype=torch.int64, device=device)
            env.step(jcell, jbeam, rb_, rs_)
        drops.append(float(env.drop_ratio()[0]))
    return sum(drops) / len(drops)


def main():
    with open(Path('experiments/array_face_s1/manifests/checkpoint_validation.json')) as f:
        val_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]
    physics = default_debug_physics_config(P_jam_W=0.1)
    for name, seed in CHECKPOINTS:
        out_dir = Path('experiments/array_face_s7/learning_repair') / name
        trainer, env_cfg = build_trainer(out_dir, seed)
        out_path = out_dir / "baseline_eval.json"
        results = json.loads(out_path.read_text())
        for aseed in (4242, 777, 31337):
            t0 = time.time()
            results[f"aseed_{aseed}"]["edf_radar_vs_jam"] = run_edf(
                trainer, env_cfg, physics, val_seeds, aseed)
            print(f"{name} aseed={aseed}: edf="
                  f"{results[f'aseed_{aseed}']['edf_radar_vs_jam']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"updated {out_path}", flush=True)


if __name__ == "__main__":
    main()
