"""Evaluation-only SNR sensitivity: re-evaluate EXISTING converged
checkpoints at 9/12/15 dB without retraining.

Cross-fire seed-01 team and the co-located control team, each evaluated at
baseline SNR 9 and 15 dB (12 dB is the published regime). Four views per
point; eta computed from the same floor-adjusted formula. Runs on CPU so it
does not contend with training. Writes <outdir>/snr_reeval.json.

Usage: python _s7_snr_reeval.py [--device cpu]
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
    S7SelfPlayTrainer, evaluate_s7,
)

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu")
args = p.parse_args()
device = args.device

CONFIGS = [
    # (out_dir, train_seed, jammer_az, label)
    ('s7_continue2_output_seed20260801', 20260801, None, 'crossfire'),
    ('s7_ablation_output_seed20260811', 20260811, '+60,+60', 'colocated'),
]
SNRS = [9.0, 15.0]   # 12 dB is the published regime


def build(out_dir, seed, jammer_az):
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
        jammer_az_deg=tuple(float(x) for x in jammer_az.split(',')) if jammer_az else None,
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
        cfg=cfg, env_cfg=env_cfg, physics=default_debug_physics_config(P_jam_W=0.1),
        radar=UPAConfig(), jammer=UPAConfig(), train_seeds=train_seeds,
        manifest_path=manifest_dir / 'ppo_train.json',
        out_dir=Path('experiments/array_face_s7/learning_repair') / out_dir,
        jammer_specs=jammer_specs, radar_specs=radar_specs,
    )
    trainer.load_selfplay(Path('experiments/array_face_s7/learning_repair')
                          / out_dir / 'selfplay_latest.pt')
    return trainer, env_cfg


def sweep_floor(env_cfg, physics, seeds):
    drops = []
    floor_cfg = replace(env_cfg, n_envs=1)
    for sd in seeds:
        env = ArrayFaceS7VecEnv(floor_cfg, physics=physics,
                                radar=UPAConfig(), jammer=UPAConfig())
        env.reset(seed=sd)
        E = floor_cfg.n_envs
        for t in range(env_cfg.horizon):
            rb_ = torch.full((E, N_RADARS), t % 25, dtype=torch.int64)
            rs_ = torch.full((E, N_RADARS), t % 2, dtype=torch.int64)
            env.step(torch.zeros(E, N_JAMMERS, 25),
                     torch.zeros(E, N_JAMMERS, dtype=torch.int64), rb_, rs_)
        drops.append(float(env.drop_ratio()[0]))
    return sum(drops) / len(drops)


with open(Path('experiments/array_face_s1/manifests/checkpoint_validation.json')) as f:
    val_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]

for out_name, seed, jaz, label in CONFIGS:
    out_dir = Path('experiments/array_face_s7/learning_repair') / out_name
    results_path = out_dir / 'snr_reeval.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for snr_db in SNRS:
        key = f'snr_{snr_db:g}db'
        if key in results:
            continue
        trainer, env_cfg = build(out_name, seed, jaz)
        env_cfg = replace(env_cfg, baseline_snr_db=snr_db)
        physics = default_debug_physics_config(P_jam_W=0.1)
        t0 = time.time()
        views = evaluate_s7(
            trainer.jam_actor, trainer.rad_actor,
            env_cfg=env_cfg, physics=physics, radar=UPAConfig(), jammer=UPAConfig(),
            scenario_seeds=val_seeds, n_action_reps=1,
            device=device, action_seed=4242)
        h = views['h2h']['mean_drop']; j = views['jam_only']['mean_drop']
        ri = 1 - views['rad_only']['mean_success']
        f = sweep_floor(env_cfg, physics, val_seeds)
        results[key] = {'h2h': h, 'jvs': j, 'rad_idle': ri, 'floor': f,
                        'eta_pct': 100 * (1 - (h - ri) / (j - f)),
                        'elapsed_s': round(time.time() - t0, 1)}
        print(f"{label} {key}: {results[key]}", flush=True)
        results_path.write_text(json.dumps(results, indent=1))
    print(f"updated {results_path}", flush=True)
