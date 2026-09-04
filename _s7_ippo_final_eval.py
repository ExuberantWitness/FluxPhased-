"""Full four-view final evaluation for an S7 IPPO checkpoint.

Output schema matches the MAPPO _s7_final_eval.py exactly (aseed_{n} with
h2h_drop/h2h_success/jam_vs_sweep_drop/rad_vs_idle_success/j1_only_drop plus
sweep_vs_idle_floor), so downstream analysis ingests both algorithms
identically.

Usage: python _s7_ippo_final_eval.py --out-dir <dir>
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
from experiments.array_face_s7.learning_repair.trainer_s7_ippo import (
    S7IPPOTrainer, evaluate_s7_ippo,
)
from paper.figures.final_eval_schema import build_metadata, wrap_final_eval

p = argparse.ArgumentParser()
p.add_argument("--out-dir", type=str, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--device", default="cuda")
args = p.parse_args()
device = args.device
out_dir = Path(args.out_dir)

cfg = S2PPOConfigV2(
    profile="array_face_s7_ippo", iterations=1,
    n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
    target_kl=0.02, per_head_entropy=True,
    entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3, "svc": 1e-2},
    entropy_anneal_frac_per_head={"cell": 0.7, "beam": 0.9, "svc": 0.5},
    use_privileged_critic=True, privileged_value_coef=0.5, distill_coef=0.1,
    seed=args.seed, train_seed=args.seed, device=device,
)
env_cfg = EnvConfig(
    n_envs=16, horizon=64, n_services=2,
    dt=1.0, active_budget_steps=63, duty_budget=1.0,
    arrival_rate_per_service=0.15,
    mission_tau_window=6, detects_required=1,
    potential_coef=0.05, gamma=0.99,
    device=device, seed=args.seed,
)
physics = default_debug_physics_config(P_jam_W=0.1)
jammer_specs = [HeadSpec("cell", "bernoulli", N_CELLS_S7, bernoulli_logit_bias=-3.0),
                HeadSpec("beam", "categorical", N_BEAM_DIRS_S7)]
radar_specs = [HeadSpec("beam", "categorical", N_BEAM_DIRS_S7),
               HeadSpec("svc", "categorical", 2)]
manifest_dir = Path('experiments/array_face_s1/manifests')
with open(manifest_dir / 'ppo_train.json') as f:
    train_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]
with open(manifest_dir / 'checkpoint_validation.json') as f:
    val_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]

trainer = S7IPPOTrainer(
    cfg=cfg, env_cfg=env_cfg, physics=physics,
    radar=UPAConfig(), jammer=UPAConfig(),
    train_seeds=train_seeds, manifest_path=manifest_dir / 'ppo_train.json',
    out_dir=out_dir, jammer_specs=jammer_specs, radar_specs=radar_specs,
)
stored = trainer.load_selfplay(out_dir / "selfplay_latest.pt")
print(f"IPPO checkpoint loaded (stored session iter={stored})", flush=True)

def sweep_vs_idle(seed_subset):
    drops = []
    floor_cfg = replace(env_cfg, n_envs=1)
    for sd in seed_subset:
        env = ArrayFaceS7VecEnv(floor_cfg, physics=physics, radar=UPAConfig(), jammer=UPAConfig())
        env.reset(seed=sd)
        E = floor_cfg.n_envs
        for t in range(env_cfg.horizon):
            b = t % 25
            rb_ = torch.full((E, N_RADARS), b, dtype=torch.int64, device=device)
            rs_ = torch.full((E, N_RADARS), t % 2, dtype=torch.int64, device=device)
            jcell = torch.zeros(E, N_JAMMERS, 25, device=device)
            jbeam = torch.zeros(E, N_JAMMERS, dtype=torch.int64, device=device)
            env.step(jcell, jbeam, rb_, rs_)
        drops.append(float(env.drop_ratio()[0]))
    return sum(drops) / len(drops)

results = {}
for aseed in [4242, 777, 31337]:
    t0 = time.time()
    views = evaluate_s7_ippo(
        trainer.jam_actors, trainer.rad_actors,
        env_cfg=env_cfg, physics=physics, radar=UPAConfig(), jammer=UPAConfig(),
        scenario_seeds=val_seeds, n_action_reps=1,
        device=device, action_seed=aseed,
    )
    results[f"aseed_{aseed}"] = {
        "h2h_drop": views["h2h"]["mean_drop"],
        "h2h_success": views["h2h"]["mean_success"],
        "jam_vs_sweep_drop": views["jam_only"]["mean_drop"],
        "rad_vs_idle_success": views["rad_only"]["mean_success"],
        "j1_only_drop": views["j1_only"]["mean_drop"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"aseed={aseed}: {results[f'aseed_{aseed}']}", flush=True)

t0 = time.time()
results["sweep_vs_idle_floor"] = {
    "drop": sweep_vs_idle(val_seeds), "elapsed_s": round(time.time() - t0, 1)}
print(f"sweep_vs_idle natural floor drop = {results['sweep_vs_idle_floor']['drop']:.4f}",
      flush=True)

out_path = out_dir / "final_eval.json"
metadata = build_metadata(
    train_seed=args.seed, algorithm="ippo", checkpoint_iteration=stored,
    n_jammers=env_cfg.n_jammers, n_radars=2,
    jammer_az_deg=env_cfg.jammer_az_deg, radar_az_deg=env_cfg.radar_az_deg,
    baseline_snr_db=env_cfg.baseline_snr_db, P_jam_W=env_cfg.P_jam_W,
    active_budget_steps=env_cfg.active_budget_steps, horizon=env_cfg.horizon,
    validation_manifest=manifest_dir / 'checkpoint_validation.json',
    action_seeds=[4242, 777, 31337], n_action_reps=1, device=device,
)
out_path.write_text(json.dumps(wrap_final_eval(results, metadata), indent=2))
print(f"wrote {out_path}", flush=True)
