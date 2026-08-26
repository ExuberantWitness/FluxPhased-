"""Four-view final evaluation for an S7 self-play checkpoint, CLI-parameterized.

Protocol mirrors S6's final eval (3 action seeds x 64 val seeds x reps=1 +
sweep_vs_idle natural floor), extended with the j1_only control view:
  h2h        — 2 learned jammers vs 2 learned radars
  jam_only   — 2 learned jammers vs scripted sweep (raw jammer power)
  rad_only   — learned radars vs idle jammers (radar competence floor)
  j1_only    — jammer 0 only vs learned radars (S7-env 1v2 control)
  sweep_vs_idle — scripted sweep vs idle jammers (natural floor)

Env config uses the clean EnvConfig defaults (baseline_snr_db=12, P_jam_W=0.1)
matching the regime the S7 seeds train in.

Usage: python _s7_final_eval.py --seed 20260801 [--device cuda]
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
p.add_argument("--seed", type=int, required=True)
p.add_argument("--device", default="cuda")
p.add_argument("--out-dir", default=None,
               help="override checkpoint dir (default: s7_selfplay_output_seed{seed})")
args = p.parse_args()
SEED = args.seed
device = args.device
out_dir = Path(args.out_dir) if args.out_dir else \
    Path(f'experiments/array_face_s7/learning_repair/s7_selfplay_output_seed{SEED}')

cfg = S2PPOConfigV2(
    profile="array_face_s7_v1", iterations=1,
    n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
    target_kl=0.02, per_head_entropy=True,
    entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3, "svc": 1e-2},
    entropy_anneal_frac_per_head={"cell": 0.7, "beam": 0.9, "svc": 0.5},
    use_privileged_critic=True, privileged_value_coef=0.5, distill_coef=0.1,
    seed=SEED, train_seed=SEED, device=device,
)
env_cfg = EnvConfig(
    n_envs=16, horizon=64, n_services=2,
    dt=1.0, active_budget_steps=63, duty_budget=1.0,
    arrival_rate_per_service=0.15,
    mission_tau_window=6, detects_required=1,
    potential_coef=0.05, gamma=0.99,
    device=device, seed=SEED,
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

trainer = S7SelfPlayTrainer(
    cfg=cfg, env_cfg=env_cfg, physics=physics,
    radar=UPAConfig(), jammer=UPAConfig(),
    train_seeds=train_seeds, manifest_path=manifest_dir / 'ppo_train.json',
    out_dir=out_dir, jammer_specs=jammer_specs, radar_specs=radar_specs,
)
stored = trainer.load_selfplay(out_dir / "selfplay_latest.pt")
print(f"checkpoint loaded (stored session iter={stored})", flush=True)

def sweep_vs_idle(seed_subset):
    """Fully-scripted reference: sweep radars vs both jammers idle (floor)."""
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
    views = evaluate_s7(
        trainer.jam_actor, trainer.rad_actor,
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
floor_drop = sweep_vs_idle(val_seeds)
results["sweep_vs_idle_floor"] = {"drop": floor_drop, "elapsed_s": round(time.time() - t0, 1)}
print(f"sweep_vs_idle natural floor drop = {floor_drop:.4f}", flush=True)

out_path = out_dir / "final_eval.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {out_path}")
