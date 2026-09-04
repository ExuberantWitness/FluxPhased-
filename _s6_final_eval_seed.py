"""Three-view final evaluation for an S6 self-play checkpoint, CLI-parameterized.

Protocol identical to _s6_final_eval.py (3 action seeds x 64 val seeds x reps=1
+ sweep_vs_idle natural floor), but the env config comes from the clean
EnvConfig() defaults (baseline_snr_db=12, P_jam_W=0.1) so the evaluation
matches the regime the snr=12 seeds were TRAINED in.
"""
import sys, json, time, argparse
sys.path.insert(0, '.')
import torch
from pathlib import Path
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s6 import EnvConfig, UPAConfig, N_CELLS_S6, N_BEAM_DIRS_S6, ArrayFaceS6VecEnv
from env.gpu.array_face_s6.geometry import N_RADARS
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s6.learning_repair.trainer_s6 import S6SelfPlayTrainer, evaluate_s6
from paper.figures.final_eval_schema import build_metadata, wrap_final_eval

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--device", default="cuda")
args = p.parse_args()
SEED = args.seed
device = args.device
out_dir = Path(f'experiments/array_face_s6/learning_repair/s6_selfplay_output_seed{SEED}')

cfg = S2PPOConfigV2(
    profile="array_face_s6_v1", iterations=1,
    n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
    target_kl=0.02, per_head_entropy=True,
    entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3, "svc": 1e-2},
    entropy_anneal_frac_per_head={"cell": 0.7, "beam": 0.9, "svc": 0.5},
    seed=SEED, train_seed=SEED, device=device,
)
# Clean S6b regime (snr=12, P_jam=0.1) — matches the training regime of the
# two replication seeds; do NOT reintroduce the stale snr=22 overrides.
env_cfg = EnvConfig(
    n_envs=16, horizon=64, n_services=2,
    dt=1.0, active_budget_steps=63, duty_budget=1.0,
    arrival_rate_per_service=0.15,
    mission_tau_window=6, detects_required=1,
    potential_coef=0.05, gamma=0.99,
    device=device, seed=SEED,
)
physics = default_debug_physics_config(P_jam_W=0.1)
jammer_specs = [HeadSpec("cell", "bernoulli", N_CELLS_S6, bernoulli_logit_bias=-3.0),
                HeadSpec("beam", "categorical", N_BEAM_DIRS_S6)]
radar_specs = [HeadSpec("beam", "categorical", N_BEAM_DIRS_S6),
               HeadSpec("svc", "categorical", 2)]
manifest_dir = Path('experiments/array_face_s1/manifests')
with open(manifest_dir / 'ppo_train.json') as f:
    train_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]
with open(manifest_dir / 'checkpoint_validation.json') as f:
    val_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]

trainer = S6SelfPlayTrainer(
    cfg=cfg, env_cfg=env_cfg, physics=physics,
    radar=UPAConfig(), jammer=UPAConfig(),
    train_seeds=train_seeds, manifest_path=manifest_dir / 'ppo_train.json',
    out_dir=out_dir, jammer_specs=jammer_specs, radar_specs=radar_specs,
)
stored = trainer.load_selfplay(out_dir / "selfplay_latest.pt")
print(f"checkpoint loaded (stored session iter={stored})", flush=True)

def sweep_vs_idle(seed_subset):
    """Fully-scripted reference: sweep radars vs idle jammer (natural floor)."""
    drops = []
    for sd in seed_subset:
        env = ArrayFaceS6VecEnv(env_cfg, physics=physics, radar=UPAConfig(), jammer=UPAConfig())
        env.reset(seed=sd)
        E = env_cfg.n_envs
        for t in range(env_cfg.horizon):
            b = t % 25
            rb_ = torch.full((E, N_RADARS), b, dtype=torch.int64, device=device)
            rs_ = torch.full((E, N_RADARS), t % 2, dtype=torch.int64, device=device)
            jcell = torch.zeros(E, 25, device=device)
            jbeam = torch.zeros(E, dtype=torch.int64, device=device)
            env.step(jcell, jbeam, rb_, rs_)
        drops.append(float(env.drop_ratio()[0]))
    return sum(drops) / len(drops)

results = {}
for aseed in [4242, 777, 31337]:
    t0 = time.time()
    views = evaluate_s6(
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
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"aseed={aseed}: {results[f'aseed_{aseed}']}", flush=True)

t0 = time.time()
floor_drop = sweep_vs_idle(val_seeds)
results["sweep_vs_idle_floor"] = {"drop": floor_drop, "elapsed_s": round(time.time() - t0, 1)}
print(f"sweep_vs_idle natural floor drop = {floor_drop:.4f}", flush=True)

out_path = out_dir / "final_eval.json"
metadata = build_metadata(
    train_seed=SEED, algorithm="mappo", checkpoint_iteration=stored,
    n_jammers=1, n_radars=N_RADARS,
    jammer_az_deg=None, radar_az_deg=None,
    baseline_snr_db=env_cfg.baseline_snr_db, P_jam_W=env_cfg.P_jam_W,
    active_budget_steps=env_cfg.active_budget_steps, horizon=env_cfg.horizon,
    validation_manifest=manifest_dir / 'checkpoint_validation.json',
    action_seeds=[4242, 777, 31337], n_action_reps=1, device=device,
    env_profile="array_face_s6_v1",
)
with open(out_path, "w") as f:
    json.dump(wrap_final_eval(results, metadata), f, indent=2)
print(f"wrote {out_path}")
