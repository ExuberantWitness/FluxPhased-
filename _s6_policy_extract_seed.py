"""Extract final-policy behavior stats from the S6 self-play checkpoint (greedy)."""
import sys, json
sys.path.insert(0, '.')
import torch
from pathlib import Path
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s6 import EnvConfig, UPAConfig, N_CELLS_S6, N_BEAM_DIRS_S6, ArrayFaceS6VecEnv
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s6.learning_repair.trainer_s6 import S6SelfPlayTrainer, N_RADARS

import sys as _sys
SEED = int(_sys.argv[1])
device = 'cuda'
out_dir = Path(f'experiments/array_face_s6/learning_repair/s6_selfplay_output_seed{SEED}')

cfg = S2PPOConfigV2(
    profile="array_face_s6_v1", iterations=1,
    n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
    target_kl=0.02, per_head_entropy=True,
    entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3, "svc": 1e-2},
    entropy_anneal_frac_per_head={"cell": 0.7, "beam": 0.9, "svc": 0.5},
    seed=SEED, train_seed=SEED, device=device,
)
env_cfg = EnvConfig(
    n_envs=16, horizon=64, n_services=2,
    dt=1.0, active_budget_steps=63, duty_budget=1.0,
    arrival_rate_per_service=0.15,
    mission_tau_window=6, detects_required=1,
    potential_coef=0.05, gamma=0.99, device=device, seed=SEED,
)
physics = default_debug_physics_config(P_jam_W=0.1)
jammer_specs = [HeadSpec("cell", "bernoulli", N_CELLS_S6, bernoulli_logit_bias=-3.0),
                HeadSpec("beam", "categorical", N_BEAM_DIRS_S6)]
radar_specs = [HeadSpec("beam", "categorical", N_BEAM_DIRS_S6),
               HeadSpec("svc", "categorical", 2)]
manifest = Path('experiments/array_face_s1/manifests/ppo_train.json')
with open(manifest) as f:
    train_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]

trainer = S6SelfPlayTrainer(
    cfg=cfg, env_cfg=env_cfg, physics=physics,
    radar=UPAConfig(), jammer=UPAConfig(),
    train_seeds=train_seeds, manifest_path=manifest, out_dir=out_dir,
    jammer_specs=jammer_specs, radar_specs=radar_specs,
)
it = trainer.load_selfplay(out_dir / "selfplay_latest.pt")
print(f"checkpoint loaded (stored session iter={it})")

with open('experiments/array_face_s1/manifests/checkpoint_validation.json') as f:
    val_seeds = [int(e["seed"]) for e in json.load(f)["entries"]][:16]

E = env_cfg.n_envs
jam_ncells, jam_duty = [], []
jam_beam_hist = torch.zeros(25); rad_beam_hist = torch.zeros(25); rad_svc_hist = torch.zeros(2)
for s in val_seeds:
    env = ArrayFaceS6VecEnv(env_cfg, physics=physics, radar=UPAConfig(), jammer=UPAConfig())
    env.reset(seed=s)
    for t in range(env_cfg.horizon):
        obs_j, obs_r = env._build_observation()
        with torch.no_grad():
            lj = trainer.jam_actor.forward(obs_j)
        cell = (lj["cell"] > 0).float()
        beam_j = lj["beam"].argmax(-1)
        jam_ncells.append(cell.sum(-1).float().mean().item())
        jam_duty.append((cell.sum(-1) > 0).float().mean().item())
        jam_beam_hist[beam_j.cpu()] += 1
        for r in range(N_RADARS):
            with torch.no_grad():
                lr = trainer.rad_actor.forward(obs_r[:, r])
            rad_beam_hist[lr["beam"].argmax(-1).cpu()] += 1
            rad_svc_hist[lr["svc"].argmax(-1).cpu()] += 1
        rb_ = torch.zeros(E, N_RADARS, dtype=torch.int64, device=device)
        rs_ = torch.zeros(E, N_RADARS, dtype=torch.int64, device=device)
        for r in range(N_RADARS):
            with torch.no_grad():
                lr = trainer.rad_actor.forward(obs_r[:, r])
            rb_[:, r] = lr["beam"].argmax(-1)
            rs_[:, r] = lr["svc"].argmax(-1)
        env.step(cell, beam_j, rb_, rs_)

import statistics as st
jb = (jam_beam_hist / jam_beam_hist.sum()).tolist()
rb = (rad_beam_hist / rad_beam_hist.sum()).tolist()
rs = (rad_svc_hist / rad_svc_hist.sum()).tolist()
print(f"jammer: mean_cells(greedy)={st.mean(jam_ncells):.2f}  duty={st.mean(jam_duty):.3f}")
print("jammer beam hist:", " ".join(f"{v:.2f}" for v in jb))
print("radar  beam hist:", " ".join(f"{v:.2f}" for v in rb))
print(f"radar  svc mix: svc0={rs[0]:.3f} svc1={rs[1]:.3f}")
top_j = sorted(range(25), key=lambda i: -jb[i])[:4]
top_r = sorted(range(25), key=lambda i: -rb[i])[:4]
print(f"jammer top beams (idx:share): {[(i, round(jb[i],3)) for i in top_j]}")
print(f"radar  top beams (idx:share): {[(i, round(rb[i],3)) for i in top_r]}")
# radar beam az decomposition: idx = az + 5*el
from collections import Counter
az_r = Counter((i % 5) for i in range(25) for _ in range(int(rb[i]*1000)))
el_r = Counter((i // 5) for i in range(25) for _ in range(int(rb[i]*1000)))
az_j = Counter((i % 5) for i in range(25) for _ in range(int(jb[i]*1000)))
print("jammer az marg:", {k: round(v/1000,3) for k,v in sorted(az_j.items())})
print("radar  az marg:", {k: round(v/1000,3) for k,v in sorted(az_r.items())})
print("radar  el marg:", {k: round(v/1000,3) for k,v in sorted(el_r.items())})
