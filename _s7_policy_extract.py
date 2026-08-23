"""Greedy behavior extraction for an S7 checkpoint — incl. cross-assignment.

Answers the S7 headline behavior question: do the two jammers cross-assign
(jammer k suppresses radar k) or double-beam one radar? Prints:
  - per-jammer mean cells / duty
  - per-jammer beam histogram + top beams
  - per-radar beam histogram + az/el marginals
  - cross-assignment matrix: for each (jammer k, radar r) the mean JNR
    contribution when jammer k is active (who is hurting whom)

Usage: python _s7_policy_extract.py <seed>
"""
import sys, json
sys.path.insert(0, '.')
import torch
from pathlib import Path
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s7 import (
    EnvConfig, UPAConfig, N_CELLS_S7, N_BEAM_DIRS_S7, N_JAMMERS, N_RADARS,
    ArrayFaceS7VecEnv,
)
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec, sample_multihead
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s7.learning_repair.trainer_s7 import S7SelfPlayTrainer

SEED = int(sys.argv[1])
device = sys.argv[2] if len(sys.argv) > 2 else 'cuda'  # 'cpu' avoids GPU contention
MODE = sys.argv[3] if len(sys.argv) > 3 else 'greedy'   # 'greedy' | 'stochastic'
out_dir = Path(f'experiments/array_face_s7/learning_repair/s7_selfplay_output_seed{SEED}')

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
    potential_coef=0.05, gamma=0.99, device=device, seed=SEED,
)
physics = default_debug_physics_config(P_jam_W=0.1)
jammer_specs = [HeadSpec("cell", "bernoulli", N_CELLS_S7, bernoulli_logit_bias=-3.0),
                HeadSpec("beam", "categorical", N_BEAM_DIRS_S7)]
radar_specs = [HeadSpec("beam", "categorical", N_BEAM_DIRS_S7),
               HeadSpec("svc", "categorical", 2)]
manifest = Path('experiments/array_face_s1/manifests/ppo_train.json')
with open(manifest) as f:
    train_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]
trainer = S7SelfPlayTrainer(
    cfg=cfg, env_cfg=env_cfg, physics=physics,
    radar=UPAConfig(), jammer=UPAConfig(),
    train_seeds=train_seeds, manifest_path=manifest, out_dir=out_dir,
    jammer_specs=jammer_specs, radar_specs=radar_specs,
)
it = trainer.load_selfplay(out_dir / "selfplay_latest.pt")
print(f"checkpoint loaded (stored session iter={it})")

with open('experiments/array_face_s1/manifests/checkpoint_validation.json') as f:
    val_seeds = [int(e["seed"]) for e in json.load(f)["entries"]][:16]

E, K, R = env_cfg.n_envs, N_JAMMERS, N_RADARS
jam_ncells = [[] for _ in range(K)]
jam_duty = [[] for _ in range(K)]
jam_beam_hist = [torch.zeros(25) for _ in range(K)]
rad_beam_hist = torch.zeros(25); rad_svc_hist = torch.zeros(2)
assign = torch.zeros(K, R)  # mean per-pair JNR contribution (dB, active only)
assign_cnt = torch.zeros(K, R)

for s in val_seeds:
    env = ArrayFaceS7VecEnv(env_cfg, physics=physics, radar=UPAConfig(), jammer=UPAConfig())
    env.reset(seed=s)
    gen = torch.Generator(device=device).manual_seed(4242 + s)
    for t in range(env_cfg.horizon):
        obs_j, obs_r = env._build_observation()
        mask_cell, mask_beam = env._compute_masks()
        cells, beams = [], []
        for k in range(K):
            if MODE == 'stochastic':
                with torch.no_grad():
                    a_j, _ = sample_multihead(
                        trainer.jam_actor, obs_j[:, k],
                        {"cell": mask_cell[:, k], "beam": mask_beam[:, k]}, gen)
                cell = a_j["cell"].float()
                beam_k = a_j["beam"]
            else:
                with torch.no_grad():
                    lj = trainer.jam_actor.forward(obs_j[:, k])
                cell = (lj["cell"] > 0).float()
                beam_k = lj["beam"].argmax(-1)
            jam_ncells[k].append(cell.sum(-1).float().mean().item())
            jam_duty[k].append((cell.sum(-1) > 0).float().mean().item())
            jam_beam_hist[k][beam_k.cpu()] += 1
            cells.append(cell); beams.append(beam_k)
        for r in range(R):
            with torch.no_grad():
                lr = trainer.rad_actor.forward(obs_r[:, r])
            rad_beam_hist[lr["beam"].argmax(-1).cpu()] += 1
            rad_svc_hist[lr["svc"].argmax(-1).cpu()] += 1
        rb_ = torch.zeros(E, R, dtype=torch.int64, device=device)
        rs_ = torch.zeros(E, R, dtype=torch.int64, device=device)
        for r in range(R):
            with torch.no_grad():
                lr = trainer.rad_actor.forward(obs_r[:, r])
            rb_[:, r] = lr["beam"].argmax(-1)
            rs_[:, r] = lr["svc"].argmax(-1)
        jcell = torch.stack(cells, dim=1)
        jbeam = torch.stack(beams, dim=1)
        (o1, o2), (rj, rr), done, info = env.step(jcell, jbeam, rb_, rs_)
        jp = info.get("jnr_per")  # [E, K, R]
        if jp is not None:
            fin = torch.isfinite(jp)
            assign += jp.where(fin, torch.zeros_like(jp)).sum(0)
            assign_cnt += fin.sum(0).float()

import statistics as st
print(f"\n=== {MODE} behavior (16 val seeds x 64 steps) ===")
for k in range(K):
    print(f"jammer {k}: mean_cells={st.mean(jam_ncells[k]):.2f}  duty={st.mean(jam_duty[k]):.3f}")
    jb = (jam_beam_hist[k] / jam_beam_hist[k].sum()).tolist()
    top = sorted(range(25), key=lambda i: -jb[i])[:4]
    print(f"  top beams (idx:share): {[(i, round(jb[i], 3)) for i in top]}")
    az_marg = {a: round(sum(jb[a + 5*e] for e in range(5)), 3) for a in range(5)}
    print(f"  az marg: {az_marg}")
rb = (rad_beam_hist / rad_beam_hist.sum()).tolist()
top = sorted(range(25), key=lambda i: -rb[i])[:4]
print(f"radars top beams (idx:share): {[(i, round(rb[i], 3)) for i in top]}")
az_marg = {a: round(sum(rb[a + 5*e] for e in range(5)), 3) for a in range(5)}
el_marg = {e: round(sum(rb[a + 5*e] for a in range(5)), 3) for e in range(5)}
print(f"radars az marg: {az_marg}")
print(f"radars el marg: {el_marg}")
print(f"radar svc mix: svc0={rad_svc_hist[0].item()/rad_svc_hist.sum().item():.3f}")
print("\ncross-assignment matrix (mean per-pair JNR contribution dB, active only):")
for k in range(K):
    row = []
    for r in range(R):
        c = assign_cnt[k, r].item()
        row.append(f"{assign[k, r].item()/max(c, 1e-6):.1f}dB" if c > 0 else "idle")
    print(f"  jammer {k}: {row}")
