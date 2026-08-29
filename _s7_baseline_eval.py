"""Evaluation-only scripted baselines against converged S7 checkpoints.

Adds four baseline views that need no training, using the exact final-eval
protocol (64 validation scenario seeds x 3 action seeds, single lane):

  random_radar_vs_jam — uniform-random legal radar beam/service vs trained jammers
  greedy_radar_vs_jam — both radar heads chase the hottest pending (svc, az)
                        read from the radar observation (no privileged state)
  random_jam_vs_rad   — uniform-random legal cell/beam jammers vs trained radars
  stare_jam_vs_rad    — each jammer stares at its same-sign radar site
                        (beam computed from pair bearings), one cell, until
                        the energy budget is spent

Trained-vs-trained (h2h), sweep-radar (jam_vs_sweep) and idle-jammer
(rad_vs_idle) references come from the existing final_eval.json.

Usage: python _s7_baseline_eval.py [--device cuda]
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
from env.gpu.array_face_s7.array_factor import BEAM_AZ_DEG_S7
from env.gpu.array_face_s7.geometry import pair_bearings_for, JAMMER_AZ_DEG, RADAR_AZ_DEG
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s7.learning_repair.trainer_s7 import S7SelfPlayTrainer
from experiments.array_face_s7.learning_repair.trainer_s7 import sample_multihead

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda")
args = p.parse_args()
device = args.device

CHECKPOINTS = [
    's7_continue2_output_seed20260801',
    's7_seed02_cont_output_seed20260802',
    's7_seed03_cont_output_seed20260803',
]
SEED_BY_NAME = {'s7_continue2_output_seed20260801': 20260801,
                's7_seed02_cont_output_seed20260802': 20260802,
                's7_seed03_cont_output_seed20260803': 20260803}

# Stare beams: for each jammer k, the horizon-plane beam closest to the
# bearing of radar k (same-sign pairing) from that jammer site.
_az_rad, _el_rad = pair_bearings_for(JAMMER_AZ_DEG, RADAR_AZ_DEG, torch.device('cpu'))
import math
STARE_BEAM = []
for k in range(N_JAMMERS):
    bearing_deg = math.degrees(float(_az_rad[k][k]))
    az_idx = min(range(len(BEAM_AZ_DEG_S7)),
                 key=lambda i: abs(BEAM_AZ_DEG_S7[i] - bearing_deg))
    STARE_BEAM.append(az_idx + 5 * 2)  # + el row 2 (0-deg elevation plane)
print(f"stare beams per jammer: {STARE_BEAM} (bearings "
      f"{[round(math.degrees(float(_az_rad[k][k])), 1) for k in range(N_JAMMERS)]} deg)",
      flush=True)


def build_trainer(out_dir: Path, seed: int) -> S7SelfPlayTrainer:
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


def run_view(trainer, env_cfg, physics, val_seeds, mode, action_seed):
    """One baseline view: mode picks which side is scripted."""
    jam_actor, rad_actor = trainer.jam_actor, trainer.rad_actor
    eval_cfg = replace(env_cfg, n_envs=1)
    E, K, R = 1, N_JAMMERS, N_RADARS
    drops = []
    for sd in val_seeds:
        env = ArrayFaceS7VecEnv(eval_cfg, physics=physics,
                                radar=UPAConfig(), jammer=UPAConfig())
        env.reset(seed=sd)
        gen = torch.Generator(device=device).manual_seed(action_seed + sd)
        for t in range(env_cfg.horizon):
            obs_j, obs_r = env._build_observation()
            mask_cell, mask_beam = env._compute_masks()
            # ---- jammer side ----
            if mode in ("random_radar_vs_jam",):
                cells, beams = [], []
                for k in range(K):
                    with torch.no_grad():
                        a_j, _ = sample_multihead(
                            jam_actor, obs_j[:, k],
                            {"cell": mask_cell[:, k], "beam": mask_beam[:, k]}, gen)
                    cells.append(a_j["cell"]); beams.append(a_j["beam"])
                jcell = torch.stack(cells, dim=1)
                jbeam = torch.stack(beams, dim=1)
            elif mode == "random_jam_vs_rad":
                jcell = torch.zeros(E, K, 25, device=device)
                jbeam = torch.zeros(E, K, dtype=torch.int64, device=device)
                for k in range(K):
                    legal = mask_beam[0, k].nonzero(as_tuple=True)[0]
                    legal_cells = mask_cell[0, k].nonzero(as_tuple=True)[0]
                    if len(legal_cells):
                        pick = legal_cells[torch.randint(len(legal_cells), (1,),
                                                         generator=gen,
                                                         device=legal_cells.device)]
                        jcell[0, k, int(pick)] = 1.0
                    jbeam[0, k] = legal[torch.randint(len(legal), (1,), generator=gen,
                                                      device=legal.device)]
            elif mode == "stare_jam_vs_rad":
                jcell = torch.zeros(E, K, 25, device=device)
                jbeam = torch.zeros(E, K, dtype=torch.int64, device=device)
                for k in range(K):
                    if mask_cell[0, k].any():
                        jcell[0, k, 0] = 1.0
                        jbeam[0, k] = STARE_BEAM[k]
            else:  # idle jammers (radar-baseline reference)
                jcell = torch.zeros(E, K, 25, device=device)
                jbeam = torch.zeros(E, K, dtype=torch.int64, device=device)
            # ---- radar side ----
            rb_ = torch.zeros(E, R, dtype=torch.int64, device=device)
            rs_ = torch.zeros(E, R, dtype=torch.int64, device=device)
            if mode in ("random_jam_vs_rad", "stare_jam_vs_rad", "idle_ref"):
                for r in range(R):
                    masks = {"beam": env._radar_mask_beam,
                             "svc": env._radar_mask_svc}
                    with torch.no_grad():
                        a_r, _ = sample_multihead(rad_actor, obs_r[:, r], masks, gen)
                    rb_[:, r] = a_r["beam"]; rs_[:, r] = a_r["svc"]
            elif mode == "random_radar_vs_jam":
                lb = env._radar_mask_beam[0].nonzero(as_tuple=True)[0]
                ls = env._radar_mask_svc[0].nonzero(as_tuple=True)[0]
                for r in range(R):
                    rb_[0, r] = lb[torch.randint(len(lb), (1,), generator=gen,
                                                 device=lb.device)]
                    rs_[0, r] = ls[torch.randint(len(ls), (1,), generator=gen,
                                                 device=ls.device)]
            else:  # greedy_radar_vs_jam: chase hottest pending (svc, az) from obs
                for r in range(R):
                    pm = obs_r[0, r, 1:11].reshape(2, 5)  # [svc, az] pending map
                    svc_idx = int(pm.sum(dim=1).argmax())
                    az_idx = int(pm[svc_idx].argmax())
                    rb_[0, r] = az_idx + 10  # horizon-plane beam at that azimuth
                    rs_[0, r] = svc_idx
            env.step(jcell, jbeam, rb_, rs_)
        drops.append(float(env.drop_ratio()[0]))
    return sum(drops) / len(drops)


def main():
    with open(Path('experiments/array_face_s1/manifests/checkpoint_validation.json')) as f:
        val_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]
    physics = default_debug_physics_config(P_jam_W=0.1)
    modes = ["random_radar_vs_jam", "greedy_radar_vs_jam",
             "random_jam_vs_rad", "stare_jam_vs_rad"]
    for name in CHECKPOINTS:
        out_dir = Path('experiments/array_face_s7/learning_repair') / name
        trainer, env_cfg = build_trainer(out_dir, SEED_BY_NAME[name])
        results = {}
        for aseed in (4242, 777, 31337):
            t0 = time.time()
            results[f"aseed_{aseed}"] = {
                m: run_view(trainer, env_cfg, physics, val_seeds, m, aseed)
                for m in modes
            }
            results[f"aseed_{aseed}"]["elapsed_s"] = round(time.time() - t0, 1)
            print(f"{name} aseed={aseed}: "
                  f"{ {m: round(v, 4) for m, v in results[f'aseed_{aseed}'].items() if m != 'elapsed_s'} }",
                  flush=True)
        out_path = out_dir / "baseline_eval.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
