"""Smoke test: train 2 iterations of the n=3 jammer game end-to-end."""
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
import torch
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s7 import EnvConfig, UPAConfig, N_CELLS_S7, N_BEAM_DIRS_S7
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s7.learning_repair.trainer_s7 import S7SelfPlayTrainer

with open('experiments/array_face_s1/manifests/ppo_train.json') as f:
    train_seeds = [int(e["seed"]) for e in json.load(f)["entries"]]

cfg = S2PPOConfigV2(
    profile="array_face_s7_v1", iterations=4, n_envs=16, horizon=64,
    actor_lr=3e-5, critic_lr=1e-3, target_kl=0.02, per_head_entropy=True,
    entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3, "svc": 1e-2},
    entropy_anneal_frac_per_head={"cell": 0.7, "beam": 0.9, "svc": 0.5},
    use_privileged_critic=True, privileged_value_coef=0.5, distill_coef=0.1,
    seed=997, train_seed=997, device="cuda",
)
env_cfg = EnvConfig(n_envs=16, horizon=64, n_services=2, dt=1.0,
                    active_budget_steps=63, duty_budget=1.0,
                    arrival_rate_per_service=0.15, mission_tau_window=6,
                    detects_required=1, potential_coef=0.05, gamma=0.99,
                    n_jammers=3, device="cuda", seed=997)
trainer = S7SelfPlayTrainer(
    cfg=cfg, env_cfg=env_cfg, physics=default_debug_physics_config(P_jam_W=0.1),
    radar=UPAConfig(), jammer=UPAConfig(), train_seeds=train_seeds,
    manifest_path=Path('experiments/array_face_s1/manifests/ppo_train.json'),
    out_dir=Path('experiments/array_face_s7/learning_repair/_smoke_n3'),
    jammer_specs=[HeadSpec("cell", "bernoulli", N_CELLS_S7, bernoulli_logit_bias=-3.0),
                  HeadSpec("beam", "categorical", N_BEAM_DIRS_S7)],
    radar_specs=[HeadSpec("beam", "categorical", N_BEAM_DIRS_S7),
                 HeadSpec("svc", "categorical", 2)],
)
print(f"obs dims: jam={trainer._obs_dim_jam} rad={trainer._obs_dim_rad} "
      f"priv_jam={trainer._priv_dim_jam} priv_rad={trainer._priv_dim_rad}")
for _ in range(2):
    m = trainer.train_iteration()
    print({k: round(v, 4) if isinstance(v, float) else v
           for k, v in m.items() if k in
           ('iteration', 'rollout_drop', 'rollout_success', 'jammer_entropy',
            'radar_entropy')})
ckpt = trainer.out_dir / 'selfplay_latest.pt'
trainer.save_selfplay(ckpt)
it = trainer.load_selfplay(ckpt)
print(f'checkpoint round-trip ok, iter={it}')
