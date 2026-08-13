"""Non-learning baselines for S2 and S3: random, greedy, always-idle.

Evaluates three fixed policies on both S2 and S3 environments using the same
64 validation seeds (checkpoint_validation manifest) as the trained PPO actors.
No training — just rollout + drop_ratio.

Policies:
  random:      uniform over legal actions (uses env.sample_action_rng)
  greedy:      always jam_svc_0, broadside beam (idx 2), all cells on (S3)
               respects energy mask — idles when tokens exhausted
  always_idle: base=ACTION_IDLE (lower bound on performance)

Usage: python run_baselines.py
Output: baseline_results.json + console table
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.g3_bsta_lite.action_contract import ACTION_IDLE
from env.gpu.array_face_s2 import (
    EnvConfig as EnvConfigS2, ArrayFaceS2VecEnv,
    RadarULAConfig, JammerULAConfig,
)
from env.gpu.array_face_s3 import (
    EnvConfig as EnvConfigS3, ArrayFaceS3VecEnv,
    N_CELLS,
)

MANIFEST_DIR = HERE.parents[1] / "array_face_s1" / "manifests"
N_ACTION_REPS = 4
ACTION_SEED = 4242  # same as PPO eval


def load_seeds(name: str) -> list[int]:
    with open(MANIFEST_DIR / f"{name}.json") as f:
        m = json.load(f)
    return [int(e["seed"]) for e in m["entries"]]


# ---------------------------------------------------------------------------
# Policy helpers — return action tensors given the env's current mask
# ---------------------------------------------------------------------------

def _greedy_base(E, mask_base, device):
    """Jam svc_0 (action=1) if legal, else idle."""
    base = torch.ones(E, dtype=torch.int64, device=device)  # jam_svc_0
    can_jam = mask_base[:, 1].to(torch.bool)
    base = torch.where(can_jam, base, torch.zeros_like(base))
    return base


def _greedy_beam(E, device):
    """Broadside = beam index 2."""
    return torch.full((E,), 2, dtype=torch.int64, device=device)


def _all_cells_on(E, device):
    return torch.ones((E, N_CELLS), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# S2 evaluation
# ---------------------------------------------------------------------------

def eval_s2(policy: str, env_cfg, physics, radar, jammer, seeds, device):
    gen = torch.Generator(device=device).manual_seed(ACTION_SEED)
    per_seed_drops = []
    for sd in seeds:
        rep_drops = []
        for rep in range(N_ACTION_REPS):
            env = ArrayFaceS2VecEnv(env_cfg, physics=physics, radar=radar, jammer=jammer)
            env.reset(seed=sd)
            for t in range(env_cfg.horizon):
                mask_base, mask_beam = env._compute_mask()
                E = env.E
                if policy == "random":
                    base, beam = env.sample_action_rng()
                    # respect mask: if illegal, fall back to idle
                    legal = mask_base.gather(1, base.unsqueeze(1)).squeeze(1)
                    base = torch.where(legal, base, torch.zeros_like(base))
                elif policy == "greedy":
                    base = _greedy_base(E, mask_base, device)
                    beam = _greedy_beam(E, device)
                elif policy == "always_idle":
                    base = torch.zeros(E, dtype=torch.int64, device=device)
                    beam = torch.zeros(E, dtype=torch.int64, device=device)
                env.step(base, beam)
            rep_drops.append(float(env.drop_ratio()[0]))
        per_seed_drops.append(sum(rep_drops) / len(rep_drops))
    macro = sum(per_seed_drops) / len(per_seed_drops)
    return {"macro_mean_drop": macro, "per_seed_drops": per_seed_drops}


# ---------------------------------------------------------------------------
# S3 evaluation
# ---------------------------------------------------------------------------

def eval_s3(policy: str, env_cfg, physics, radar, jammer, seeds, device):
    gen = torch.Generator(device=device).manual_seed(ACTION_SEED)
    per_seed_drops = []
    for sd in seeds:
        rep_drops = []
        for rep in range(N_ACTION_REPS):
            env = ArrayFaceS3VecEnv(env_cfg, physics=physics, radar=radar, jammer=jammer)
            env.reset(seed=sd)
            for t in range(env_cfg.horizon):
                mask_base, mask_beam, mask_cell = env._compute_mask()
                E = env.E
                if policy == "random":
                    base, beam, cell = env.sample_action_rng()
                    legal = mask_base.gather(1, base.unsqueeze(1)).squeeze(1)
                    base = torch.where(legal, base, torch.zeros_like(base))
                elif policy == "greedy":
                    base = _greedy_base(E, mask_base, device)
                    beam = _greedy_beam(E, device)
                    cell = _all_cells_on(E, device)
                elif policy == "always_idle":
                    base = torch.zeros(E, dtype=torch.int64, device=device)
                    beam = torch.zeros(E, dtype=torch.int64, device=device)
                    cell = torch.zeros((E, N_CELLS), dtype=torch.float32, device=device)
                env.step(base, beam, cell)
            rep_drops.append(float(env.drop_ratio()[0]))
        per_seed_drops.append(sum(rep_drops) / len(rep_drops))
    macro = sum(per_seed_drops) / len(per_seed_drops)
    return {"macro_mean_drop": macro, "per_seed_drops": per_seed_drops}


def main():
    device = "cpu"  # baselines are tiny; CPU is fine and avoids GPU contention
    seeds = load_seeds("checkpoint_validation")
    print(f"Baseline evaluation  seeds={len(seeds)}  reps={N_ACTION_REPS}")
    print(f"{'Policy':<15} {'S2@16':>12} {'S3@63':>12}")
    print("-" * 42)

    physics = default_debug_physics_config(P_jam_W=2.0)
    radar = RadarULAConfig()
    jammer = JammerULAConfig()

    # S2 env config (matches PPO training: active_budget_steps=16, duty=0.25)
    env_cfg_s2 = EnvConfigS2(
        n_envs=16, horizon=64, n_services=2, dt=1.0, P_jam_W=2.0,
        active_budget_steps=16, duty_budget=0.25,
        arrival_rate_per_service=0.15, baseline_snr_db=22.0,
        mission_tau_window=6, detects_required=1, device=device, seed=0,
    )
    # S3 env config (matches PPO training: active_budget_steps=63, duty=1.0)
    env_cfg_s3 = EnvConfigS3(
        n_envs=16, horizon=64, n_services=2, dt=1.0, P_jam_W=2.0,
        active_budget_steps=63, duty_budget=1.0,
        arrival_rate_per_service=0.15, baseline_snr_db=22.0,
        mission_tau_window=6, detects_required=1, device=device, seed=0,
    )

    results = {}
    for policy in ["always_idle", "random", "greedy"]:
        r_s2 = eval_s2(policy, env_cfg_s2, physics, radar, jammer, seeds, device)
        r_s3 = eval_s3(policy, env_cfg_s3, physics, radar, jammer, seeds, device)
        results[policy] = {"s2_16": r_s2, "s3_63": r_s3}
        print(f"{policy:<15} {r_s2['macro_mean_drop']:>12.4f} {r_s3['macro_mean_drop']:>12.4f}")

    out_path = HERE / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")

    # Reference: trained PPO results for context
    print("\n--- Reference (trained PPO) ---")
    print(f"{'PPO S2@16':<15} {'0.2114':>12} {'':>12}  (3-seed mean)")
    print(f"{'PPO S3@63':<15} {'':>12} {'0.4230':>12}  (1-seed, multi-seed pending)")


if __name__ == "__main__":
    main()
