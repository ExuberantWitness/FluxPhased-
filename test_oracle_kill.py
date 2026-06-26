"""Ground-truth oracle kill test (bypasses PPO).

Purpose: determine if the laser kill mechanics in vec_battlefield.step_lasers
actually work when given a perfect aim + sustained fire. If kills don't happen
here, the environment is broken and no policy can succeed.

Three scenarios (each runs E envs × N pulses):
  A.OracleAlwaysFire:   aim = true enemy pos, fire_on = True every pulse
  B.OracleBernoulli:    aim = true enemy pos, fire_on ~ Bernoulli(0.5)
  C.OracleBias270m:     aim = enemy pos + 270m offset (mimics Kalman bias), fire_on = True

Expected outcomes if env is correct:
  A → kill_rate ~1.0 by pulse 20-30 (illumination_time accumulates fast)
  B → kill_rate ~0.3-0.5 (needs sustained fire; 0.5^20 ≈ 1e-6 per opportunity but
      many opportunities across 12 envs × 500 pulses)
  C → kill_rate ~0.0 (270m > 50m kill_radius → never in_range)

If A fails, env has a real bug in step_lasers.
If A passes but C passes too, kill_radius check is broken.
If A passes and C fails, kill logic is correct → problem is sensing/policy.
"""
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from radar_sim.gpu.vec_mfar_env import MFARVecEnv
from training.laser.episode import LaserEpisodeRunner


class ScriptedOracle:
    """Trainer-like object that returns oracle commander actions.

    Three modes:
      - "always_fire_truth": aim at true nearest enemy, fire=True
      - "bernoulli_truth":   aim at true nearest enemy, fire ~ Bernoulli(0.5)
      - "always_fire_bias":  aim at true enemy + fixed 270m x-offset, fire=True
    """

    def __init__(self, mode: str, bias_m: float = 270.0, seed: int = 42):
        self.mode = mode
        self.bias_m = bias_m
        self.min_radar_baseline_m = 0.0  # skip baseline enforcement
        self.reward_shaper = None
        self.kalman_tracker = None
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def get_own_actions(self, env, team: int, deterministic: bool = False,
                        spectrum=None, events=None):
        E = env.num_envs
        dev = torch.device(env.device)
        n_teams = env.n_teams
        r_per_team = env.n_radars // n_teams
        r_start = team * r_per_team
        r_end = r_start + r_per_team

        enemy_team = 1 - team
        enemy_idx = env.battlefield.team_radar_indices[enemy_team]
        enemy_pos = env.radar_pos[:, enemy_idx, :]  # [E, R/2, 3]
        enemy_alive = env.battlefield.alive[:, enemy_idx]  # [E, R/2]
        # Mask dead enemies with huge position so they don't get picked
        enemy_pos_masked = enemy_pos + (~enemy_alive).unsqueeze(-1).float() * 1e6
        # Nearest alive enemy per env
        # Use drone position as reference to find "nearest" — but oracle cheats:
        # picks the enemy that's nearest to DRONE (which is at center, fixed).
        # Simpler: just pick enemy 0 of the opposing team (oracle doesn't need
        # to be smart, just needs to HIT something alive).
        target = enemy_pos_masked[:, 0, :]  # [E, 3]
        # If enemy 0 dead, fall back to enemy 1
        any_alive = enemy_alive.any(dim=-1)  # [E]
        target = torch.where(
            enemy_alive[:, 0].unsqueeze(-1),
            enemy_pos[:, 0, :],
            enemy_pos[:, 1, :] if enemy_pos.shape[1] > 1 else enemy_pos[:, 0, :],
        )

        # Apply mode-specific offset
        if self.mode == "always_fire_bias":
            target = target.clone()
            target[:, 0] += self.bias_m  # x-axis offset

        half_x = float(env.map_size[0]) / 2.0
        half_y = float(env.map_size[1]) / 2.0

        # Build commander_action[E, 5]
        cmd = torch.zeros(E, 5, device=dev)
        # Fire bit: +1 = fire (env decode: >0.5)
        if self.mode == "bernoulli_truth":
            fire_draws = torch.rand(E, generator=self.rng, device="cpu").to(dev)
            cmd[:, 0] = torch.where(fire_draws < 0.5, 1.0, -1.0)
        else:
            cmd[:, 0] = 1.0  # always fire
        # Aim normalized [-1, 1]
        cmd[:, 1] = (target[:, 0] / half_x).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        cmd[:, 2] = (target[:, 1] / half_y).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        cmd[:, 3] = 0.0  # z=0 (ground radars)
        cmd[:, 4] = 0.0  # reserved

        # radar_actions: list of [E, action_dim] per own radar (zeros are fine,
        # they only affect TX waveform, not the kill check)
        action_dim = env.action_dim
        radar_actions = [torch.zeros(E, action_dim, device=dev) for _ in range(r_per_team)]

        return {
            "radar_actions": radar_actions,
            "commander_action": cmd,
            "transition": {},
            "r_start": r_start,
            "r_end": r_end,
        }


def run_scenario(mode: str, n_episodes: int = 5, max_steps: int = 500):
    """Run one scenario for n_episodes × max_steps control steps.

    Returns: dict with kill_count, total_episodes, dist_min_avg, illum_max_avg
    """
    cfg_path = "configs/ablation_f1f8/v3_scaling.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg.get("env", {})
    # Override num_envs to 4 (small) to coexist with v5c training on the same GPU.
    # Env mechanics are identical; just fewer parallel envs.
    env = MFARVecEnv(
        num_envs=4,
        n_radars=env_cfg.get("n_radars", 4),
        rows=env_cfg.get("rows", 25),
        cols=env_cfg.get("cols", 25),
        fc=10e9,
        bandwidth=env_cfg.get("bandwidth", 200e6),
        prf=env_cfg.get("prf", 10e3),
        pulses_per_cpi=env_cfg.get("pulses_per_cpi", 4),
        fft_size=env_cfg.get("fft_size", 64),
        tx_power_w=env_cfg.get("tx_power_w", 1.0),
        n_teams=env_cfg.get("n_teams", 2),
        device=env_cfg.get("device", "cuda"),
        kill_radius_m=env_cfg.get("kill_radius_m", 0.2),
        illumination_time_s=env_cfg.get("illumination_time_s", 0.002),
        drone_altitude_m=env_cfg.get("drone_altitude_m", 3000.0),
        map_size=env_cfg.get("map_size", [20000.0, 20000.0]),
        vehicle_speed_ms=env_cfg.get("vehicle_speed_ms", 20.0),
        reward_config=cfg.get("reward_shaping", {}),
    )

    oracle = ScriptedOracle(mode=mode, bias_m=270.0)
    runner = LaserEpisodeRunner(env, pulses_per_control=cfg.get("training", {}).get("pulses_per_control", 5), device=env.device)

    total_kills = 0
    total_timeouts = 0
    dist_min_samples = []
    illum_max_samples = []
    kill_step_samples = []

    for ep in range(n_episodes):
        runner.reset(red_trainer=oracle, blue_trainer=oracle)
        ep_kills = 0
        last_step = 0
        for step in range(max_steps):
            with torch.no_grad():
                step_out = runner.step_control(oracle, oracle, deterministic=False)
            result = step_out["result"]
            if result is None:
                break
            last_step = step
            if result["dones"].any():
                # Count kills across all done envs (regardless of "winner")
                kills = result.get("kills")
                if kills is not None:
                    ep_kills += int(kkills_count(kills))
                else:
                    # Fall back: a "done" with a winner != -1 is a kill event
                    ep_kills += int((result["dones"] & (result["winners"] >= 0)).sum().item())
                break

        # Sample diagnostics from end of episode
        bf = env.battlefield
        enemy_idx_0 = bf.team_radar_indices[1]
        # Dist from team 0's commander_aim to nearest enemy
        if hasattr(bf.drone, "_commander_aim"):
            aim_t0 = bf.drone._commander_aim[:, 0, :]  # [E, 3]
            enemy_pos = env.radar_pos[:, enemy_idx_0, :]  # [E, R/2, 3]
            dist = (aim_t0.unsqueeze(1) - enemy_pos).norm(dim=-1).min(dim=-1).values
            dist_min_samples.append(float(dist.mean().item()))
        illum_max_samples.append(float(bf.laser.illumination_time[:, 0].max().item()))

        if ep_kills > 0:
            total_kills += ep_kills
            kill_step_samples.append(last_step)
        else:
            total_timeouts += 1
        print(f"  [{mode}] ep {ep+1}/{n_episodes}: last_step={last_step+1}, "
              f"this_ep_kills={ep_kills}, dist_end={dist_min_samples[-1]:.2f}m, "
              f"illum_max={illum_max_samples[-1]*1000:.2f}ms", flush=True)

    return {
        "mode": mode,
        "n_episodes": n_episodes,
        "total_kills": total_kills,
        "total_timeouts": total_timeouts,
        "dist_end_avg_m": float(np.mean(dist_min_samples)) if dist_min_samples else float("nan"),
        "illum_max_avg_ms": float(np.mean(illum_max_samples) * 1000) if illum_max_samples else float("nan"),
        "kill_step_avg": float(np.mean(kill_step_samples)) if kill_step_samples else float("nan"),
    }


def kkills_count(kills_tensor):
    """Count total kills across all envs/teams from a kill tensor.

    kills shape: [E, n_teams, n_enemy] or similar. Count all True values.
    """
    if isinstance(kills_tensor, torch.Tensor):
        return int(kills_tensor.sum().item())
    return 0


def main():
    print("=" * 70)
    print("ORACLE KILL TEST — bypasses PPO to verify env kill mechanics")
    print("=" * 70)
    print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
    print()

    results = []
    for mode in ["always_fire_truth", "bernoulli_truth", "always_fire_bias"]:
        print(f"\n>>> Scenario: {mode}")
        r = run_scenario(mode=mode, n_episodes=5, max_steps=300)
        results.append(r)
        print(f"  → kills={r['total_kills']}, timeouts={r['total_timeouts']}, "
              f"dist_end_avg={r['dist_end_avg_m']:.2f}m, "
              f"illum_max_avg={r['illum_max_avg_ms']:.2f}ms")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    a = next(r for r in results if r["mode"] == "always_fire_truth")
    b = next(r for r in results if r["mode"] == "bernoulli_truth")
    c = next(r for r in results if r["mode"] == "always_fire_bias")

    print(f"A.OracleAlwaysFire:    kills={a['total_kills']}/{a['n_episodes']}eps  "
          f"illum_max={a['illum_max_avg_ms']:.2f}ms")
    print(f"B.OracleBernoulli:     kills={b['total_kills']}/{b['n_episodes']}eps  "
          f"illum_max={b['illum_max_avg_ms']:.2f}ms")
    print(f"C.OracleBias270m:      kills={c['total_kills']}/{c['n_episodes']}eps  "
          f"illum_max={c['illum_max_avg_ms']:.2f}ms")
    print()
    if a["total_kills"] == 0:
        print(">>> ENV BUG: oracle with perfect aim + always-fire cannot kill.")
        print(">>> step_lasers kill logic is broken. PPO cannot fix this.")
    elif c["total_kills"] > 0:
        print(">>> ENV BUG: oracle with 270m offset still kills — kill_radius broken.")
    else:
        print(">>> ENV OK: oracle can kill. Problem is sensing/policy, not kill mechanics.")
        if b["total_kills"] == 0:
            print(">>> NOTE: Bernoulli(0.5) fire gives 0 kills → fire_commitment signal")
            print(">>>       is critical for any real policy to succeed.")


if __name__ == "__main__":
    main()
