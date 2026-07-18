"""决定性对照 (用户 2026-07-18 提出):
BC vs BC — 胜任经典能不能 track+kill 一个会规避的目标?

如果 BC vs BC 也 ~0 init/kill → 规避压倒跟踪 = 对称 floor (不是 RL 输给 BC)
如果 BC vs BC 有 kill → 规避目标可跟 → RL 只是缺主动感知策略 → 不是 floor
"""
import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")
import torch
import numpy as np
from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


def configure_channels(env, mode: str = "orthogonal"):
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    freqs = torch.zeros(E, T, R, device=env.device)
    fc = env.fc_hz
    cs = env.channel_spacing_hz
    for e in range(E):
        if mode == "orthogonal":
            freqs[e, 0, 0] = fc
            freqs[e, 0, 1] = fc + cs
            freqs[e, 1, 0] = fc
            freqs[e, 1, 1] = fc + cs
        else:
            freqs[e, 0, :] = fc
            freqs[e, 1, :] = fc
    env.set_radar_freqs(freqs)


def run_bc_vs_bc(n_episodes: int = 50, horizon: int = 200,
                 channel_mode: str = "orthogonal"):
    """Returns per-team kills + tracker init stats for BC vs BC."""
    env = TwoTeamVecEnv(n_envs=16, device="cuda", episode_steps=horizon,
                        geometry=RANDOM_GEOMETRY, seed=9000)
    bc = BlindClassicalCommander()
    per_ep_t0_kills, per_ep_t1_kills = [], []
    per_ep_t0_init, per_ep_t1_init = [], []
    per_ep_t0_dwell_max, per_ep_t1_dwell_max = [], []

    for ep in range(n_episodes):
        env.seed = 9000 + ep
        env._reset_count = ep
        env.reset()
        configure_channels(env, channel_mode)
        for step in range(horizon):
            a0 = bc.get_action(env, 0)
            a1 = bc.get_action(env, 1)
            action = combine_team_actions(env, a0, a1)
            obs, reward, done, info = env.step(action)
        # End of episode stats
        for e in range(16):
            per_ep_t0_kills.append(int(info["team_kills"][e, 0].item()))
            per_ep_t1_kills.append(int(info["team_kills"][e, 1].item()))
            per_ep_t0_init.append(int(env.tracker_initialized[e, 0].sum().item()))
            per_ep_t1_init.append(int(env.tracker_initialized[e, 1].sum().item()))
            # max dwell over episode = final radar_E (since it accumulates)
            per_ep_t0_dwell_max.append(float(env.radar_E[e, 1].sum().item()))  # t0 lase t1
            per_ep_t1_dwell_max.append(float(env.radar_E[e, 0].sum().item()))

    return {
        "t0_kills_mean": np.mean(per_ep_t0_kills),
        "t1_kills_mean": np.mean(per_ep_t1_kills),
        "t0_init_mean": np.mean(per_ep_t0_init),
        "t1_init_mean": np.mean(per_ep_t1_init),
        "t0_dwell_max_mean": np.mean(per_ep_t0_dwell_max),
        "t1_dwell_max_mean": np.mean(per_ep_t1_dwell_max),
        "n_episodes": n_episodes * 16,
    }


for mode in ["orthogonal", "same_channel"]:
    print(f"\n=== BC vs BC ({mode}) ===")
    r = run_bc_vs_bc(n_episodes=10, horizon=200, channel_mode=mode)
    print(f"  BC team0 kills: {r['t0_kills_mean']:.3f} (mean over {r['n_episodes']} eps)")
    print(f"  BC team1 kills: {r['t1_kills_mean']:.3f}")
    print(f"  BC team0 tracker_init: {r['t0_init_mean']:.2f}/2 radars (end-of-ep)")
    print(f"  BC team1 tracker_init: {r['t1_init_mean']:.2f}/2 radars")
    print(f"  BC team0 final radar_E[enemy]: {r['t0_dwell_max_mean']:.3f} (e_kill=2.0)")
    print(f"  BC team1 final radar_E[enemy]: {r['t1_dwell_max_mean']:.3f}")
    total_kills = r['t0_kills_mean'] + r['t1_kills_mean']
    print(f"  Σ kills (both teams): {total_kills:.3f}")
