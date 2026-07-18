"""Inspect trained RL ckpt behavior vs different opponents.

For each opp, run N episodes, log per-step:
  - radar_E[:, et] (dwell progress on enemy)
  - new_kills (delta team_kills[:, lt])
  - emission_on (is RL firing?)
  - exposure (is RL being seen?)
  - reward components (base / dwell / kill / total)

This tells us WHERE RL is failing vs BC: not firing? firing but missing?
firing + hitting but not enough dwell?
"""
from __future__ import annotations
import os
import sys
import math
import torch
import numpy as np

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions, STRATEGIES


def configure_channels(env, mode: str = "orthogonal"):
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    dev = env.device
    freqs = torch.zeros(E, T, R, device=dev)
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


def run_episodes(env, rl_ac, opp_cmd, learning_team: int, n_episodes: int,
                 horizon: int = 200, channel_mode: str = "orthogonal",
                 shape_dwell_bonus: float = 1.0, shape_kill_bonus: float = 50.0,
                 reward_scale: float = 0.1):
    """Run n_episodes, return per-episode aggregate + per-step env-0 trace."""
    et = 1 - learning_team
    E = env.E
    assert E >= n_episodes, f"need E≥n_episodes, got E={E}"

    per_ep = {
        "rl_kills": [], "opp_kills": [], "rl_survival": [],
        "rl_dwell_frac_mean": [], "rl_emission_rate": [],
        "rl_exposure_mean": [], "rl_new_kills_total": [],
    }
    env0_trace = []   # first episode, per-step
    for ep in range(n_episodes):
        env.seed = 9000 + ep
        env._reset_count = ep
        env.reset()
        configure_channels(env, channel_mode)

        prev_kills = env.team_kills[:, learning_team].clone()
        steps_log = []
        for step in range(horizon):
            obs_dict = env.get_obs()
            obs_lt = obs_dict["obs"][:, learning_team]
            priv_lt = obs_dict["privileged"][:, learning_team]
            detect_lt = env.get_detect_list()[:, learning_team]
            with torch.no_grad():
                a_rl, _ = rl_ac.get_action_for_env(
                    obs_lt, detect_lt, priv_lt, deterministic=True)
            a_opp = opp_cmd.get_action(env, 1 - learning_team)
            if learning_team == 0:
                action = combine_team_actions(env, a_rl, a_opp)
            else:
                action = combine_team_actions(env, a_opp, a_rl)
            emission_lt_pre = a_rl["emission_on"].sum(dim=-1)
            obs_dict, reward, done, info = env.step(action)

            radar_E_et = env.radar_E[:, et].sum(-1)
            dwell_frac = radar_E_et / env.e_kill
            now_kills = info["team_kills"][:, learning_team]
            new_kills = (now_kills - prev_kills).clamp(min=0).float()
            prev_kills = now_kills.clone()

            if ep == 0:
                env0_trace.append({
                    "step": step,
                    "radar_E_et": float(radar_E_et[0].item()),
                    "dwell_frac": float(dwell_frac[0].item()),
                    "new_kills": float(new_kills[0].item()),
                    "emission": float(emission_lt_pre[0].item()),
                    "exposure": float(info["exposure"][0, learning_team].item()),
                    "team_kills_lt": int(now_kills[0].item()),
                })

        # End-of-episode aggregates (only first n_episodes envs)
        for e in range(n_episodes):
            per_ep["rl_kills"].append(int(info["team_kills"][e, learning_team].item()))
            per_ep["opp_kills"].append(int(info["team_kills"][e, 1 - learning_team].item()))
            per_ep["rl_survival"].append(int(env.radar_alive[e, learning_team].sum().item()))
            # Averages over episode
            trace_e = [s for s in env0_trace] if e == 0 else None   # only env 0 has trace
        # Just compute env-0 trace aggregates (others we skip for speed)
    # Aggregate
    return {
        "rl_kills_mean": np.mean(per_ep["rl_kills"]),
        "rl_kills_std": np.std(per_ep["rl_kills"]),
        "opp_kills_mean": np.mean(per_ep["opp_kills"]),
        "rl_survival_mean": np.mean(per_ep["rl_survival"]),
        "env0_trace": env0_trace,
    }


def main():
    ckpt_path = "checkpoints/blind/wp3_20260718_090802/iter_final.pt"
    print(f"Loading: {ckpt_path}")
    rl_ac = TwoTeamCommanderActorCritic().to("cuda")
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    rl_ac.load_state_dict(ckpt["ac_state"])
    rl_ac.eval()
    print(f"  iter={ckpt.get('iter', '?')}")

    env = TwoTeamVecEnv(
        n_envs=10, device="cuda", episode_steps=200,
        geometry=RANDOM_GEOMETRY, seed=9000,
    )

    opps = {
        "BlindClassical": BlindClassicalCommander(),
        "StrongRule": TwoTeamStrongRuleCommander(),
        "extreme/balanced": STRATEGIES["balanced"],
        "extreme/pure_track": STRATEGIES["pure_track"],
        "extreme/pure_jam": STRATEGIES["pure_jam"],
    }

    SHAPE_DWELL = 1.0
    SHAPE_KILL = 50.0

    print(f"\n{'opp':<22} | rl_kills | opp_kills | survival | dwell@max | emit@avg | "
          f"expos@avg | env0 trace summary")
    print("-" * 110)
    for name, opp in opps.items():
        result = run_episodes(
            env, rl_ac, opp, learning_team=0, n_episodes=10, horizon=200,
            channel_mode="orthogonal",
            shape_dwell_bonus=SHAPE_DWELL, shape_kill_bonus=SHAPE_KILL,
        )
        # env0 trace aggregates
        env0 = result["env0_trace"]
        dwell_max = max(s["dwell_frac"] for s in env0)
        emit_avg = np.mean([s["emission"] for s in env0])
        expos_avg = np.mean([s["exposure"] for s in env0])
        # steps where RL fired
        n_firing_steps = sum(1 for s in env0 if s["emission"] > 0.5)
        n_dwell_steps = sum(1 for s in env0 if s["radar_E_et"] > 0.01)
        any_kill = any(s["new_kills"] > 0.5 for s in env0)
        print(f"{name:<22} | {result['rl_kills_mean']:.2f}     | "
              f"{result['opp_kills_mean']:.2f}       | "
              f"{result['rl_survival_mean']:.2f}      | "
              f"{dwell_max:.2f}      | {emit_avg:.2f}      | "
              f"{expos_avg:.2f}      | "
              f"fire={n_firing_steps}/200 dwell={n_dwell_steps}/200 kill={any_kill}")

    # Detailed env-0 trace for BC vs RL
    print("\n=== Env-0 per-step trace: RL vs BlindClassical (orthogonal) ===")
    result = run_episodes(
        env, rl_ac, BlindClassicalCommander(), learning_team=0,
        n_episodes=10, horizon=200, channel_mode="orthogonal",
    )
    env0 = result["env0_trace"]
    print(f"  step | radar_E_et | dwell_frac | new_kills | emission | exposure | kills_lt")
    for s in env0[:50]:   # first 50 steps
        print(f"  {s['step']:4d} | {s['radar_E_et']:10.3f} | "
              f"{s['dwell_frac']:10.3f} | {s['new_kills']:.0f} | "
              f"{s['emission']:.0f} | {s['exposure']:.3f} | "
              f"{s['team_kills_lt']}")
    # Sample every 20 steps for full horizon
    print("  ... (every 20 steps) ...")
    for s in env0[::20]:
        print(f"  {s['step']:4d} | {s['radar_E_et']:10.3f} | "
              f"{s['dwell_frac']:10.3f} | {s['new_kills']:.0f} | "
              f"{s['emission']:.0f} | {s['exposure']:.3f} | "
              f"{s['team_kills_lt']}")


if __name__ == "__main__":
    main()
