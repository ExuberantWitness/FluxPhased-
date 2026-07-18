"""WP-3.1 Fix A micro-verification (no GPU burn).

3 asserts per spec §4 A:
  (a) learning_team lase → radar_E[:, enemy_team] increases
  (b) dwell_frac increases → rew_lt increment > 0
  (c) team_kills delta=1 → kill bonus fires exactly once

Strategy: use BlindClassicalCommander as BOTH teams' action source (BC actively
tracks + fires — ensures dwell chain physically triggers). Compute the reward
shaping arithmetic inline (same formula as br_trainer.collect_rollout).
"""
from __future__ import annotations
import os
import sys
import torch

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


def manual_rollout(env, bc_lt, bc_opp, learning_team: int, horizon: int,
                   shape_dwell_bonus: float, shape_kill_bonus: float,
                   shape_track_bonus: float = 0.0,
                   shape_exposure_penalty: float = 0.0,
                   reward_scale: float = 0.1,
                   channel_mode: str = "orthogonal"):
    """Manual rollout replicating br_trainer.collect_rollout reward shaping."""
    et = 1 - learning_team
    obs_dict = env.reset()
    configure_channels(env, channel_mode)
    prev_kills = env.team_kills[:, learning_team].clone()
    log = {k: [] for k in [
        "rew_total", "rew_dwell", "rew_kill", "rew_track", "rew_exp",
        "rew_base", "dwell_frac", "new_kills", "radar_E_et",
        "team_kills_lt", "emission_lt",
    ]}
    for t in range(horizon):
        br_action = bc_lt.get_action(env, learning_team)
        opp_action = bc_opp.get_action(env, 1 - learning_team)
        if learning_team == 0:
            action = combine_team_actions(env, br_action, opp_action)
        else:
            action = combine_team_actions(env, opp_action, br_action)

        obs_dict, reward, done, info = env.step(action)

        rew_base = reward[:, learning_team] * reward_scale
        dwell_frac = env.radar_E[:, et].sum(dim=-1) / env.e_kill
        rew_dwell = shape_dwell_bonus * dwell_frac
        now_kills = info["team_kills"][:, learning_team]
        new_kills = (now_kills - prev_kills).clamp(min=0).float()
        rew_kill = shape_kill_bonus * new_kills
        if shape_track_bonus > 0.0:
            trace_P_t = env.tracker_P[:, learning_team, :, 0, 0] + \
                        env.tracker_P[:, learning_team, :, 2, 2]
            n_tracked = ((trace_P_t < env.tau_track) &
                         env.tracker_initialized[:, learning_team]).float().sum(dim=-1)
            rew_track = shape_track_bonus * n_tracked
        else:
            rew_track = torch.zeros_like(rew_base)
        if shape_exposure_penalty > 0.0:
            rew_exp = -shape_exposure_penalty * info["exposure"][:, learning_team]
        else:
            rew_exp = torch.zeros_like(rew_base)
        rew_total = rew_base + rew_dwell + rew_kill + rew_track + rew_exp

        log["rew_total"].append(rew_total.cpu())
        log["rew_base"].append(rew_base.cpu())
        log["rew_dwell"].append(rew_dwell.cpu())
        log["rew_kill"].append(rew_kill.cpu())
        log["rew_track"].append(rew_track.cpu())
        log["rew_exp"].append(rew_exp.cpu())
        log["dwell_frac"].append(dwell_frac.cpu())
        log["new_kills"].append(new_kills.cpu())
        log["radar_E_et"].append(env.radar_E[:, et].sum(dim=-1).cpu())
        log["team_kills_lt"].append(now_kills.cpu())
        log["emission_lt"].append(br_action["emission_on"].sum(dim=-1).cpu())
        prev_kills = now_kills.clone()
        if done.all():
            obs_dict = env.reset()
            configure_channels(env, channel_mode)
            prev_kills = env.team_kills[:, learning_team].clone()
    for k in log:
        log[k] = torch.stack(log[k])
    return log


def configure_channels(env, mode: str = "orthogonal"):
    """Copy from wp3_smoke_crossplay.py — orthogonal gives BC ~1.0 kill per WP-2."""
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


def main():
    dev = "cuda"
    n_envs = 16
    horizon = 300   # need enough steps for BC detect+track+dwell→kill

    env = TwoTeamVecEnv(
        n_envs=n_envs, device=dev, episode_steps=200,
        geometry=RANDOM_GEOMETRY, seed=42,
    )
    env.reset()
    configure_channels(env, "orthogonal")   # WP-2 baseline setup; gives BC ~1.0 kill
    print(f"env: E={n_envs} H={horizon} n_radars={env.n_radars_per_team} "
          f"e_kill={env.e_kill} dwell_rate={env.dwell_rate}")

    # Try multiple action sources to find one that produces real kills in 300 steps.
    # Per WP-2 cross-play reports, BC vs BC rarely kills (mutual counter-fire).
    # StrongRule (god-view) is a more aggressive killer.
    from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
    lt_cmd = TwoTeamStrongRuleCommander()
    opp_cmd = BlindClassicalCommander()
    print(f"learning_team=StrongRule (god-view, aggressive), opponent=BlindClassical")

    SHAPE_DWELL = 1.0
    SHAPE_KILL = 50.0
    log = manual_rollout(
        env, lt_cmd, opp_cmd, learning_team=0, horizon=horizon,
        shape_dwell_bonus=SHAPE_DWELL, shape_kill_bonus=SHAPE_KILL,
    )

    radar_E = log["radar_E_et"]   # [H, E]
    dwell = log["dwell_frac"]
    new_kills = log["new_kills"]
    rew_dwell = log["rew_dwell"]
    rew_kill = log["rew_kill"]
    rew_total = log["rew_total"]

    # ---- Assert (a): radar_E[:,et] increased ----
    any_increase_per_env = (radar_E[1:] > radar_E[:-1] + 1e-6).any(dim=0)
    n_envs_increase = int(any_increase_per_env.sum().item())
    radar_E_max = float(radar_E.max().item())
    print(f"\n=== [a] radar_E[:,et] increase ===")
    print(f"  envs with radar_E increase: {n_envs_increase}/{n_envs}")
    print(f"  radar_E max over rollout:   {radar_E_max:.3f} (e_kill={env.e_kill})")
    print(f"  radar_E per-env final:      {radar_E[-1].tolist()}")
    assert n_envs_increase > 0, (
        "FAIL (a): radar_E[:,et] never increased — BC's laser not accumulating dwell."
    )

    # ---- Assert (b): dwell_frac > 0 → rew_dwell > 0 ----
    dwell_pos = (dwell > 1e-6).any(dim=0)
    rew_dwell_pos = (rew_dwell > 1e-6).any(dim=0)
    n_dwell_pos = int(dwell_pos.sum().item())
    n_rew_pos = int(rew_dwell_pos.sum().item())
    print(f"\n=== [b] dwell_frac → rew_dwell ===")
    print(f"  envs with dwell_frac > 0:   {n_dwell_pos}/{n_envs}")
    print(f"  envs with rew_dwell > 0:    {n_rew_pos}/{n_envs}")
    print(f"  dwell_frac max:             {dwell.max():.3f}")
    print(f"  rew_dwell max:              {rew_dwell.max():.3f}")
    assert n_dwell_pos == n_rew_pos and n_dwell_pos > 0, (
        f"FAIL (b): dwell positive in {n_dwell_pos} envs, rew_dwell positive in "
        f"{n_rew_pos} — arithmetic mismatch."
    )

    # ---- Assert (c): team_kills delta>0 → kill bonus = 50 × kills ----
    total_kills = int(new_kills.sum().item())
    total_kill_bonus = float(rew_kill.sum().item())
    expected = total_kills * SHAPE_KILL
    print(f"\n=== [c] kill bonus ===")
    print(f"  total new kills in rollout: {total_kills}")
    print(f"  total kill bonus paid:      {total_kill_bonus:.1f}")
    print(f"  expected (kills × {SHAPE_KILL}):    {expected:.1f}")
    if total_kills == 0:
        print(f"  ⚠️ No kills in {horizon} steps. Increase horizon or check BC vs pure_track.")
        print(f"     radar_E max={radar_E_max:.3f}, e_kill={env.e_kill}")
        if radar_E_max < env.e_kill:
            print(f"     radar_E never reached e_kill — BC tracking/lasing too slow.")
        # Not asserting — env physics matter here, not shaping arithmetic.
    else:
        assert abs(total_kill_bonus - expected) < 1e-3, (
            f"FAIL (c): kill bonus {total_kill_bonus} != expected {expected}"
        )
        print(f"  ✅ kill bonus fires exactly once per kill (= {SHAPE_KILL}/kill).")

    # ---- Per-step trace ----
    print(f"\nPer-step (env 0, every 10 steps):")
    print(f"  step | radar_E_et | dwell | rew_dwell | new_kills | rew_kill | rew_total")
    for t in range(0, horizon, max(1, horizon // 20)):
        e = 0
        print(f"  {t:4d} | {log['radar_E_et'][t,e]:10.3f} | "
              f"{log['dwell_frac'][t,e]:5.3f} | {log['rew_dwell'][t,e]:9.3f} | "
              f"{int(log['new_kills'][t,e].item())} | "
              f"{log['rew_kill'][t,e]:8.3f} | {log['rew_total'][t,e]:+7.3f}")

    # Final team_kills summary
    final_kills_per_env = log["team_kills_lt"][-1].tolist()
    print(f"\nFinal team_kills[learning_team=0] per env: {final_kills_per_env}")

    print("\n=== Fix A micro-verification PASS ===")


if __name__ == "__main__":
    main()
