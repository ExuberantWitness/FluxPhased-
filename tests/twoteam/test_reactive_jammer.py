"""WP-C R2 unit tests: ReactiveJammerCommander behavior."""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import torch
from env.gpu.twoteam import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.baselines.reactive_jammer_commander import ReactiveJammerCommander
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


def _make_env(n_envs=4, episode_steps=50):
    return TwoTeamVecEnv(
        n_envs=n_envs, device="cuda",
        episode_steps=episode_steps,
        geometry=MIRROR_GEOMETRY,
        team_offset_m=2500.0,
    )


def test_reactive_jammer_outputs_all_action_keys():
    """ReactiveJammer outputs the 6-key action dict including channel_select."""
    env = _make_env()
    env.reset()
    jammer = ReactiveJammerCommander(jam_fraction=1e-4)

    a = jammer.get_action(env, team=0)
    for key in ("task_alloc", "beam_target", "laser_target",
                "emission_on", "freq_hop_rate", "channel_select"):
        assert key in a, f"ReactiveJammer missing key: {key}"


def test_reactive_jammer_jam_fraction_in_alloc():
    """task_alloc[jam] equals jam_fraction (apple-to-apple with fixed jammer)."""
    env = _make_env()
    env.reset()
    jam_frac = 1e-4
    jammer = ReactiveJammerCommander(jam_fraction=jam_frac)

    a = jammer.get_action(env, team=0)
    # f_emit = task_alloc[detect] + task_alloc[track] + task_alloc[jam]
    # (comm excluded). Should equal jam_frac since reactive is jam-only.
    f_emit = a["task_alloc"][..., :3].sum(dim=-1).max().item()
    assert abs(f_emit - jam_frac) < 1e-3, \
        f"reactive f_emit {f_emit} should equal jam_fraction {jam_frac}"


def test_reactive_jammer_channel_select_within_bounds():
    """channel_select ∈ [0, n_channels-1]."""
    env = _make_env()
    env.reset()
    jammer = ReactiveJammerCommander(jam_fraction=1e-4)

    a = jammer.get_action(env, team=0)
    cs = a["channel_select"]
    assert (cs >= 0).all() and (cs < env.n_channels).all()


def test_reactive_jammer_picks_lowest_trace_p_victim():
    """ReactiveJammer targets victim radar with lowest trace_P (best-tracked)."""
    env = _make_env()
    env.reset()
    # Manually set tracker_P so victim radar 0 is much better tracked than radar 1
    with torch.no_grad():
        env.tracker_P[:, 1, 0, 0, 0] = 0.01   # victim radar 0 tight lock
        env.tracker_P[:, 1, 0, 2, 2] = 0.01
        env.tracker_P[:, 1, 1, 0, 0] = 100.0  # victim radar 1 lost lock
        env.tracker_P[:, 1, 1, 2, 2] = 100.0
        env.tracker_initialized[:, 1] = True

    jammer = ReactiveJammerCommander(jam_fraction=1e-4)
    a = jammer.get_action(env, team=0)
    # beam_target should be 0 (best-tracked victim radar)
    assert (a["beam_target"] == 0).all(), \
        f"beam_target should be 0 (lowest trace_P victim), got {a['beam_target']}"


def test_reactive_jammer_runs_full_episode():
    """ReactiveJammer runs cleanly with env.step for a full episode."""
    env = _make_env(n_envs=4, episode_steps=30)
    jammer = ReactiveJammerCommander(jam_fraction=1e-3)

    env.reset()
    for step in range(30):
        a_jam = jammer.get_action(env, team=0)
        # Opponent: just zeros (placeholder)
        E, R = env.E, env.n_radars_per_team
        a_opp = {
            "task_alloc": torch.ones(E, R, 4, device=env.device) / 4,
            "beam_target": torch.zeros(E, R, dtype=torch.long, device=env.device),
            "laser_target": torch.zeros(E, dtype=torch.long, device=env.device),
            "emission_on": torch.ones(E, R, device=env.device),
            "freq_hop_rate": torch.ones(E, R, device=env.device),
            "channel_select": torch.zeros(E, R, dtype=torch.long, device=env.device),
        }
        action = combine_team_actions(env, a_jam, a_opp)
        obs, reward, done, info = env.step(action)
        if done.all():
            break

    # Sanity: no NaN in key env state
    assert not torch.isnan(env.tracker_P).any(), "tracker_P NaN after episode"
    assert not torch.isnan(env.radar_freq_hz).any(), "radar_freq_hz NaN"


if __name__ == "__main__":
    test_reactive_jammer_outputs_all_action_keys()
    test_reactive_jammer_jam_fraction_in_alloc()
    test_reactive_jammer_channel_select_within_bounds()
    test_reactive_jammer_picks_lowest_trace_p_victim()
    test_reactive_jammer_runs_full_episode()
    print("All ReactiveJammer tests PASS")
