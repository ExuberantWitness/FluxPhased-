"""WP-C R3 unit tests: channel_select action interface (env.step + AC head)."""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import pytest
import torch
from env.gpu.twoteam import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic


def _make_env(n_envs=4, episode_steps=50):
    return TwoTeamVecEnv(
        n_envs=n_envs, device="cuda",
        episode_steps=episode_steps,
        geometry=MIRROR_GEOMETRY,
        team_offset_m=2500.0,
    )


def test_env_step_accepts_channel_select():
    """env.step consumes channel_select and updates radar_freq_hz."""
    env = _make_env()
    env.reset()
    fc = env.fc_hz
    spacing = env.channel_spacing_hz

    # Before: all radars at fc
    assert torch.allclose(env.radar_freq_hz, torch.full_like(env.radar_freq_hz, fc))

    # Action: set channel_select = 3 for all radars
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    cs = torch.full((E, T, R), 3, dtype=torch.long, device=env.device)
    action = {
        "task_alloc": torch.ones(E, T, R, 4, device=env.device) / 4,
        "beam_target": torch.zeros(E, T, R, dtype=torch.long, device=env.device),
        "laser_target": torch.zeros(E, T, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(E, T, R, device=env.device),
        "freq_hop_rate": torch.ones(E, T, R, device=env.device),
        "channel_select": cs,
    }
    env.step(action)

    expected = fc + 3 * spacing
    assert torch.allclose(env.radar_freq_hz, torch.full_like(env.radar_freq_hz, expected)), \
        f"Expected all freq = {expected}, got max={env.radar_freq_hz.max().item()}"


def test_env_step_no_channel_select_keeps_reset_freq():
    """Without channel_select, env keeps reset freq (backward compat)."""
    env = _make_env()
    env.reset()
    fc = env.fc_hz
    # Set custom freq via set_radar_freqs
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    custom_freq = torch.full((E, T, R), fc + 5 * env.channel_spacing_hz, device=env.device)
    env.set_radar_freqs(custom_freq)

    action = {
        "task_alloc": torch.ones(E, T, R, 4, device=env.device) / 4,
        "beam_target": torch.zeros(E, T, R, dtype=torch.long, device=env.device),
        "laser_target": torch.zeros(E, T, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(E, T, R, device=env.device),
        "freq_hop_rate": torch.ones(E, T, R, device=env.device),
        # No channel_select
    }
    env.step(action)
    assert torch.allclose(env.radar_freq_hz, custom_freq), \
        "env should preserve set_radar_freqs when channel_select absent"


def test_ac_has_channel_select_head():
    """AC actor outputs channel_select in action dict.

    WP-3 M0: AC signature requires detect_list [B, K_max, 5].
    """
    env = _make_env()
    ac = TwoTeamCommanderActorCritic(
        obs_dim=env.obs_dim, privileged_dim=env.privileged_dim,
        n_fn=env.n_fn, n_aperture=env.n_radars_per_team,
        n_enemy=env.n_radars_per_team, freq_hop_max=env.freq_hop_max,
        n_channels=env.n_channels,
    ).to(env.device)

    obs = torch.randn(8, env.obs_dim, device=env.device)
    detect = torch.randn(8, env.k_max, 5, device=env.device)
    priv = torch.randn(8, env.privileged_dim, device=env.device)
    action, log_prob, value, value_local = ac(obs, detect, priv)

    assert "channel_select" in action, "AC action missing channel_select"
    assert action["channel_select"].shape == (8, env.n_radars_per_team)
    assert action["channel_select"].dtype == torch.long
    assert (action["channel_select"] >= 0).all() and \
           (action["channel_select"] < env.n_channels).all()


def test_ac_evaluate_actions_includes_channel_select():
    """AC evaluate_actions computes log_prob including channel_select head.

    WP-3 M0: AC signature requires detect_list; beam_target removed.
    """
    env = _make_env()
    ac = TwoTeamCommanderActorCritic(
        obs_dim=env.obs_dim, privileged_dim=env.privileged_dim,
        n_fn=env.n_fn, n_aperture=env.n_radars_per_team,
        n_enemy=env.n_radars_per_team, freq_hop_max=env.freq_hop_max,
        n_channels=env.n_channels,
    ).to(env.device)

    B = 8
    obs = torch.randn(B, env.obs_dim, device=env.device)
    detect = torch.randn(B, env.k_max, 5, device=env.device)
    priv = torch.randn(B, env.privileged_dim, device=env.device)
    action = {
        "task_alloc": torch.softmax(torch.randn(B, env.n_radars_per_team, env.n_fn, device=env.device), dim=-1),
        "beam_direction": torch.zeros(B, env.n_radars_per_team, device=env.device),
        "laser_target": torch.zeros(B, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(B, env.n_radars_per_team, device=env.device),
        "freq_hop_rate": torch.ones(B, env.n_radars_per_team, device=env.device) * 2.0,
        "channel_select": torch.zeros(B, env.n_radars_per_team, dtype=torch.long, device=env.device),
    }
    log_prob, value, value_local, entropy = ac.evaluate_actions(obs, detect, action, priv)

    assert log_prob.shape == (B,)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


def test_mirror_symmetry_preserved_with_channel_select():
    """In mirror env, identical channel_select from both teams → no asymmetry."""
    env = _make_env()
    env.reset()
    E, T, R = env.E, env.n_teams, env.n_radars_per_team

    # Both teams issue identical mirror-symmetric channel_select
    cs = torch.zeros(E, T, R, dtype=torch.long, device=env.device)
    cs[:, 0, 0] = 0   # team A radar 0 → ch0
    cs[:, 0, 1] = 1   # team A radar 1 → ch1
    cs[:, 1, 0] = 0   # team B radar 0 → ch0 (mirror)
    cs[:, 1, 1] = 1   # team B radar 1 → ch1 (mirror)

    action = {
        "task_alloc": torch.ones(E, T, R, 4, device=env.device) / 4,
        "beam_target": torch.zeros(E, T, R, dtype=torch.long, device=env.device),
        "laser_target": torch.zeros(E, T, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(E, T, R, device=env.device) * 0.5,
        "freq_hop_rate": torch.ones(E, T, R, device=env.device),
        "channel_select": cs,
    }
    env.step(action)

    # team_A freq should equal team_B freq (mirror symmetric)
    freq_A = env.radar_freq_hz[:, 0, :]
    freq_B = env.radar_freq_hz[:, 1, :]
    assert torch.allclose(freq_A, freq_B, atol=1.0), \
        f"Mirror symmetry broken: team_A vs team_B freq differ by " \
        f"{(freq_A - freq_B).abs().max().item():.1f} Hz"


if __name__ == "__main__":
    test_env_step_accepts_channel_select()
    test_env_step_no_channel_select_keeps_reset_freq()
    test_ac_has_channel_select_head()
    test_ac_evaluate_actions_includes_channel_select()
    test_mirror_symmetry_preserved_with_channel_select()
    print("All channel_select tests PASS")
