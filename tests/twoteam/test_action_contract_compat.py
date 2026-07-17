"""WP-1 M3: action contract backward compatibility.

Verifies soft transition (M3) — env accepts BOTH legacy beam_target{0,1} and
new beam_direction ∈ [-π, π] continuous azimuth. M4 will hard-cut the legacy.

Test matrix:
  1. AC forward outputs beam_direction ∈ [-π, π] (Beta sample rescaled).
  2. Env step works with new beam_direction (continuous azimuth).
  3. Env step works with legacy beam_target (no beam_direction key).
  4. When BOTH keys present, env prefers beam_direction.
  5. AC evaluate_actions handles legacy actions (no beam_direction) gracefully.
  6. AC evaluate_actions matches forward log_prob when beam_direction included.
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import math
import torch
from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions, ExtremeCommander


def configure_distinct_channels(env):
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    freqs = torch.zeros(E, T, R, device=env.device)
    fc = env.fc_hz
    for e in range(E):
        freqs[e, 0, 0] = fc
        freqs[e, 0, 1] = fc + env.channel_spacing_hz
        freqs[e, 1, 0] = fc + 2 * env.channel_spacing_hz
        freqs[e, 1, 1] = fc + 3 * env.channel_spacing_hz
    env.set_radar_freqs(freqs)


def test_ac_outputs_beam_direction():
    """AC forward produces beam_direction ∈ [-π, π] of shape [B, n_aperture].

    WP-3 M0: AC now requires `detect_list` [E, K_max, 5] (DeepSets encoder).
    """
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=30, geometry=MIRROR_GEOMETRY)
    env.reset()
    ac = TwoTeamCommanderActorCritic(
        obs_dim=env.obs_dim, privileged_dim=env.privileged_dim,
        n_fn=env.n_fn, n_aperture=env.n_radars_per_team,
        n_enemy=env.n_radars_per_team, freq_hop_max=env.freq_hop_max,
        n_channels=env.n_channels,
    ).to(env.device)

    obs_dict = env.get_obs()
    detect_lt = env.get_detect_list()[:, 0]   # WP-3 M0/M1
    action, _, _, _ = ac(obs_dict["obs"][:, 0], detect_lt, obs_dict["privileged"][:, 0])

    assert "beam_direction" in action, "AC action missing beam_direction"
    assert action["beam_direction"].shape == (4, env.n_radars_per_team)
    # Range check: [-π, π] (with small numerical margin)
    bd_min = action["beam_direction"].min().item()
    bd_max = action["beam_direction"].max().item()
    assert -math.pi - 1e-3 <= bd_min and bd_max <= math.pi + 1e-3, (
        f"beam_direction out of [-π, π]: min={bd_min}, max={bd_max}"
    )
    print(f"AC outputs beam_direction in [{bd_min:.3f}, {bd_max:.3f}]")


def test_env_step_with_new_beam_direction():
    """Env step succeeds with action containing beam_direction (no beam_target needed? no — beam_target still required by combine_team_actions)."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=30, geometry=MIRROR_GEOMETRY,
                        p_fa=1e-3)
    env.reset()
    configure_distinct_channels(env)
    E = env.E

    # Construct action with beam_direction pointing at known enemy azimuth
    # Team A[0] at (-2500, 750), enemy B[0] at (2500, -750)
    # az from A[0] to B[0] = atan2(-1500, 5000) ≈ -0.291 rad
    az_to_enemy0 = math.atan2(-1500.0, 5000.0)
    az_to_enemy1 = math.atan2(-1500.0 - 0.0, 5000.0)   # A[0] to B[1] at (2500, 750) = atan2(0, 5000) = 0
    az_to_enemy1 = math.atan2(750.0 - 750.0, 5000.0)   # = 0

    beam_direction = torch.zeros(E, 2, 2, device=env.device)
    # Team 0 (A): both apertures point at enemy 0 (-0.291 rad)
    beam_direction[:, 0, :] = az_to_enemy0
    # Team 1 (B): mirror — both apertures point at enemy 0 (A[0]) from B[0]'s perspective
    # B[0] at (2500, -750), enemy A[0] at (-2500, 750)
    # az = atan2(750-(-750), -2500-2500) = atan2(1500, -5000) = π - 0.291 = 2.85 rad
    beam_direction[:, 1, :] = math.atan2(1500.0, -5000.0)

    action = {
        "task_alloc": torch.full((E, 2, 2, 4), 0.25, device=env.device),
        "beam_target": torch.zeros(E, 2, 2, dtype=torch.long, device=env.device),   # ignored
        "beam_direction": beam_direction,
        "laser_target": torch.zeros(E, 2, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(E, 2, 2, dtype=torch.bool, device=env.device),
        "freq_hop_rate": torch.ones(E, 2, 2, device=env.device),
    }
    obs, r, d, info = env.step(action)

    # Verify beam_az matches what we provided
    bd_used = env.radar_beam_az[0, 0, 0].item()
    assert abs(bd_used - az_to_enemy0) < 1e-4, (
        f"env didn't use beam_direction: got {bd_used:.4f}, expected {az_to_enemy0:.4f}"
    )
    print(f"✅ env used beam_direction: beam_az={bd_used:.4f} (provided {az_to_enemy0:.4f})")


def test_env_step_with_legacy_beam_target():
    """Env step works with action containing only beam_target (legacy, no beam_direction)."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=30, geometry=MIRROR_GEOMETRY)
    env.reset()
    E = env.E

    action = {
        "task_alloc": torch.full((E, 2, 2, 4), 0.25, device=env.device),
        "beam_target": torch.zeros(E, 2, 2, dtype=torch.long, device=env.device),
        "laser_target": torch.zeros(E, 2, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(E, 2, 2, dtype=torch.bool, device=env.device),
        "freq_hop_rate": torch.ones(E, 2, 2, device=env.device),
    }
    # No beam_direction key — env should fall back to legacy path
    obs, r, d, info = env.step(action)
    # Legacy path: beam_target=0 → az from A[0] to enemy 0 = -0.291 rad
    bd_used = env.radar_beam_az[0, 0, 0].item()
    expected = math.atan2(-1500.0, 5000.0)
    assert abs(bd_used - expected) < 1e-4, (
        f"legacy beam_target→az wrong: got {bd_used:.4f}, expected {expected:.4f}"
    )
    print(f"✅ legacy beam_target fallback OK: beam_az={bd_used:.4f}")


def test_env_prefers_beam_direction_when_both_present():
    """When BOTH beam_direction and beam_target are in action, env uses beam_direction."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=30, geometry=MIRROR_GEOMETRY)
    env.reset()
    E = env.E

    # beam_target would give -0.291, beam_direction overrides to +1.5
    overridden_az = 1.5
    beam_direction = torch.full((E, 2, 2), overridden_az, device=env.device)
    action = {
        "task_alloc": torch.full((E, 2, 2, 4), 0.25, device=env.device),
        "beam_target": torch.zeros(E, 2, 2, dtype=torch.long, device=env.device),
        "beam_direction": beam_direction,
        "laser_target": torch.zeros(E, 2, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(E, 2, 2, dtype=torch.bool, device=env.device),
        "freq_hop_rate": torch.ones(E, 2, 2, device=env.device),
    }
    env.step(action)
    bd_used = env.radar_beam_az[0, 0, 0].item()
    assert abs(bd_used - overridden_az) < 1e-4, (
        f"env didn't prefer beam_direction: got {bd_used}, expected {overridden_az}"
    )
    print(f"✅ env prefers beam_direction when both present: {bd_used:.4f}")


def test_ac_evaluate_actions_handles_no_beam_target():
    """WP-3 M0: AC evaluate_actions no longer accepts beam_target (god-view killed).

    Confirms action dict has no beam_target key and evaluate_actions runs cleanly.
    """
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=30, geometry=MIRROR_GEOMETRY)
    env.reset()
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
    # Blind action (no beam_target — god-view removed in WP-3 M0)
    action = {
        "task_alloc": torch.softmax(torch.randn(B, env.n_radars_per_team, env.n_fn, device=env.device), dim=-1),
        "beam_direction": torch.zeros(B, env.n_radars_per_team, device=env.device),
        "laser_target": torch.zeros(B, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(B, env.n_radars_per_team, device=env.device),
        "freq_hop_rate": torch.full((B, env.n_radars_per_team), 2.0, device=env.device),
        "channel_select": torch.zeros(B, env.n_radars_per_team, dtype=torch.long, device=env.device),
    }
    log_prob, value, value_local, entropy = ac.evaluate_actions(obs, detect, action, priv)

    assert log_prob.shape == (B,)
    assert torch.isfinite(log_prob).all(), "log_prob has NaN/inf"
    assert torch.isfinite(entropy).all(), "entropy has NaN/inf"
    # WP-3 M0 contract: AC must NOT have beam_target_head attribute
    assert not hasattr(ac, "beam_target_head"), (
        "AC still has beam_target_head — god-view leak not fully removed")
    print(f"AC evaluate_actions blind OK (no beam_target in action; no beam_target_head attr); "
          f"log_prob mean={log_prob.mean().item():.2f}")


def test_ac_evaluate_actions_consistent_with_forward():
    """forward log_prob matches evaluate_actions when action includes beam_direction.

    WP-3 M0: AC signature now requires detect_list (DeepSets encoder).
    """
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=30, geometry=MIRROR_GEOMETRY)
    env.reset()
    ac = TwoTeamCommanderActorCritic(
        obs_dim=env.obs_dim, privileged_dim=env.privileged_dim,
        n_fn=env.n_fn, n_aperture=env.n_radars_per_team,
        n_enemy=env.n_radars_per_team, freq_hop_max=env.freq_hop_max,
        n_channels=env.n_channels,
    ).to(env.device)

    obs = env.get_obs()["obs"][:, 0]
    detect = env.get_detect_list()[:, 0]
    priv = env.get_obs()["privileged"][:, 0]
    a, lp_fwd, _, _ = ac(obs, detect, priv)
    lp_eval, _, _, _ = ac.evaluate_actions(obs, detect, a, priv)
    diff = (lp_fwd - lp_eval).abs().max().item()
    assert diff < 1e-4, f"forward vs evaluate log_prob diff: {diff}"
    print(f"AC forward/evaluate consistent with beam_direction: diff={diff:.2e}")


def test_combine_team_actions_handles_beam_direction():
    """combine_team_actions stacks beam_direction when both teams provide it.

    WP-3 M0: AC now requires detect_list (DeepSets encoder).
    """
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=30, geometry=MIRROR_GEOMETRY)
    env.reset()
    ac = TwoTeamCommanderActorCritic(
        obs_dim=env.obs_dim, privileged_dim=env.privileged_dim,
    ).to(env.device)

    obs_dict = env.get_obs()
    detect_0 = env.get_detect_list()[:, 0]
    detect_1 = env.get_detect_list()[:, 1]
    a_t0, _, _, _ = ac(obs_dict["obs"][:, 0], detect_0, obs_dict["privileged"][:, 0])
    a_t1, _, _, _ = ac(obs_dict["obs"][:, 1], detect_1, obs_dict["privileged"][:, 1])
    combined = combine_team_actions(env, a_t0, a_t1)

    assert "beam_direction" in combined, "combined action missing beam_direction"
    assert combined["beam_direction"].shape == (env.E, 2, env.n_radars_per_team)

    # Also verify legacy commander (no beam_direction) doesn't break combine
    ec = ExtremeCommander("test", [[0.25] * 4, [0.25] * 4])
    a0_legacy = ec.get_action(env, 0)
    a1_legacy = ec.get_action(env, 1)
    combined_legacy = combine_team_actions(env, a0_legacy, a1_legacy)
    assert "beam_direction" not in combined_legacy, (
        "combine_team_actions shouldn't add beam_direction when teams don't provide it"
    )
    print(f"combine_team_actions stacks beam_direction when both teams provide it; "
          f"omits when neither does")


if __name__ == "__main__":
    print("=== WP-1 M3: action contract backward compat ===")
    print()
    tests = [
        test_ac_outputs_beam_direction,
        test_env_step_with_new_beam_direction,
        test_env_step_with_legacy_beam_target,
        test_env_prefers_beam_direction_when_both_present,
        test_ac_evaluate_actions_handles_no_beam_target,
        test_ac_evaluate_actions_consistent_with_forward,
        test_combine_team_actions_handles_beam_direction,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
        print()
    print("🎉 all M3 action-contract-compat tests PASS")
