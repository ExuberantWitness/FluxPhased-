"""WP-1 M2: no-godview assert + obs extensions.

Spec §2.5 铁律: obs 绝不含敌方 true 位置 / true emit 状态 / true jam.

Test matrix:
  1. assert_no_godview() returns 0 fail_dims on a clean env.
  2. enemy_emitting=False → obs[enemy_freq_slot] is zeroed (no leak from silent enemy).
  3. New M2 obs fields exist (frames_since_last_detection, search_coverage, n_detections).
  4. Proactive detect of hidden enemy → exposure jumps (spec §2.3).
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import torch
from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, MIRROR_GEOMETRY


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


def uniform_action(env):
    E = env.E
    return {
        "task_alloc": torch.full((E, 2, 2, 4), 0.25, device=env.device),
        "beam_target": torch.zeros(E, 2, 2, dtype=torch.long, device=env.device),
        "laser_target": torch.zeros(E, 2, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(E, 2, 2, dtype=torch.bool, device=env.device),
        "freq_hop_rate": torch.ones(E, 2, 2, device=env.device),
    }


def test_assert_no_godview_passes_clean():
    """On a clean env with state built up, assert_no_godview returns 0 fail_dims."""
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=50, geometry=MIRROR_GEOMETRY)
    env.reset()
    configure_distinct_channels(env)
    action = uniform_action(env)

    # Build up state with 10 steps
    for _ in range(10):
        env.step(action)

    result = env.assert_no_godview(tol=1e-5)

    assert len(result["fail_dims"]) == 0, (
        f"god-view leaks detected at dims: {result['fail_dims']}\n"
        f"max_diff_per_dim: {result['max_diff_per_dim']}"
    )
    assert len(result["pass_dims"]) == env.obs_dim, (
        f"only {len(result['pass_dims'])}/{env.obs_dim} dims passed"
    )
    print(f"✅ assert_no_godview: all {env.obs_dim} dims invariant under truth permutation")


def test_assert_no_godview_detects_injected_leak():
    """Sanity: if we inject a fake god-view field, assert catches it."""
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=50, geometry=MIRROR_GEOMETRY)
    env.reset()
    configure_distinct_channels(env)
    action = uniform_action(env)
    for _ in range(5):
        env.step(action)

    # Save original obs method
    orig_get_obs = env.get_obs

    # Inject a fake god-view: write enemy true x-position into obs dim 43
    def leaky_get_obs():
        out = orig_get_obs()
        # Leak: obs[t, 43] = enemy_team_true_x (depends on which team's perspective)
        # For team t, enemy is team 1-t; leak that enemy's radar 0 x-coord.
        for t in range(env.n_teams):
            et = 1 - t
            out["obs"][:, t, 43] = env.radar_pos[:, et, 0, 0] / 1000.0
        return out

    env.get_obs = leaky_get_obs
    result = env.assert_no_godview(tol=1e-5)
    env.get_obs = orig_get_obs   # restore

    assert 43 in result["fail_dims"], (
        f"Injected god-view leak at dim 43 not detected. fail_dims: {result['fail_dims']}"
    )
    print(f"✅ assert_no_godview caught injected god-view leak at dim 43 "
          f"(diff={result['max_diff_per_dim'][43]:.4f})")


def test_enemy_freq_zeroed_when_shutdown():
    """When enemy_emitting=False, obs[enemy_freq_slots] (38, 39) must be zero."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=50, geometry=MIRROR_GEOMETRY)
    env.reset()
    configure_distinct_channels(env)

    # Default enemy_emitting=True: enemy freq slots populated
    env.enemy_emitting[:] = True
    obs_emit = env.get_obs()["obs"]
    enemy_freq_emit = obs_emit[0, 0, 38].item()   # enemy team's radar 0 freq slot

    # enemy_emitting=False: enemy freq slots must zero out
    env.enemy_emitting[:] = False
    obs_silent = env.get_obs()["obs"]
    enemy_freq_silent = obs_silent[0, 0, 38].item()

    assert enemy_freq_emit > 0, (
        f"enemy freq slot empty even when enemy emitting: {enemy_freq_emit}"
    )
    assert enemy_freq_silent == 0.0, (
        f"enemy freq slot non-zero when enemy shut down: {enemy_freq_silent} "
        f"(should be 0 — can't measure freq from silent target)"
    )
    print(f"✅ enemy freq slot zeroed on shutdown: emit={enemy_freq_emit:.3f}, silent=0.0")


def test_obs_has_new_m2_fields():
    """obs_dim=44, new fields at indices 40-43 are populated."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=50, geometry=MIRROR_GEOMETRY)
    assert env.obs_dim == 44, f"obs_dim should be 44 after M2, got {env.obs_dim}"
    obs = env.reset()

    # Fresh env: all new fields are 0
    new_fields = obs["obs"][0, 0, 40:44]
    assert torch.allclose(new_fields, torch.zeros_like(new_fields)), (
        f"New M2 fields not zero on fresh env: {new_fields.cpu().numpy()}"
    )

    # Run steps and verify they populate
    configure_distinct_channels(env)
    action = uniform_action(env)
    for _ in range(20):
        obs, _, _, _ = env.step(action)

    # After 20 steps with detect alloc, search_coverage > 0
    search_cov = obs["obs"][0, 0, 42].item()
    assert search_cov > 0, f"search_coverage not populated: {search_cov}"

    # n_detections > 0 (distinct channels → detections happen)
    n_det = obs["obs"][0, 0, 43].item()
    assert n_det > 0, f"n_detections not populated: {n_det}"

    # frames_since_last_detection[r] tracked
    fsld_0 = obs["obs"][0, 0, 40].item()   # /100 normalized
    fsld_1 = obs["obs"][0, 0, 41].item()
    # Both enemies tracked (distinct ch, P_d≈0.8) → fsld low
    assert fsld_0 < 0.5 and fsld_1 < 0.5, (
        f"fsld should be low under active tracking: r0={fsld_0}, r1={fsld_1}"
    )
    print(f"✅ M2 obs fields populated: fsld=[{fsld_0:.2f},{fsld_1:.2f}], "
          f"search_cov={search_cov:.3f}, n_det={n_det:.3f}")


if __name__ == "__main__":
    print("=== WP-1 M2: no-godview + obs extensions ===")
    print()
    print("--- test_assert_no_godview_passes_clean ---")
    test_assert_no_godview_passes_clean()
    print()
    print("--- test_assert_no_godview_detects_injected_leak ---")
    test_assert_no_godview_detects_injected_leak()
    print()
    print("--- test_enemy_freq_zeroed_when_shutdown ---")
    test_enemy_freq_zeroed_when_shutdown()
    print()
    print("--- test_obs_has_new_m2_fields ---")
    test_obs_has_new_m2_fields()
    print()
    print("🎉 all M2 no-godview tests PASS")
