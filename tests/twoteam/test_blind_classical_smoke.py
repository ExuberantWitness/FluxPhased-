"""WP-2 M2: BlindClassicalCommander smoke tests.

Spec §3 ③ requires the blind classical commander to:
  - Output beam_direction (no god-view beam_target)
  - Search when slot uninit, track when init
  - Survive a full episode without crash

These are smoke tests — capability verification (kill rates, interference
gradient) is in test_blind_classical_low_interference.py / _high_interference.py.
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import math
import torch
from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


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


def make_env(n_envs=4, episode_steps=200, p_fa=1e-6):
    env = TwoTeamVecEnv(n_envs=n_envs, device="cuda", episode_steps=episode_steps,
                        geometry=MIRROR_GEOMETRY, seed=42, p_fa=p_fa)
    env.reset()
    configure_distinct_channels(env)
    return env


def test_blind_commander_outputs_beam_direction():
    """Action dict has beam_direction [E, R] ∈ [-π, π], no beam_target key."""
    env = make_env()
    cmd = BlindClassicalCommander()
    a0 = cmd.get_action(env, team=0)
    assert "beam_direction" in a0, "missing beam_direction in action"
    assert "beam_target" not in a0, "should not emit legacy beam_target"
    bd = a0["beam_direction"]
    assert bd.shape == (env.E, env.n_radars_per_team), (
        f"beam_direction shape {bd.shape} != {(env.E, env.n_radars_per_team)}"
    )
    assert (bd >= -math.pi - 1e-4).all() and (bd <= math.pi + 1e-4).all(), (
        f"beam_direction out of [-π, π]: min={bd.min():.3f}, max={bd.max():.3f}"
    )
    print(f"✅ outputs beam_direction: shape={tuple(bd.shape)}, "
          f"range=[{bd.min():.3f}, {bd.max():.3f}]")


def test_blind_commander_search_when_uninit():
    """All slots uninit → beam_direction points at a search cell, not origin.

    Fresh env: tracker_initialized all False. Commander should pick low-coverage
    cells. Search cells span [-π, π] so beam_direction should NOT be uniformly 0
    (origin) — different apertures should aim at different cells.
    """
    env = make_env()
    cmd = BlindClassicalCommander()
    a0 = cmd.get_action(env, team=0)
    bd = a0["beam_direction"]                                              # [E, R]
    # Two apertures should aim at different cells (round-robin through order)
    diff = (bd[:, 0] - bd[:, 1]).abs()
    assert diff.mean() > 1e-3, (
        f"aperture 0 and 1 picked same search cell: |diff| mean={diff.mean():.4f}"
    )
    # And both should be valid cell centers
    cell_width = 2.0 * math.pi / env.n_search_cells
    cell_centers = torch.arange(env.n_search_cells, device=env.device).float() * cell_width - math.pi
    # Distance from each beam_az to nearest cell center should be 0 (it's a cell center)
    for k in range(env.n_radars_per_team):
        bd_k = bd[:, k]
        dists = (bd_k.unsqueeze(-1) - cell_centers.unsqueeze(0)).abs()    # [E, n_cells]
        nearest = dists.min(dim=-1).values                                # [E]
        assert nearest.max() < 1e-4, (
            f"aperture {k} beam_az not at search cell center: max dist={nearest.max():.4f}"
        )
    print(f"✅ search mode: aperture 0/1 pick distinct cells, "
          f"mean |diff|={diff.mean():.3f} rad")


def test_blind_commander_tracks_when_init():
    """After slot init, beam_direction points at tracker belief azimuth.

    Force tracker_x to known position, verify beam_az = atan2(dy, dx).
    """
    env = make_env()
    # Force tracker slot 0 team 0 to a known belief
    env.tracker_x[:, 0, 0, 0] = env.radar_pos[:, 0, 0, 0] + 1000.0   # +x offset
    env.tracker_x[:, 0, 0, 2] = env.radar_pos[:, 0, 0, 1]            # same y
    env.tracker_initialized[:, 0, 0] = True
    cmd = BlindClassicalCommander()
    a0 = cmd.get_action(env, team=0)
    bd_slot0 = a0["beam_direction"][:, 0]                              # [E]
    # atan2(0, 1000) = 0 (pointing +x)
    err = (bd_slot0 - 0.0).abs().max().item()
    assert err < 1e-3, (
        f"after init slot 0 beam_az should be 0 (+x), got max err={err:.4f}"
    )
    print(f"✅ track mode: slot 0 init at +x → beam_az = {bd_slot0.mean():.4f} (= 0)")


def test_blind_commander_no_godview():
    """Blind commander must pass env.assert_no_godview (no enemy-truth leak)."""
    env = make_env()
    cmd = BlindClassicalCommander()
    # Run a few steps so obs state has tracker / jam populated
    a0 = cmd.get_action(env, team=0)
    a1 = cmd.get_action(env, team=1)
    action = combine_team_actions(env, a0, a1)
    for _ in range(5):
        env.step(action)
    result = env.assert_no_godview(tol=1e-5)
    n_fail = len(result["fail_dims"])
    assert n_fail == 0, (
        f"god-view leak in {n_fail} obs dims: {result['fail_dims'][:5]}..."
    )
    print(f"✅ no-godview assert: {len(result['pass_dims'])}/{env.obs_dim} dims invariant")


def test_blind_commander_runs_full_episode():
    """200 steps, both teams running BlindClassical — no crash, no NaN."""
    env = make_env(episode_steps=200)
    cmd = BlindClassicalCommander()
    for step in range(env.episode_steps):
        a0 = cmd.get_action(env, team=0)
        a1 = cmd.get_action(env, team=1)
        action = combine_team_actions(env, a0, a1)
        obs, reward, done, info = env.step(action)
        assert not torch.isnan(reward).any(), f"NaN reward at step {step}"
        assert not torch.isnan(env.tracker_x).any(), f"NaN tracker_x at step {step}"
        if done.all():
            break
    n_kills = info["team_kills"].sum().item()
    print(f"✅ full episode ran: {step+1} steps, total kills={n_kills}, "
          f"trace_P mean={info['mean_trace_P'].mean():.3f}")


if __name__ == "__main__":
    print("=== WP-2 M2: BlindClassicalCommander smoke ===")
    print()
    tests = [
        test_blind_commander_outputs_beam_direction,
        test_blind_commander_search_when_uninit,
        test_blind_commander_tracks_when_init,
        test_blind_commander_no_godview,
        test_blind_commander_runs_full_episode,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
        print()
    print("🎉 all M2 BlindClassical smoke tests PASS")
