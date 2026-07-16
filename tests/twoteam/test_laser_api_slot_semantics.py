"""WP-2 M0: laser slot-id semantics + belief check.

spec §3 ⑥: fire control only on active + confident tracks. With the new
slot-id semantics, laser energy accumulates on the enemy the slot is *actually*
tracking (by belief position), not on the slot index. Slot mis-tracked → miss.

Tests:
  1. Slot 0 init at enemy 0's position → laser_target=0 adds energy to enemy 0.
  2. Force slot 0's belief to enemy 1's position → laser_target=0 adds energy
     to enemy 1 (because slot 0 belief is near enemy 1, not enemy 0).
  3. Slot 0 uninit → laser_target=0 ineffective.
  4. Mirror symmetry preserved under the new laser semantics.
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import math
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


def _set_slot_belief(env, team, slot, enemy_team, enemy_r):
    """Force tracker slot [team, slot] to track enemy_team's radar enemy_r."""
    enemy_pos = env.radar_pos[:, enemy_team, enemy_r]              # [E, 2]
    env.tracker_x[:, team, slot, 0] = enemy_pos[:, 0]
    env.tracker_x[:, team, slot, 2] = enemy_pos[:, 1]
    env.tracker_P[:, team, slot] = torch.eye(4, device=env.device) * 0.001
    env.tracker_initialized[:, team, slot] = True


def _laser_action(env, slot=0):
    """Laser-only action: all teams fire laser at given slot."""
    E = env.E
    return {
        "task_alloc": torch.full((E, 2, 2, 4), 0.25, device=env.device),
        "beam_target": torch.zeros(E, 2, 2, dtype=torch.long, device=env.device),
        "laser_target": torch.full((E, 2), slot, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(E, 2, 2, dtype=torch.bool, device=env.device),
        "freq_hop_rate": torch.ones(E, 2, 2, device=env.device),
    }


def test_laser_hits_when_slot_tracks_enemy():
    """Slot 0 belief at enemy 0's true pos → laser adds energy to enemy 0."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=10,
                        geometry=MIRROR_GEOMETRY, laser_hit_radius_m=50.0)
    env.reset()
    configure_distinct_channels(env)
    # Slot 0 of team 0 → tracks enemy team 1 radar 0 (B0)
    _set_slot_belief(env, team=0, slot=0, enemy_team=1, enemy_r=0)
    # Slot 0 of team 1 → tracks enemy team 0 radar 0 (A0)
    _set_slot_belief(env, team=1, slot=0, enemy_team=0, enemy_r=0)

    E_before = env.radar_E.clone()
    env.step(_laser_action(env, slot=0))
    E_after = env.radar_E.clone()

    delta_t0 = (E_after[:, 1, 0] - E_before[:, 1, 0]).mean().item()   # team 0 hit enemy team 1 radar 0
    delta_t1 = (E_after[:, 0, 0] - E_before[:, 0, 0]).mean().item()   # team 1 hit enemy team 0 radar 0
    expected_delta = env.dwell_rate * env.dt   # accum_dt per step
    assert delta_t0 > expected_delta * 0.5, (
        f"team 0 slot 0 didn't hit enemy radar 0: delta={delta_t0:.4f}, expected≈{expected_delta:.4f}"
    )
    assert delta_t1 > expected_delta * 0.5, (
        f"team 1 slot 0 didn't hit enemy radar 0: delta={delta_t1:.4f}, expected≈{expected_delta:.4f}"
    )
    print(f"✅ slot 0 → enemy 0: delta_t0={delta_t0:.4f}, delta_t1={delta_t1:.4f}")


def test_laser_misses_when_slot_mis_tracked():
    """Slot 0 belief forced to enemy 1's position → laser hits enemy 1, NOT 0."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=10,
                        geometry=MIRROR_GEOMETRY, laser_hit_radius_m=50.0)
    env.reset()
    configure_distinct_channels(env)
    # Force slot 0 → enemy radar 1 (mis-tracked relative to slot convention)
    _set_slot_belief(env, team=0, slot=0, enemy_team=1, enemy_r=1)
    _set_slot_belief(env, team=1, slot=0, enemy_team=0, enemy_r=1)

    E_before = env.radar_E.clone()
    env.step(_laser_action(env, slot=0))
    E_after = env.radar_E.clone()

    # Slot 0 belief is at enemy 1's pos → laser should add energy to enemy 1, NOT 0
    delta_enemy0 = (E_after[:, 1, 0] - E_before[:, 1, 0]).mean().item()
    delta_enemy1 = (E_after[:, 1, 1] - E_before[:, 1, 1]).mean().item()
    assert delta_enemy0 < 1e-6, (
        f"slot mis-tracked but laser hit enemy 0 (should miss): delta_enemy0={delta_enemy0:.4f}"
    )
    assert delta_enemy1 > 0.05, (
        f"slot mis-tracked laser should hit enemy 1 (belief target): delta_enemy1={delta_enemy1:.4f}"
    )
    print(f"✅ slot mis-tracked: laser hits belief target (enemy 1: Δ={delta_enemy1:.4f}), "
          f"NOT slot index (enemy 0: Δ={delta_enemy0:.4f})")


def test_laser_no_hit_when_slot_uninit():
    """Slot 0 uninitialized → laser ineffective regardless of position."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=10,
                        geometry=MIRROR_GEOMETRY, laser_hit_radius_m=50.0)
    env.reset()
    configure_distinct_channels(env)
    # Block all detections during step (so slot stays uninit through laser logic)
    env.enemy_emitting[:] = False
    assert not env.tracker_initialized[:, :, 0].any(), "slot 0 should be uninit by default"

    E_before = env.radar_E.clone()
    env.step(_laser_action(env, slot=0))
    E_after = env.radar_E.clone()

    delta = (E_after - E_before).abs().max().item()
    assert delta < 1e-6, (
        f"laser added energy despite slot uninit: max delta={delta:.4f}"
    )
    print(f"✅ uninit slot → laser ineffective (max ΔE = {delta:.2e})")


def test_laser_no_hit_when_belief_far_from_enemy():
    """Slot 0 init at a far position → laser misses (no enemy within hit_radius)."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=10,
                        geometry=MIRROR_GEOMETRY, laser_hit_radius_m=50.0)
    env.reset()
    configure_distinct_channels(env)
    # Block detections to prevent slot 0 from re-initializing on real enemy
    env.enemy_emitting[:] = False
    # Force slot 0's belief to a far position (origin, no enemy within 50m)
    env.tracker_x[:, :, 0, 0] = 0.0
    env.tracker_x[:, :, 0, 2] = 0.0
    env.tracker_P[:, :, 0] = torch.eye(4, device=env.device) * 0.001
    env.tracker_initialized[:, :, 0] = True

    E_before = env.radar_E.clone()
    env.step(_laser_action(env, slot=0))
    E_after = env.radar_E.clone()

    delta = (E_after - E_before).abs().max().item()
    assert delta < 1e-6, (
        f"laser added energy despite no enemy near belief: max delta={delta:.4f}"
    )
    print(f"✅ slot belief far from any enemy → laser misses (max ΔE = {delta:.2e})")


def test_laser_mirror_symmetric():
    """Laser accum is mirror-symmetric when both teams have mirrored beliefs."""
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=10,
                        geometry=MIRROR_GEOMETRY, laser_hit_radius_m=50.0)
    env.reset()
    configure_distinct_channels(env)
    _set_slot_belief(env, team=0, slot=0, enemy_team=1, enemy_r=0)
    _set_slot_belief(env, team=1, slot=0, enemy_team=0, enemy_r=0)

    E_before = env.radar_E.clone()
    env.step(_laser_action(env, slot=0))
    E_after = env.radar_E.clone()

    # team 0 hit enemy team 1 radar 0 == team 1 hit enemy team 0 radar 0 (mirror)
    delta_team0_kills = (E_after[:, 1, 0] - E_before[:, 1, 0]).cpu().numpy()
    delta_team1_kills = (E_after[:, 0, 0] - E_before[:, 0, 0]).cpu().numpy()
    asym = abs(delta_team0_kills.mean() - delta_team1_kills.mean())
    assert asym < 1e-5, (
        f"laser accum asymmetry under mirror: {asym:.2e}"
    )
    print(f"✅ laser accum mirror-symmetric: asym={asym:.2e}")


if __name__ == "__main__":
    print("=== WP-2 M0: laser slot-id semantics ===")
    print()
    tests = [
        test_laser_hits_when_slot_tracks_enemy,
        test_laser_misses_when_slot_mis_tracked,
        test_laser_no_hit_when_slot_uninit,
        test_laser_no_hit_when_belief_far_from_enemy,
        test_laser_mirror_symmetric,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
        print()
    print("🎉 all M0 laser-API tests PASS")
