"""WP-1 M2: proactive detect of hidden enemy → exposure jumps.

Spec §2.3: 主动探隐藏敌方 → 抬 exposure. Represents "active reveal" moment.

Scenario:
  1. Run 10 steps with enemy_emitting=False (track lost, fsld grows).
  2. Re-enable enemy emission. Next step detects hidden enemy → proactive bonus.
  3. Verify exposure jump > regular emit_increment + bonus amount per event.
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import numpy as np
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


def test_proactive_detect_raises_exposure():
    """When hidden enemy is re-detected, exposure jumps by emit + proactive_bonus."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=50, geometry=MIRROR_GEOMETRY)
    env.reset()
    configure_distinct_channels(env)
    action = uniform_action(env)

    # Phase 1: shut down enemies, accumulate fsld
    env.enemy_emitting[:] = False
    for _ in range(10):
        env.step(action)

    fsld_before = env.frames_since_last_detection.clone()
    assert int(fsld_before.max().item()) >= 10, (
        f"fsld didn't grow during shutdown: {fsld_before.max().item()}"
    )

    # Phase 2: re-enable emission → proactive detect event fires
    env.enemy_emitting[:] = True
    exp_pre = env.exposure.clone()
    env.step(action)
    exp_post_step1 = env.exposure.clone()
    delta_step1 = (exp_post_step1 - exp_pre)[0]   # [T]

    # Phase 3: subsequent step (no proactive event, only regular emit)
    exp_pre2 = env.exposure.clone()
    env.step(action)
    exp_post_step2 = env.exposure.clone()
    delta_step2 = (exp_post_step2 - exp_pre2)[0]

    # Compute regular emit_increment expected value
    emit_inc_expected = (
        env.emit_power_per_subarray * env.n_subarrays * 1.0 * env.dt *
        (0.25 + 0.25 + 0.25) * 2   # 2 apertures
    )

    # Step 1 delta should be emit_inc + proactive_bonus (for each event)
    # Step 2 delta should be emit_inc only
    bonus_measured = (delta_step1 - delta_step2)
    bonus_expected = env.proactive_detect_exposure_bonus

    # Sanity: deltas are positive
    assert (delta_step1 > 0).all(), f"step 1 exposure delta not positive: {delta_step1}"
    assert (delta_step2 > 0).all(), f"step 2 exposure delta not positive: {delta_step2}"

    # Step 1 should be > step 2 (proactive bonus on top)
    assert (delta_step1 > delta_step2 + 0.01).all(), (
        f"proactive bonus not visible: step1={delta_step1.cpu().numpy()}, "
        f"step2={delta_step2.cpu().numpy()}, bonus_measured={bonus_measured.cpu().numpy()}"
    )

    # Measured bonus should be close to expected (per event × n_events)
    # Both teams have 2 apertures detecting → 2 proactive events each
    expected_total_bonus = 2 * bonus_expected   # 2 events per team
    bonus_diff = (bonus_measured[0].item() - expected_total_bonus)
    assert abs(bonus_diff) < 0.05, (
        f"proactive bonus mismatch: measured={bonus_measured[0].item():.4f}, "
        f"expected={expected_total_bonus:.4f} (2 events × {bonus_expected})"
    )

    print(f"✅ proactive detect → exposure jump:")
    print(f"   step1 delta = {delta_step1.cpu().numpy()} (emit + 2·bonus)")
    print(f"   step2 delta = {delta_step2.cpu().numpy()} (emit only)")
    print(f"   bonus diff  = {bonus_diff:.4f} (measured-expected)")


def test_no_proactive_bonus_under_continuous_tracking():
    """If target was tracked continuously (fsld low), no proactive bonus fires."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=50, geometry=MIRROR_GEOMETRY)
    env.reset()
    configure_distinct_channels(env)
    action = uniform_action(env)

    # Run 5 steps with continuous emission (no shutdown phase)
    # First step gets proactive bonus (fsld init=0 doesn't count? Let me check)
    # Actually: fsld starts at 0. Threshold is fsld > 5. So step 0 fsld_pre=0 → no proactive.
    # Step 1: if detected, fsld_pre=0 (reset to 0 on step 0) → no proactive.
    # Continuously tracked targets never have fsld > 5 → no proactive bonus.

    deltas = []
    for step in range(10):
        exp_pre = env.exposure.clone()
        env.step(action)
        delta = (env.exposure - exp_pre)[0]   # [T]
        deltas.append(delta.cpu().numpy())

    deltas = torch.tensor(np.array(deltas))

    # All deltas should be roughly equal (no proactive spikes)
    delta_max = deltas.max()
    delta_min = deltas.min()
    spread = (delta_max - delta_min)
    assert spread < 0.05, (
        f"delta spread too large (proactive bonus may be spurious): "
        f"min={delta_min:.4f}, max={delta_max:.4f}, spread={spread:.4f}"
    )
    print(f"✅ no proactive bonus under continuous tracking: "
          f"delta range [{delta_min:.4f}, {delta_max:.4f}], spread={spread:.4f}")


if __name__ == "__main__":
    print("=== WP-1 M2: proactive detect → exposure ===")
    print()
    print("--- test_proactive_detect_raises_exposure ---")
    test_proactive_detect_raises_exposure()
    print()
    print("--- test_no_proactive_bonus_under_continuous_tracking ---")
    test_no_proactive_bonus_under_continuous_tracking()
    print()
    print("🎉 all M2 proactive-detect tests PASS")
