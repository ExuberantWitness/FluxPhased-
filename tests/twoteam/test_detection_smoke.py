"""WP-1 M1 smoke test for detection chain.

Verifies:
  1. Detections are produced when geometry + SNR permit.
  2. K_max padding: mask zeros after last real/FA detection (no garbage past count).
  3. p_fa=1e-3 stress produces >=1 false alarm over a 30-step episode.
  4. Real detections have finite SNR > threshold - 2·width.
  5. z positions of real detections are within ~10·σ_range of true enemy position.
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import torch
from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, MIRROR_GEOMETRY


def configure_distinct_channels(env):
    """4 distinct channels to avoid co-channel interference (isolation for chain test)."""
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


def test_real_detections_produced():
    """At least 10 real detections over 20 steps × E=2 with distinct channels."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=50, geometry=MIRROR_GEOMETRY)
    env.reset()
    configure_distinct_channels(env)
    action = uniform_action(env)

    total_real = 0
    for step in range(20):
        obs, reward, done, info = env.step(action)
        d = env._last_detections
        real_mask = d.mask & ~d.is_false_alarm
        total_real += int(real_mask.sum().item())

    assert total_real >= 10, f"Too few real detections: {total_real} (expected >=10)"
    print(f"✅ {total_real} real detections over 20 steps × E=2 (>= 10)")


def test_kmax_padding_correct():
    """After packing real+FA into K_max slots, mask must be False for slots >= count.

    Detection packing (detection.py L267-287) iterates count then writes z/mask/is_fa
    at index `count` and increments. Slots beyond `count` must retain default zeros.
    """
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=50, geometry=MIRROR_GEOMETRY,
                        p_fa=1e-3)   # higher P_fa to fill some slots
    env.reset()
    configure_distinct_channels(env)
    action = uniform_action(env)

    for step in range(10):
        env.step(action)
        d = env._last_detections
        # For each (env, team), check that mask is monotonically packed (True then False).
        for e in range(d.mask.shape[0]):
            for t in range(d.mask.shape[1]):
                mask_et = d.mask[e, t]   # [K_max]
                # Find first False after a True — there should be no True after the first False.
                seen_false = False
                for k in range(d.mask.shape[2]):
                    if not mask_et[k].item():
                        seen_false = True
                    elif seen_false:
                        # True after False = padding violation
                        assert False, (
                            f"K_max packing violation: mask[{e},{t},{k}]=True after "
                            f"a False slot. Mask: {mask_et.cpu().tolist()}"
                        )
    print("✅ K_max padding correct (all mask slots packed front-to-back)")


def test_false_alarms_under_stress():
    """p_fa=1e-3 stress produces >=1 false alarm over 30 steps × E=2."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=50, geometry=MIRROR_GEOMETRY,
                        p_fa=1e-3)
    env.reset()
    configure_distinct_channels(env)
    action = uniform_action(env)

    total_fa = 0
    for step in range(30):
        env.step(action)
        d = env._last_detections
        fa_mask = d.mask & d.is_false_alarm
        total_fa += int(fa_mask.sum().item())

    # 30 steps × 2 envs × 2 teams × 84 cells × 1e-3 P_fa ≈ 10 expected FAs
    assert total_fa >= 1, f"No false alarms under p_fa=1e-3 stress (expected >=1, got {total_fa})"
    print(f"✅ {total_fa} false alarms produced under p_fa=1e-3 stress (>= 1)")


def test_real_detection_snr_reasonable():
    """Real detections have SNR > threshold - 2·width (i.e., detection was plausible)."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=50, geometry=MIRROR_GEOMETRY)
    env.reset()
    configure_distinct_channels(env)
    action = uniform_action(env)

    snr_min_threshold = env.detect_threshold_db - 2 * env.detect_width_db

    for step in range(20):
        env.step(action)
        d = env._last_detections
        real_mask = d.mask & ~d.is_false_alarm
        if real_mask.any():
            real_snrs = d.snr_db[real_mask]
            # All real detections should have SNR at least near threshold
            # (P_detect is sigmoid((SNR - thr)/width); SNR < thr - 2·width → P_d < 4.7e-4 → very unlikely)
            min_snr = real_snrs.min().item()
            assert min_snr > snr_min_threshold, (
                f"Real detection with SNR={min_snr:.2f} dB below reasonable threshold "
                f"({snr_min_threshold:.2f} dB = detect_thr - 2·width)"
            )
    print(f"✅ All real detections have SNR > {snr_min_threshold:.1f} dB (thr-2·width)")


def test_real_detection_z_near_truth():
    """Real detection z positions are within 10·σ_range of true enemy position."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=50, geometry=MIRROR_GEOMETRY)
    env.reset()
    configure_distinct_channels(env)
    action = uniform_action(env)

    max_err_m = 50.0   # σ_range at SNR=20dB is ~1.5m; 10·σ = 15m; 50m is very generous
    violations = 0

    for step in range(20):
        env.step(action)
        d = env._last_detections
        # For each (env, team), each real detection should be near an enemy position
        for e in range(d.mask.shape[0]):
            for t in range(d.mask.shape[1]):
                et = 1 - t
                enemy_positions = env.radar_pos[e, et]   # [R, 2]
                for k in range(d.mask.shape[2]):
                    if bool(d.mask[e, t, k]) and not bool(d.is_false_alarm[e, t, k]):
                        z = d.z[e, t, k]   # [2]
                        # Distance to nearest enemy
                        dists = (enemy_positions - z.unsqueeze(0)).norm(dim=-1)   # [R]
                        min_dist = dists.min().item()
                        if min_dist > max_err_m:
                            violations += 1

    assert violations == 0, (
        f"{violations} real detections > {max_err_m}m from any enemy "
        f"(should be within 10·σ_range ≈ 15m)"
    )
    print(f"✅ All real detection z within {max_err_m}m of true enemy position")


if __name__ == "__main__":
    print("=== WP-1 M1: detection chain smoke tests ===")
    print()
    print("--- test_real_detections_produced ---")
    test_real_detections_produced()
    print()
    print("--- test_kmax_padding_correct ---")
    test_kmax_padding_correct()
    print()
    print("--- test_false_alarms_under_stress ---")
    test_false_alarms_under_stress()
    print()
    print("--- test_real_detection_snr_reasonable ---")
    test_real_detection_snr_reasonable()
    print()
    print("--- test_real_detection_z_near_truth ---")
    test_real_detection_z_near_truth()
    print()
    print("🎉 all M1 detection smoke tests PASS")
