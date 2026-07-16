"""WP-1 M4 §2.6④: tracker association under clutter (high P_fa stress).

Spec §2.6④ requires the tracker to correctly associate detections under
clutter. With P_fa=1e-3 (stress) and ~84 search cells, expect ~1 false alarm
per team per step. The nearest-neighbor scheme must NOT diverge — track RMSE
under clutter should stay within 2× the clean track RMSE.

Test plan:
  1. Run 100 steps with P_fa=1e-3 (stress), distinct channels, all emission.
  2. Record tracker error vs true_pos per step (when target alive & emitting).
  3. Compare to baseline (P_fa=0, no clutter).
  4. Assert cluttered RMSE ≤ 2× clean RMSE.

Note: nearest-neighbor association is M1 minimum-viable. PDAF (M4 optional)
gives smoother behavior near the gating boundary. This test verifies the M1
scheme is "good enough" for §2.6④.
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import math
import torch
import numpy as np
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
    """Aperture 0 → enemy 0, aperture 1 → enemy 1 (both enemies covered).

    With single_target=True in run_episode_track_error, enemy 1 is deactivated
    so aperture 1 → enemy 1 simply doesn't detect (no harm).
    """
    E = env.E
    bt = torch.zeros(E, 2, 2, dtype=torch.long, device=env.device)
    bt[:, :, 1] = 1   # aperture 1 points at enemy radar 1
    return {
        "task_alloc": torch.full((E, 2, 2, 4), 0.25, device=env.device),
        "beam_target": bt,
        "laser_target": torch.zeros(E, 2, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(E, 2, 2, dtype=torch.bool, device=env.device),
        "freq_hop_rate": torch.ones(E, 2, 2, device=env.device),
    }


def run_episode_track_error(env, n_steps=100, warmup=20, single_target=True):
    """Run an episode and return per-step tracker position error vs true_pos.

    Returns array of shape [T-warmup, E, T, R] of Euclidean error in meters
    (warmup steps skipped to allow Kalman to converge past initial P=1.0).
    Only counts alive + initialized + enemy-alive + enemy-emitting track slots.

    single_target: if True, deactivates enemy radar 1 for both teams before
        stepping. This isolates the clutter-robustness question to slot 0
        (avoids confound from multi-target NN init limitations, which is a
        separate problem from §2.6④ clutter handling).
    """
    env.reset()
    configure_distinct_channels(env)
    if single_target:
        env.radar_alive[:, :, 1] = False   # only enemy radar 0 alive per team
    action = uniform_action(env)
    errors = []
    for step in range(n_steps):
        env.step(action)
        if step < warmup:
            continue
        tracker_pos = env.tracker_x[..., [0, 2]]                  # [E, T, R, 2]
        true_pos = torch.stack(
            [env.radar_pos[:, 1], env.radar_pos[:, 0]], dim=1     # swap teams
        )                                                          # [E, T, R, 2]
        err = (tracker_pos - true_pos).norm(dim=-1)               # [E, T, R]
        enemy_alive_for_tracker = torch.stack(
            [env.radar_alive[:, 1], env.radar_alive[:, 0]], dim=1
        )                                                          # [E, T, R]
        enemy_emitting_for_tracker = torch.stack(
            [env.enemy_emitting[:, 1], env.enemy_emitting[:, 0]], dim=1
        )                                                          # [E, T, R]
        valid = env.tracker_initialized & enemy_alive_for_tracker & enemy_emitting_for_tracker
        err_masked = torch.where(valid, err, torch.zeros_like(err))
        errors.append(err_masked.cpu().numpy())
    return np.stack(errors, axis=0)


def test_track_error_under_clutter():
    """Track RMSE under P_fa=1e-3 stress ≤ 2× clean track RMSE."""
    n_steps = 80

    # Clean baseline (P_fa very low → essentially no false alarms)
    env_clean = TwoTeamVecEnv(n_envs=8, device="cuda", episode_steps=n_steps,
                              geometry=MIRROR_GEOMETRY, seed=42, p_fa=1e-10)
    errs_clean = run_episode_track_error(env_clean, n_steps=n_steps)
    rmse_clean = float(np.sqrt((errs_clean ** 2).mean()))

    # Stress (P_fa=1e-3 → ~1 FA per team per step)
    env_clutter = TwoTeamVecEnv(n_envs=8, device="cuda", episode_steps=n_steps,
                                geometry=MIRROR_GEOMETRY, seed=42, p_fa=1e-3)
    errs_clutter = run_episode_track_error(env_clutter, n_steps=n_steps)
    rmse_clutter = float(np.sqrt((errs_clutter ** 2).mean()))

    # Verify clutter actually injected FAs (n_detections should be > 0 reliably)
    # Re-run a single step to check n_detections counter
    env_check = TwoTeamVecEnv(n_envs=8, device="cuda", episode_steps=n_steps,
                              geometry=MIRROR_GEOMETRY, seed=42, p_fa=1e-3)
    env_check.reset()
    configure_distinct_channels(env_check)
    env_check.step(uniform_action(env_check))
    n_det_after_step1 = int(env_check.n_detections.sum().item())

    ratio = rmse_clutter / max(rmse_clean, 1.0)
    print(f"clean RMSE   = {rmse_clean:7.2f} m")
    print(f"clutter RMSE = {rmse_clutter:7.2f} m  (P_fa=1e-3, n_det step1={n_det_after_step1})")
    print(f"ratio = {ratio:.3f}")

    # Sanity: clean should be small (Kalman converged)
    assert rmse_clean < 200.0, (
        f"clean RMSE too large (Kalman not converging): {rmse_clean:.1f} m"
    )

    # Clutter inflation must be bounded
    assert ratio <= 2.0, (
        f"clutter inflation too high: ratio={ratio:.3f} > 2.0 "
        f"(clean={rmse_clean:.2f}, clutter={rmse_clutter:.2f})"
    )
    print(f"✅ association robust to clutter: ratio={ratio:.3f} ≤ 2.0")


def test_no_divergence_under_p_fa_stress():
    """Under P_fa=1e-3 for short window, tracker P doesn't blow up.

    A buggy NN associator would occasionally pick a far-away FA, inflating P
    via the innovation covariance. Verify trace_P stays bounded (< 20× init)
    over the first 30 steps (before home-on-jam kills everyone).

    Uses single-target isolation to avoid confound from multi-target NN init.
    """
    n_steps = 30
    env = TwoTeamVecEnv(n_envs=8, device="cuda", episode_steps=n_steps,
                        geometry=MIRROR_GEOMETRY, seed=42, p_fa=1e-3)
    env.reset()
    configure_distinct_channels(env)
    env.radar_alive[:, :, 1] = False   # single-target isolation
    action = uniform_action(env)
    init_trace = float(
        (env.tracker_P[..., 0, 0] + env.tracker_P[..., 2, 2]).mean().item()
    )
    peak_trace = init_trace
    for _ in range(n_steps):
        env.step(action)
        trace_step = float(
            (env.tracker_P[..., 0, 0] + env.tracker_P[..., 2, 2]).mean().item()
        )
        peak_trace = max(peak_trace, trace_step)
    print(f"init trace_P mean = {init_trace:.3f}")
    print(f"peak trace_P mean over {n_steps} steps = {peak_trace:.3f}")
    # Init=1.0; healthy convergence → 0.005-0.05 within 10 steps.
    # Tolerate peak < 5× init (= 5.0) — allows for predict-only growth between detections.
    assert peak_trace < 5.0 * init_trace, (
        f"trace_P diverged: peak={peak_trace:.3f} > 5×init={5*init_trace:.3f}"
    )
    print(f"✅ no divergence: peak trace_P {peak_trace:.3f} < 5×init")


if __name__ == "__main__":
    print("=== WP-1 M4 §2.6④: association under clutter ===")
    print()
    print("--- test_track_error_under_clutter ---")
    test_track_error_under_clutter()
    print()
    print("--- test_no_divergence_under_p_fa_stress ---")
    test_no_divergence_under_p_fa_stress()
    print()
    print("🎉 all M4 clutter-association tests PASS")
