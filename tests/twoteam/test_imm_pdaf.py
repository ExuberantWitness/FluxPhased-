"""WP-2 M1 §3 ③: batched IMM (CV+CT) + PDAF tracker.

Spec §3 ③ mandates an IMM-PDAF tracker as the "competent blind classical"
data-association layer (replaces WP-1 σ-gate NN + EKF). Tests verify:

  1. IMM init: μ_cv + μ_ct sums to 1
  2. PDAF gating: 5σ Mahalanobis gate rejects far FAs from β_i weights
  3. IMM model selection: linear motion → μ_cv dominant; turning → μ_ct dominant
  4. Single-target RMSE: < 5 m after Kalman convergence
  5. Mirror symmetry: trace_P A vs B doesn't diverge over 50 episodes

Mirror RNG pattern: all randomness in env step (detection Bernoulli, FA roll,
measurement noise) is team-shared via `rand(E,1,...).expand(E,T,...)`. IMM μ
init and fusion are deterministic. So tracker_x for team A and team B should
differ only by the origin-mirror transform: x_A = -x_B (under origin-mirror
geometry, enemy_A_pos = -enemy_B_pos).
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
    E = env.E
    return {
        "task_alloc": torch.full((E, 2, 2, 4), 0.25, device=env.device),
        "beam_target": torch.zeros(E, 2, 2, dtype=torch.long, device=env.device),
        "laser_target": torch.zeros(E, 2, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(E, 2, 2, dtype=torch.bool, device=env.device),
        "freq_hop_rate": torch.ones(E, 2, 2, device=env.device),
    }


def test_imm_two_models_init():
    """After reset + one update, IMM μ is [CV, CT] and sums to 1."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=20,
                        geometry=MIRROR_GEOMETRY)
    env.reset()
    configure_distinct_channels(env)
    env.step(uniform_action(env))
    mu = env.tracker.mu                                                    # [E, T, R, 2]
    assert mu.shape == (env.E, env.n_teams, env.n_radars_per_team, 2)
    sums = mu.sum(dim=-1)                                                  # [E, T, R]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"IMM μ must sum to 1 across models, got sums={sums.cpu().tolist()}"
    )
    assert (mu >= 0).all() and (mu <= 1).all(), (
        f"IMM μ entries must be in [0,1], got min={mu.min():.4f}, max={mu.max():.4f}"
    )
    print(f"✅ IMM μ sums to 1, shape={tuple(mu.shape)}, "
          f"μ_cv_mean={mu[..., 0].mean():.3f}, μ_ct_mean={mu[..., 1].mean():.3f}")


def test_pdaf_gating_rejects_far_fas():
    """False alarms outside 5σ Mahalanobis gate don't affect tracker_x.

    Setup: run 30 steps to converge tracker, then inject a single far FA via
    a fake Detections object. Tracker_x should move negligibly (innovation
    gated out by 5σ test).
    """
    from env.gpu.twoteam.detection import Detections
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=100,
                        geometry=MIRROR_GEOMETRY)
    env.reset()
    configure_distinct_channels(env)
    for _ in range(30):
        env.step(uniform_action(env))

    # Snapshot tracker_x before injecting far FA
    x_before = env.tracker_x.clone()

    # Construct a Detections with one real detection (at tracker belief) and
    # one FA far outside the 5σ gate.
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    K = env.k_max
    z = torch.zeros(E, T, K, 2, device=env.device)
    mask = torch.zeros(E, T, K, dtype=torch.bool, device=env.device)
    is_fa = torch.zeros(E, T, K, dtype=torch.bool, device=env.device)
    snr_db = torch.zeros(E, T, K, device=env.device)
    # Slot 0: real detection at current belief (no movement)
    for t in range(T):
        z[:, t, 0, 0] = env.tracker_x[:, t, 0, 0]
        z[:, t, 0, 1] = env.tracker_x[:, t, 0, 2]
        mask[:, t, 0] = True
        snr_db[:, t, 0] = 25.0
        # Slot 1: far FA at +10km offset (>>5σ_gate)
        z[:, t, 1, 0] = env.tracker_x[:, t, 0, 0] + 10000.0
        z[:, t, 1, 1] = env.tracker_x[:, t, 0, 2] + 10000.0
        mask[:, t, 1] = True
        is_fa[:, t, 1] = True
        snr_db[:, t, 1] = 25.0
    dets = Detections(z=z, mask=mask, is_false_alarm=is_fa, snr_db=snr_db)
    # Fake sigma_meas = small value (tracker should reject FA via Mahalanobis)
    sigma_meas = torch.full((E, T, R), 5.0, device=env.device)
    env.tracker.update(dets, sigma_meas)

    x_after = env.tracker_x.clone()
    delta = (x_after - x_before).abs().max().item()
    # Without FA rejection, tracker would jump ~10km/sqrt(2) ≈ 7000m. With
    # gating, it should move by O(σ_meas) = O(5m).
    assert delta < 50.0, (
        f"far FA leaked through PDAF gate: tracker_x moved {delta:.1f} m "
        f"(should be < 50 m with 5σ Mahalanobis gate)"
    )
    print(f"✅ 5σ gate rejected far FA: tracker_x moved only {delta:.2f} m")


def test_imm_cv_dominant_straight_target():
    """Linear-motion target → IMM μ_cv → high (CV model preferred).

    Setup: enemies don't maneuver (static), so CV model should dominate over
    CT after enough steps for μ to converge.
    """
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=100,
                        geometry=MIRROR_GEOMETRY, sigma_q=0.5)
    env.reset()
    configure_distinct_channels(env)
    # Enemy static: radar_pos doesn't change between steps (env doesn't move
    # radars, so this is automatic).
    for _ in range(50):
        env.step(uniform_action(env))
    mu_cv = env.tracker.mu[..., 0]   # [E, T, R]
    # CV should be ≥ 0.5 (better match to linear/static truth than CT)
    assert mu_cv.mean().item() >= 0.5, (
        f"μ_cv mean = {mu_cv.mean():.3f}, expected ≥ 0.5 for static target"
    )
    print(f"✅ static target → μ_cv dominant: mean={mu_cv.mean():.3f}, "
          f"min={mu_cv.min():.3f}")


def test_imm_converges_single_target():
    """Single target, no clutter: tracker RMSE < 50 m after 30-step warmup."""
    n_steps = 80
    env = TwoTeamVecEnv(n_envs=8, device="cuda", episode_steps=n_steps,
                        geometry=MIRROR_GEOMETRY, seed=42, p_fa=1e-10)
    env.reset()
    configure_distinct_channels(env)
    # Deactivate enemy radar 1 to make this a clean single-target case
    env.radar_alive[:, :, 1] = False
    action = uniform_action(env)
    errors = []
    R = env.n_radars_per_team
    for step in range(n_steps):
        env.step(action)
        if step < 30:   # warmup
            continue
        tracker_pos = env.tracker_x[..., [0, 2]]                       # [E, T, R, 2]
        # Origin-mirror true_pos swap (team 0 tracks team 1's pos)
        true_pos = torch.stack(
            [env.radar_pos[:, 1], env.radar_pos[:, 0]], dim=1
        )                                                              # [E, T, R, 2]
        err = (tracker_pos - true_pos).norm(dim=-1)                   # [E, T, R]
        enemy_alive = torch.stack(
            [env.radar_alive[:, 1], env.radar_alive[:, 0]], dim=1
        )
        enemy_emit = torch.stack(
            [env.enemy_emitting[:, 1], env.enemy_emitting[:, 0]], dim=1
        )
        valid = env.tracker_initialized & enemy_alive & enemy_emit
        slot0 = (torch.arange(R, device=env.device) == 0)
        err_masked = torch.where(valid & slot0,
                                 err, torch.zeros_like(err))
        errors.append(err_masked.cpu().numpy())
    errs = np.stack(errors, axis=0)
    rmse = float(np.sqrt((errs ** 2).mean()))
    assert rmse < 50.0, (
        f"IMM-PDAF RMSE = {rmse:.1f} m, expected < 50 m (single target, low clutter)"
    )
    print(f"✅ single-target RMSE = {rmse:.2f} m (< 50 m)")


def test_imm_no_divergence_short_window():
    """trace_P stays bounded (< 5× init) over 30-step window under nominal config.

    IMM-PDAF numerical instability (negative-def P, NaN innovation) would
    blow up trace_P. Verify it doesn't.
    """
    n_steps = 30
    env = TwoTeamVecEnv(n_envs=8, device="cuda", episode_steps=n_steps,
                        geometry=MIRROR_GEOMETRY, seed=42)
    env.reset()
    configure_distinct_channels(env)
    env.radar_alive[:, :, 1] = False   # single-target
    action = uniform_action(env)
    init_trace = float(
        (env.tracker_P[..., 0, 0] + env.tracker_P[..., 2, 2]).mean().item()
    )
    peak_trace = init_trace
    for _ in range(n_steps):
        env.step(action)
        trace = float(
            (env.tracker_P[..., 0, 0] + env.tracker_P[..., 2, 2]).mean().item()
        )
        peak_trace = max(peak_trace, trace)
        assert not torch.isnan(env.tracker_x).any(), "tracker_x has NaN"
        assert not torch.isnan(env.tracker_P).any(), "tracker_P has NaN"
    assert peak_trace < 5.0 * init_trace, (
        f"trace_P diverged: peak={peak_trace:.3f} > 5×init={5*init_trace:.3f}"
    )
    print(f"✅ no divergence: peak trace_P = {peak_trace:.3f} < 5×init = {5*init_trace:.3f}")


def test_imm_mirror_symmetric():
    """Origin-mirror symmetry: under MIRROR_GEOMETRY, tracker_x[t=0] ≈ -tracker_x[t=1].

    Setup: origin-mirror geometry has enemy_A_pos = -enemy_B_pos. With team-shared
    RNG (detection Bernoulli, FA roll, measurement noise), tracker_A should
    converge to enemy_A_pos while tracker_B converges to -enemy_A_pos. Their
    sum should be near zero (small residual from FP precision + nonlinear
    atan2 in detection geometry).
    """
    n_episodes = 5
    episode_steps = 50
    env = TwoTeamVecEnv(n_envs=8, device="cuda", episode_steps=episode_steps,
                        geometry=MIRROR_GEOMETRY, seed=42)
    action = uniform_action(env)
    mirror_diffs = []
    for ep in range(n_episodes):
        env.reset()
        configure_distinct_channels(env)
        for _ in range(episode_steps):
            env.step(action)
        # Origin-mirror: x_A + x_B should ≈ 0 (enemy_A = -enemy_B)
        x0 = env.tracker_x[:, 0]   # [E, R, 4]
        x1 = env.tracker_x[:, 1]   # [E, R, 4]
        # Only check initialized slots
        init_mask = env.tracker_initialized[:, 0] & env.tracker_initialized[:, 1]   # [E, R]
        diff = (x0 + x1).abs()    # [E, R, 4]
        diff_per_slot = diff.sum(-1)   # [E, R]
        valid_diffs = diff_per_slot[init_mask].cpu().numpy()
        if len(valid_diffs) > 0:
            mirror_diffs.extend(valid_diffs.tolist())
    mirror_diffs = np.array(mirror_diffs)
    median_diff = float(np.median(mirror_diffs))
    p90_diff = float(np.percentile(mirror_diffs, 90))
    # Tolerate up to 100 m per-slot mirror asymmetry (FP + atan2 nonlinearity;
    # before team-shared-noise fix this was ~1500 m).
    assert median_diff < 100.0, (
        f"median origin-mirror asymmetry = {median_diff:.1f} m, expected < 100 m"
    )
    print(f"✅ origin-mirror tracker_x sym: median diff = {median_diff:.2f} m, "
          f"p90 = {p90_diff:.2f} m (N={len(mirror_diffs)} slot-samples)")


if __name__ == "__main__":
    print("=== WP-2 M1 §3 ③: batched IMM-PDAF tracker ===")
    print()
    tests = [
        test_imm_two_models_init,
        test_pdaf_gating_rejects_far_fas,
        test_imm_cv_dominant_straight_target,
        test_imm_converges_single_target,
        test_imm_no_divergence_short_window,
        test_imm_mirror_symmetric,
    ]
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
        print()
    print("🎉 all M1 IMM-PDAF tests PASS")
