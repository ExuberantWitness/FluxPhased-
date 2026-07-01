"""Direct test of Kalman fused_sensing bias.

Builds a real MFARVecEnv, runs fused_sensing on the actual commander obs,
and measures the distance between the Kalman-fused enemy position (obs[68:72])
and the true enemy position (env.radar_pos).

Three configurations tested:
  1. fused-only (no tracking) — info-filter of 2 own radars
  2. tracked, first call (warm-start triggers) — should be tight
  3. tracked, second call (no reset, like prod bug) — measures the "stale state" path

If bias > 10m for any config, that's the bug.
"""
import os
import sys
import math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.gpu.vec_mfar_env import MFARVecEnv
from algo._shared.laser.sensing import fused_sensing, KalmanTracker


def get_commander_obs(env):
    """Pull the commander obs from the env (mirrors vec_drone.get_commander_obs)."""
    radar_latents = env.battlefield.drone.get_radar_latents(env.radar_pos) \
        if hasattr(env.battlefield.drone, "get_radar_latents") else None
    if radar_latents is None:
        # Fall back to zero latents (we only care about positions for sensing test)
        radar_latents = torch.zeros(env.num_envs, env.n_radars, 32, device=env.device)
    return env.battlefield.drone.get_commander_obs(env.radar_pos, radar_latents)


def measure_bias(mode: str, n_episodes: int = 5, reset_each_ep: bool = True):
    """Run env resets, apply fused_sensing at step 0, measure (fused - true) distance.

    reset_each_ep=True simulates the FIXED training path (trainer.reset_episode
    is called per ep). False simulates the BUG (no reset → KF carries stale state).
    """
    env = MFARVecEnv(
        num_envs=4,
        n_radars=4,
        rows=25, cols=25,
        fc=10e9,
        bandwidth=200e6,
        prf=10e3,
        pulses_per_cpi=4,
        fft_size=64,
        tx_power_w=1.0,
        n_teams=2,
        device="cuda",
        kill_radius_m=0.2,
        illumination_time_s=0.002,
        drone_altitude_m=3000.0,
        map_size=[20000.0, 20000.0],
        vehicle_speed_ms=20.0,
    )

    half_x = float(env.map_size[0]) / 2.0
    half_y = float(env.map_size[1]) / 2.0

    from algo._shared.laser.sensing import enforce_radar_baseline
    enforce_radar_baseline(env, min_baseline_m=5000.0)

    range_sigma_m = 0.05
    crossrange_factor = 7.4e-5
    track = mode == "tracked"
    tracker = KalmanTracker(track_q_m=0.02, track_burnin=120, acq_baseline_m=0.0) \
        if track else None

    all_err = []

    for ep in range(n_episodes):
        env.reset()
        enforce_radar_baseline(env, min_baseline_m=5000.0)
        if tracker is not None and reset_each_ep:
            tracker.reset()

        obs = get_commander_obs(env)
        true_e0_x = obs[:, :, 68].clone() * half_x
        true_e0_y = obs[:, :, 69].clone() * half_y

        obs_noisy = fused_sensing(
            obs.clone(), half_x, half_y,
            range_sigma_m, crossrange_factor,
            tracker=tracker,
            jam_gain=0.0, exposure_gain=0.0, jam_level=None,
        )
        fused_e0_x = obs_noisy[:, :, 68] * half_x
        fused_e0_y = obs_noisy[:, :, 69] * half_y

        err_x = fused_e0_x - true_e0_x
        err_y = fused_e0_y - true_e0_y
        err = torch.sqrt(err_x ** 2 + err_y ** 2)
        all_err.append(float(err.mean().item()))

        print(f"  [{mode} reset={reset_each_ep}] ep {ep}: "
              f"step-0 fused err mean {err.mean().item():.3f}m, "
              f"max {err.max().item():.3f}m", flush=True)

    return {
        "mode": mode,
        "reset_each_ep": reset_each_ep,
        "mean_err_m": float(np.mean(all_err)),
        "max_err_m": float(np.max(all_err)),
        "first_ep_err_m": all_err[0],
        "later_ep_mean_err_m": float(np.mean(all_err[1:])) if len(all_err) > 1 else float("nan"),
    }


def main():
    print("=" * 70)
    print("KALMAN FUSED_SENSING BIAS TEST")
    print("=" * 70)
    print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
    print()

    for mode in ["fused", "tracked"]:
        for reset_each_ep in [True, False]:
            print(f">>> Mode: {mode}, reset_each_ep: {reset_each_ep}")
            r = measure_bias(mode=mode, n_episodes=5, reset_each_ep=reset_each_ep)
            print(f"  mean_err_m:           {r['mean_err_m']:.3f} m")
            print(f"  first_ep_err_m:       {r['first_ep_err_m']:.3f} m")
            print(f"  later_ep_mean_err_m:  {r['later_ep_mean_err_m']:.3f} m")
            print()
            if reset_each_ep:
                if r["mean_err_m"] > 10.0:
                    print(f"  >>> BUG: with reset, error still > 10m — Kalman broken")
                elif r["mean_err_m"] > 1.0:
                    print(f"  >>> SUSPICIOUS: with reset, error 1-10m")
                else:
                    print(f"  >>> OK: with reset, error < 1m — Kalman converges")
            else:
                if r["later_ep_mean_err_m"] > 100.0:
                    print(f"  >>> STALE-STATE BUG confirmed: without reset, "
                          f"later-ep error = {r['later_ep_mean_err_m']:.0f}m")
            print()


if __name__ == "__main__":
    main()
