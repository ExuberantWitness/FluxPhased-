"""WP-2 M3: BlindClassical capability proof + failure mode.

Spec §3 ③ hard requirement:
  ① low-interference + enemy always emitting → BlindClassical searches, detects,
     tracks, kills (kill_rate ≥ 0.5 — at least 1 enemy killed per episode)
  ② high-interference (heavy jam + duty cycle + same channel + clutter) →
     BlindClassical track breaks, kill chain collapses (kill_rate ≤ 0.3, and
     ≤ ½ of low-interference kill rate)
  ③ if high-interference still produces kills → interference isn't strong
     enough; spec mandate to加大

Scenario:
  - Team A (subject): BlindClassicalCommander
  - Team B (adversary): fixed action — jam_fraction + duty cycle + channel
  - 20 episodes × 200 steps per cell
  - Compare kill_rate(A) in low vs high interference
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import math
import torch
import numpy as np
from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


def make_env(n_envs=8, episode_steps=200, p_fa=1e-6):
    env = TwoTeamVecEnv(n_envs=n_envs, device="cuda", episode_steps=episode_steps,
                        geometry=MIRROR_GEOMETRY, seed=42, p_fa=p_fa)
    env.reset()
    return env


def set_channels(env, mode: str):
    """Configure per-radar frequencies.

    'orthogonal': each radar on its own channel (no intra- or cross-team overlap)
    'same': all radars on channel 0 (worst-case co-channel interference)
    """
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    freqs = torch.zeros(E, T, R, device=env.device)
    fc = env.fc_hz
    if mode == "orthogonal":
        for e in range(E):
            freqs[e, 0, 0] = fc
            freqs[e, 0, 1] = fc + env.channel_spacing_hz
            freqs[e, 1, 0] = fc + 2 * env.channel_spacing_hz
            freqs[e, 1, 1] = fc + 3 * env.channel_spacing_hz
    else:   # 'same'
        freqs[:] = fc
    env.set_radar_freqs(freqs)


def adversary_action(env, jam_fraction: float, duty_on_fraction: float,
                     emission_pattern_step: int):
    """Fixed adversary (team B) action — jam-heavy task_alloc + duty-cycle emit.

    jam_fraction: 0.0 to 0.6 — fraction of aperture power to jam (vs detect/track)
    duty_on_fraction: 0.0 to 1.0 — fraction of steps adversary emits
    emission_pattern_step: current step count (for duty cycle phase)
    """
    E = env.E
    R = env.n_radars_per_team
    dev = env.device
    # Task alloc: [detect, track, jam, comm]
    # Adversary focus: jam_fraction to jam, rest split among track + comm + detect
    remaining = max(1e-3, 1.0 - jam_fraction - 0.10)   # reserve 0.10 for comm
    alloc = torch.tensor(
        [0.05, remaining * 0.7, jam_fraction, 0.10], device=dev
    ).expand(E, R, 4).clone()
    # Normalize (safety)
    alloc = alloc / alloc.sum(dim=-1, keepdim=True)

    # Beam direction: aim at team A's centroid (approximate — adversary has
    # its own tracker but for test purposes use a fixed az)
    own_pos_B = env.radar_pos[:, 1]                                      # [E, R, 2]
    enemy_pos_A = env.radar_pos[:, 0]                                    # [E, R, 2]
    # Aim aperture 0 at A0, aperture 1 at A1
    beam_az = torch.zeros(E, R, device=dev)
    for k in range(R):
        delta = enemy_pos_A[:, k] - own_pos_B[:, k]
        beam_az[:, k] = torch.atan2(delta[:, 1], delta[:, 0])

    # Duty cycle: emit only during ON phase
    period = 5   # 5-step period
    on_steps = max(1, int(round(period * duty_on_fraction)))
    is_on = (emission_pattern_step % period) < on_steps
    emit_val = 1.0 if is_on else 0.0
    emission_on = torch.full((E, R), emit_val, device=env.device)

    # Laser: fire at slot 0 (BlindClassical A's tracker slot 0 may be tracking B0)
    laser_target = torch.zeros(E, dtype=torch.long, device=dev)

    return {
        "task_alloc": alloc,
        "beam_direction": beam_az,
        "laser_target": laser_target,
        "emission_on": emission_on,
        "freq_hop_rate": torch.ones(E, R, device=dev),
    }


def run_episodes(env, cmd, jam_fraction: float, duty_on_fraction: float,
                 n_episodes: int):
    """Run n_episodes episodes; return per-episode team-A kills, trace_P, search_cov."""
    kills_A = []
    trace_P_final = []
    search_cov_final = []
    track_active_frac = []
    for ep in range(n_episodes):
        env.reset()
        set_channels(env, "orthogonal" if jam_fraction < 0.2 else "same")
        ep_track_active = []
        for step in range(env.episode_steps):
            a_A = cmd.get_action(env, team=0)
            a_B = adversary_action(env, jam_fraction, duty_on_fraction, step)
            action = combine_team_actions(env, a_A, a_B)
            obs, reward, done, info = env.step(action)
            # Track active: fraction of slots with trace_P < tau_track AND init'd
            trace_P_slots = env.tracker_P[:, 0, :, 0, 0] + env.tracker_P[:, 0, :, 2, 2]
            active = ((trace_P_slots < env.tau_track) & env.tracker_initialized[:, 0]).float()
            ep_track_active.append(active.mean().item())
            if done.all():
                break
        kills_A.append(info["team_kills"][:, 0].cpu().numpy())
        trace_P_final.append(info["mean_trace_P"][:, 0].cpu().numpy())
        search_cov_final.append(env.search_coverage[:, 0].cpu().numpy())
        track_active_frac.append(np.mean(ep_track_active))
    return {
        "kills_A": np.concatenate(kills_A),
        "trace_P": np.concatenate(trace_P_final),
        "search_cov": np.concatenate(search_cov_final),
        "track_active_frac": np.array(track_active_frac),
    }


def test_blind_classical_low_interference():
    """Low-interference: adversary pure-track + always emitting + orthogonal ch.

    Expect: BlindClassical(A) kills ≥ 0.5 per episode on average.
    """
    n_episodes = 20
    env = make_env(n_envs=8, episode_steps=200, p_fa=1e-6)
    cmd = BlindClassicalCommander()
    result = run_episodes(env, cmd, jam_fraction=0.0, duty_on_fraction=1.0,
                          n_episodes=n_episodes)
    kill_rate = float(result["kills_A"].mean())
    track_active = float(result["track_active_frac"].mean())
    search_cov = float(result["search_cov"].mean())
    print(f"low-interference: kill_rate={kill_rate:.3f}, "
          f"track_active={track_active:.3f}, search_cov={search_cov:.3f}")
    assert kill_rate >= 0.5, (
        f"BlindClassical kill_rate = {kill_rate:.3f} < 0.5 in low interference — "
        f"capability proof failed (must search→detect→track→kill)"
    )
    print(f"✅ low-interference kill_rate = {kill_rate:.3f} ≥ 0.5")


def test_blind_classical_high_interference():
    """High-interference: adversary heavy jam + duty 40% + same channel + clutter.

    Expect: kill_rate ≤ 0.3 AND ≤ ½ of low-interference kill_rate.
    If still kills well → spec §3 ③ demands加大 interference.
    """
    n_episodes = 20
    # First get baseline from low-interference
    env_low = make_env(n_envs=8, episode_steps=200, p_fa=1e-6)
    cmd = BlindClassicalCommander()
    result_low = run_episodes(env_low, cmd, jam_fraction=0.0, duty_on_fraction=1.0,
                              n_episodes=n_episodes)
    kill_rate_low = float(result_low["kills_A"].mean())

    # Now stress
    env_high = make_env(n_envs=8, episode_steps=200, p_fa=1e-3)
    result_high = run_episodes(env_high, cmd, jam_fraction=0.45,
                               duty_on_fraction=0.6, n_episodes=n_episodes)
    kill_rate_high = float(result_high["kills_A"].mean())
    track_active_high = float(result_high["track_active_frac"].mean())
    search_cov_high = float(result_high["search_cov"].mean())
    print(f"high-interference: kill_rate={kill_rate_high:.3f}, "
          f"track_active={track_active_high:.3f}, search_cov={search_cov_high:.3f}")
    print(f"low kill_rate = {kill_rate_low:.3f}, high = {kill_rate_high:.3f}, "
          f"ratio = {kill_rate_high/max(kill_rate_low, 1e-3):.3f}")
    assert kill_rate_high <= 0.3, (
        f"BlindClassical kill_rate = {kill_rate_high:.3f} > 0.3 under heavy jam — "
        f"spec §3 ③ requires kill collapse; interference may be too weak"
    )
    assert kill_rate_high <= 0.5 * kill_rate_low, (
        f"high kill_rate ({kill_rate_high:.3f}) > ½ low ({kill_rate_low:.3f}) — "
        f"interference gradation insufficient"
    )
    print(f"✅ high-interference kill collapse: {kill_rate_high:.3f} ≤ 0.3 "
          f"and ≤ ½ low ({kill_rate_low:.3f})")


if __name__ == "__main__":
    print("=== WP-2 M3: BlindClassical interference gradation ===")
    print()
    print("--- test_blind_classical_low_interference ---")
    test_blind_classical_low_interference()
    print()
    print("--- test_blind_classical_high_interference ---")
    test_blind_classical_high_interference()
    print()
    print("🎉 all M3 interference-gradation tests PASS")
