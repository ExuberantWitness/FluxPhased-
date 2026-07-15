"""WP-C D1 — Verify truly dynamic enemy > fixed enemy (blocker for R3).

Diagnosis of R2 0/3 fail:
  R2's "fixed" jammer read env.radar_freq_hz[:, team] for channel_select.
  Under mirror orth config (victim radar 0 = ch0, radar 1 = ch1), the
  "fixed" enemy landed on the SAME [ch0, ch1] as the victim → it was
  channel-following without explicit logic. The "reactive" enemy did the
  same with one channel concentrated. Apple-to-apple, both ended up on
  the victim's channels → no difference.

D1 fix:
  1. TRUE FIXED: both enemy radars stay on ch0 regardless of victim
     (constant channel jammer). Victim radar on ch1 is free → baseline
     can still track with one radar.
  2. ADAPTIVE SPECTRUM (channel-split follower): enemy radar i → victim
     radar i's channel + beam. Both victim channels jammed simultaneously.
     Victim cannot escape by re-allocating one radar.

PASS criterion: at jam_fraction = 1e-4, AdaptiveSpectrum raises competent
baseline trace_P by ≥ 50% vs TrueFixed, OR reduces kills by ≥ 0.3.
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import argparse
import math
import torch
import numpy as np

from env.gpu.twoteam import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.baselines.adaptive_spectrum_jammer import (
    AdaptiveSpectrumJammer, TrueFixedJammer,
)
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


def configure_channels(env, mode: str):
    """StrongRule + orth wrapper: ch0 / ch1 mirror-symmetric."""
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    dev = env.device
    freqs = torch.zeros(E, T, R, device=dev)
    ch_spacing = env.channel_spacing_hz
    fc = env.fc_hz
    for e in range(E):
        if mode == "orthogonal":
            freqs[e, 0, 0] = fc
            freqs[e, 0, 1] = fc + ch_spacing
            freqs[e, 1, 0] = fc
            freqs[e, 1, 1] = fc + ch_spacing
        else:
            freqs[e, 0, :] = fc
            freqs[e, 1, :] = fc
    env.set_radar_freqs(freqs)


def run_scenario(jam_fraction: float, enemy_type: str,
                 n_episodes: int, n_envs: int, episode_steps: int,
                 team_offset_m: float = 2500.0):
    """Team A = enemy (true-fixed or adaptive-split), Team B = StrongRule + orth."""
    env = TwoTeamVecEnv(
        n_envs=n_envs, device="cuda",
        episode_steps=episode_steps,
        geometry=MIRROR_GEOMETRY,
        team_offset_m=team_offset_m,
    )

    episodes_done = 0
    metrics = {
        "mean_trace_P": [],
        "kill_capacity": [],
        "team_a_kills": [],
        "jnr_at_victim": [],
    }

    while episodes_done < n_episodes:
        env.reset()
        configure_channels(env, "orthogonal")
        rule_b = TwoTeamStrongRuleCommander()

        if enemy_type == "fixed":
            enemy_a = TrueFixedJammer(jam_fraction=jam_fraction)
        elif enemy_type == "adaptive":
            enemy_a = AdaptiveSpectrumJammer(jam_fraction=jam_fraction)
        else:
            raise ValueError(f"unknown enemy_type: {enemy_type}")

        ep_trace_P_sum = torch.zeros(env.E, device=env.device)
        ep_jnr_sum = torch.zeros(env.E, device=env.device)
        ep_step_count = 0
        last_info = None

        for step in range(episode_steps):
            a_t0 = enemy_a.get_action(env, team=0)
            a_t1 = rule_b.get_action(env, team=1)
            action = combine_team_actions(env, a_t0, a_t1)

            obs, reward, done, info = env.step(action)
            last_info = info

            # IQ JNR cross-ref
            jnr_mat = env.iq.compute_jnr_matrix(
                pos=env.radar_pos, beam_az=env.radar_beam_az,
                alloc=action["task_alloc"], freq_hz=env.radar_freq_hz,
                emission_on=action["emission_on"],
                hop_rate=action.get("freq_hop_rate",
                                    torch.ones(env.E, env.n_teams, env.n_radars_per_team,
                                               device=env.device)),
                radar_alive=env.radar_alive,
            )
            jnr_at_victim = jnr_mat[:, [0, 1], 2].sum(dim=-1)
            ep_jnr_sum += jnr_at_victim

            # trace_P for victim team B (team 1) on alive radars
            trace_P = env.tracker_P[:, 1, :, 0, 0] + env.tracker_P[:, 1, :, 2, 2]
            enemy_alive = env.radar_alive[:, 0]
            alive_mask = enemy_alive.float()
            n_alive = alive_mask.sum(dim=-1).clamp(min=1.0)
            ep_trace_P_sum += (trace_P * alive_mask).sum(dim=-1) / n_alive

            ep_step_count += 1
            if done.all():
                break

        kills_b = last_info["team_kills"][:, 1].float()
        kills_a = last_info["team_kills"][:, 0].float()
        for e in range(env.E):
            if episodes_done >= n_episodes:
                break
            sc = max(ep_step_count, 1)
            metrics["mean_trace_P"].append(ep_trace_P_sum[e].item() / sc)
            metrics["kill_capacity"].append(kills_b[e].item())
            metrics["team_a_kills"].append(kills_a[e].item())
            metrics["jnr_at_victim"].append(ep_jnr_sum[e].item() / sc)
            episodes_done += 1

    return {k: np.array(v) for k, v in metrics.items()}


# ---------- main -------------------------------------------------------------

F_EMIT_GRID = [1e-6, 1e-5, 1e-4, 1e-3]
QUICK_F_EMIT_GRID = [1e-5, 1e-4, 1e-3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=32)
    args = p.parse_args()

    f_emit_grid = QUICK_F_EMIT_GRID if args.quick else F_EMIT_GRID
    n_ep = 20 if args.quick else args.episodes

    print(f"=== WP-C D1: Dynamic enemy verify (true-fixed vs adaptive-split) ===")
    print(f"  tau_track      = 4.0 (σ_pos = 2m)")
    print(f"  baseline       = StrongRule + orth channel config (competent)")
    print(f"  enemy fixed    = TrueFixedJammer (both radars on ch0)")
    print(f"  enemy adaptive = AdaptiveSpectrumJammer (split-follow victim)")
    print(f"  apple-to-apple = same jam_fraction (= f_emit_A)")
    print(f"  geometry       = MIRROR, team_offset=2500m")
    print(f"  episodes       = {n_ep} × {args.episode_steps} steps per scenario")
    print()

    results = {}
    for f_emit in f_emit_grid:
        for enemy in ["fixed", "adaptive"]:
            m = run_scenario(jam_fraction=f_emit, enemy_type=enemy,
                             n_episodes=n_ep, n_envs=args.n_envs,
                             episode_steps=args.episode_steps)
            results[(f_emit, enemy)] = m
            jnr_dB = 10 * math.log10(max(m["jnr_at_victim"].mean(), 1e-15))
            print(f"[jam_frac={f_emit:.0e} enemy={enemy:>9}] "
                  f"JNR_victim={jnr_dB:>6.1f}dB "
                  f"trace_P={m['mean_trace_P'].mean():>8.3f} "
                  f"kills_B={m['kill_capacity'].mean():.2f} "
                  f"kills_A={m['team_a_kills'].mean():.2f}")

    # ----- degradation comparison table -----
    print()
    print("=" * 100)
    print("D1 DEGRADATION TABLE — competent baseline vs {true-fixed, adaptive-split}")
    print("=" * 100)
    print(f"{'jam_frac':>9} | {'fixed_trP':>10} {'adapt_trP':>10} {'trP_ratio':>10} | "
          f"{'fixed_k':>8} {'adapt_k':>8} {'kills_Δ':>8}")
    print("-" * 90)
    for f_emit in f_emit_grid:
        trP_f = results[(f_emit, "fixed")]["mean_trace_P"].mean()
        trP_a = results[(f_emit, "adaptive")]["mean_trace_P"].mean()
        k_f = results[(f_emit, "fixed")]["kill_capacity"].mean()
        k_a = results[(f_emit, "adaptive")]["kill_capacity"].mean()
        ratio = trP_a / max(trP_f, 1e-9)
        print(f"{f_emit:>9.0e} | {trP_f:>10.3f} {trP_a:>10.3f} {ratio:>9.2f}× | "
              f"{k_f:>8.2f} {k_a:>8.2f} {k_a-k_f:>+8.2f}")

    # ----- PASS criterion -----
    print()
    print("=" * 100)
    print("D1 VERDICT — adaptive-split enemy harder than true-fixed?")
    print("=" * 100)
    pass_count = 0
    for f_emit in f_emit_grid:
        trP_f = results[(f_emit, "fixed")]["mean_trace_P"].mean()
        trP_a = results[(f_emit, "adaptive")]["mean_trace_P"].mean()
        k_f = results[(f_emit, "fixed")]["kill_capacity"].mean()
        k_a = results[(f_emit, "adaptive")]["kill_capacity"].mean()
        # PASS if adaptive reduces kills by ≥ 0.3 OR raises trace_P by ≥ 50%
        kills_gap = k_f - k_a
        trP_ratio = trP_a / max(trP_f, 1e-9)
        sub_pass = (kills_gap >= 0.3) or (trP_ratio >= 1.5)
        marker = "✓" if sub_pass else "✗"
        if sub_pass:
            pass_count += 1
        print(f"  jam_frac={f_emit:.0e}: kills_gap={kills_gap:+.2f}, "
              f"trP_ratio={trP_ratio:.2f}× {marker}")

    print(f"\n  {pass_count}/{len(f_emit_grid)} points show adaptive-split "
          f"meaningfully harder than true-fixed")
    if pass_count >= 2:
        print("  → D1 PASS: dynamic coordination skill needed, proceed to D3")
    elif pass_count == 1:
        print("  → D1 MARGINAL: dynamic harder at one point; investigate "
              "regime before D3")
    else:
        print("  → D1 FAIL: even adaptive-split not harder than fixed → "
              "no dynamic difficulty → IET floor")


if __name__ == "__main__":
    main()
