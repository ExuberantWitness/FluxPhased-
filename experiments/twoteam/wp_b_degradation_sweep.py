"""WP-B: Strong classical degradation curve under interference intensity sweep.

Per FLUXPH_MARL_RESEARCH_PLAN.md WP-B + handoff spec:
  - Demonstrate, not gate: classical failure at high interference is expected.
  - Find D_c: realistic working segment where classical goes from "degraded"
    to "fully collapsed" — this is the WP-C RL regime.
  - Intra-team self-blinding (coordination gap evidence): same-channel team-B
    breaks EVEN WITH NO ENEMY (kills drop from 2.0 → 0.85, loss 6% → 91%).

Sweep design:
  Two parallel sweeps, both with channel_mode ∈ {same, orthogonal}:

  Sweep A — f_emit_A (enemy jam power) at fixed 5km geometry:
      [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
      Maps to victim-side JNR ∈ [-inf, 10, 20, 30, 40, 50, 60, 70] dB
      THIS IS THE DEGRADATION CURVE — narrow power range to find D_c.

  Sweep B — engagement range at full-power enemy (f_emit_A=1.0):
      [2500, 5000, 10000, 30000, 100000, 300000] m
      Sanity check: at all realistic ranges, full-power enemy saturates.

Baseline: TwoTeamStrongRuleCommander (existing anti-strawman rule).
Enemy: fixed-action jammer with parametric f_emit_A.

Per scenario, run N episodes × episode_steps, record for team B (the baseline):
  - mean_trace_P      — tracking quality (tau_track=0.04 boundary)
  - track_loss_rate   — fraction of steps where trace_P > tau_track
  - kill_capacity     — mean team_kills per episode (out of 2)
  - JNR_at_victim     — direct read of IQ physics JNR (cross-reference)

Usage:
  python experiments/twoteam/wp_b_degradation_sweep.py
  python experiments/twoteam/wp_b_degradation_sweep.py --quick
  python experiments/twoteam/wp_b_degradation_sweep.py --episodes 200
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import argparse
import math
import torch
import numpy as np

from env.gpu.twoteam import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


# ---------- scenario builders -----------------------------------------------

def build_enemy_action(env, f_emit_A: float):
    """Team-A jammer: f_emit_A fraction to jam, rest to comm (no emit).

    f_emit_A can be arbitrarily small (P_tx scales linearly); f_emit_safe
    floors only affect D_eff for beamwidth, not P_tx.
    """
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    dev = env.device
    alloc = torch.zeros(E, T, R, 4, device=dev)
    if f_emit_A > 0:
        alloc[:, 0, :, 2] = f_emit_A            # team A jam fraction
        alloc[:, 0, :, 3] = max(0.0, 1.0 - f_emit_A)  # rest comm
    else:
        alloc[:, 0, :, 3] = 1.0                   # all comm, no emit
        # emission_on will be False (handled by caller)
    return {
        "task_alloc": alloc,
        "beam_target": torch.zeros(E, T, R, dtype=torch.long, device=dev),
        "laser_target": torch.zeros(E, T, dtype=torch.long, device=dev),
        "emission_on": torch.ones(E, T, R, dtype=torch.bool, device=dev) if f_emit_A > 0
                       else torch.zeros(E, T, R, dtype=torch.bool, device=dev),
        "freq_hop_rate": torch.ones(E, T, R, device=dev),
    }


def configure_channels(env, mode: str):
    """Set per-radar frequencies (mirror-symmetric across teams)."""
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    dev = env.device
    freqs = torch.zeros(E, T, R, device=dev)
    ch_spacing = env.channel_spacing_hz
    fc = env.fc_hz
    for e in range(E):
        if mode == "same":
            freqs[e, 0, :] = fc
            freqs[e, 1, :] = fc
        elif mode == "orthogonal":
            freqs[e, 0, 0] = fc
            freqs[e, 0, 1] = fc + ch_spacing
            freqs[e, 1, 0] = fc
            freqs[e, 1, 1] = fc + ch_spacing
        else:
            raise ValueError(f"unknown channel mode: {mode}")
    env.set_radar_freqs(freqs)


# ---------- single-scenario runner ------------------------------------------

def run_scenario(f_emit_A: float, channel_mode: str,
                 team_offset_m: float, n_episodes: int,
                 n_envs: int, episode_steps: int):
    """Run N episodes; team A = fixed jammer at f_emit_A, team B = StrongRule."""
    env = TwoTeamVecEnv(
        n_envs=n_envs, device="cuda",
        episode_steps=episode_steps,
        geometry=MIRROR_GEOMETRY,
        team_offset_m=team_offset_m,
    )

    episodes_done = 0
    metrics = {
        "mean_trace_P": [],
        "track_loss_rate": [],
        "kill_capacity": [],
        "team_a_kills": [],
        "jnr_at_victim": [],
    }

    while episodes_done < n_episodes:
        env.reset()
        configure_channels(env, channel_mode)
        rule_b = TwoTeamStrongRuleCommander()

        ep_trace_P_sum = torch.zeros(env.E, device=env.device)
        ep_track_loss_sum = torch.zeros(env.E, device=env.device)
        ep_jnr_sum = torch.zeros(env.E, device=env.device)
        ep_step_count = 0
        last_info = None

        for step in range(episode_steps):
            a_enemy = build_enemy_action(env, f_emit_A)
            a_t0 = {k: v[:, 0] for k, v in a_enemy.items()}
            a_t1 = rule_b.get_action(env, team=1)
            action = combine_team_actions(env, a_t0, a_t1)

            obs, reward, done, info = env.step(action)
            last_info = info

            # Read IQ JNR for cross-reference
            jnr_mat = env.iq.compute_jnr_matrix(
                pos=env.radar_pos, beam_az=env.radar_beam_az,
                alloc=action["task_alloc"], freq_hz=env.radar_freq_hz,
                emission_on=action["emission_on"],
                hop_rate=action.get("freq_hop_rate",
                                    torch.ones(env.E, env.n_teams, env.n_radars_per_team,
                                               device=env.device)),
                radar_alive=env.radar_alive,
            )
            jnr_at_victim = jnr_mat[:, [0, 1], 2].sum(dim=-1)  # team_A → team_B_radar_0
            ep_jnr_sum += jnr_at_victim

            trace_P = env.tracker_P[:, 1, :, 0, 0] + env.tracker_P[:, 1, :, 2, 2]
            enemy_alive = env.radar_alive[:, 0]
            alive_mask = enemy_alive.float()
            n_alive = alive_mask.sum(dim=-1).clamp(min=1.0)
            ep_trace_P_sum += (trace_P * alive_mask).sum(dim=-1) / n_alive

            init = env.tracker_initialized[:, 1]
            lost = ((trace_P > env.tau_track) & init & enemy_alive).float()
            ep_track_loss_sum += lost.sum(dim=-1) / n_alive

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
            metrics["track_loss_rate"].append(ep_track_loss_sum[e].item() / sc)
            metrics["kill_capacity"].append(kills_b[e].item())
            metrics["team_a_kills"].append(kills_a[e].item())
            metrics["jnr_at_victim"].append(ep_jnr_sum[e].item() / sc)
            episodes_done += 1

    return {k: np.array(v) for k, v in metrics.items()}


# ---------- main sweep ------------------------------------------------------

DEFAULT_F_EMIT_GRID = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
QUICK_F_EMIT_GRID = [0.0, 1e-5, 1e-3, 1e-1, 1.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--channels", default="both",
                   choices=["both", "same", "orthogonal"])
    p.add_argument("--team-offset-m", type=float, default=2500.0,
                   help="engagement range for f_emit sweep")
    args = p.parse_args()

    f_emit_grid = QUICK_F_EMIT_GRID if args.quick else DEFAULT_F_EMIT_GRID
    ch_modes = ["same", "orthogonal"] if args.channels == "both" else [args.channels]

    print(f"=== WP-B degradation sweep (f_emit axis) ===")
    print(f"team_offset_m: {args.team_offset_m}, episodes/scenario: {args.episodes}")
    print(f"f_emit_grid: {f_emit_grid}")
    print(f"channel_modes: {ch_modes}")
    print()

    results = {}
    for ch in ch_modes:
        for f_emit in f_emit_grid:
            m = run_scenario(f_emit_A=f_emit, channel_mode=ch,
                             team_offset_m=args.team_offset_m,
                             n_episodes=args.episodes,
                             n_envs=args.n_envs,
                             episode_steps=args.episode_steps)
            results[(f_emit, ch)] = m
            jnr_dB = 10 * math.log10(max(m["jnr_at_victim"].mean(), 1e-15))
            enemy_tag = "no-enemy" if f_emit == 0 else "jam     "
            print(f"[f_emit={f_emit:.0e} ch={ch:>10} {enemy_tag}] "
                  f"JNR_victim={jnr_dB:>6.1f}dB "
                  f"trace_P={m['mean_trace_P'].mean():>7.3f} "
                  f"loss={m['track_loss_rate'].mean()*100:>5.1f}% "
                  f"kills_B={m['kill_capacity'].mean():.2f} "
                  f"kills_A={m['team_a_kills'].mean():.2f}")

    # ----- degradation-curve tables -----
    print()
    print("=" * 100)
    print(f"DEGRADATION CURVES — StrongRule (team B) vs enemy f_emit @ range={args.team_offset_m}m")
    print("=" * 100)

    for metric_name, label in [("mean_trace_P", "trace_P"),
                                ("track_loss_rate", "loss%"),
                                ("kill_capacity", "kills_B"),
                                ("jnr_at_victim", "JNR_lin")]:
        print(f"\n--- {label} vs f_emit_A ---")
        print(f"{'f_emit':>9}", end="")
        for ch in ch_modes:
            print(f" | {ch:>14}", end="")
        print()
        print("-" * (10 + 17 * len(ch_modes)))
        for f_emit in f_emit_grid:
            print(f"{f_emit:>9.0e}", end="")
            for ch in ch_modes:
                val = results[(f_emit, ch)][metric_name].mean()
                if metric_name == "track_loss_rate":
                    print(f" | {val*100:>13.1f}%", end="")
                elif metric_name == "jnr_at_victim":
                    print(f" | {val:>14.2e}", end="")
                else:
                    print(f" | {val:>14.3f}", end="")
            print()

    # ----- D_c (collapse points) -----
    print()
    print("=" * 100)
    print("D_c ANALYSIS — classical collapse points (track_loss crosses 50%)")
    print("=" * 100)
    for ch in ch_modes:
        prev_loss = 0.0
        prev_f = 0.0
        d_c = None
        for f_emit in f_emit_grid:
            loss = results[(f_emit, ch)]["track_loss_rate"].mean()
            if prev_loss < 0.5 <= loss:
                d_c = (prev_f, f_emit)
                break
            prev_loss = loss
            prev_f = f_emit
        if d_c:
            print(f"[{ch:>12}] D_c ∈ ({d_c[0]:.0e}, {d_c[1]:.0e}] f_emit "
                  f"(loss jumped {prev_loss*100:.1f}% → {results[(d_c[1], ch)]['track_loss_rate'].mean()*100:.1f}%)")
        else:
            max_loss = max(results[(f, ch)]["track_loss_rate"].mean() for f in f_emit_grid)
            min_loss = min(results[(f, ch)]["track_loss_rate"].mean() for f in f_emit_grid)
            if max_loss < 0.5:
                print(f"[{ch:>12}] classical survives full sweep (loss max {max_loss*100:.1f}%)")
            else:
                print(f"[{ch:>12}] classical already failing at f_emit=0 (intra-team mutual alone → "
                      f"loss={results[(0.0, ch)]['track_loss_rate'].mean()*100:.1f}%)")

    # ----- coordination gap -----
    if "same" in ch_modes and "orthogonal" in ch_modes:
        print()
        print("=" * 100)
        print("COORDINATION GAP — orthogonal vs same channel")
        print("(isolates intra-team mutual interference; same channels = team-B radars blind each other)")
        print("=" * 100)
        print(f"{'f_emit':>9} | {'same_trP':>10} {'orth_trP':>10} {'gap×':>6} | "
              f"{'same_kill':>9} {'orth_kill':>9} {'gapΔ':>6} | {'same_loss%':>10} {'orth_loss%':>10}")
        print("-" * 105)
        for f_emit in f_emit_grid:
            tP_s = results[(f_emit, "same")]["mean_trace_P"].mean()
            tP_o = results[(f_emit, "orthogonal")]["mean_trace_P"].mean()
            k_s = results[(f_emit, "same")]["kill_capacity"].mean()
            k_o = results[(f_emit, "orthogonal")]["kill_capacity"].mean()
            l_s = results[(f_emit, "same")]["track_loss_rate"].mean()
            l_o = results[(f_emit, "orthogonal")]["track_loss_rate"].mean()
            gap_trP = tP_s / max(tP_o, 1e-9)
            print(f"{f_emit:>9.0e} | {tP_s:>10.3f} {tP_o:>10.3f} {gap_trP:>6.2f}× | "
                  f"{k_s:>9.2f} {k_o:>9.2f} {k_o-k_s:>+6.2f} | {l_s*100:>9.1f}% {l_o*100:>9.1f}%")


if __name__ == "__main__":
    main()
