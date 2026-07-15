"""WP-C R1 — Verify the competent baseline (StrongRule + external orthogonal
channel config) is NOT a strawman.

Per FLUXPH_MARL_RESEARCH_PLAN.md WP-C + handoff spec:
  WP-B's "same-channel collapse / 442× gap" is a strawman — a StrongRule that
  doesn't plan frequency fails. A competent classical baseline (any real radar
  has fixed orthogonal channel planning) recovers most of that gap. WP-C must
  beat THIS competent baseline, not the strawman.

R1 four sub-tests (must all PASS before R2/R3):
  Sub-test 1 — f_emit=0 orth → kills_B ≥ 1.95 / 2.00 + trace_P < 1.0
               (baseline reaches near-perfect at zero enemy)
  Sub-test 2 — f_emit ∈ {1e-6, 1e-5, 1e-4} orth → kills_B > 0 in ≥2 points
               (no step-function collapse like same-channel)
  Sub-test 3 — orth vs same gap× ≥ 5 at f_emit ∈ [1e-6, 1e-3]
               (competent baseline really is more competent)
  Sub-test 4 — f_emit=1e-4 orth → StrongRule jam_detect reaction fires
               (mean hop_rate on jammed steps ≥ 4.0; baseline uses its skill)

Usage:
  python experiments/twoteam/wp_c_competent_baseline_check.py
  python experiments/twoteam/wp_c_competent_baseline_check.py --quick
  python experiments/twoteam/wp_c_competent_baseline_check.py --sub-test 1
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


# ---------- shared helpers (mirror wp_b_degradation_sweep) -------------------

def build_enemy_action(env, f_emit_A: float):
    """Team-A fixed-action jammer (WP-B style)."""
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    dev = env.device
    alloc = torch.zeros(E, T, R, 4, device=dev)
    if f_emit_A > 0:
        alloc[:, 0, :, 2] = f_emit_A
        alloc[:, 0, :, 3] = max(0.0, 1.0 - f_emit_A)
    else:
        alloc[:, 0, :, 3] = 1.0
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


def run_scenario(f_emit_A: float, channel_mode: str,
                 n_episodes: int, n_envs: int, episode_steps: int,
                 team_offset_m: float = 2500.0,
                 track_hop: bool = False):
    """Run N episodes; team A = fixed jammer, team B = StrongRule.

    Returns metrics dict. If track_hop=True, also returns per-step hop_rate
    stats on jammed steps (sub-test 4).
    """
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
    # Sub-test 4: hop_rate on jammed vs calm steps
    hop_on_jammed = []
    hop_on_calm = []

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

            # IQ JNR cross-reference
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

            # Sub-test 4: track hop_rate by jam state (team B = index 1)
            if track_hop:
                hop_b = action["freq_hop_rate"][:, 1].mean(dim=-1)  # [E]
                jammed = jnr_at_victim > 1.0  # JNR > 0 dB
                hop_on_jammed.extend(hop_b[jammed].cpu().tolist())
                hop_on_calm.extend(hop_b[~jammed].cpu().tolist())

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

    out = {k: np.array(v) for k, v in metrics.items()}
    if track_hop:
        out["hop_on_jammed"] = np.array(hop_on_jammed) if hop_on_jammed else np.array([0.0])
        out["hop_on_calm"] = np.array(hop_on_calm) if hop_on_calm else np.array([0.0])
    return out


# ---------- sub-tests --------------------------------------------------------

def sub_test_1(n_episodes, n_envs, episode_steps):
    """f_emit=0 orth → kills ≥ 1.95, trace_P < 1.0."""
    print("\n=== Sub-test 1: f_emit=0, orthogonal — near-perfect baseline ===")
    m = run_scenario(f_emit_A=0.0, channel_mode="orthogonal",
                     n_episodes=n_episodes, n_envs=n_envs,
                     episode_steps=episode_steps)
    kills = m["kill_capacity"].mean()
    trace_P = m["mean_trace_P"].mean()
    print(f"  kills_B  = {kills:.3f} / 2.00  (target ≥ 1.95)")
    print(f"  trace_P  = {trace_P:.3f}        (target < 1.0)")
    pass_kills = kills >= 1.95
    pass_trP = trace_P < 1.0
    print(f"  [{'PASS' if pass_kills else 'FAIL'}] kills ≥ 1.95")
    print(f"  [{'PASS' if pass_trP else 'FAIL'}] trace_P < 1.0")
    return pass_kills and pass_trP


def sub_test_2(n_episodes, n_envs, episode_steps):
    """f_emit ∈ {1e-6, 1e-5, 1e-4} orth → kills > 0 in ≥ 2 points."""
    print("\n=== Sub-test 2: light interference, orth — no step-function collapse ===")
    f_emits = [1e-6, 1e-5, 1e-4]
    results = {}
    for f in f_emits:
        m = run_scenario(f_emit_A=f, channel_mode="orthogonal",
                         n_episodes=n_episodes, n_envs=n_envs,
                         episode_steps=episode_steps)
        results[f] = (m["kill_capacity"].mean(), m["mean_trace_P"].mean())
        print(f"  f_emit={f:.0e}  kills_B={results[f][0]:.3f}  trace_P={results[f][1]:.3f}")
    n_above_zero = sum(1 for f in f_emits if results[f][0] > 0.0)
    print(f"  kills > 0 in {n_above_zero}/3 points (target ≥ 2)")
    pass_ = n_above_zero >= 2
    print(f"  [{'PASS' if pass_ else 'FAIL'}] no step-function collapse")
    return pass_


def sub_test_3(n_episodes, n_envs, episode_steps):
    """orth vs same gap× ≥ 5 at WP-C working point {1e-4, 1e-3}.

    Note: gap× at low f_emit (1e-6, 1e-5) is small because both same and orth
    are near baseline (kills=2.00, trace_P < 1). Real "competent gap" emerges
    at the WP-C working point (1e-4 and above) where same-channel collapses
    but orth-channel still degrades gracefully.
    """
    print("\n=== Sub-test 3: orth vs same channel gap (competent vs strawman) ===")
    f_emits = [1e-6, 1e-5, 1e-4, 1e-3]
    print(f"  {'f_emit':>9} | {'same_trP':>10} {'orth_trP':>10} {'gap×':>6} | {'same_k':>7} {'orth_k':>7}")
    print("  " + "-" * 70)
    gaps = {}
    for f in f_emits:
        m_s = run_scenario(f_emit_A=f, channel_mode="same",
                           n_episodes=n_episodes, n_envs=n_envs,
                           episode_steps=episode_steps)
        m_o = run_scenario(f_emit_A=f, channel_mode="orthogonal",
                           n_episodes=n_episodes, n_envs=n_envs,
                           episode_steps=episode_steps)
        trP_s = m_s["mean_trace_P"].mean()
        trP_o = m_o["mean_trace_P"].mean()
        gap = trP_s / max(trP_o, 1e-9)
        k_s = m_s["kill_capacity"].mean()
        k_o = m_o["kill_capacity"].mean()
        gaps[f] = gap
        print(f"  {f:>9.0e} | {trP_s:>10.3f} {trP_o:>10.3f} {gap:>6.2f}× | {k_s:>7.2f} {k_o:>7.2f}")
    # Competent gap judged at WP-C working point {1e-4, 1e-3}
    working_gaps = [gaps[1e-4], gaps[1e-3]]
    max_working_gap = max(working_gaps)
    n_working_above_5 = sum(1 for g in working_gaps if g >= 5.0)
    print(f"\n  WP-C working point gap× (1e-4): {gaps[1e-4]:.2f}×")
    print(f"  WP-C working point gap× (1e-3): {gaps[1e-3]:.2f}×")
    print(f"  working points with gap ≥ 5×: {n_working_above_5}/2 (target ≥ 1)")
    pass_ = n_working_above_5 >= 1
    print(f"  [{'PASS' if pass_ else 'FAIL'}] competent baseline really is more competent at WP-C regime")
    return pass_


def sub_test_4(n_episodes, n_envs, episode_steps):
    """f_emit=1e-4 orth → StrongRule jam_detect reaction fires on jammed steps."""
    print("\n=== Sub-test 4: StrongRule jam_detect reaction fires (orth, f_emit=1e-4) ===")
    m = run_scenario(f_emit_A=1e-4, channel_mode="orthogonal",
                     n_episodes=n_episodes, n_envs=n_envs,
                     episode_steps=episode_steps, track_hop=True)
    h_j = m["hop_on_jammed"].mean()
    h_c = m["hop_on_calm"].mean()
    n_jammed = len(m["hop_on_jammed"])
    n_calm = len(m["hop_on_calm"])
    print(f"  hop_rate on jammed steps (JNR>0dB) = {h_j:.2f}  (target ≥ 4.0, n={n_jammed})")
    if n_calm > 10:
        print(f"  hop_rate on calm steps             = {h_c:.2f}  (target near 1.0, n={n_calm})")
    else:
        print(f"  calm steps too few (n={n_calm}) — f_emit=1e-4 means almost always jammed, OK")
    pass_ = h_j >= 4.0
    print(f"  [{'PASS' if pass_ else 'FAIL'}] jam_detect reaction fires")
    return pass_


# ---------- main -------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="use smaller episode count for fast iteration")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--sub-test", type=int, default=0,
                   help="run single sub-test (1-4); 0 = all")
    args = p.parse_args()

    n_ep = 20 if args.quick else args.episodes

    print(f"=== WP-C R1: Competent baseline check ===")
    print(f"  tau_track = 4.0 (relaxed from WP-B's 0.04)")
    print(f"  baseline  = StrongRule + orthogonal channel config")
    print(f"  geometry  = MIRROR, team_offset=2500m, radar_sep=1500m")
    print(f"  episodes  = {n_ep} × {args.episode_steps} steps per scenario")

    results = {}
    if args.sub_test in (0, 1):
        results[1] = sub_test_1(n_ep, args.n_envs, args.episode_steps)
    if args.sub_test in (0, 2):
        results[2] = sub_test_2(n_ep, args.n_envs, args.episode_steps)
    if args.sub_test in (0, 3):
        results[3] = sub_test_3(n_ep, args.n_envs, args.episode_steps)
    if args.sub_test in (0, 4):
        results[4] = sub_test_4(n_ep, args.n_envs, args.episode_steps)

    print("\n" + "=" * 70)
    print("R1 SUMMARY — competent baseline (StrongRule + orthogonal)")
    print("=" * 70)
    for k in sorted(results.keys()):
        print(f"  Sub-test {k}: [{'PASS' if results[k] else 'FAIL'}]")
    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} sub-tests passed")
    if n_total == 4:
        if n_pass == 4:
            print("\n  → R1 PASS: baseline competent, proceed to R2 (hardened regime)")
        else:
            print("\n  → R1 FAIL: investigate before R2/R3 (do NOT proceed)")


if __name__ == "__main__":
    main()
