"""WP-C R2 — Hardened regime: competent baseline vs dynamic (reactive) enemy.

Per WP-C plan: prove that a dynamic enemy (channel-following reactive jammer)
degrades the competent baseline (StrongRule + orthogonal fixed channel config)
more than a fixed-action jammer does — this is the coordination-difficulty
evidence that motivates R3 RL.

Sweep design (vs WP-B fixed jammer):
  axis 1: f_emit_A ∈ [1e-6, 1e-5, 1e-4, 1e-3]  (jam power)
  axis 2: enemy_type ∈ {fixed, reactive}        (WP-B vs WP-C dynamic)
  fixed geometry: team_offset=2500m, mirror

For competent baseline (team B = StrongRule + external orth channel config),
record: mean_trace_P, kills_B, hop_rate use.

Expected: reactive enemy tracks victim's fixed ch0/ch1 → competent baseline
degrades harder than vs fixed jammer (kills drop faster, trace_P higher).
This is the "no trivial fixed solution" evidence for WP-C coordination skill.

PASS criterion: at f_emit=1e-4, reactive enemy reduces kills_B by ≥ 0.3 vs
fixed jammer (or raises trace_P by ≥ 50% if kills already 0).
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import argparse
import math
import torch
import numpy as np

from env.gpu.twoteam import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.baselines.reactive_jammer_commander import ReactiveJammerCommander
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


# ---------- channel config (R1 baseline uses orth) --------------------------

def configure_channels(env, mode: str):
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


def build_fixed_enemy_action(env, f_emit_A: float):
    """WP-B style fixed fraction jammer. Returns per-team slice (team 0)."""
    E, R = env.E, env.n_radars_per_team
    dev = env.device
    alloc = torch.zeros(E, R, 4, device=dev)
    if f_emit_A > 0:
        alloc[:, :, 2] = f_emit_A
        alloc[:, :, 3] = max(0.0, 1.0 - f_emit_A)
    else:
        alloc[:, :, 3] = 1.0
    # WP-C R3: hold current env freq (channel_select from env state).
    ch_idx = ((env.radar_freq_hz[:, 0, :] - env.fc_hz)
              / env.channel_spacing_hz).round().long().clamp(0, env.n_channels - 1)
    return {
        "task_alloc": alloc,
        "beam_target": torch.zeros(E, R, dtype=torch.long, device=dev),
        "laser_target": torch.zeros(E, dtype=torch.long, device=dev),
        "emission_on": torch.ones(E, R, dtype=torch.bool, device=dev) if f_emit_A > 0
                       else torch.zeros(E, R, dtype=torch.bool, device=dev),
        "freq_hop_rate": torch.ones(E, R, device=dev),
        "channel_select": ch_idx,
    }


# ---------- single-scenario runner ------------------------------------------

def run_scenario(f_emit_A: float, enemy_type: str,
                 n_episodes: int, n_envs: int, episode_steps: int,
                 team_offset_m: float = 2500.0):
    """Team A = enemy (fixed or reactive), Team B = StrongRule+orth baseline."""
    env = TwoTeamVecEnv(
        n_envs=n_envs, device="cuda",
        episode_steps=episode_steps,
        geometry=MIRROR_GEOMETRY,
        team_offset_m=team_offset_m,
    )

    # Apple-to-apple: reactive uses SAME jam fraction as fixed (f_emit_A).
    # The only difference is strategy (fixed vs channel-following reactive),
    # NOT raw power. JNR difference then reflects pure strategy effect.
    reactive_jam_frac = f_emit_A

    episodes_done = 0
    metrics = {
        "mean_trace_P": [],
        "kill_capacity": [],
        "team_a_kills": [],
        "jnr_at_victim": [],
        "hop_rate_jammed": [],
    }

    while episodes_done < n_episodes:
        env.reset()
        configure_channels(env, "orthogonal")   # baseline gets competent config
        rule_b = TwoTeamStrongRuleCommander()
        reactive_a = ReactiveJammerCommander(
            jam_fraction=reactive_jam_frac
        ) if enemy_type == "reactive" and f_emit_A > 0 else None

        ep_trace_P_sum = torch.zeros(env.E, device=env.device)
        ep_jnr_sum = torch.zeros(env.E, device=env.device)
        ep_hop_jammed_sum = torch.zeros(env.E, device=env.device)
        ep_step_count = 0
        last_info = None

        for step in range(episode_steps):
            if enemy_type == "fixed":
                a_t0 = build_fixed_enemy_action(env, f_emit_A)
            else:  # reactive — channel-following via channel_select action (no wrapper mutation)
                if reactive_a is None:
                    # f_emit=0 reactive: same as fixed no-enemy
                    a_t0 = build_fixed_enemy_action(env, 0.0)
                else:
                    a_t0 = reactive_a.get_action(env, team=0)

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

            trace_P = env.tracker_P[:, 1, :, 0, 0] + env.tracker_P[:, 1, :, 2, 2]
            enemy_alive = env.radar_alive[:, 0]
            alive_mask = enemy_alive.float()
            n_alive = alive_mask.sum(dim=-1).clamp(min=1.0)
            ep_trace_P_sum += (trace_P * alive_mask).sum(dim=-1) / n_alive

            hop_b = action["freq_hop_rate"][:, 1].mean(dim=-1)
            jammed = jnr_at_victim > 1.0
            ep_hop_jammed_sum += (hop_b * jammed.float())

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
            metrics["hop_rate_jammed"].append(ep_hop_jammed_sum[e].item() / sc)
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

    print(f"=== WP-C R2: Hardened regime (competent baseline vs dynamic enemy) ===")
    print(f"  tau_track = 4.0")
    print(f"  baseline  = StrongRule + orth channel config (competent, from R1)")
    print(f"  enemy     = {{fixed (WP-B style), reactive (channel-following)}}")
    print(f"  geometry  = MIRROR, team_offset=2500m")
    print(f"  episodes  = {n_ep} × {args.episode_steps} steps per scenario")
    print()

    results = {}
    for f_emit in f_emit_grid:
        for enemy in ["fixed", "reactive"]:
            m = run_scenario(f_emit_A=f_emit, enemy_type=enemy,
                             n_episodes=n_ep, n_envs=args.n_envs,
                             episode_steps=args.episode_steps)
            results[(f_emit, enemy)] = m
            jnr_dB = 10 * math.log10(max(m["jnr_at_victim"].mean(), 1e-15))
            print(f"[f_emit={f_emit:.0e} enemy={enemy:>8}] "
                  f"JNR_victim={jnr_dB:>6.1f}dB "
                  f"trace_P={m['mean_trace_P'].mean():>8.3f} "
                  f"kills_B={m['kill_capacity'].mean():.2f} "
                  f"kills_A={m['team_a_kills'].mean():.2f} "
                  f"hop_jammed={m['hop_rate_jammed'].mean():.2f}")

    # ----- degradation comparison table -----
    print()
    print("=" * 100)
    print("R2 DEGRADATION TABLE — competent baseline vs {fixed, reactive} enemy")
    print("=" * 100)
    print(f"{'f_emit':>9} | {'fixed_trP':>10} {'react_trP':>10} {'trP_ratio':>10} | "
          f"{'fixed_k':>8} {'react_k':>8} {'kills_Δ':>8}")
    print("-" * 90)
    for f_emit in f_emit_grid:
        trP_f = results[(f_emit, "fixed")]["mean_trace_P"].mean()
        trP_r = results[(f_emit, "reactive")]["mean_trace_P"].mean()
        k_f = results[(f_emit, "fixed")]["kill_capacity"].mean()
        k_r = results[(f_emit, "reactive")]["kill_capacity"].mean()
        ratio = trP_r / max(trP_f, 1e-9)
        print(f"{f_emit:>9.0e} | {trP_f:>10.3f} {trP_r:>10.3f} {ratio:>9.2f}× | "
              f"{k_f:>8.2f} {k_r:>8.2f} {k_r-k_f:>+8.2f}")

    # ----- PASS criterion -----
    print()
    print("=" * 100)
    print("R2 VERDICT — dynamic enemy harder than fixed?")
    print("=" * 100)
    pass_count = 0
    for f_emit in f_emit_grid:
        trP_f = results[(f_emit, "fixed")]["mean_trace_P"].mean()
        trP_r = results[(f_emit, "reactive")]["mean_trace_P"].mean()
        k_f = results[(f_emit, "fixed")]["kill_capacity"].mean()
        k_r = results[(f_emit, "reactive")]["kill_capacity"].mean()
        # PASS if reactive reduces kills by ≥ 0.3 OR raises trace_P by ≥ 50%
        kills_gap = k_f - k_r
        trP_ratio = trP_r / max(trP_f, 1e-9)
        sub_pass = (kills_gap >= 0.3) or (trP_ratio >= 1.5)
        marker = "✓" if sub_pass else "✗"
        if sub_pass:
            pass_count += 1
        print(f"  f_emit={f_emit:.0e}: kills_gap={kills_gap:+.2f}, trP_ratio={trP_ratio:.2f}× {marker}")

    print(f"\n  {pass_count}/{len(f_emit_grid)} points show reactive enemy meaningfully harder")
    if pass_count >= 2:
        print("  → R2 PASS: dynamic coordination skill needed, proceed to R3")
    elif pass_count == 1:
        print("  → R2 MARGINAL: enemy harder at one point; consider tightening reactive strategy")
    else:
        print("  → R2 FAIL: reactive enemy not harder than fixed — investigate "
              "(jam_fraction scaling? mirror geometry? target selection?)")


if __name__ == "__main__":
    main()
