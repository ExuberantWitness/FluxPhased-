"""WP0 validation harness for the TAES env.

Per TAES_IMPLEMENTATION_SPEC.md §1.6, the WP0 sanity gate is:
  - Single-target no-EW: classical achieves kill_rate ≈ 1.0,
    time-to-kill ≈ E_kill / dwell_rate (= 20 steps), trace_P ≈ PCRLB.
  - Multi-target + L3: classical kill rate drops, time-to-kill explodes
    (hard regime established).

Runs a grid of {N_targets × jammer_level × exposure} cells, ≥5 seeds each.
Reports mean ± std for kill_rate, ttk, survival, track_loss_rate, trace_P/PCRLB.

Usage:
    python -m algo._shared.pilot.taes.run_wp0_validation
    python -m algo._shared.pilot.taes.run_wp0_validation --cells n1_l0 n4_l0 n4_l3 n8_l3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Optional

# Make project importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from env.gpu.taes.taes_env import TAESVecEnv
from env.gpu.qos_rrm.adversary import make_jammer
from algo._shared.laser.crlb import PCRLBTracker
from algo._shared.baselines.taes_classical_commander import (
    TaesClassicalCommander,
    TaesGreedyCommander,
)


CELLS_DEFAULT = [
    "n1_l0",     # 1 target, no EW (sanity)
    "n1_l3",     # 1 target + L3 (single-tgt adaptive should still kill)
    "n4_l0",     # 4 targets, no EW
    "n4_l1",     # 4 targets + L1 reactive
    "n4_l3",     # 4 targets + L3 learned (hard regime)
    "n8_l0",     # 8 targets, no EW
    "n8_l3",     # 8 targets + L3 (hardest)
]


def parse_cell(cell: str):
    """Parse 'n4_l3' → (n_targets=4, jammer='L3')."""
    parts = cell.split("_")
    n_str, j_str = parts[0], parts[1]
    n_targets = int(n_str[1:])
    jammer = j_str.upper()
    return n_targets, jammer


def run_cell(
    cell: str,
    n_seeds: int = 5,
    n_envs: int = 4,
    episode_steps: int = 600,
    device: str = "cuda",
    seed_offset: int = 0,
    classical_type: str = "strong",
    dt: float = 0.1,
    e_kill: float = 2.0,
    dwell_rate: float = 1.0,
    jam_gain: float = 8.0,
    exposure_gain: float = 50.0,
) -> Dict:
    """Run a single validation cell across seeds, return aggregated metrics."""
    n_targets, jammer_level = parse_cell(cell)
    print(f"\n=== Cell {cell} (N={n_targets}, jammer={jammer_level}) "
          f"seeds={n_seeds} envs={n_envs} classical={classical_type} ===")

    all_metrics = []
    for seed in range(seed_offset, seed_offset + n_seeds):
        env = TAESVecEnv(
            n_envs=n_envs,
            n_targets=n_targets,
            device=device,
            dt=dt,
            episode_steps=episode_steps,
            e_kill=e_kill,
            dwell_rate=dwell_rate,
            jam_gain=jam_gain,
            exposure_gain=exposure_gain,
            seed=seed * 100 + 7,
        )
        obs = env.reset()

        # Jammer
        if jammer_level == "L0":
            # "L0" in cell naming means no EW (jam=0); use L0 with jam_level=0
            jammer = make_jammer("L0", jam_level=0.0)
        elif jammer_level == "L1":
            jammer = make_jammer("L1", tau=8, base_jam=0.3, max_jam=1.0, adaptivity=0.7)
        elif jammer_level == "L3":
            # Note: WP0 uses random-init L3 (no training yet). Still produces
            # adaptive behavior via the MLP reacting to red task histogram.
            jammer = make_jammer("L3", base_jam=0.3, device=device)
        else:
            raise ValueError(f"Unknown jammer level: {jammer_level}")

        jammer.reset(n_envs, 1, device)
        # Init jam history
        env._last_jam = torch.zeros(n_envs, device=device)

        # Classical commander
        if classical_type == "strong":
            commander = TaesClassicalCommander()
        elif classical_type == "greedy":
            commander = TaesGreedyCommander()
        else:
            raise ValueError(f"Unknown classical_type: {classical_type}")

        # PCRLB tracker (per-env, per-target)
        pcrlb = PCRLBTracker(n_envs=n_envs, n_max=env.N_max, device=device,
                              dt=dt, sigma_q=env.sigma_q)

        # Episode loop
        seed_metrics = {
            "kill_rate": [],
            "ttk_first": [],
            "survived": [],
            "track_loss_rate": [],
            "E_progress_end": [],
            "exposure_end": [],
            "trace_P_over_pcrlb": [],
            "n_steps": [],
        }

        # Per-env accumulators for trace_P/PCRLB running mean
        trace_pcratio_sum = torch.zeros(n_envs, device=device)
        trace_pcratio_count = torch.zeros(n_envs, device=device)
        track_loss_rate_sum = torch.zeros(n_envs, device=device)
        track_loss_rate_count = torch.zeros(n_envs, device=device)

        done_mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
        final_metrics = [None] * n_envs

        for step_i in range(episode_steps):
            action = commander.step(env)
            obs, reward, done, info = env.step(action, jammer=jammer)

            # PCRLB update (only for cells where we want to verify tracking bound)
            # Use un-jammed sigma for PCRLB lower bound (the "best case")
            rs = torch.tensor(env.range_sigma_m, device=device).expand(n_envs, env.N_max)
            bs = torch.tensor(env.bearing_sigma_rad, device=device).expand(n_envs, env.N_max)
            pcrlb.update(env.target_pos, env.radar_pos, rs, bs,
                          env.target_alive_mask, use_range_bearing=True)

            # Accumulate trace_P/PCRLB over alive targets (running mean)
            trace_P = (env.tracker_P[..., 0, 0] + env.tracker_P[..., 2, 2])
            pcrlb_v = pcrlb.get_pcrlb()
            alive_f = env.target_alive_mask.float()
            n_alive = alive_f.sum(dim=1)
            ratios = (trace_P / pcrlb_v.clamp(min=1e-9)) * alive_f
            ratios_sum = ratios.sum(dim=1)
            valid = n_alive > 0
            trace_pcratio_sum = trace_pcratio_sum + torch.where(
                valid, ratios_sum / n_alive.clamp(min=1), torch.zeros_like(trace_pcratio_sum))
            trace_pcratio_count = trace_pcratio_count + valid.float()

            # Accumulate track_loss_rate
            track_loss_rate_sum = track_loss_rate_sum + torch.where(
                valid, info["track_loss_rate"], torch.zeros_like(track_loss_rate_sum))
            track_loss_rate_count = track_loss_rate_count + valid.float()

            # On done, snapshot
            newly_done = done & (~done_mask)
            if newly_done.any():
                for e_idx in range(n_envs):
                    if newly_done[e_idx] and final_metrics[e_idx] is None:
                        n_total = env.target_n_actual[e_idx].item()
                        n_killed = env.target_killed[e_idx].sum().item()
                        kill_rate = n_killed / max(n_total, 1)
                        ttk = info["time_to_kill_first"][e_idx].item()
                        survived = env.own_alive[e_idx].item()
                        cnt = max(trace_pcratio_count[e_idx].item(), 1.0)
                        tl_cnt = max(track_loss_rate_count[e_idx].item(), 1.0)
                        final_metrics[e_idx] = {
                            "kill_rate": kill_rate,
                            "ttk_first": ttk if kill_rate > 0 else float(episode_steps),
                            "survived": float(survived),
                            "track_loss_rate": float(track_loss_rate_sum[e_idx].item() / tl_cnt),
                            "E_progress_end": info["E_progress_mean"][e_idx].item(),
                            "exposure_end": env.exposure[e_idx].item(),
                            "trace_P_over_pcrlb": float(trace_pcratio_sum[e_idx].item() / cnt),
                            "n_steps": env.step_idx[e_idx].item(),
                        }
                done_mask = done_mask | done

            if done_mask.all():
                break

        # Snapshot unfinished envs
        for e_idx in range(n_envs):
            if final_metrics[e_idx] is None:
                n_total = env.target_n_actual[e_idx].item()
                n_killed = env.target_killed[e_idx].sum().item()
                kill_rate = n_killed / max(n_total, 1)
                cnt = max(trace_pcratio_count[e_idx].item(), 1.0)
                tl_cnt = max(track_loss_rate_count[e_idx].item(), 1.0)
                final_metrics[e_idx] = {
                    "kill_rate": kill_rate,
                    "ttk_first": float(episode_steps),
                    "survived": float(env.own_alive[e_idx].item()),
                    "track_loss_rate": float(track_loss_rate_sum[e_idx].item() / tl_cnt),
                    "E_progress_end": 0.0,
                    "exposure_end": env.exposure[e_idx].item(),
                    "trace_P_over_pcrlb": float(trace_pcratio_sum[e_idx].item() / cnt),
                    "n_steps": env.step_idx[e_idx].item(),
                }

        # Aggregate over envs (each env is one episode in this seed)
        for m in final_metrics:
            for k, v in m.items():
                if k not in seed_metrics:
                    seed_metrics[k] = []
                seed_metrics[k].append(v)

        # Compute seed-level mean
        seed_summary = {k: float(np.mean(v)) for k, v in seed_metrics.items()}
        all_metrics.append(seed_summary)

        # Per-seed print
        kill_rates = [m["kill_rate"] for m in final_metrics]
        print(f"  seed {seed:3d}: kill_rate={np.mean(kill_rates):.2f} "
              f"(±{np.std(kill_rates):.2f}, n={len(kill_rates)} envs)  "
              f"ttk_first={seed_summary['ttk_first']:.0f}  "
              f"surv={seed_summary['survived']:.2f}  "
              f"track_loss={seed_summary['track_loss_rate']:.2f}")

    # Aggregate over seeds
    summary = {}
    for k in all_metrics[0].keys():
        vals = [m[k] for m in all_metrics]
        summary[k + "_mean"] = float(np.mean(vals))
        summary[k + "_std"] = float(np.std(vals))
        summary[k + "_all"] = vals

    print(f"\n  SUMMARY {cell}: kill_rate={summary['kill_rate_mean']:.3f}±{summary['kill_rate_std']:.3f} "
          f"ttk={summary['ttk_first_mean']:.0f}±{summary['ttk_first_std']:.0f} "
          f"surv={summary['survived_mean']:.2f}±{summary['survived_std']:.2f} "
          f"track_loss={summary['track_loss_rate_mean']:.3f} "
          f"P/PCRLB={summary['trace_P_over_pcrlb_mean']:.2f}")

    return {"cell": cell, "n_targets": n_targets, "jammer": jammer_level,
            "n_seeds": n_seeds, "summary": summary, "per_seed": all_metrics}


def _trace_over_pcrlb(env: TAESVecEnv, pcrlb: PCRLBTracker, e_idx: int) -> float:
    """Compute mean trace_P/PCRLB over alive targets in env e_idx."""
    trace_P = (env.tracker_P[e_idx, :, 0, 0] + env.tracker_P[e_idx, :, 2, 2])
    pcrlb_v = pcrlb.get_pcrlb()[e_idx]
    alive = env.target_alive_mask[e_idx].float()
    if alive.sum() < 1:
        return 0.0
    ratios = trace_P / pcrlb_v.clamp(min=1e-9)
    return float(((ratios * alive).sum() / alive.sum()).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", default=CELLS_DEFAULT)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--classical", choices=["strong", "greedy"], default="strong")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--e-kill", type=float, default=2.0)
    parser.add_argument("--dwell-rate", type=float, default=1.0)
    parser.add_argument("--jam-gain", type=float, default=8.0)
    parser.add_argument("--exposure-gain", type=float, default=50.0)
    parser.add_argument("--out", type=str,
                        default="experiments/wp0_validation/results.json")
    parser.add_argument("--report", type=str,
                        default="experiments/wp0_validation/WP0_VERDICT.md")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"WP0 Validation")
    print(f"  cells: {args.cells}")
    print(f"  seeds: {args.seeds}, envs: {args.envs}, steps: {args.steps}")
    print(f"  classical: {args.classical}")
    print(f"  physics: dt={args.dt}, e_kill={args.e_kill}, dwell_rate={args.dwell_rate}")
    print(f"  EW: jam_gain={args.jam_gain}, exposure_gain={args.exposure_gain}")

    all_results = []
    t0 = time.time()
    for cell in args.cells:
        result = run_cell(
            cell=cell,
            n_seeds=args.seeds,
            n_envs=args.envs,
            episode_steps=args.steps,
            classical_type=args.classical,
            dt=args.dt,
            e_kill=args.e_kill,
            dwell_rate=args.dwell_rate,
            jam_gain=args.jam_gain,
            exposure_gain=args.exposure_gain,
        )
        all_results.append(result)
    elapsed = time.time() - t0

    # Save JSON
    # Strip "all" lists for JSON brevity (keep mean/std)
    json_results = []
    for r in all_results:
        jr = {"cell": r["cell"], "n_targets": r["n_targets"], "jammer": r["jammer"],
              "n_seeds": r["n_seeds"],
              "summary": {k: v for k, v in r["summary"].items() if not k.endswith("_all")}}
        json_results.append(jr)
    with open(args.out, "w") as f:
        json.dump({"elapsed_sec": elapsed, "results": json_results}, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # Generate verdict report
    _write_verdict(args.report, all_results, args, elapsed)
    print(f"Verdict saved to {args.report}")


def _write_verdict(path: str, results: List[Dict], args, elapsed: float):
    """Generate WP0_VERDICT.md per spec §1.6 sanity criteria."""
    lines = []
    lines.append("# WP0 Validation Verdict\n")
    lines.append(f"**Run**: dt={args.dt}, e_kill={args.e_kill}, dwell_rate={args.dwell_rate}, "
                  f"jam_gain={args.jam_gain}, exposure_gain={args.exposure_gain}\n")
    lines.append(f"**Cells**: {len(results)} | **Seeds**: {args.seeds} | "
                  f"**Envs/seed**: {args.envs} | **Steps**: {args.steps}\n")
    lines.append(f"**Elapsed**: {elapsed:.1f}s\n\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Cell | N | Jammer | kill_rate | ttk_first | survived | track_loss | P/PCRLB | exposure_end |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for r in results:
        s = r["summary"]
        lines.append(
            f"| {r['cell']} | {r['n_targets']} | {r['jammer']} | "
            f"{s['kill_rate_mean']:.2f}±{s['kill_rate_std']:.2f} | "
            f"{s['ttk_first_mean']:.0f}±{s['ttk_first_std']:.0f} | "
            f"{s['survived_mean']:.2f} | "
            f"{s['track_loss_rate_mean']:.3f} | "
            f"{s['trace_P_over_pcrlb_mean']:.2f} | "
            f"{s['exposure_end_mean']:.1f} |\n"
        )

    # Gate check
    lines.append("\n## Gate Checks (spec §1.6)\n")
    sanity_pass = True
    hard_regime_pass = True

    # Find specific cells
    n1_l0 = next((r for r in results if r["cell"] == "n1_l0"), None)
    n4_l3 = next((r for r in results if r["cell"] == "n4_l3"), None)
    n4_l0 = next((r for r in results if r["cell"] == "n4_l0"), None)
    n8_l3 = next((r for r in results if r["cell"] == "n8_l3"), None)

    expected_ttk = args.e_kill / args.dwell_rate / args.dt
    if n1_l0 is not None:
        s = n1_l0["summary"]
        kill_ok = s["kill_rate_mean"] > 0.8
        ttk_ok = abs(s["ttk_first_mean"] - expected_ttk) < 0.5 * expected_ttk
        pcratio_ok = 0.5 < s["trace_P_over_pcrlb_mean"] < 3.0
        sanity_pass = kill_ok and ttk_ok and pcratio_ok
        lines.append(f"### Sanity (n1_l0): kill={s['kill_rate_mean']:.2f} "
                      f"(target >0.8: **{'PASS' if kill_ok else 'FAIL'}**), "
                      f"ttk={s['ttk_first_mean']:.0f} (target ≈{expected_ttk:.0f}: "
                      f"**{'PASS' if ttk_ok else 'FAIL'}**), "
                      f"P/PCRLB={s['trace_P_over_pcrlb_mean']:.2f} "
                      f"(target [0.5, 3.0]: **{'PASS' if pcratio_ok else 'FAIL'}**)\n")
    else:
        lines.append("### Sanity: n1_l0 cell NOT RUN — cannot verify sanity\n")
        sanity_pass = False

    if n4_l3 is not None and n4_l0 is not None:
        s_hard = n4_l3["summary"]
        s_easy = n4_l0["summary"]
        kill_drop = s_easy["kill_rate_mean"] - s_hard["kill_rate_mean"]
        hard_kill_ok = kill_drop > 0.20
        ttk_blow = s_hard["ttk_first_mean"] > 2.0 * s_easy["ttk_first_mean"]
        hard_regime_pass = hard_kill_ok and ttk_blow
        lines.append(f"\n### Hard regime (n4_l0 vs n4_l3): "
                      f"kill drop = {kill_drop:.2f} (target >0.20: "
                      f"**{'PASS' if hard_kill_ok else 'FAIL'}**), "
                      f"ttk blow-up {s_easy['ttk_first_mean']:.0f}→{s_hard['ttk_first_mean']:.0f} "
                      f"(target 2×: **{'PASS' if ttk_blow else 'FAIL'}**)\n")

    if n8_l3 is not None:
        s = n8_l3["summary"]
        lines.append(f"\n### Extreme regime (n8_l3): kill={s['kill_rate_mean']:.2f}, "
                      f"ttk={s['ttk_first_mean']:.0f}, "
                      f"survived={s['survived_mean']:.2f}\n")

    lines.append(f"\n## Verdict\n")
    if sanity_pass and hard_regime_pass:
        lines.append("✅ **WP0 PASS** — env mechanics correct, hard regime established.\n")
        lines.append("Proceed to WP1: build IMM-PDAF (Stone Soup) + fictitious-play baseline, "
                      "verify strong classical is non-strawman at low difficulty.\n")
    elif sanity_pass and not hard_regime_pass:
        lines.append("⚠️ **WP0 PARTIAL** — sanity OK, hard regime NOT YET established.\n")
        lines.append("**Action**: tighten kill-chain coupling (smaller decay_factor, "
                      "lower tau_track_scale, higher jam_gain) or strengthen L3 jammer.\n")
    else:
        lines.append("❌ **WP0 FAIL** — sanity check failed, env mechanics broken.\n")
        lines.append("**Action**: debug env before proceeding. Common causes: tau_track "
                      "missetuned, E_kill/dwell_rate wrong, tracker diverging.\n")

    with open(path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
