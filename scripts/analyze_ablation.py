#!/usr/bin/env python3
"""Analyze ablation logs and pick the winner.

Metrics (in priority order):
  1. kill_rate trajectory (most important — does policy learn to kill?)
  2. PPO policy_loss recovery (non-zero = learning signal)
  3. kill_radius anneal events (curriculum progress)
  4. avg_r trend (reward growth)
  5. entropy stability (didn't collapse)

Usage: python scripts/analyze_ablation.py
"""
import re
import os
import sys
import json
from pathlib import Path

LOG_DIR = Path("logs/ablation_f1f8")
VARIANTS = ["v4_control", "v1_conservative", "v2_aggressive", "v3_scaling"]


def parse_log(log_path: Path) -> dict:
    """Extract key metrics from a training log."""
    if not log_path.exists():
        return {}

    text = log_path.read_text(errors="replace")
    lines = text.splitlines()

    # kill_rate values (from "kill_rate=X.XX" in [League] messages)
    kill_rates = []
    policy_losses = []
    value_losses = []
    entropies = []
    avg_rewards = []
    iterations = []
    kr_anneals = []
    ppo_updates = 0

    kr_pat = re.compile(r'kill_radius anneal: ([\d.]+)m → ([\d.]+)m')
    pl_pat = re.compile(r'pl=([-\d.]+)')
    iter_pat = re.compile(r'Iteration (\d+):')
    kill_rate_pat = re.compile(r'kill_rate=([\d.]+)')
    avg_r_pat = re.compile(r'avg_r=([-\d.]+)')
    ppo_pat = re.compile(r'\[PPO\]')

    for line in lines:
        m = kr_pat.search(line)
        if m:
            kr_anneals.append(float(m.group(2)))

        m = iter_pat.search(line)
        if m:
            iterations.append(int(m.group(1)))

        m = kill_rate_pat.search(line)
        if m:
            kill_rates.append(float(m.group(1)))

        m = pl_pat.search(line)
        if m:
            try:
                policy_losses.append(float(m.group(1)))
            except ValueError:
                pass

        m = avg_r_pat.search(line)
        if m:
            try:
                avg_rewards.append(float(m.group(1)))
            except ValueError:
                pass

        if ppo_pat.search(line):
            ppo_updates += 1

    return {
        "log_file": str(log_path),
        "n_iterations": len(set(iterations)),
        "max_iter": max(iterations) if iterations else 0,
        "n_kr_anneals": len(kr_anneals),
        "final_kr": kr_anneals[-1] if kr_anneals else None,
        "kr_trajectory": kr_anneals,
        "kill_rate_max": max(kill_rates) if kill_rates else None,
        "kill_rate_last20_avg": (sum(kill_rates[-20:]) / min(20, len(kill_rates)))
                                 if kill_rates else None,
        "policy_loss_last20_avg": (sum(policy_losses[-20:]) / min(20, len(policy_losses)))
                                    if policy_losses else None,
        "policy_loss_n_samples": len(policy_losses),
        "avg_reward_last20_avg": (sum(avg_rewards[-20:]) / min(20, len(avg_rewards)))
                                  if avg_rewards else None,
        "n_ppo_updates": ppo_updates,
    }


def main():
    results = {}
    for v in VARIANTS:
        # Find latest log for this variant
        logs = sorted(LOG_DIR.glob(f"{v}_*.log"))
        if not logs:
            results[v] = {"error": "no log file"}
            continue
        results[v] = parse_log(logs[-1])

    # Print summary
    print("=" * 80)
    print("ABLATION COMPARISON — F1 Reward Tuning × F2-F8 Code Fixes")
    print("=" * 80)
    print(f"{'Metric':<30} {'v4 control':>14} {'v1 conservative':>16} {'v2 aggressive':>15} {'v3 scaling':>12}")
    print("-" * 80)

    def cell(v, key, fmt="{:.4f}"):
        r = results.get(v, {})
        val = r.get(key)
        if val is None:
            return "N/A".rjust(14 if v == "v4_control" else 16 if v == "v1_conservative" else 15 if v == "v2_aggressive" else 12)
        try:
            return fmt.format(float(val))
        except (ValueError, TypeError):
            return str(val)

    metrics = [
        ("n_iterations",            "{:d}",   "Iterations"),
        ("n_kr_anneals",            "{:d}",   "KR anneals"),
        ("final_kr",                "{:.3f}", "Final kill_radius (m)"),
        ("kill_rate_max",           "{:.4f}", "Max kill_rate"),
        ("kill_rate_last20_avg",    "{:.4f}", "kill_rate (last 20 avg)"),
        ("policy_loss_last20_avg",  "{:.4f}", "policy_loss (last 20 avg)"),
        ("policy_loss_n_samples",   "{:d}",   "policy_loss samples"),
        ("avg_reward_last20_avg",   "{:.2f}",  "avg_r (last 20 avg)"),
        ("n_ppo_updates",           "{:d}",   "PPO update count"),
    ]

    for key, fmt, label in metrics:
        row = f"{label:<30}"
        for v in VARIANTS:
            r = results.get(v, {})
            val = r.get(key)
            if val is None:
                cell_str = "N/A"
            else:
                try:
                    cell_str = fmt.format(val)
                except (ValueError, TypeError):
                    cell_str = str(val)
            row += f" {cell_str:>14}"
        print(row)

    print()
    print("=" * 80)
    print("WINNER PICK")
    print("=" * 80)

    # Score: prefer variants that (a) annealed kill_radius, (b) have rising kill_rate,
    # (c) have non-zero policy_loss (learning signal).
    scores = {}
    for v in VARIANTS:
        r = results.get(v, {})
        score = 0.0
        # +3 if kill_radius annealed at least once
        score += 3.0 * min(r.get("n_kr_anneals", 0), 3)
        # +2 per 0.1 of kill_rate above 0.1 (max ~10)
        kr_max = r.get("kill_rate_max") or 0.0
        score += 2.0 * max(0.0, kr_max - 0.1) * 10
        # +1 if policy_loss is non-zero and small (healthy)
        pl_avg = r.get("policy_loss_last20_avg")
        if pl_avg is not None and abs(pl_avg) > 1e-6:
            score += 1.0
        # +1 per PPO update (normalized)
        n_updates = r.get("n_ppo_updates", 0)
        score += min(n_updates / 100.0, 2.0)
        scores[v] = score

    for v, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {v:<20} score={s:.2f}")

    winner = max(scores, key=scores.get)
    print(f"\n  WINNER: {winner}")

    # Save full results
    out = LOG_DIR / "ablation_summary.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results saved to: {out}")


if __name__ == "__main__":
    main()
