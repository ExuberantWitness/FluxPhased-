"""Bet B Step 0 — Rule sensitivity sweep (cheap gate).

Sweep 6 physics axes + geometry, measure StrongRule's WR vs pure_track baseline
across the grid. If rule is robust across all off-nominal points, Bet B's premise
("rule is brittle off-nominal") is dead -> pivot to IET floor.

Cheap (~30-60 min): 14 grid points x 200 ep each (bidirectional) x horizon=200.

For each grid point:
  - Build env with grid-specific config (overrides nominal defaults)
  - Direction A: Rule @ t0 vs pure_track @ t1
  - Direction B: pure_track @ t0 vs Rule @ t1
  - Average win rate from Rule's POV
  - Bootstrap 95% CI

Output: experiments/twoteam/rule_sensitivity_sweep.md
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import numpy as np
import torch

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import (
    TwoTeamVecEnv, MIRROR_GEOMETRY, RANDOM_GEOMETRY,
)
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.pilot.twoteam.extreme_commanders import STRATEGIES
from algo._shared.pilot.twoteam.run_g0_gate import (
    run_episodes_two_commanders, bootstrap_ci,
)


# --- Nominal env config (matches TwoTeamVecEnv defaults) ---
NOMINAL = dict(
    range_sigma_m=0.05,
    sigma_q=2.0,
    exposure_gain=200.0,
    radar_separation_m=1500.0,
    map_size_m=8000.0,
    geometry=MIRROR_GEOMETRY,
)

# --- Sweep grid: 1 nominal + 11 variants (5 axes x 2 levels + geometry) = 12 ---
# NOTE: jam_gain axis removed at WP-A end (scalar_jam_mul mode deleted; IQ-native
# physics now drives interference, with P_tx / G_max / aperture_D as the new knobs).
GRID = [
    ("nominal",            {}),
    # range_sigma_m (sensor precision)
    ("range_sigma=0.02",   {"range_sigma_m": 0.02}),
    ("range_sigma=0.10",   {"range_sigma_m": 0.10}),
    # sigma_q (target dynamics)
    ("sigma_q=1.0",        {"sigma_q": 1.0}),
    ("sigma_q=4.0",        {"sigma_q": 4.0}),
    # exposure_gain (exposure sensitivity)
    ("exposure=100",       {"exposure_gain": 100.0}),
    ("exposure=400",       {"exposure_gain": 400.0}),
    # radar_separation_m (formation tightness)
    ("radar_sep=1000",     {"radar_separation_m": 1000.0}),
    ("radar_sep=2000",     {"radar_separation_m": 2000.0}),
    # map_size_m (engagement range)
    ("map_size=6000",      {"map_size_m": 6000.0}),
    ("map_size=10000",     {"map_size_m": 10000.0}),
    # geometry (random vs mirror)
    ("geometry=RANDOM",    {"geometry": RANDOM_GEOMETRY}),
]


def build_env(grid_kwargs, n_envs, horizon, seed):
    """Build env with NOMINAL overridden by grid-specific kwargs."""
    cfg = dict(NOMINAL)
    cfg.update(grid_kwargs)
    return TwoTeamVecEnv(
        n_envs=n_envs, device="cuda", episode_steps=horizon, seed=seed, **cfg,
    )


def _wrap_commander(cmd):
    def fn(env, team, _c=cmd):
        return _c.get_action(env, team)
    return fn


def _win_perspective_rule(winner_arr, rule_is_team0):
    """1.0 = rule won, 0.0 = rule lost, 0.5 = draw."""
    if rule_is_team0:
        win = (winner_arr == 0).astype(float)
        lose = (winner_arr == 1).astype(float)
    else:
        win = (winner_arr == 1).astype(float)
        lose = (winner_arr == 0).astype(float)
    draw = ((winner_arr != 0) & (winner_arr != 1)).astype(float) * 0.5
    return win + draw


def eval_rule_vs_pure_track(env, n_episodes, horizon):
    """Bidirectional eval: Rule vs pure_track. Returns dict with rule's POV metrics."""
    rule = TwoTeamStrongRuleCommander()
    pure_track = STRATEGIES["pure_track"]
    rule_fn = _wrap_commander(rule)
    pt_fn = _wrap_commander(pure_track)

    # Direction A: rule @ t0 vs pure_track @ t1
    resA = run_episodes_two_commanders(
        env, rule_fn, pt_fn, n_episodes, horizon, seed_base=1000,
    )
    # Direction B: pure_track @ t0 vs rule @ t1
    resB = run_episodes_two_commanders(
        env, pt_fn, rule_fn, n_episodes, horizon, seed_base=2000,
    )

    winA = _win_perspective_rule(resA["winner"], rule_is_team0=True)
    winB = _win_perspective_rule(resB["winner"], rule_is_team0=False)
    wins = np.concatenate([winA, winB])

    wr_mean, wr_lo, wr_hi = bootstrap_ci(wins, n_boot=10000)

    draw_rate = float(np.concatenate([
        (resA["winner"] == -1).astype(float),
        (resB["winner"] == -1).astype(float),
    ]).mean())

    rule_kills = float(np.concatenate([
        resA["kills_t0"], resB["kills_t1"],
    ]).mean())
    pt_kills = float(np.concatenate([
        resA["kills_t1"], resB["kills_t0"],
    ]).mean())

    return {
        "win_rate_mean": wr_mean,
        "win_rate_ci": (wr_lo, wr_hi),
        "n_episodes_total": len(wins),
        "rule_kills_mean": rule_kills,
        "pt_kills_mean": pt_kills,
        "draw_rate": draw_rate,
    }


def render_markdown(results, nominal_wr, cliff_threshold, out_path, cli_args):
    """Render sweep report as markdown."""
    lines = []
    lines.append("# Bet B Step 0 — Rule Sensitivity Sweep (Cheap Gate)\n")
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## Setup\n")
    lines.append(f"- Players: TwoTeamStrongRuleCommander vs `pure_track` baseline")
    lines.append(f"- Episodes per direction: {cli_args['n_episodes']} → "
                 f"total per grid: {2 * cli_args['n_episodes']}")
    lines.append(f"- Horizon: {cli_args['horizon']}, n_envs: {cli_args['n_envs']}")
    lines.append(f"- Nominal WR (rule vs pure_track): **{nominal_wr:.3f}**")
    lines.append(f"- Cliff definition: WR drop > {cliff_threshold:.2f} from nominal")
    lines.append(f"- Total grid points: {len(results)} (1 nominal + {len(results)-1} variants)\n")

    lines.append("## Sweep results\n")
    lines.append("| Config | Rule WR | 95% CI | Δ from nominal | Draw | "
                 "Rule kills | PT kills | Cliff? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        delta = r["win_rate_mean"] - nominal_wr
        is_cliff = "**CLIFF**" if (-delta) > cliff_threshold else ""
        marker = " ← NOMINAL" if r["name"] == "nominal" else ""
        lines.append(
            f"| `{r['name']}`{marker} | **{r['win_rate_mean']:.3f}** | "
            f"[{r['win_rate_ci'][0]:.3f}, {r['win_rate_ci'][1]:.3f}] | "
            f"{delta:+.3f} | {r['draw_rate']:.2f} | "
            f"{r['rule_kills_mean']:.2f} | {r['pt_kills_mean']:.2f} | {is_cliff} |"
        )

    variants = [r for r in results if r["name"] != "nominal"]
    cliffs = [r for r in variants
              if (nominal_wr - r["win_rate_mean"]) > cliff_threshold]
    n_cliffs = len(cliffs)
    n_variants = len(variants)
    brittleness = n_cliffs / n_variants if n_variants else 0.0

    # Also count "rule loses to pure_track" (WR < 0.5) — strongest cliff signal
    loses_to_pt = [r for r in variants if r["win_rate_mean"] < 0.5]
    n_loses = len(loses_to_pt)

    lines.append("\n## Verdict\n")
    lines.append(f"- **Cliff count** (WR drop > {cliff_threshold:.2f}): "
                 f"**{n_cliffs} / {n_variants}** "
                 f"(brittleness score = {brittleness:.2f})")
    lines.append(f"- **Rule loses to pure_track** (WR < 0.5): "
                 f"**{n_loses} / {n_variants}**")

    if n_cliffs <= 1:
        verdict = ("rule 全程稳健 — Bet B 前提死 → 退 IET 地板")
        emoji = "❌"
    elif n_cliffs <= 4:
        verdict = ("中度脆 — DR 训练专攻 cliff 轴")
        emoji = "⚠️"
    else:
        verdict = ("高度脆 — 全轴 DR 训练")
        emoji = "✅"
    lines.append(f"- **Verdict**: {emoji} {verdict}")

    if cliffs:
        lines.append("\n### Cliff list (rule drops > {:.2f} from nominal)".format(cliff_threshold))
        for c in cliffs:
            lines.append(f"- `{c['name']}`: WR = {c['win_rate_mean']:.3f} "
                         f"(Δ = {c['win_rate_mean'] - nominal_wr:+.3f}, "
                         f"CI [{c['win_rate_ci'][0]:.3f}, {c['win_rate_ci'][1]:.3f}])")

    if loses_to_pt:
        lines.append("\n### Rule loses to pure_track (WR < 0.5)")
        for c in loses_to_pt:
            lines.append(f"- `{c['name']}`: WR = {c['win_rate_mean']:.3f}")

    lines.append("\n## Decision tree\n")
    if n_cliffs <= 1:
        lines.append("→ **Hard stop**. Bet B premise dead. **Pivot to IET floor**.")
        lines.append("  - 3-line near-Nash evidence (G0 #3 + V1 exploits + rule design) is strong.")
        lines.append("  - Sensitivity sweep adds 4th line: 'rule is also robust to off-nominal physics.'")
        lines.append("  - AppInt pivot dead. IET is the right venue.")
    else:
        lines.append("→ **Bet B alive**. Proceed to Step 1 (DR-PPO training).")
        cliff_axes = sorted(set(c["name"].split("=")[0] for c in cliffs))
        lines.append(f"  - Focus DR distribution on cliff axes: `{cliff_axes}`")
        lines.append("  - 6-axis full DR also OK (training cost same).")

    lines.append("\n## Per-axis reading\n")
    axes_seen = []
    for r in variants:
        axis = r["name"].split("=")[0]
        if axis not in axes_seen:
            axes_seen.append(axis)
    for axis in axes_seen:
        axis_variants = [r for r in variants if r["name"].startswith(axis + "=")]
        if not axis_variants:
            continue
        deltas = [r["win_rate_mean"] - nominal_wr for r in axis_variants]
        min_delta = min(deltas)
        max_delta = max(deltas)
        lines.append(f"- **{axis}**: Δ range [{min_delta:+.3f}, {max_delta:+.3f}] "
                     f"({len(axis_variants)} variants)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written: {out_path}")
    return n_cliffs, n_loses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100,
                   help="Episodes per direction (total = 2x this)")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--cliff-threshold", type=float, default=0.15,
                   help="WR drop from nominal that counts as a cliff")
    p.add_argument("--out", type=str,
                   default="experiments/twoteam/rule_sensitivity_sweep.md")
    args = p.parse_args()

    print(f"[gpu] Using device: cuda")
    print(f"[gpu] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[gpu] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)
    print("Bet B Step 0 — Rule Sensitivity Sweep (Cheap Gate)")
    print("=" * 70)
    print(f"\nSweep: {len(GRID)} grid points (1 nominal + {len(GRID)-1} variants)")
    print(f"Each grid: {args.episodes} ep x 2 directions x horizon={args.horizon} "
          f"= {2*args.episodes} ep total")
    print(f"Estimated total: ~{len(GRID) * 2 * args.episodes} episodes")

    results = []
    nominal_wr = None
    t_start = time.time()
    for i, (name, kwargs) in enumerate(GRID):
        print(f"\n[{i+1}/{len(GRID)}] {name}")
        t0 = time.time()
        env = build_env(kwargs, args.n_envs, args.horizon, seed=42)
        r = eval_rule_vs_pure_track(env, args.episodes, args.horizon)
        r["name"] = name
        r["grid_kwargs"] = kwargs
        elapsed = time.time() - t0
        delta_str = ""
        if nominal_wr is not None and name != "nominal":
            delta_str = f"  Δ={r['win_rate_mean'] - nominal_wr:+.3f}"
        print(f"  WR={r['win_rate_mean']:.3f} CI=[{r['win_rate_ci'][0]:.3f}, {r['win_rate_ci'][1]:.3f}]"
              f"  draw={r['draw_rate']:.2f}  kills={r['rule_kills_mean']:.2f} vs {r['pt_kills_mean']:.2f}"
              f"{delta_str}  ({elapsed:.1f}s)")
        if name == "nominal":
            nominal_wr = r["win_rate_mean"]
        results.append(r)
        del env
        torch.cuda.empty_cache()

    total_elapsed = time.time() - t_start
    print(f"\nTotal sweep time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    cli_args = {
        "n_episodes": args.episodes, "horizon": args.horizon,
        "n_envs": args.n_envs,
    }
    n_cliffs, n_loses = render_markdown(
        results, nominal_wr, args.cliff_threshold, args.out, cli_args,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Nominal WR (rule vs pure_track): {nominal_wr:.3f}")
    print(f"  Cliff count: {n_cliffs} / {len(results)-1} "
          f"(threshold {args.cliff_threshold})")
    print(f"  Rule loses to pure_track: {n_loses} / {len(results)-1}")
    if n_cliffs <= 1:
        print(f"\n  ❌ Verdict: rule robust → Bet B premise dead → pivot to IET")
    elif n_cliffs <= 4:
        print(f"\n  ⚠️ Verdict: moderate brittleness → DR train focused on cliff axes")
    else:
        print(f"\n  ✅ Verdict: high brittleness → full-axis DR train")


if __name__ == "__main__":
    main()
