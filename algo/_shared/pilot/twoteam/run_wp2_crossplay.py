"""WP2 cross-play tournament: Elo + non-transitivity + G2 baseline check.

Per plan snuggly-exploring-parrot.md Step 4 + WP2_BC_LEAGUE_PLAN.md.

Setup:
  - Method set: {league-commander, StrongRule, MAPPO baseline, IPPO baseline,
                 7 extreme, 3 candidate-exploit}
  - All-vs-all bidirectional round-robin (A@t0 vs B@t1 + B@t0 vs A@t1)
  - Held-out seeds (different from training seeds) for honest eval

Outputs:
  - Cross-play win rate matrix (n × n)
  - Elo ratings (1e4 bootstrap → 95% CI)
  - Head-to-head league vs rule (per-seed + mean + CI)
  - Non-transitivity: DFS cycle detection on directed graph (A→B if WR > 0.55)
  - G2: league Elo > MAPPO Elo, CI excludes 0

Output file: experiments/twoteam/wp2_crossplay_report.md
"""

from __future__ import annotations
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from typing import Dict, List, Tuple, Callable, Optional

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.extreme_commanders import STRATEGIES, combine_team_actions
from algo._shared.pilot.twoteam.run_g0_gate import (
    run_episodes_two_commanders, bootstrap_ci,
)


# ----------------------------------------------------------------------
# Method loading (returns dict {name: action_fn})
# ----------------------------------------------------------------------

def _wrap_commander(cmd):
    def fn(env, team, _c=cmd):
        return _c.get_action(env, team)
    return fn


def _wrap_ac_ckpt(ckpt_path: str, device: str = "cuda", deterministic: bool = True):
    ac = TwoTeamCommanderActorCritic().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ac_state", ckpt)
    ac.load_state_dict(state)
    ac.eval()

    @torch.no_grad()
    def fn(env, team):
        obs_dict = env.get_obs()
        detect_t = env.get_detect_list()[:, team]   # WP-3 M0/M1
        action, _ = ac.get_action_for_env(
            obs_dict["obs"][:, team], detect_t, obs_dict["privileged"][:, team],
            deterministic=deterministic,
        )
        return action

    def cleanup():
        nonlocal ac
        del ac
        torch.cuda.empty_cache()

    return fn, cleanup


def load_methods(args) -> Dict[str, dict]:
    """Load all eval methods. Each value is {action_fn, cleanup, kind}."""
    methods = {}

    # Always include StrongRule + extremes + exploits
    methods["strong_rule"] = {
        "action_fn": _wrap_commander(TwoTeamStrongRuleCommander()),
        "cleanup": lambda: None, "kind": "rule",
    }
    extreme_names = ["pure_track", "pure_jam", "pure_comm", "pure_detect",
                     "balanced", "balanced_jam_heavy", "track_agile"]
    for nm in extreme_names:
        methods[f"extreme/{nm}"] = {
            "action_fn": _wrap_commander(STRATEGIES[nm]),
            "cleanup": lambda: None, "kind": "extreme",
        }
    exploit_names = ["jam_spread", "hard_jam_focus", "track_heavy_agile"]
    for nm in exploit_names:
        methods[f"exploit/{nm}"] = {
            "action_fn": _wrap_commander(STRATEGIES[nm]),
            "cleanup": lambda: None, "kind": "script",
        }

    # AC checkpoint methods (league, MAPPO, IPPO)
    if args.league_ckpt:
        fn, cleanup = _wrap_ac_ckpt(args.league_ckpt, deterministic=True)
        methods["league"] = {"action_fn": fn, "cleanup": cleanup, "kind": "checkpoint"}
    if args.mappo_ckpt:
        fn, cleanup = _wrap_ac_ckpt(args.mappo_ckpt, deterministic=True)
        methods["MAPPO"] = {"action_fn": fn, "cleanup": cleanup, "kind": "checkpoint"}
    if args.ippo_ckpt:
        fn, cleanup = _wrap_ac_ckpt(args.ippo_ckpt, deterministic=True)
        methods["IPPO"] = {"action_fn": fn, "cleanup": cleanup, "kind": "checkpoint"}

    return methods


# ----------------------------------------------------------------------
# Bidirectional cross-play
# ----------------------------------------------------------------------

def _winner_arr_to_povA(winner: np.ndarray) -> np.ndarray:
    """winner codes: 0=t0, 1=t1, -1=draw → A's POV: 1=win, 0=loss, 0.5=draw."""
    return np.where(winner == 0, 1.0, np.where(winner == 1, 0.0, 0.5))


def play_bidirectional(
    env, A_fn, B_fn, n_episodes: int, horizon: int, seed_base: int = 5000,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """A vs B, both directions. Returns (A_pov_winrate, all_A_wins, all_kill_deltas)."""
    # Direction 1: A@t0, B@t1
    res1 = run_episodes_two_commanders(env, A_fn, B_fn, n_episodes, horizon,
                                       seed_base=seed_base)
    # Direction 2: B@t0, A@t1 (flip perspective)
    res2 = run_episodes_two_commanders(env, B_fn, A_fn, n_episodes, horizon,
                                       seed_base=seed_base + n_episodes)

    wins1 = _winner_arr_to_povA(res1["winner"])                 # A is t0
    wins2 = 1.0 - _winner_arr_to_povA(res2["winner"])           # A is t1 → flip
    all_wins = np.concatenate([wins1, wins2])

    kill_delta_1 = res1["kills_t0"] - res1["kills_t1"]          # A is t0
    kill_delta_2 = res2["kills_t1"] - res2["kills_t0"]          # A is t1
    all_kill_deltas = np.concatenate([kill_delta_1, kill_delta_2])

    return float(all_wins.mean()), all_wins, all_kill_deltas


def run_tournament(
    methods: Dict[str, dict], env, n_episodes: int, horizon: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Run all-vs-all bidirectional tournament.

    Returns:
        wr_matrix: [n_methods, n_methods] — wr_matrix[i, j] = win rate of method i vs j (i's POV)
        kd_matrix: [n_methods, n_methods] — mean kill delta (i's kills − j's kills)
        names: method names in matrix order
    """
    names = list(methods.keys())
    n = len(names)
    wr_matrix = np.full((n, n), np.nan)
    kd_matrix = np.full((n, n), np.nan)
    np.fill_diagonal(wr_matrix, 0.5)
    np.fill_diagonal(kd_matrix, 0.0)

    print(f"\nRunning {n}x{n} tournament ({n * (n - 1)} pairings, "
          f"{2 * n_episodes} episodes each)...")
    t0 = time.time()
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_fn = methods[names[i]]["action_fn"]
            B_fn = methods[names[j]]["action_fn"]
            wr_mean, all_wins, all_kd = play_bidirectional(
                env, A_fn, B_fn, n_episodes, horizon, seed_base=5000 + i * 100 + j)
            wr_matrix[i, j] = wr_mean
            wr_matrix[j, i] = 1.0 - wr_mean
            kd_matrix[i, j] = float(all_kd.mean())
            kd_matrix[j, i] = -float(all_kd.mean())
            done += 1
            elapsed = time.time() - t0
            print(f"  [{done}/{n * (n - 1) // 2}] {names[i]:25s} vs {names[j]:25s}  "
                  f"WR(i→j)={wr_mean:.3f}  t={elapsed:.1f}s",
                  flush=True)

    return wr_matrix, kd_matrix, names


# ----------------------------------------------------------------------
# Elo from win rate matrix
# ----------------------------------------------------------------------

def compute_elo(wr_matrix: np.ndarray, k: float = 32.0, n_rounds: int = 50,
                n_boot: int = 1000, rng_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Elo ratings from win-rate matrix.

    Uses logistic Elo update over n_rounds passes. Bootstrap by resampling
    the off-diagonal entries' implied win rates (treat each entry as a Bernoulli
    sample with p = wr_matrix[i,j]).

    Returns:
        elo: [n_methods] final Elo ratings
        elo_ci: [n_methods, 2] bootstrap 95% CI
    """
    n = wr_matrix.shape[0]
    rng = np.random.RandomState(rng_seed)

    # Point estimate
    elo = np.zeros(n)
    for _ in range(n_rounds):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p = wr_matrix[i, j]
                # Add tiny noise to avoid exact 0/1 saturation
                p_noisy = np.clip(p + rng.normal(0, 0.01), 1e-3, 1 - 1e-3)
                expected = 1.0 / (1.0 + 10 ** ((elo[j] - elo[i]) / 400.0))
                update = k * (p_noisy - expected)
                elo[i] += update

    # Bootstrap CI: resample off-diagonal entries as Bernoulli(p_ij)
    elo_boot = np.zeros((n_boot, n))
    for b in range(n_boot):
        wr_b = wr_matrix.copy()
        for i in range(n):
            for j in range(n):
                if i != j:
                    wr_b[i, j] = rng.binomial(1, wr_matrix[i, j])
                    # If both directions played, ensure consistency isn't required
        e = np.zeros(n)
        for _ in range(n_rounds):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    p = wr_b[i, j]
                    p_noisy = np.clip(p + rng.normal(0, 0.01), 1e-3, 1 - 1e-3)
                    expected = 1.0 / (1.0 + 10 ** ((e[j] - e[i]) / 400.0))
                    e[i] += k * (p_noisy - expected)
        elo_boot[b] = e

    elo_ci = np.zeros((n, 2))
    for i in range(n):
        elo_ci[i, 0] = np.percentile(elo_boot[:, i], 2.5)
        elo_ci[i, 1] = np.percentile(elo_boot[:, i], 97.5)

    return elo, elo_ci


# ----------------------------------------------------------------------
# Non-transitivity detection (DFS cycle search)
# ----------------------------------------------------------------------

def detect_cycles(wr_matrix: np.ndarray, names: List[str],
                  threshold: float = 0.55, max_cycle_len: int = 3) -> List[List[str]]:
    """Detect directed cycles in the dominance graph (i→j if WR(i,j) > threshold).

    Returns list of cycles (each as a list of method names).
    """
    n = len(names)
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and wr_matrix[i, j] > threshold:
                adj[i].append(j)

    cycles = []
    seen_cycle_keys = set()

    def _dfs(start: int, current: int, path: List[int], visited: set):
        if len(path) > max_cycle_len:
            return
        for nxt in adj[current]:
            if nxt == start and len(path) >= 2:
                # Found a cycle: path → start
                key = tuple(sorted(path))
                if key not in seen_cycle_keys:
                    seen_cycle_keys.add(key)
                    cycles.append([names[p] for p in path])
            elif nxt not in visited and nxt > start:
                # Only explore nodes > start to dedupe cycles by smallest element
                visited.add(nxt)
                _dfs(start, nxt, path + [nxt], visited)
                visited.discard(nxt)

    for s in range(n):
        _dfs(s, s, [s], {s})

    return cycles


# ----------------------------------------------------------------------
# Report rendering
# ----------------------------------------------------------------------

def render_report(
    out_path: str, names: List[str], wr_matrix: np.ndarray, kd_matrix: np.ndarray,
    elo: np.ndarray, elo_ci: np.ndarray, cycles: List[List[str]],
    args: dict, league_idx: Optional[int] = None, mappo_idx: Optional[int] = None,
    rule_idx: Optional[int] = None,
):
    """Render cross-play report as markdown."""
    n = len(names)
    lines = []
    lines.append("# WP2 Cross-Play Tournament Report\n")
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## Setup\n")
    lines.append(f"- Methods: {n}")
    lines.append(f"- Episodes per pairing (each direction): {args['n_episodes']}")
    lines.append(f"- Total episodes per pairing (bidirectional): {2 * args['n_episodes']}")
    lines.append(f"- Horizon: {args['horizon']}, n_envs: {args['n_envs']}")
    lines.append(f"- Dominance threshold for cycle detection: WR > {args['cycle_threshold']}\n")

    lines.append("## Cross-play win-rate matrix\n")
    lines.append("Rows = player, Cols = opponent. Values = row's win rate (draw=0.5).\n")
    header = "| player \\ opponent | " + " | ".join(f"`{nm}`" for nm in names) + " |"
    sep = "|" + "---|" * (n + 1)
    lines.append(header)
    lines.append(sep)
    for i, nm in enumerate(names):
        row_vals = []
        for j in range(n):
            if i == j:
                row_vals.append("—")
            else:
                row_vals.append(f"{wr_matrix[i, j]:.3f}")
        lines.append(f"| `{nm}` | " + " | ".join(row_vals) + " |")

    lines.append("\n## Elo ratings\n")
    lines.append("| Method | Elo | 95% CI |")
    lines.append("|---|---|---|")
    # Sort by Elo descending
    elo_order = np.argsort(-elo)
    for idx in elo_order:
        nm = names[idx]
        e = elo[idx]
        lo, hi = elo_ci[idx]
        lines.append(f"| `{nm}` | {e:+.1f} | [{lo:+.1f}, {hi:+.1f}] |")

    lines.append("\n## G1 — League vs StrongRule (head-to-head)\n")
    if league_idx is not None and rule_idx is not None:
        wr = wr_matrix[league_idx, rule_idx]
        kd = kd_matrix[league_idx, rule_idx]
        verdict = "G1 PASS" if (wr > 0.5) else "G1 FAIL"
        lines.append(f"- Win rate (league's POV): **{wr:.3f}**")
        lines.append(f"- Kill delta (league − rule): {kd:+.3f}")
        lines.append(f"- Verdict (threshold WR > 0.5): **{verdict}**")
        if wr > 0.5:
            lines.append("  - Note: bootstrap CI on this single entry needs full per-episode"
                        " data; the bidirectional win-rate is averaged across all episodes.")
    else:
        lines.append("- League or StrongRule method not present in this tournament.")

    lines.append("\n## G1-clean — Non-transitivity check\n")
    if cycles:
        lines.append(f"⚠️ **{len(cycles)} dominance cycle(s) detected** (length ≤ "
                     f"{args['cycle_len']}):")
        for c in cycles[:20]:
            lines.append(f"  - {' → '.join(f'`{nm}`' for nm in c)} → `{c[0]}`")
        lines.append("\nIf league wins G1 but is in a cycle, the win is **rock-paper-scissors**, "
                     "not robust dominance → not a clean TAES champion.")
    else:
        lines.append(f"✅ No dominance cycles detected at threshold WR > {args['cycle_threshold']}.")
        if league_idx is not None:
            league_row = wr_matrix[league_idx]
            n_strong_wins = int((league_row > args['cycle_threshold']).sum())
            n_losses = int((league_row < 0.5).sum())
            lines.append(f"- League dominates {n_strong_wins} of {n - 1} opponents "
                         f"(WR > {args['cycle_threshold']}).")
            lines.append(f"- League loses to {n_losses} of {n - 1} opponents (WR < 0.5).")

    lines.append("\n## G2 — League vs MAPPO baseline\n")
    if league_idx is not None and mappo_idx is not None:
        league_elo = elo[league_idx]
        mappo_elo = elo[mappo_idx]
        diff = league_elo - mappo_elo
        # CI on the difference (rough — assumes independence)
        lo_diff = (elo_ci[league_idx, 0] - elo_ci[mappo_idx, 1])
        hi_diff = (elo_ci[league_idx, 1] - elo_ci[mappo_idx, 0])
        verdict = "G2 PASS" if (diff > 0 and lo_diff > 0) else "G2 FAIL"
        lines.append(f"- League Elo: {league_elo:+.1f}  (CI [{elo_ci[league_idx, 0]:+.1f}, {elo_ci[league_idx, 1]:+.1f}])")
        lines.append(f"- MAPPO Elo:  {mappo_elo:+.1f}  (CI [{elo_ci[mappo_idx, 0]:+.1f}, {elo_ci[mappo_idx, 1]:+.1f}])")
        lines.append(f"- Difference: {diff:+.1f}  (rough CI [{lo_diff:+.1f}, {hi_diff:+.1f}])")
        lines.append(f"- Verdict (league > MAPPO, CI excludes 0): **{verdict}**")
    else:
        lines.append("- Either league or MAPPO baseline not present in this tournament. "
                     "Run `--mappo-ckpt` flag to enable G2 check.")

    lines.append("\n## WP2 decision tree\n")
    g1_pass = (league_idx is not None and rule_idx is not None and wr_matrix[league_idx, rule_idx] > 0.5)
    g1_clean = (not cycles) and g1_pass
    g2_pass = (league_idx is not None and mappo_idx is not None and
               elo[league_idx] > elo[mappo_idx])
    if g1_pass and g1_clean:
        lines.append("**✅ G1 PASS + G1-clean (robust dominance) → TAES champion established**")
        lines.append("  → Proceed to WP3 (operational envelope, ablations, CRLB, statistics)")
    elif g1_pass and not g1_clean:
        lines.append("**⚠️ G1 PASS but non-transitive (rock-paper-scissors) → IET (non-transitive dynamics)**")
        lines.append("  → Still a contribution, but NOT a clean 'RL > classical' claim")
    elif g1_pass and not g2_pass:
        lines.append("**⚠️ G1 PASS but G2 FAIL → marginal: league doesn't clearly beat MAPPO**")
        lines.append("  → Review whether MAPPO got same compute budget; may need ablation")
    else:
        lines.append("**❌ G1 FAIL (league doesn't beat rule) → IET floor + Bet B (generalization)**")
        lines.append("  → Rule ≈ Nash is the honest answer; pivot to robustness/generalization eval")

    lines.append("\n## Kill-delta matrix (reference)\n")
    lines.append("Values = row's kills − column's kills (mean per episode).\n")
    lines.append(header)
    lines.append(sep)
    for i, nm in enumerate(names):
        row_vals = []
        for j in range(n):
            if i == j:
                row_vals.append("—")
            else:
                row_vals.append(f"{kd_matrix[i, j]:+.2f}")
        lines.append(f"| `{nm}` | " + " | ".join(row_vals) + " |")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written: {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--league-ckpt", type=str, default=None,
                   help="League final checkpoint (e.g. checkpoints/twoteam/wp2_league/iter_final.pt)")
    p.add_argument("--mappo-ckpt", type=str, default=None,
                   help="MAPPO baseline checkpoint (no league, same compute budget)")
    p.add_argument("--ippo-ckpt", type=str, default=None,
                   help="IPPO baseline checkpoint")
    p.add_argument("--n-episodes", type=int, default=30,
                   help="Episodes per direction per pairing (total = 2x this)")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--cycle-threshold", type=float, default=0.55,
                   help="WR threshold for dominance edge in non-transitivity graph")
    p.add_argument("--cycle-len", type=int, default=3,
                   help="Max cycle length to search")
    p.add_argument("--out", type=str, default="experiments/twoteam/wp2_crossplay_report.md")
    args = p.parse_args()

    print(f"[gpu] Using device: cuda")
    print(f"[gpu] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[gpu] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)
    print("WP2 Cross-Play Tournament")
    print("=" * 70)

    env = TwoTeamVecEnv(
        n_envs=args.n_envs, device="cuda", episode_steps=args.horizon,
        geometry=RANDOM_GEOMETRY, seed=700,
    )

    methods = load_methods(args)
    print(f"\nLoaded {len(methods)} methods:")
    for name, m in methods.items():
        print(f"  - {name:30s}  (kind: {m['kind']})")

    # Run tournament
    wr_matrix, kd_matrix, names = run_tournament(
        methods, env, n_episodes=args.n_episodes, horizon=args.horizon,
    )

    # Elo
    print("\nComputing Elo...")
    elo, elo_ci = compute_elo(wr_matrix, k=32.0, n_rounds=50, n_boot=1000)

    # Non-transitivity
    print("Detecting non-transitivity...")
    cycles = detect_cycles(wr_matrix, names,
                           threshold=args.cycle_threshold,
                           max_cycle_len=args.cycle_len)
    if cycles:
        print(f"  ⚠️ {len(cycles)} cycle(s) detected")
    else:
        print(f"  ✅ No cycles at threshold {args.cycle_threshold}")

    # Indices for gate checks
    league_idx = names.index("league") if "league" in names else None
    mappo_idx = names.index("MAPPO") if "MAPPO" in names else None
    rule_idx = names.index("strong_rule") if "strong_rule" in names else None

    # Render report
    render_report(
        args.out, names, wr_matrix, kd_matrix, elo, elo_ci, cycles,
        cli_args_to_dict(args),
        league_idx=league_idx, mappo_idx=mappo_idx, rule_idx=rule_idx,
    )

    # Save raw matrices
    raw_path = os.path.join(os.path.dirname(args.out) or ".", "wp2_crossplay_raw.npz")
    np.savez(raw_path,
             names=np.array(names),
             wr_matrix=wr_matrix, kd_matrix=kd_matrix,
             elo=elo, elo_ci=elo_ci)
    print(f"Raw matrices → {raw_path}")

    # Cleanup
    for m in methods.values():
        m["cleanup"]()

    # Summary print
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Elo ranking:")
    for idx in np.argsort(-elo):
        nm = names[idx]
        e = elo[idx]
        lo, hi = elo_ci[idx]
        marker = " ← CHAMPION" if nm == "league" else (" ← BASELINE" if nm in ("MAPPO", "IPPO") else "")
        print(f"    {nm:30s}  Elo={e:+.1f} (CI [{lo:+.1f}, {hi:+.1f}]){marker}")
    if league_idx is not None and rule_idx is not None:
        wr = wr_matrix[league_idx, rule_idx]
        verdict = "G1 PASS" if wr > 0.5 else "G1 FAIL"
        print(f"\n  G1 (league vs rule): WR={wr:.3f} → {verdict}")
    if cycles:
        print(f"\n  ⚠️ Non-transitivity: {len(cycles)} cycle(s) at WR>{args.cycle_threshold}")
    else:
        print(f"\n  ✅ No dominance cycles")


def cli_args_to_dict(args) -> dict:
    return {
        "n_episodes": args.n_episodes, "horizon": args.horizon,
        "n_envs": args.n_envs,
        "cycle_threshold": args.cycle_threshold, "cycle_len": args.cycle_len,
    }


if __name__ == "__main__":
    main()
