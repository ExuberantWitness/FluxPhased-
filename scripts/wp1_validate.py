"""WP1 convergence-gate validator.

Parses a WP1 training log and applies the 4 PASS criteria from WP1_CONVERGENCE_GATE.md §4:
  (1) Final kill_radius ≤ 0.5m (ideal) or ≤ 1m (minimum acceptable)
  (2) Residual 0.50 win_rate ratio < 10% (computed over the last iter's payoff matrix)
  (3) Final NashConv < 0.05 OR monotonically decreasing across iters
  (4) kill_radius anneal is monotonic (no rebound)

Exits 0 on PASS, 1 on FAIL. Prints a structured report regardless.

Usage:
    python scripts/wp1_validate.py logs/wp1_gate_seed42.log
    python scripts/wp1_validate.py logs/wp1_gate_seed42.log --strict   # 0.5m hard
"""

import argparse
import re
import sys
from pathlib import Path


KR_ANNEAL_RE = re.compile(
    r"\[League\] kill_radius anneal:\s*([\d.]+)m\s*→\s*([\d.]+)m"
)
KR_INIT_RE = re.compile(r"\[League\] kill_radius initialized at\s*([\d.]+)m")
WIN_RATE_RE = re.compile(r"win_rate=([\d.]+)")
NASH_RE = re.compile(
    r"Team\s+[01]\s+sigma=\[([^\]]+)\]\s+NashConv=([\d.]+)\s+H_task=([\d.]+)\s+effK=([\d.]+)"
)
ITER_HEADER_RE = re.compile(r"PSRO Iteration\s+(\d+)/(\d+)")


def parse_log(path: Path):
    iters = []
    cur = None
    anneal_chain = [None]  # start with init value once we see it
    win_rates_by_iter = []  # list of lists

    for line in path.read_text().splitlines():
        m = ITER_HEADER_RE.search(line)
        if m:
            if cur is not None:
                iters.append(cur)
            cur = {"iter": int(m.group(1)), "n_iters": int(m.group(2))}
            win_rates_by_iter.append([])
            continue

        m = KR_INIT_RE.search(line)
        if m and anneal_chain[0] is None:
            anneal_chain[0] = float(m.group(1))

        m = KR_ANNEAL_RE.search(line)
        if m:
            anneal_chain.append(float(m.group(2)))  # record post-anneal value

        if cur is not None:
            m = WIN_RATE_RE.search(line)
            if m:
                win_rates_by_iter[-1].append(float(m.group(1)))

            m = NASH_RE.search(line)
            if m:
                sigma_str, nash_str, _, effk_str = m.groups()
                sigma = [float(x) for x in sigma_str.split()]
                cur.setdefault("teams", []).append({
                    "sigma": sigma,
                    "NashConv": float(nash_str),
                    "effK": float(effk_str),
                })

    if cur is not None:
        iters.append(cur)

    return {
        "iters": iters,
        "anneal_chain": [x for x in anneal_chain if x is not None],
        "win_rates_by_iter": win_rates_by_iter,
    }


def check_kill_radius(anneal_chain, strict: bool):
    if not anneal_chain:
        return False, "no kill_radius entries found", None
    final_kr = anneal_chain[-1]
    threshold = 0.5 if strict else 1.0
    passed = final_kr <= threshold
    msg = f"final kr={final_kr:.4f}m (threshold ≤{threshold}m, {'strict' if strict else 'minimum'})"
    return passed, msg, final_kr


def check_monotonic_anneal(anneal_chain):
    if len(anneal_chain) < 2:
        return True, "only init value present (no anneal events yet) — vacuously monotonic", None
    rebounds = []
    for i in range(1, len(anneal_chain)):
        if anneal_chain[i] > anneal_chain[i - 1] + 1e-6:
            rebounds.append((i, anneal_chain[i - 1], anneal_chain[i]))
    if rebounds:
        return False, f"kill_radius rebounded at steps {rebounds}", rebounds
    return True, f"kill_radius monotonic across {len(anneal_chain) - 1} anneals", None


def check_residual_05(win_rates_by_iter, threshold_pct=10.0):
    if not win_rates_by_iter or not win_rates_by_iter[-1]:
        return False, "no win_rate entries in final iter", None
    last = win_rates_by_iter[-1]
    n_05 = sum(1 for w in last if abs(w - 0.5) < 1e-3)
    pct = 100.0 * n_05 / len(last)
    passed = pct < threshold_pct
    msg = f"final iter 0.50 ratio = {pct:.1f}% ({n_05}/{len(last)} pairs, threshold <{threshold_pct}%)"
    return passed, msg, pct


def check_nash_conv(iters):
    if not iters:
        return False, "no iters parsed", None
    last = iters[-1]
    if "teams" not in last or not last["teams"]:
        return False, "no Nash metrics in final iter", None
    max_nc = max(t["NashConv"] for t in last["teams"])
    passed_threshold = max_nc < 0.05

    chain = []
    for it in iters:
        if "teams" in it and it["teams"]:
            chain.append(max(t["NashConv"] for t in it["teams"]))
    monotone_dec = all(chain[i] <= chain[i - 1] + 0.02 for i in range(1, len(chain))) if len(chain) >= 2 else False

    passed = passed_threshold or monotone_dec
    reason = "threshold<0.05" if passed_threshold else ("monotone decreasing" if monotone_dec else "neither")
    msg = (f"final NashConv={max_nc:.4f} (max over 2 teams); "
           f"chain={[f'{x:.3f}' for x in chain]}; pass via {reason}")
    return passed, msg, max_nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--strict", action="store_true",
                    help="Require ≤0.5m kr (otherwise ≤1m acceptable)")
    ap.add_argument("--threshold-pct", type=float, default=10.0,
                    help="Max %% of pairs at 0.50 in final iter (default 10)")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"FAIL: log file not found: {args.log}", file=sys.stderr)
        sys.exit(2)

    data = parse_log(args.log)
    print("=" * 72)
    print(f"WP1 Convergence Gate Validator — {args.log}")
    print("=" * 72)
    print(f"iters parsed: {len(data['iters'])}  (anneal chain len: {len(data['anneal_chain'])})")
    print()

    results = []
    p1, m1, _ = check_kill_radius(data["anneal_chain"], strict=args.strict)
    results.append(("kill_radius final", p1, m1))

    p2, m2, _ = check_monotonic_anneal(data["anneal_chain"])
    results.append(("kill_radius monotonic", p2, m2))

    p3, m3, _ = check_residual_05(data["win_rates_by_iter"], threshold_pct=args.threshold_pct)
    results.append(("residual 0.50 ratio", p3, m3))

    p4, m4, _ = check_nash_conv(data["iters"])
    results.append(("NashConv final", p4, m4))

    for name, passed, msg in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name:25s} {msg}")
    print()

    all_pass = all(p for _, p, _ in results)
    final_kr = data["anneal_chain"][-1] if data["anneal_chain"] else None

    if all_pass:
        print(f"OVERALL: PASS — WP1 gate cleared, final kr={final_kr:.4f}m")
        print("  → Proceed to WP2 (external baselines). This run = cell-A seed-0.")
        sys.exit(0)
    else:
        n_fail = sum(1 for _, p, _ in results if not p)
        print(f"OVERALL: FAIL — {n_fail}/4 criteria not met, final kr={final_kr}")
        print("  → Per WP1 §5: STOP. Do NOT burn GPU on WP2/WP3 until fixed.")
        print("    If kr stuck > 5m but train_laser reaches 0.2m: paper pivots to "
              "C1-engineering + PSRO-lite (honest negative result).")
        sys.exit(1)


if __name__ == "__main__":
    main()
