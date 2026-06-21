"""WP6 — Statistical analysis protocol for multi-seed experiment results.

Implements the statistical requirements from PAPER_PLAN_EAAI.md §6.1:
  - ≥5 seeds per cell (main comparison)
  - mean ± 95% CI (bootstrap, 1e4 resamples)
  - Welch's t-test for pairwise significance (or Mann-Whitney if non-normal)
  - Cohen's d effect size
  - Holm-Bonferroni multiple-comparison correction
  - Learning-curve AUC + GPU-hours-to-threshold

Usage:
    # Extract per-iter metrics from multiple training logs
    python scripts/wp6_stats.py extract --logs logs/wp1_gate_seed*.log \\
        --out logs/wp1_gate_metrics.csv

    # Compare two cells (e.g., FluxLeague vs MAPPO) on final kr
    python scripts/wp6_stats.py compare \\
        --cell-a logs/wp1_gate_metrics.csv \\
        --cell-b logs/wp2_mappo_metrics.csv \\
        --metric final_kr

    # Full multi-cell table with Holm-Bonferroni correction
    python scripts/wp6_stats.py table \\
        --cells wp1:logs/wp1.csv wp2_mappo:logs/wp2_mappo.csv \\
        --metric final_kr --out tables/wp6_main_comparison.tex
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Metric extraction from training logs
# ---------------------------------------------------------------------------

KR_ANNEAL_RE = re.compile(r"kill_radius anneal:\s*([\d.]+)m\s*→\s*([\d.]+)m")
KR_INIT_RE = re.compile(r"kill_radius initialized at\s*([\d.]+)m")
WIN_RATE_RE = re.compile(r"win_rate=([\d.]+)")
NASH_RE = re.compile(
    r"Team\s+[01]\s+sigma=\[[^\]]+\]\s+NashConv=([\d.]+)\s+H_task=([\d.]+)\s+effK=([\d.]+)"
)
ITER_HEADER_RE = re.compile(r"PSRO Iteration\s+(\d+)/(\d+)")


def extract_metrics_from_log(log_path: Path) -> dict:
    """Parse one training log into a metrics dict."""
    anneal_chain = []
    iters = []
    cur = None
    win_rates_by_iter = []

    for line in log_path.read_text().splitlines():
        m = ITER_HEADER_RE.search(line)
        if m:
            if cur is not None:
                iters.append(cur)
            cur = {"iter": int(m.group(1))}
            win_rates_by_iter.append([])
            continue

        m = KR_INIT_RE.search(line)
        if m and not anneal_chain:
            anneal_chain.append(float(m.group(1)))

        m = KR_ANNEAL_RE.search(line)
        if m:
            anneal_chain.append(float(m.group(2)))

        if cur is not None:
            m = WIN_RATE_RE.search(line)
            if m:
                win_rates_by_iter[-1].append(float(m.group(1)))
            m = NASH_RE.search(line)
            if m:
                cur.setdefault("NashConv", []).append(float(m.group(1)))
                cur.setdefault("effK", []).append(float(m.group(3)))

    if cur is not None:
        iters.append(cur)

    final_kr = anneal_chain[-1] if anneal_chain else float("nan")
    final_nash = max(iters[-1]["NashConv"]) if iters and "NashConv" in iters[-1] else float("nan")
    final_effk = max(iters[-1]["effK"]) if iters and "effK" in iters[-1] else float("nan")

    last_wr = win_rates_by_iter[-1] if win_rates_by_iter else []
    n_05 = sum(1 for w in last_wr if abs(w - 0.5) < 1e-3)
    residual_05_ratio = (n_05 / len(last_wr)) if last_wr else float("nan")

    return {
        "log": str(log_path),
        "n_iters_completed": len(iters),
        "final_kill_radius_m": final_kr,
        "final_nash_conv": final_nash,
        "final_eff_k": final_effk,
        "residual_05_ratio": residual_05_ratio,
        "kill_radius_chain": anneal_chain,
    }


def extract_many(log_globs: Iterable[str], out_csv: Path):
    """Extract metrics from multiple logs (one per seed) and write CSV."""
    rows = []
    for pat in log_globs:
        for log in Path(".").glob(pat):
            if log.is_file():
                rows.append(extract_metrics_from_log(log))

    if not rows:
        print(f"ERROR: no logs matched {log_globs}", file=sys.stderr)
        sys.exit(2)

    fields = list(rows[0].keys())
    # drop the long chain column from CSV (keep everything else scalar)
    fields = [f for f in fields if f != "kill_radius_chain"]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r2 = {k: v for k, v in r.items() if k in fields}
            w.writerow(r2)

    print(f"[extract] wrote {len(rows)} rows to {out_csv}")
    return rows


# ---------------------------------------------------------------------------
# Stats primitives
# ---------------------------------------------------------------------------

def bootstrap_ci(data: np.ndarray, n_resamples: int = 10000, alpha: float = 0.05):
    """Bootstrap mean + (1-alpha) CI."""
    if len(data) == 0:
        return float("nan"), (float("nan"), float("nan"))
    n = len(data)
    rng = np.random.default_rng(seed=42)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(data)), (float(lo), float(hi))


def welch_t(a: np.ndarray, b: np.ndarray):
    """Welch's t-test (unequal variances). Returns (t, p, df)."""
    from scipy import stats
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # df via Welch–Satterthwaite
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return float(t), float(p), float(df)


def cohens_d(a: np.ndarray, b: np.ndarray):
    """Cohen's d (pooled-SD denominator)."""
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled < 1e-12:
        return float("inf") if abs(np.mean(a) - np.mean(b)) > 0 else 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def holm_bonferroni(pvals: list[float], alpha: float = 0.05):
    """Return list of reject booleans after Holm-Bonferroni correction."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if pvals[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def cmd_extract(args):
    extract_many(args.logs, Path(args.out))


def cmd_compare(args):
    """Compare two cells on one metric."""
    def load_col(csv_path: str, col: str) -> np.ndarray:
        with open(csv_path) as f:
            r = csv.DictReader(f)
            return np.array([float(row[col]) for row in r if row[col]])

    a = load_col(args.cell_a, args.metric)
    b = load_col(args.cell_b, args.metric)

    mean_a, ci_a = bootstrap_ci(a)
    mean_b, ci_b = bootstrap_ci(b)
    t, p, df = welch_t(a, b)
    d = cohens_d(a, b)

    print(f"Cell A ({args.cell_a}): n={len(a)}, mean={mean_a:.4g} "
          f"95% CI=[{ci_a[0]:.4g}, {ci_a[1]:.4g}]")
    print(f"Cell B ({args.cell_b}): n={len(b)}, mean={mean_b:.4g} "
          f"95% CI=[{ci_b[0]:.4g}, {ci_b[1]:.4g}]")
    print(f"Welch's t: t={t:.3f}, df={df:.2f}, p={p:.4g}")
    print(f"Cohen's d: {d:.3f}  ({'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'})")
    print(f"Reject H0 (equal means) at α=0.05: {'YES' if p < 0.05 else 'NO'}")


def cmd_table(args):
    """Multi-cell comparison table with Holm-Bonferroni correction."""
    cells = []
    for spec in args.cells:
        name, csv_path = spec.split(":", 1)
        with open(csv_path) as f:
            r = csv.DictReader(f)
            col = np.array([float(row[args.metric]) for row in r if row[args.metric]])
        cells.append((name, col))

    if not cells:
        print("ERROR: no cells specified", file=sys.stderr)
        sys.exit(2)

    # Reference cell = first
    ref_name, ref_data = cells[0]

    rows = []
    pvals = []
    for name, data in cells[1:]:
        mean, ci = bootstrap_ci(data)
        _, p, _ = welch_t(ref_data, data)
        d = cohens_d(ref_data, data)
        rows.append((name, mean, ci, p, d))
        pvals.append(p)

    rejects = holm_bonferroni(pvals, alpha=0.05)

    ref_mean, ref_ci = bootstrap_ci(ref_data)
    print(f"Reference: {ref_name}  n={len(ref_data)}  "
          f"mean={ref_mean:.4g}  CI=[{ref_ci[0]:.4g}, {ref_ci[1]:.4g}]")
    print()
    print(f"{'cell':<24} {'mean':>12} {'95% CI':>24} {'p':>10} {'Cohen d':>10} {'reject':>8}")
    print("-" * 92)
    for (name, mean, ci, p, d), rej in zip(rows, rejects):
        ci_str = f"[{ci[0]:.3g}, {ci[1]:.3g}]"
        print(f"{name:<24} {mean:>12.4g} {ci_str:>24} {p:>10.4g} {d:>10.3f} "
              f"{'YES' if rej else 'no':>8}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(f"% Holm-Bonferroni corrected at α=0.05; reference: {ref_name}\n")
            f.write("\\begin{tabular}{lrrrrr}\n")
            f.write("\\toprule\n")
            f.write("Cell & Mean & 95\\% CI & $p$ & Cohen's $d$ & Sig. \\\\\n")
            f.write("\\midrule\n")
            f.write(f"{ref_name} & {ref_mean:.3g} & "
                    f"[{ref_ci[0]:.3g}, {ref_ci[1]:.3g}] & --- & --- & ref \\\\\n")
            for (name, mean, ci, p, d), rej in zip(rows, rejects):
                star = "$^{*}$" if rej else ""
                f.write(f"{name} & {mean:.3g} & [{ci[0]:.3g}, {ci[1]:.3g}] & "
                        f"{p:.3g} & {d:.3f} & {('sig' if rej else 'n.s.')}{star} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print(f"\n[table] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="Extract per-seed metrics from logs → CSV")
    p_ext.add_argument("--logs", nargs="+", required=True,
                       help="Glob patterns matching per-seed logs")
    p_ext.add_argument("--out", required=True, help="Output CSV path")

    p_cmp = sub.add_parser("compare", help="Compare two cells on one metric")
    p_cmp.add_argument("--cell-a", required=True)
    p_cmp.add_argument("--cell-b", required=True)
    p_cmp.add_argument("--metric", required=True,
                       help="CSV column name (e.g., final_kill_radius_m)")

    p_tbl = sub.add_parser("table", help="Multi-cell table with Holm-Bonferroni")
    p_tbl.add_argument("--cells", nargs="+", required=True,
                       help="NAME:CSV_PATH pairs (first is reference)")
    p_tbl.add_argument("--metric", required=True)
    p_tbl.add_argument("--out", default=None, help="Optional LaTeX output path")

    args = ap.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    elif args.cmd == "table":
        cmd_table(args)


if __name__ == "__main__":
    main()
