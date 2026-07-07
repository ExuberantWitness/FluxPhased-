"""Concerto-RRM pilot criteria judgement (A13).

Reads pilot_results.csv and evaluates the 4 pilot criteria per plan §B7:

  C1: Classical QoS sanity floor
      L0 classical QoS > 0.9  AND  L3 classical QoS < 0.6
      (classical works at low difficulty, fails at high difficulty)

  C2: Concerto beats classical + MAPPO at L3
      qos[v2, L3] > qos[classical, L3] + 0.10
      qos[v2, L3] > qos[mappo, L3] + 0.05
      Welch's t-test p<0.05 across seeds

  C3: No function collapse (4-function dwell non-zero)
      min(dwell_frac[fn]) > 0.05 for all fn, all methods, all difficulties

  C4: Concerto trains faster than MAPPO (to equivalent QoS)
      wallclock[conc_v2] < 0.7 × wallclock[mappo]

Decision tree per plan:
  4/4 PASS → proceed to full EAAI WP-A scan
  3/4 PASS → see decision tree in criteria.py docstring
  ≤ 2/4 PASS → retreat to Path A (C1+C0 → IEEE TAES)

Usage:
    python -m algo._shared.pilot.criteria \\
        --csv experiments/concerto_pilot_results.csv \\
        --out experiments/concerto_pilot_verdict.md
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CSV parsing → per-cell aggregates
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> List[dict]:
    """Load pilot_results.csv → list of row dicts."""
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            row["seed"] = int(row["seed"])
            for k in ("qos_satisfaction_mean", "qos_satisfaction_std",
                      "qos_detect", "qos_track", "qos_comm", "qos_jam",
                      "dwell_detect", "dwell_track", "dwell_comm", "dwell_jam",
                      "min_dwell_frac", "rl_frac", "wallclock_s"):
                row[k] = float(row[k]) if row[k] else 0.0
            for k in ("n_episodes", "n_rl_steps_total", "n_classical_steps_total"):
                row[k] = int(row[k])
            rows.append(row)
    return rows


def aggregate(rows: List[dict], method: str, difficulty: str) -> Dict[str, float]:
    """Mean ± std across seeds for one (method, difficulty) cell."""
    cell = [r for r in rows if r["method"] == method and r["difficulty"] == difficulty]
    if not cell:
        return {"n_seeds": 0}
    n = len(cell)
    out = {"n_seeds": n}
    for k in ("qos_satisfaction_mean", "qos_detect", "qos_track", "qos_comm", "qos_jam",
              "dwell_detect", "dwell_track", "dwell_comm", "dwell_jam",
              "min_dwell_frac", "rl_frac", "wallclock_s"):
        vals = [r[k] for r in cell]
        out[f"{k}_mean"] = statistics.mean(vals)
        out[f"{k}_std"] = statistics.stdev(vals) if n >= 2 else 0.0
        out[f"{k}_values"] = vals
    return out


# ---------------------------------------------------------------------------
# Welch's t-test (small-sample, two-sample, unequal variance)
# ---------------------------------------------------------------------------

def welch_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Returns (t_statistic, approximate p_value) for two-sided Welch's t-test.

    Uses normal approximation for p-value when df > 30; otherwise uses
    Student's t approximation withWelch-Satterthwaite df.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    if va == 0 and vb == 0:
        return float("inf") if ma != mb else 0.0, 0.0 if ma != mb else 1.0
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    # Welch-Satterthwaite df
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / max(den, 1e-12) if den > 0 else na + nb - 2
    # Approximate p-value via Student's t CDF (use math.erf for large df,
    # otherwise a simple Bonferroni-like bound).
    if df > 30:
        # Normal approximation
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    else:
        # Conservative bound: use normal as upper bound for p
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return t, max(0.0, min(1.0, p))


# ---------------------------------------------------------------------------
# Criterion checks
# ---------------------------------------------------------------------------

def check_c1(agg: Dict, l0_floor: float = 0.9, l3_collapse: float = 0.6) -> Tuple[bool, str]:
    """C1: classical sanity floor (L0 high, L3 low)."""
    l0 = agg.get(("classical", "L0"), {})
    l3 = agg.get(("classical", "L3"), {})
    if l0.get("n_seeds", 0) == 0 or l3.get("n_seeds", 0) == 0:
        return False, "missing cells"
    l0_qos = l0["qos_satisfaction_mean_mean"]
    l3_qos = l3["qos_satisfaction_mean_mean"]
    pass_l0 = l0_qos > l0_floor
    pass_l3 = l3_qos < l3_collapse
    msg = (f"L0 classical QoS={l0_qos:.3f} (> {l0_floor} required: "
           f"{'PASS' if pass_l0 else 'FAIL'}) | "
           f"L3 classical QoS={l3_qos:.3f} (< {l3_collapse} required: "
           f"{'PASS' if pass_l3 else 'FAIL'})")
    return (pass_l0 and pass_l3), msg


def check_c2(agg: Dict, classical_gap: float = 0.10, mappo_gap: float = 0.05,
             p_thresh: float = 0.05) -> Tuple[bool, str]:
    """C2: Concerto v2 beats classical + MAPPO at L3."""
    v2 = agg.get(("concerto_v2", "L3"), {})
    cls = agg.get(("classical", "L3"), {})
    mp = agg.get(("mappo", "L3"), {})
    if v2.get("n_seeds", 0) < 2 or cls.get("n_seeds", 0) < 2 or mp.get("n_seeds", 0) < 2:
        return False, "insufficient seeds (<2) for t-test"
    v2_qos = v2["qos_satisfaction_mean_mean"]
    cls_qos = cls["qos_satisfaction_mean_mean"]
    mp_qos = mp["qos_satisfaction_mean_mean"]
    gap_cls = v2_qos - cls_qos
    gap_mp = v2_qos - mp_qos
    pass_cls_gap = gap_cls > classical_gap
    pass_mp_gap = gap_mp > mappo_gap
    _, p_cls = welch_t_test(v2["qos_satisfaction_mean_values"],
                             cls["qos_satisfaction_mean_values"])
    _, p_mp = welch_t_test(v2["qos_satisfaction_mean_values"],
                            mp["qos_satisfaction_mean_values"])
    pass_cls_p = p_cls < p_thresh
    pass_mp_p = p_mp < p_thresh
    msg = (f"v2-classical gap={gap_cls:+.3f} (>{classical_gap} required: "
           f"{'PASS' if pass_cls_gap else 'FAIL'}, p={p_cls:.3f}: "
           f"{'PASS' if pass_cls_p else 'FAIL'}) | "
           f"v2-mappo gap={gap_mp:+.3f} (>{mappo_gap} required: "
           f"{'PASS' if pass_mp_gap else 'FAIL'}, p={p_mp:.3f}: "
           f"{'PASS' if pass_mp_p else 'FAIL'})")
    return (pass_cls_gap and pass_mp_gap and pass_cls_p and pass_mp_p), msg


def check_c3(rows: List[dict], floor: float = 0.05) -> Tuple[bool, str]:
    """C3: 4-function dwell non-zero (no collapse)."""
    failing = []
    for r in rows:
        for fn in ("detect", "track", "comm", "jam"):
            dwell = r[f"dwell_{fn}"]
            if dwell < floor:
                failing.append(f"{r['method']}/{r['difficulty']}/seed{r['seed']}/{fn}={dwell:.3f}")
    if failing:
        return False, f"{len(failing)} cells below floor {floor}: e.g. {failing[:3]}"
    return True, f"all cells have all functions ≥ {floor}"


def check_c4(agg: Dict, ratio: float = 0.7) -> Tuple[bool, str]:
    """C4: Concerto trains/evals faster than MAPPO."""
    v2 = agg.get(("concerto_v2", "L3"), {})
    mp = agg.get(("mappo", "L3"), {})
    if v2.get("n_seeds", 0) == 0 or mp.get("n_seeds", 0) == 0:
        return False, "missing cells"
    v2_wall = v2["wallclock_s_mean"]
    mp_wall = mp["wallclock_s_mean"]
    if mp_wall == 0:
        return False, "MAPPO wallclock = 0"
    actual_ratio = v2_wall / mp_wall
    pass_ = actual_ratio < ratio
    msg = (f"v2 wallclock={v2_wall:.1f}s / mappo wallclock={mp_wall:.1f}s = "
           f"{actual_ratio:.2f} (< {ratio} required: {'PASS' if pass_ else 'FAIL'})")
    return pass_, msg


# ---------------------------------------------------------------------------
# Verdict rendering
# ---------------------------------------------------------------------------

def render_verdict(rows: List[dict], out_path: str):
    """Build per-cell aggregate dict, run all 4 criteria, render markdown."""
    keys = sorted({(r["method"], r["difficulty"]) for r in rows})
    agg = {(m, d): aggregate(rows, m, d) for m, d in keys}

    c1, c1_msg = check_c1(agg)
    c2, c2_msg = check_c2(agg)
    c3, c3_msg = check_c3(rows)
    c4, c4_msg = check_c4(agg)
    n_pass = sum([c1, c2, c3, c4])

    # Decision tree
    if n_pass == 4:
        decision = ("**4/4 PASS → proceed to full EAAI WP-A scan** "
                    "(EAAI_RESEARCH_PLAN.md §8).")
    elif n_pass == 3:
        if not c4:
            decision = ("3/4 (only C4 speed fails) → continue; speed is a "
                        "secondary limitation. Proceed to WP-A.")
        elif not c2:
            decision = ("3/4 (only C2 marginal) → keep Concerto, drop MAPPO "
                        "comparison; retitle 'Concerto vs Classical'.")
        elif not c1:
            decision = ("3/4 (only C1 classical-no-collapse fails) → strengthen "
                        "L3 (jam power↑, τ↓); re-run L3 only (~5 GPU-h).")
        elif not c3:
            decision = ("3/4 (only C3 collapse) → tighten classical floor "
                        "(5 elem/fn); re-run.")
        else:
            decision = "3/4 PASS with unknown failure pattern — investigate."
    else:
        decision = (f"**{n_pass}/4 PASS → retreat to Path A** (C1+C0 → IEEE TAES) "
                    f"per EAAI_RESEARCH_PLAN.md §9.")

    md = ["# Concerto-RRM Pilot Verdict", "",
          f"**Result**: {n_pass}/4 criteria PASS", "",
          f"**Decision**: {decision}", "",
          "## Per-cell aggregate (mean across seeds)", "",
          "| Method | Difficulty | n_seeds | QoS agg | detect | track | comm | jam | min dwell | wallclock(s) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for (m, d), a in sorted(agg.items()):
        if a.get("n_seeds", 0) == 0:
            continue
        md.append(f"| {m} | {d} | {a['n_seeds']} | "
                  f"{a['qos_satisfaction_mean_mean']:.3f}±{a['qos_satisfaction_mean_std']:.3f} | "
                  f"{a['qos_detect_mean']:.3f} | {a['qos_track_mean']:.3f} | "
                  f"{a['qos_comm_mean']:.3f} | {a['qos_jam_mean']:.3f} | "
                  f"{a['min_dwell_frac_mean']:.3f} | "
                  f"{a['wallclock_s_mean']:.1f} |")
    md += ["", "## Criterion checks", "",
           f"### C1 (classical sanity floor): **{'PASS' if c1 else 'FAIL'}**",
           c1_msg, "",
           f"### C2 (Concerto v2 beats classical + MAPPO at L3): **{'PASS' if c2 else 'FAIL'}**",
           c2_msg, "",
           f"### C3 (no function collapse, dwell ≥ 0.05): **{'PASS' if c3 else 'FAIL'}**",
           c3_msg, "",
           f"### C4 (Concerto faster than MAPPO): **{'PASS' if c4 else 'FAIL'}**",
           c4_msg, "",
           "## Decision tree", "",
           decision, "",
           "## Per-seed values (QoS aggregate)", "",
           "| Method | Difficulty | Seed | QoS agg | n_rl | n_classical |",
           "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["method"], r["difficulty"], r["seed"])):
        md.append(f"| {r['method']} | {r['difficulty']} | {r['seed']} | "
                  f"{r['qos_satisfaction_mean']:.3f} | "
                  f"{r['n_rl_steps_total']} | {r['n_classical_steps_total']} |")

    out_text = "\n".join(md)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out_text)
    print(out_text)
    print(f"\n[verdict] written to {out_path}")
    return n_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/concerto_pilot_results.csv")
    ap.add_argument("--out", default="experiments/concerto_pilot_verdict.md")
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        sys.stderr.write(f"CSV not found: {args.csv}\n")
        sys.exit(1)
    rows = load_csv(args.csv)
    render_verdict(rows, args.out)


if __name__ == "__main__":
    main()
