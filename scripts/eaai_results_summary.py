"""EAAI results summary generator.

Consumes JSON outputs from all WP eval scripts and produces:
  - Consolidated markdown table for paper §5 (Core Metrics)
  - PASS/FAIL report per EAAI gate
  - Consolidated JSON for figure pipeline

Inputs (all optional, skip if missing):
  logs/wp3_robustness_eval.json     (WP3.2 robustness sweep)
  logs/wp3_crlb_achieved.json       (WP3.1 achieved RMSE / CRLB)
  logs/wp4_generalization.json      (WP4 OOD generalization)
  logs/wp2_main_comparison.json     (WP2 main win-rate matrix)

Outputs:
  logs/eaai_summary.md              paper-ready summary table
  logs/eaai_summary.json            structured pass/fail report

Usage:
    python scripts/eaai_results_summary.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
        return None


def _fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{digits}f}%"


def _fmt_float(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def summarize_wp3_robustness(data: dict) -> dict:
    """WP3.2 robustness sweep summary."""
    if not data:
        return {"status": "missing"}
    results = data.get("results", {})
    baseline = results.get("baseline", {})
    cells = {k: v for k, v in results.items() if k != "baseline"}
    return {
        "status": "ok",
        "baseline_kill_rate": baseline.get("kill_rate"),
        "baseline_illum": baseline.get("mean_illumination_progress"),
        "n_cells_pass": data.get("n_cells_pass"),
        "n_cells_total": data.get("n_cells_total"),
        "overall_pass": data.get("overall_pass"),
        "cells": cells,
    }


def summarize_wp3_crlb(data: dict) -> dict:
    """WP3.1 achieved RMSE / CRLB ratio."""
    if not data:
        return {"status": "missing"}
    return {
        "status": "ok",
        "achieved_rmse_m": data.get("achieved", {}).get("rmse_m"),
        "crlb_tracked_m": data.get("crlb", {}).get("rmse_tracked_m"),
        "crlb_static_m": data.get("crlb", {}).get("rmse_static_m"),
        "ratio": data.get("ratio"),
        "ratio_target": data.get("ratio_target"),
        "pass": data.get("pass"),
        "baseline_m": data.get("crlb", {}).get("mean_baseline_m"),
        "target_range_m": data.get("crlb", {}).get("mean_target_range_m"),
        "gdop": data.get("crlb", {}).get("gdop"),
    }


def summarize_wp4_generalization(data: dict) -> dict:
    """WP4 OOD generalization matrix."""
    if not data:
        return {"status": "missing"}
    results = data.get("results", {})
    baseline = results.get("baseline", {})
    conditions = {k: v for k, v in results.items() if k != "baseline"}
    return {
        "status": "ok",
        "baseline_kill_rate": baseline.get("kill_rate"),
        "n_conditions_pass": data.get("n_conditions_pass"),
        "n_conditions_total": data.get("n_conditions_total"),
        "overall_pass": data.get("overall_pass"),
        "conditions": conditions,
    }


def summarize_wp2_main(data: dict) -> dict:
    """WP2 main comparison win-rate matrix."""
    if not data:
        return {"status": "missing"}
    return {
        "status": "ok",
        "entries": data.get("entry_names", []),
        "matrix": data.get("matrix"),
        "per_entry": data.get("per_entry", {}),
    }


def render_markdown(
    wp3_rob: dict, wp3_crlb: dict, wp4_gen: dict, wp2_main: dict,
    checkpoint_dir: str, output_path: Path,
):
    """Render paper-ready summary markdown."""
    lines = []
    lines.append("# EAAI Evaluation Summary")
    lines.append("")
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Checkpoint**: `{checkpoint_dir}`")
    lines.append("")

    # WP3.1 CRLB
    lines.append("## WP3.1 — Physical Anchor (CRLB Ratio)")
    lines.append("")
    if wp3_crlb["status"] == "ok":
        c = wp3_crlb
        lines.append("| Metric | Value | Target | PASS |")
        lines.append("|---|---|---|---|")
        lines.append(f"| Achieved RMSE | {_fmt_float(c['achieved_rmse_m'], 4)} m | — | — |")
        lines.append(f"| CRLB (tracked, N=120) | {_fmt_float(c['crlb_tracked_m'], 4)} m | — | — |")
        lines.append(f"| CRLB (static, N=1) | {_fmt_float(c['crlb_static_m'], 4)} m | — | — |")
        lines.append(f"| **Ratio achieved/CRLB** | **{_fmt_float(c['ratio'], 3)}×** | "
                     f"≤ {c['ratio_target']:.1f}× | "
                     f"{'✓' if c['pass'] else '✗'} |")
        lines.append(f"| Deployment baseline | {_fmt_float(c['baseline_m'], 0)} m | "
                     f"≥ 5000 m | — |")
        lines.append(f"| Target range | {_fmt_float(c['target_range_m'], 0)} m | — | — |")
        lines.append(f"| GDOP | {_fmt_float(c['gdop'], 2)} | < 5 | — |")
    else:
        lines.append("_No data — run `scripts/wp3_crlb_achieved.py`._")
    lines.append("")

    # WP3.2 Robustness
    lines.append("## WP3.2 — Robustness Sweep (5 Damage Cells)")
    lines.append("")
    if wp3_rob["status"] == "ok":
        r = wp3_rob
        lines.append(f"Baseline (no damage): kill_rate = "
                     f"{_fmt_float(r['baseline_kill_rate'], 3)}, "
                     f"illum = {_fmt_float(r['baseline_illum'], 4)}")
        lines.append("")
        lines.append("| Damage Cell | kill_rate | Retention | illum | Retention | PASS |")
        lines.append("|---|---|---|---|---|---|")
        for name, m in r["cells"].items():
            lines.append(
                f"| `{name}` | "
                f"{_fmt_float(m['kill_rate'], 3)} | "
                f"{_fmt_pct(m.get('kill_rate_retention'))} | "
                f"{_fmt_float(m['mean_illumination_progress'], 4)} | "
                f"{_fmt_pct(m.get('illum_retention'))} | "
                f"{'✓' if m.get('pass') else '✗'} |"
            )
        lines.append("")
        lines.append(f"**Overall**: {r['n_cells_pass']}/{r['n_cells_total']} cells pass "
                     f"(threshold 70% kill_rate retention) "
                     f"{'✓' if r['overall_pass'] else '✗'}")
    else:
        lines.append("_No data — run `scripts/wp3_robustness_eval.py`._")
    lines.append("")

    # WP4 Generalization
    lines.append("## WP4 — OOD Generalization")
    lines.append("")
    if wp4_gen["status"] == "ok":
        g = wp4_gen
        lines.append(f"Train baseline: kill_rate = "
                     f"{_fmt_float(g['baseline_kill_rate'], 3)}")
        lines.append("")
        lines.append("| OOD Condition | Axis | kill_rate | Retention | PASS |")
        lines.append("|---|---|---|---|---|")
        for name, m in g["conditions"].items():
            lines.append(
                f"| `{name}` | {m.get('variation_axis', '?')} | "
                f"{_fmt_float(m['kill_rate'], 3)} | "
                f"{_fmt_pct(m.get('kill_rate_retention'))} | "
                f"{'✓' if m.get('pass') else '✗'} |"
            )
        lines.append("")
        lines.append(f"**Overall**: {g['n_conditions_pass']}/{g['n_conditions_total']} "
                     f"conditions pass "
                     f"{'✓' if g['overall_pass'] else '✗'}")
    else:
        lines.append("_No data — run `scripts/wp4_generalization_eval.py`._")
    lines.append("")

    # WP2 Main comparison
    lines.append("## WP2 — Main Comparison (Win-Rate Matrix)")
    lines.append("")
    if wp2_main["status"] == "ok":
        m_data = wp2_main
        entries = m_data["entries"]
        matrix = m_data["matrix"]
        if entries and matrix:
            # Header
            header = "| Red \\ Blue | " + " | ".join(entries) + " |"
            sep = "|" + "---|" * (len(entries) + 1)
            lines.append(header)
            lines.append(sep)
            for i, row_name in enumerate(entries):
                row = []
                for j, _ in enumerate(entries):
                    if i == j:
                        row.append("—")
                    else:
                        row.append(f"{matrix[i][j]:.3f}")
                lines.append(f"| **{row_name}** | " + " | ".join(row) + " |")
            lines.append("")
            lines.append("Per-entry aggregates (red perspective, avg vs all opponents):")
            lines.append("")
            lines.append("| Entry | Avg red win rate | Mean kill rate |")
            lines.append("|---|---|---|")
            for name, m in m_data["per_entry"].items():
                lines.append(f"| `{name}` | "
                             f"{_fmt_float(m.get('avg_red_win_rate'), 3)} | "
                             f"{_fmt_float(m.get('mean_kill_rate_as_red'), 3)} |")
    else:
        lines.append("_No data — run `scripts/wp2_main_comparison.py`._")
    lines.append("")

    # Overall EAAI gate
    lines.append("## Overall EAAI Gate Status")
    lines.append("")
    gates = []
    if wp3_crlb["status"] == "ok":
        gates.append(("WP3.1 CRLB ratio ≤ 1.3×",
                      wp3_crlb["pass"],
                      f"ratio = {wp3_crlb['ratio']:.3f}×"))
    if wp3_rob["status"] == "ok":
        gates.append(("WP3.2 robustness ≥ 70% retention (all 5 cells)",
                      wp3_rob["overall_pass"],
                      f"{wp3_rob['n_cells_pass']}/{wp3_rob['n_cells_total']} cells pass"))
    if wp4_gen["status"] == "ok":
        gates.append(("WP4 OOD generalization ≥ 70% retention (all 6 conditions)",
                      wp4_gen["overall_pass"],
                      f"{wp4_gen['n_conditions_pass']}/{wp4_gen['n_conditions_total']} conditions pass"))
    if not gates:
        lines.append("_No gate data available yet._")
    else:
        lines.append("| Gate | PASS | Detail |")
        lines.append("|---|---|---|")
        for name, passed, detail in gates:
            lines.append(f"| {name} | {'✓' if passed else '✗'} | {detail} |")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--output-md", type=Path,
                    default=Path("logs/eaai_summary.md"))
    ap.add_argument("--output-json", type=Path,
                    default=Path("logs/eaai_summary.json"))
    args = ap.parse_args()

    # Load all available JSONs
    wp3_rob_data = _load_json(args.logs_dir / "wp3_robustness_eval.json")
    wp3_crlb_data = _load_json(args.logs_dir / "wp3_crlb_achieved.json")
    wp4_gen_data = _load_json(args.logs_dir / "wp4_generalization.json")
    wp2_main_data = _load_json(args.logs_dir / "wp2_main_comparison.json")

    summary = {
        "wp3_robustness": summarize_wp3_robustness(wp3_rob_data),
        "wp3_crlb": summarize_wp3_crlb(wp3_crlb_data),
        "wp4_generalization": summarize_wp4_gen(wp4_gen_data),
        "wp2_main_comparison": summarize_wp2_main(wp2_main_data),
    }

    # Find checkpoint dir from any of the inputs
    checkpoint_dir = "unknown"
    for d in (wp3_rob_data, wp3_crlb_data, wp4_gen_data, wp2_main_data):
        if d and "checkpoint_dir" in d:
            checkpoint_dir = d["checkpoint_dir"]
            break

    # Write JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_dir": checkpoint_dir,
        **summary,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[JSON] wrote {args.output_json}")

    # Write markdown
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    render_markdown(summary["wp3_robustness"], summary["wp3_crlb"],
                    summary["wp4_generalization"], summary["wp2_main_comparison"],
                    checkpoint_dir, args.output_md)
    print(f"[MD  ] wrote {args.output_md}")
    print()

    # Brief stdout summary
    print("=" * 60)
    print("EAAI Summary (brief)")
    print("=" * 60)
    if summary["wp3_crlb"]["status"] == "ok":
        c = summary["wp3_crlb"]
        print(f"WP3.1 CRLB ratio: {c['ratio']:.3f}× "
              f"(target ≤ {c['ratio_target']:.1f}×) "
              f"{'PASS' if c['pass'] else 'FAIL'}")
    if summary["wp3_robustness"]["status"] == "ok":
        r = summary["wp3_robustness"]
        print(f"WP3.2 robustness: {r['n_cells_pass']}/{r['n_cells_total']} cells pass "
              f"{'PASS' if r['overall_pass'] else 'FAIL'}")
    if summary["wp4_generalization"]["status"] == "ok":
        g = summary["wp4_generalization"]
        print(f"WP4 OOD generalization: {g['n_conditions_pass']}/{g['n_conditions_total']} conditions pass "
              f"{'PASS' if g['overall_pass'] else 'FAIL'}")
    if summary["wp2_main_comparison"]["status"] == "ok":
        m = summary["wp2_main_comparison"]
        print(f"WP2 entries compared: {len(m['entries'])}")


if __name__ == "__main__":
    main()
