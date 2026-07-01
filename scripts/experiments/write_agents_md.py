#!/usr/bin/env python3
"""
write_agents_md.py — generate the per-run AGENTS.md.

The AGENTS.md file is the entry point for any future AI agent (or human) who
lands in this directory cold. It answers four questions in 60 seconds:
  1. What was this run trying to do?
  2. Did it pass its gate?
  3. How do I reproduce / extend it?
  4. Where is everything else?

USAGE:
  python scripts/experiments/write_agents_md.py \
      --run-dir experiments/phase1.5_mappo_seed42 \
      --baseline-dir experiments/phase1_pfsp_seed42
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def render_run(run_dir: Path, baseline_dir: Path | None) -> str:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        m = _load(metrics_path)
    else:
        m = {"final": {}, "iter_count": 0, "eval_count": 0}
    try:
        meta = _load(run_dir / "metadata.json")
    except FileNotFoundError:
        meta = {}
    final = m.get("final") or {}

    # Optional comparison values
    b_final = None
    if baseline_dir and (baseline_dir / "metrics.json").is_file():
        bm = _load(baseline_dir / "metrics.json")
        b_final = bm.get("final") or {}

    is_mappo = "mappo" in run_dir.name.lower()
    purpose = (
        "**MAPPO baseline** — same env/reward/curriculum as PfspFix, but switches the "
        "critic to a **team critic** (CTDE) and **disables PFSP** (uniform opponent "
        "sampling). Answers the EAAI reviewer question: *why AlphaStar league instead "
        "of MAPPO?*"
    ) if is_mappo else (
        "**PfspFix reference baseline** — verified recipe (commit `911c5ef`), seed=42 "
        "bit-exact. This is the reference every Phase 1.5+ run compares against."
    )

    gate_status = "N/A (reference)"
    if is_mappo and b_final:
        cr = final.get("cum_red", float("nan"))
        if cr >= 0.75:
            gate_status = "PASS ✅ (cum_red ≥ 0.75)"
        elif cr < 0.60:
            gate_status = "FAIL ❌ (cum_red < 0.60)"
        else:
            gate_status = f"MARGINAL ⚠️ (cum_red={cr:.3f})"

    lines = [
        f"# {run_dir.name}", "",
        f"> {purpose}", "",
        "## TL;DR", "",
    ]
    if m.get("iter_count", 0) == 0:
        lines += [
            "> _Training has not started yet, or `metrics.json` has not been generated. "
            "> Run `parse_log_to_metrics.py` after / during training to populate this section._",
            "",
        ]
    elif final:
        lines += [
            f"- **kr (final train)**: `{final.get('kr_train_m', 'n/a')}m`  "
            f"(curriculum floor: 0.5m)",
            f"- **eval kill_rate @ iter {final.get('iter', '?')}**: "
            f"`{final.get('eval_kill_rate', 'n/a')}`",
            f"- **cum red / blue / draw**: "
            f"`{final.get('cum_red', 0):.3f} / {final.get('cum_blue', 0):.3f} / "
            f"{final.get('cum_draw', 0):.3f}`",
            f"- **aim residual**: `{final.get('aim_res_m', 'n/a')}m`",
            f"- **adv_std (last)**: `{final.get('adv_std_last', 'n/a')}`  "
            f"(health: 1e-3 < x < 50)",
            f"- **cmd policy_loss (last)**: `{final.get('cmd_pl_last', 'n/a')}`  "
            f"(collapse watch: |x| > 1e-4)",
            "",
        ]

    if is_mappo and b_final:
        if m.get("iter_count", 0) == 0:
            lines += [
                "## Gate vs PfspFix baseline", "",
                "- **Gate status**: PENDING (training has not finished)",
                f"- PfspFix reference: cum_red={b_final.get('cum_red', 0):.3f}, "
                f"aim_res={b_final.get('aim_res_m', 0):.3f}m, "
                f"eval_kr={b_final.get('eval_kill_rate', 0):.3f}",
                f"- This run: _(populated by `compare_runs.py` once training finishes)_",
                "- See `comparison.md` for full PASS/FAIL table and per-iter deltas.",
                "",
            ]
        else:
            lines += [
                "## Gate vs PfspFix baseline", "",
                f"- **Gate status**: {gate_status}",
                f"- PfspFix reference: cum_red={b_final.get('cum_red', 0):.3f}, "
                f"aim_res={b_final.get('aim_res_m', 0):.3f}m, "
                f"eval_kr={b_final.get('eval_kill_rate', 0):.3f}",
                f"- This run:        cum_red={final.get('cum_red', 0):.3f}, "
                f"aim_res={final.get('aim_res_m', 0):.3f}m, "
                f"eval_kr={final.get('eval_kill_rate', 0):.3f}",
                f"- See `comparison.md` for full PASS/FAIL table and per-iter deltas.",
                "",
            ]

    lines += [
        "## Files in this directory", "",
        "| File | What |",
        "|---|---|",
        "| `config.yaml` | Frozen config snapshot (survives edits to live yaml) |",
        "| `metadata.json` | seed, git commit, host, reproduce_cmd, wall-clock |",
        "| `metrics.json` | Per-iter structured metrics (machine-readable) |",
        "| `train.log` | Raw stdout from training |",
        "| `figures/*.png` | Curve plots (kr, cum_red, aim_res, adv_std, cmd_pl, eval_kr) |",
        "| `reproduce.sh` | One-command re-run |",
        "| `AGENTS.md` | This file |",
        "| `comparison.md` | (Candidate runs only) Diff vs PfspFix baseline |",
        "",
        "## Reproduce", "",
        "```bash",
        meta.get("reproduce_cmd",
                 "bash scripts/run_train.sh <config> <log_path>"),
        "```",
        "",
        f"_Commit: `{meta.get('git', {}).get('commit', 'unknown')[:12]}`  "
        f"branch: `{meta.get('git', {}).get('branch', 'unknown')}`  "
        f"dirty: `{meta.get('git', {}).get('dirty', 'unknown')}`  "
        f"seed: `{meta.get('seed', 'unknown')}`  "
        f"wall: `{meta.get('timing', {}).get('wall_clock_hours', 'unknown')}h`_",
        "",
        "## Parse / re-plot", "",
        "```bash",
        "# regenerate metrics.json from train.log",
        "python scripts/experiments/parse_log_to_metrics.py \\",
        f"    --log {Path(meta.get('log_path', 'train.log')).name} \\",
        "    --out metrics.json --run-id " + run_dir.name,
        "",
        "# regenerate figures",
        "python scripts/experiments/make_figures.py \\",
        f"    --run {run_dir.name}:metrics.json --out-dir figures",
        "```",
        "",
    ]
    return "\n".join(lines)


def render_index(run_dirs: list[Path]) -> str:
    lines = [
        "# Experiments index", "",
        "> Headline numbers across all runs in this directory.", "",
        "> 📄 **Full experiment report**: [`../PHASE1_5_MAPPO_REPORT.md`](../PHASE1_5_MAPPO_REPORT.md) — setup, results, reproduction, discussion.",
        "",
        "> Add a new run by copying `scripts/experiments/_template/` (TODO) or by ",
        "> following the recipe in `scripts/experiments/README.md`.", "",
        "## Run comparison", "",
        "| Run | seed | iters | kr (m) | eval_kr | cum_red | aim_res (m) | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in sorted(run_dirs):
        mpath = d / "metrics.json"
        if not mpath.is_file():
            lines.append(f"| {d.name} | — | — | — | — | — | — | (no metrics yet) |")
            continue
        m = _load(mpath)
        f = m.get("final") or {}
        verdict = ""
        if "mappo" in d.name.lower():
            cr = f.get("cum_red", 0)
            verdict = "PASS ✅" if cr >= 0.75 else (
                "FAIL ❌" if cr < 0.60 else "MARGINAL ⚠️")
        else:
            verdict = "reference"
        seed_val = m.get("seed_detected") or "?"
        lines.append(
            f"| [{d.name}]({d.name}/AGENTS.md) "
            f"| {seed_val} | {m.get('iter_count', '?')} "
            f"| {f.get('kr_m', 'n/a')} "
            f"| {f.get('eval_kill_rate', 'n/a'):.3f} "
            f"| {f.get('cum_red', 'n/a'):.3f} "
            f"| {f.get('aim_res_m', 'n/a')} "
            f"| {verdict} |"
        )
    lines += [
        "",
        "## Tooling", "",
        "- `scripts/experiments/parse_log_to_metrics.py` — log → metrics.json",
        "- `scripts/experiments/make_figures.py` — metrics.json → PNG curves",
        "- `scripts/experiments/compare_runs.py` — two metrics.json → comparison.md",
        "- `scripts/experiments/write_metadata.py` — seed/commit/reproduce → metadata.json",
        "- `scripts/experiments/write_agents_md.py` — this file (regenerates AGENTS.md)",
        "",
        "## Reproducing the whole comparison", "",
        "```bash",
        "# 1. run PfspFix (already done, ~3h)",
        "bash scripts/run_train.sh configs/laser_25x25_pro6000_stable.yaml logs/phase1_seed42_run1.log",
        "",
        "# 2. run MAPPO (~3h)",
        "bash scripts/run_train.sh configs/laser_25x25_mappo.yaml logs/phase1.5_mappo.log",
        "",
        "# 3. regenerate every artifact",
        "for d in experiments/phase1_*/ experiments/phase1.5_*/ ; do",
        "  python scripts/experiments/parse_log_to_metrics.py \\",
        "      --log $d/train.log --out $d/metrics.json --run-id $(basename $d)",
        "  python scripts/experiments/make_figures.py \\",
        "      --run $(basename $d):$d/metrics.json --out-dir $d/figures",
        "  python scripts/experiments/write_agents_md.py --run-dir $d",
        "done",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="generate per-run AGENTS.md for this directory")
    ap.add_argument("--baseline-dir", type=Path, default=None,
                    help="baseline run directory (for comparison line)")
    ap.add_argument("--index", action="store_true",
                    help="regenerate the top-level experiments/AGENTS.md index")
    ap.add_argument("--experiments-root", type=Path,
                    default=Path("experiments"))
    args = ap.parse_args()

    if args.index:
        root = args.experiments_root
        run_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        out = render_index(run_dirs)
        (root / "AGENTS.md").write_text(out, encoding="utf-8")
        print(f"wrote {root / 'AGENTS.md'} ({len(run_dirs)} runs)")
        return 0

    if not args.run_dir:
        ap.error("--run-dir or --index required")
    text = render_run(args.run_dir, args.baseline_dir)
    out_path = args.run_dir / "AGENTS.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
