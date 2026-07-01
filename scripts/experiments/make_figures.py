#!/usr/bin/env python3
"""
make_figures.py — render PNG curves from metrics.json files.

WHAT:
  Given one or more metrics.json files, produces PNG curves for the headline
  metrics that the EAAI paper needs: kr trajectory, cum_red/blue/draw, aim
  residual, advantage std (PPO health), cmd policy_loss (collapse watch).

USAGE:
  # single run
  python scripts/experiments/make_figures.py \
      --run phase1_pfsp_seed42:experiments/phase1_pfsp_seed42/metrics.json \
      --out-dir experiments/phase1_pfsp_seed42/figures

  # compare two runs (overlays curves with legend)
  python scripts/experiments/make_figures.py \
      --run PfspFix:experiments/phase1_pfsp_seed42/metrics.json \
      --run MAPPO:experiments/phase1.5_mappo_seed42/metrics.json \
      --out-dir experiments/comparison_figures

OUTPUT: <out-dir>/{kr_curve,cum_red_curve,aim_res_curve,adv_std_curve,cmd_pl_curve}.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iters(metrics: dict) -> list[int]:
    return [r["iter"] for r in metrics["per_iter"]]


def _eval_iters(metrics: dict) -> list[int]:
    return [r["iter"] for r in metrics["per_eval"]]


def _plot_curve(runs: list[tuple[str, dict]], x_fn, y_fn, xlabel: str, ylabel: str,
                title: str, out_path: Path, log_y: bool = False,
                hlines: list[tuple[float, str, str]] | None = None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, m in runs:
        xs = x_fn(m)
        ys = [y_fn(r) for r in (m["per_iter"] if x_fn is _iters else m["per_eval"])]
        # filter mismatches defensively
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pairs:
            continue
        xs2, ys2 = zip(*pairs)
        ax.plot(xs2, ys2, marker="o", label=name, linewidth=1.8)
    if hlines:
        for y, lbl, color in hlines:
            ax.axhline(y, color=color, linestyle="--", alpha=0.6, label=lbl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("symlog")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="NAME:PATH",
                    help="display-name:metrics.json (repeat for overlay)")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    runs: list[tuple[str, dict]] = []
    for spec in args.run:
        name, path = spec.split(":", 1)
        runs.append((name, _load(Path(path))))

    out = args.out_dir

    # 1. kr trajectory (train + eval-next on same axes)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, m in runs:
        its = [r["iter"] for r in m["per_iter"]]
        krs = [r["kr_m"] for r in m["per_iter"]]
        ax.plot(its, krs, marker="o", label=f"{name} (train)", linewidth=1.8)
        eits = [r["iter"] for r in m["per_eval"]]
        ekr = [r["kr_next_m"] for r in m["per_eval"]]
        ax.plot(eits, ekr, marker="x", linestyle="--", alpha=0.6,
                label=f"{name} (eval→next)", linewidth=1.5)
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.6, label="kr floor (0.5m)")
    ax.set_yscale("log")
    ax.set_xlabel("PSRO iteration")
    ax.set_ylabel("kill radius (m, log)")
    ax.set_title("kill-radius curriculum trajectory")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "kr_curve.png", dpi=140)
    plt.close(fig)
    print(f"wrote {out / 'kr_curve.png'}")

    # 2. cum red / blue / draw
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, m in runs:
        eits = [r["iter"] for r in m["per_eval"]]
        ax.plot(eits, [r["cum_red"] for r in m["per_eval"]],
                marker="o", label=f"{name} red", linewidth=1.8, color="green")
        ax.plot(eits, [r["cum_blue"] for r in m["per_eval"]],
                marker="x", linestyle="--", label=f"{name} blue", linewidth=1.5, color="red")
    ax.axhline(0.8, color="black", linestyle=":", alpha=0.4, label="gate: cum_red ≥ 0.8")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("PSRO iteration")
    ax.set_ylabel("cumulative win/draw share")
    ax.set_title("self-play league: cumulative red vs blue")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cum_red_curve.png", dpi=140)
    plt.close(fig)
    print(f"wrote {out / 'cum_red_curve.png'}")

    # 3. aim residual (precision)
    _plot_curve(runs, _iters, lambda r: r["aim_res_m"],
                "PSRO iteration", "aim residual (m)",
                "commander aim residual (lower = better)",
                out / "aim_res_curve.png")

    # 4. advantage std (PPO health; should stay > 1e-3, not explode)
    _plot_curve(runs, _iters, lambda r: r["adv_std"],
                "PSRO iteration", "advantage std",
                "PPO health: advantage std (collapse watch)",
                out / "adv_std_curve.png",
                hlines=[(1e-3, "collapse floor (1e-3)", "red")])

    # 5. cmd policy_loss (should stay > 1e-4 magnitude; not collapse to 0)
    _plot_curve(runs, _iters, lambda r: r["cmd_pl"],
                "PSRO iteration", "cmd policy_loss",
                "cmd policy_loss (collapse watch)",
                out / "cmd_pl_curve.png",
                hlines=[(0.0, "zero line", "black")])

    # 6. eval_kill_rate trajectory
    _plot_curve(runs, _eval_iters, lambda r: r["eval_kill_rate"],
                "PSRO iteration", "eval kill_rate",
                "eval kill_rate (deterministic policy)",
                out / "eval_kill_rate_curve.png",
                hlines=[(0.5, "gate ≥ 0.5", "green")])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
