#!/usr/bin/env python3
"""
parse_log_to_metrics.py — convert train_laser.py stdout into structured metrics.json.

WHAT:
  Reads a `logs/*.log` file produced by `training.train_laser`, extracts per-PSRO-iter
  and per-eval metrics, and writes a JSON document that downstream agents (and the
  comparison script) can read without re-parsing free-form text.

WHY:
  Free-form logs are fragile (humans can read them, agents can't reliably). This
  parser pins the exact fields we care about for the league / MAPPO / IPPO
  comparison: kr trajectory, kill_rate, advantage std (PPO health), cmd policy_loss
  (collapse watch), aim residual, cum red/blue, winrate vs opponent pool.

USAGE:
  python scripts/experiments/parse_log_to_metrics.py \
      --log logs/phase1_seed42_run1.log \
      --out experiments/phase1_pfsp_seed42/metrics.json

OUTPUT SCHEMA (metrics.json):
{
  "run_id": "phase1_pfsp_seed42",
  "log_path": "logs/phase1_seed42_run1.log",
  "parsed_at": "2026-07-01T...",
  "iter_count": 20,
  "final": { kr_m, eval_kill_rate, aim_res_m, cum_red, cum_blue, cum_draw, ... },
  "per_iter": [
    {"iter": 1, "kills": 180, "kill_rate": 9.0, "kr_m": 50.0,
     "cmd_pl": -0.00404, "adv_std": 10.457, "aim_res_m": 0.426,
     "log_std": -1.0, "bc_w": 5.0, "rate_p_s": 5, "wall_s": 628, "upd": 6}, ...
  ],
  "per_eval": [
    {"iter": 1, "eval_kills": 20, "eval_kill_rate": 0.833,
     "kr_next_m": 35.0, "pool_size": 1, "opp_idx": 0,
     "R": 12, "B": 0, "D": 0, "n_total": 12,
     "cum_red": 1.0, "cum_blue": 0.0, "cum_draw": 0.0, "wr_opp": 0.38, "jam": 0.0}, ...
  ]
}
"""
from __future__ import annotations
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# Patterns are intentionally strict; if a line doesn't match, we skip silently.
PSRO_RE = re.compile(
    r"^\[PSRO\s+(\d+)/(\d+)\]\s+"
    r"kills=(\d+)\s+"
    r"kill_rate=([\d.]+)\s+"
    r"min_aim_dist=(\d+)m\s+"
    r"kr=([\d.]+)m\s+"
    r"avg_cmd_r=([-\d.]+)\s+"
    r"log_std=(-?[\d.]+)\s+"
    r"bc_w=([\d.]+)\s+"
    r"rate=(\d+)p/s\s+"
    r"time=(\d+)s\s+"
    r"upd=(\d+)\s+"
    r"cmd_pl=([-\d.]+)\s+"
    r"adv_std=([\d.]+)\s+"
    r"aim_res=([\d.]+)"
)
EVAL_RE = re.compile(
    r"^\[Eval @ iter (\d+)\]\s+"
    r"eval_kills=(\d+)\s+"
    r"eval_min_aim_dist=(\d+)m\s+"
    r"eval_kill_rate=([\d.]+)\s+"
    r"kr_next=([\d.]+)m\s+"
    r"pool=(\d+)\s+"
    r"opp=(\d+)\s+\|\s+"
    r"this:R(\d+)/B(\d+)/D(\d+)\s+\|\s+"
    r"cum red=([\d.]+)\s+"
    r"blue=([\d.]+)\s+"
    r"draw=([\d.]+)\s+"
    r"\(n=(\d+)\)\s+"
    r"wr\[opp\]=([\d.]+)\s+\|\s+"
    r"jam=([\d.]+)"
)
UPDATE_RE = re.compile(
    r"^\s+\[Update\]\s+"
    r"radar_loss=([-\d.]+)\s+"
    r"cmd_loss=([-\d.]+)\s+"
    r"bc_loss=([-\d.]+)\s+"
    r"adv_std=([\d.]+)\s+"
    r"aim_res=([\d.]+)"
)
SEED_RE = re.compile(r"[Ss]eed[:=]\s*(\d+)")


def _f(s: str) -> float:
    return float(s)


def _i(s: str) -> int:
    return int(s)


def parse(log_path: Path, run_id: str) -> dict:
    per_iter: dict[int, dict] = {}
    per_eval: dict[int, dict] = {}
    per_update: dict[int, dict] = {}
    seed_val: int | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if seed_val is None:
                m = SEED_RE.search(line)
                if m:
                    seed_val = _i(m.group(1))

            m = PSRO_RE.search(line)
            if m:
                it = _i(m.group(1))
                per_iter[it] = {
                    "iter": it,
                    "total_iters": _i(m.group(2)),
                    "kills": _i(m.group(3)),
                    "kill_rate": _f(m.group(4)),
                    "min_aim_dist_m": _i(m.group(5)),
                    "kr_m": _f(m.group(6)),
                    "avg_cmd_r": _f(m.group(7)),
                    "log_std": _f(m.group(8)),
                    "bc_w": _f(m.group(9)),
                    "rate_p_s": _i(m.group(10)),
                    "wall_s": _i(m.group(11)),
                    "upd": _i(m.group(12)),
                    "cmd_pl": _f(m.group(13)),
                    "adv_std": _f(m.group(14)),
                    "aim_res_m": _f(m.group(15)),
                }
                continue

            m = EVAL_RE.search(line)
            if m:
                it = _i(m.group(1))
                per_eval[it] = {
                    "iter": it,
                    "eval_kills": _i(m.group(2)),
                    "eval_min_aim_dist_m": _i(m.group(3)),
                    "eval_kill_rate": _f(m.group(4)),
                    "kr_next_m": _f(m.group(5)),
                    "pool_size": _i(m.group(6)),
                    "opp_idx": _i(m.group(7)),
                    "R": _i(m.group(8)),
                    "B": _i(m.group(9)),
                    "D": _i(m.group(10)),
                    "cum_red": _f(m.group(11)),
                    "cum_blue": _f(m.group(12)),
                    "cum_draw": _f(m.group(13)),
                    "n_total": _i(m.group(14)),
                    "wr_opp": _f(m.group(15)),
                    "jam": _f(m.group(16)),
                }
                continue

            m = UPDATE_RE.search(line)
            if m:
                # update line follows a PSRO line — attribute to the most recent iter seen.
                if per_iter:
                    it = max(per_iter.keys())
                    per_update[it] = {
                        "iter": it,
                        "radar_loss": _f(m.group(1)),
                        "cmd_loss": _f(m.group(2)),
                        "bc_loss": _f(m.group(3)),
                        "adv_std_update": _f(m.group(4)),
                        "aim_res_update_m": _f(m.group(5)),
                    }

    iter_list = [per_iter[k] for k in sorted(per_iter.keys())]
    eval_list = [per_eval[k] for k in sorted(per_eval.keys())]
    update_list = [per_update[k] for k in sorted(per_update.keys())]

    final: dict = {}
    if eval_list:
        last = eval_list[-1]
        final = {
            "iter": last["iter"],
            "kr_m": last["kr_next_m"],
            "eval_kill_rate": last["eval_kill_rate"],
            "cum_red": last["cum_red"],
            "cum_blue": last["cum_blue"],
            "cum_draw": last["cum_draw"],
            "wr_opp": last["wr_opp"],
            "n_total": last["n_total"],
        }
    if iter_list:
        last_it = iter_list[-1]
        final.setdefault("iter", last_it["iter"])
        final["aim_res_m"] = last_it["aim_res_m"]
        final["cmd_pl_last"] = last_it["cmd_pl"]
        final["adv_std_last"] = last_it["adv_std"]
        final["kr_train_m"] = last_it["kr_m"]
        final["log_std_final"] = last_it["log_std"]
        final["bc_w_final"] = last_it["bc_w"]

    return {
        "run_id": run_id,
        "log_path": str(log_path),
        "log_filename": log_path.name,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "seed_detected": seed_val,
        "iter_count": len(iter_list),
        "eval_count": len(eval_list),
        "final": final,
        "per_iter": iter_list,
        "per_eval": eval_list,
        "per_update": update_list,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", default=None,
                    help="run identifier (defaults to stem of --out dir)")
    args = ap.parse_args()

    if not args.log.is_file():
        raise SystemExit(f"log not found: {args.log}")

    run_id = args.run_id or args.out.parent.name
    data = parse(args.log.resolve(), run_id=run_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {args.out}  iter={data['iter_count']}  eval={data['eval_count']}")
    if data["final"]:
        f = data["final"]
        print(f"  final: kr={f.get('kr_m')}m  eval_kr={f.get('eval_kill_rate')}  "
              f"cum_red={f.get('cum_red')}  aim_res={f.get('aim_res_m')}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
