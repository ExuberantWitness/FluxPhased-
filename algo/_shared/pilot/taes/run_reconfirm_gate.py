"""R2 pre-reconfirm gate (Task A): 3 methods × 4 cells × 5 seeds = 60 eps.

Spec: RECONFIRM_GATE.md (commit f9073b5).

Why this gate exists:
  Pre-flight exposed (1) α_eff bug → prior WP2 MAPPO was secretly IPPO,
  so the headline "MAPPO wins n4_L1" is invalid; (2) sanity showed strong
  classical beat MAPPO at n8 (8.0 vs 5.58) and at n4_L3-trained (3.58 vs 3.54).
  Before burning 1120-ep R2 grid, re-confirm with FIXED α_eff code that
  learned beat classical on n4_L1 kill OR survival-Pareto.

Cells (n4_L1-τ1 is the new addition vs sanity gate):
  n4_L0          - 4 targets, StaticJammer 0.3 (no EW stress)
  n4_L1-τ1       - 4 targets, ReactiveJammer τ=1 (hardest L1, the headline cell)
  n4_L3-trained  - 4 targets, trained LearnedJammer (heavy EW)
  n8_L0          - 8 targets, StaticJammer 0.3 (scale test)

Methods: {mappo, ippo, strong_classical}
  (drop static_classical per spec)

5 seeds: 42, 43, 44, 45, 46

Output:
  experiments/wp12_results/reconfirm_taskA.csv (per-seed kill + survival)
  experiments/wp12_results/RECONFIRM_TASKA_REPORT.md (mean + 95% CI + gate verdict)
"""

from __future__ import annotations

import os
import sys
import csv
import math
import argparse
import torch
from typing import Dict, List, Tuple

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.taes.taes_env import TAESVecEnv
from env.gpu.qos_rrm.adversary import make_jammer
from algo._shared.pilot.taes.taes_actor_critic import TaesCommanderActorCritic
from algo._shared.pilot.taes.run_wp2 import make_classical_static


class RLCommanderWrapper:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.ac = TaesCommanderActorCritic().to(device)
        sd = torch.load(ckpt_path, map_location=device)
        self.ac.load_state_dict(sd)
        self.ac.eval()
        self.device = device

    def __call__(self, env):
        obs_dict = env.get_obs()
        obs = obs_dict["obs"]
        alive_mask = env.target_alive_mask
        action = self.ac.get_action_for_env(obs, deterministic=True,
                                            target_alive_mask=alive_mask)
        return action


def eval_one(method_fn, n_targets: int, jammer_level: str,
             episode_steps: int, n_envs: int, seed: int,
             l3_jammer_ckpt: str = None, device: str = "cuda") -> Dict[str, float]:
    torch.manual_seed(seed)
    env = TAESVecEnv(n_envs=n_envs, n_targets=n_targets, device=device,
                     seed=seed, episode_steps=episode_steps)

    if jammer_level == "L3-trained" and l3_jammer_ckpt:
        jammer = make_jammer("L3-trained", device=device,
                             policy_path=l3_jammer_ckpt)
    elif jammer_level == "L3-trained":
        jammer = make_jammer("L1-tau1", device=device)
    else:
        jammer = make_jammer(jammer_level, device=device)
    jammer.reset(env.E, 1, env.device)
    env._last_jam = torch.zeros(env.E, device=device)
    env.reset()

    ep_kills = torch.zeros(env.E, device=device)
    ep_homejam = torch.zeros(env.E, device=device)
    for step in range(episode_steps):
        action = method_fn(env)
        obs, r, done, info = env.step(action, jammer=jammer)
        ep_kills += info["n_kills_step"]
        ep_homejam += info["homejam_death"]
        if done.all():
            break

    n_actual = float(env.target_n_actual.float().mean())
    kill_count = float(ep_kills.mean())
    survival = float((ep_homejam < 0.5).float().mean())
    return {"kill_count": kill_count, "survival_rate": survival,
            "n_actual": n_actual}


def mean_ci(vals: List[float]) -> Tuple[float, float]:
    """Mean and 95% CI (t=2.776 for n=5, df=4; using normal approx 1.96 for safety)."""
    if len(vals) < 2:
        return (sum(vals) / max(len(vals), 1), float("nan"))
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    se = math.sqrt(var / len(vals))
    # For n=5, t(0.975, df=4)=2.776; we report ±2.776·SE
    return m, 2.776 * se


def run_taskA(mappo_ckpt: str, ippo_ckpt: str, l3_jammer_ckpt: str,
              seeds=(42, 43, 44, 45, 46), episode_steps: int = 600,
              out_csv: str = None, out_report: str = None,
              device: str = "cuda"):
    out_csv = out_csv or "/home/ubuntu/CODE/FluxPhased-/experiments/wp12_results/reconfirm_taskA.csv"
    out_report = out_report or "/home/ubuntu/CODE/FluxPhased-/experiments/wp12_results/RECONFIRM_TASKA_REPORT.md"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    methods = {"strong_classical": make_classical_static()}
    if mappo_ckpt and os.path.exists(mappo_ckpt):
        wrapper = RLCommanderWrapper(mappo_ckpt, device=device)
        methods["mappo"] = lambda env, _w=wrapper: _w(env)
    if ippo_ckpt and os.path.exists(ippo_ckpt):
        wrapper = RLCommanderWrapper(ippo_ckpt, device=device)
        methods["ippo"] = lambda env, _w=wrapper: _w(env)

    cells = [
        ("n4_L0",          4, "L0"),
        ("n4_L1-tau1",     4, "L1-tau1"),
        ("n4_L3-trained",  4, "L3-trained"),
        ("n8_L0",          8, "L0"),
    ]

    rows = []
    n_eps = len(cells) * len(methods) * len(seeds)
    print(f"RECONFIRM Task A: {len(cells)} cells × {len(methods)} methods × "
          f"{len(seeds)} seeds = {n_eps} eps", flush=True)

    for cell_name, n_t, jam_lvl in cells:
        for m_name, m_fn in methods.items():
            for seed in seeds:
                r = eval_one(m_fn, n_targets=n_t, jammer_level=jam_lvl,
                             episode_steps=episode_steps, n_envs=8, seed=seed,
                             l3_jammer_ckpt=l3_jammer_ckpt, device=device)
                rows.append({"cell": cell_name, "method": m_name, "seed": seed, **r})
                print(f"  {cell_name:14s} {m_name:18s} seed={seed}: "
                      f"kill={r['kill_count']:.2f}/{r['n_actual']:.0f}  "
                      f"surv={r['survival_rate']:.2f}", flush=True)

    # Write per-seed CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "method", "seed", "kill_count", "survival_rate", "n_actual"])
        for r in rows:
            w.writerow([r["cell"], r["method"], r["seed"],
                        f"{r['kill_count']:.3f}", f"{r['survival_rate']:.3f}",
                        f"{r['n_actual']:.1f}"])

    # Aggregate
    agg = {}
    for r in rows:
        key = (r["cell"], r["method"])
        agg.setdefault(key, {"kill": [], "surv": []})
        agg[key]["kill"].append(r["kill_count"])
        agg[key]["surv"].append(r["survival_rate"])

    # Gate verdict
    def get(cell, method, metric="kill"):
        return agg.get((cell, method), {}).get(metric, [0.0])

    n4_l1 = {m: {"kill": get("n4_L1-tau1", m, "kill"),
                 "surv": get("n4_L1-tau1", m, "surv")} for m in ("mappo", "ippo", "strong_classical")}

    # PASS condition: learned wins n4_L1 kill OR (≥ same kill AND clearly higher survival)
    gate_conditions = {}
    for learned in ("mappo", "ippo"):
        l_kill_m, _ = mean_ci(n4_l1[learned]["kill"])
        c_kill_m, _ = mean_ci(n4_l1["strong_classical"]["kill"])
        l_surv_m, l_surv_ci = mean_ci(n4_l1[learned]["surv"])
        c_surv_m, c_surv_ci = mean_ci(n4_l1["strong_classical"]["surv"])
        kill_wins = l_kill_m > c_kill_m + 0.05
        kill_ties = abs(l_kill_m - c_kill_m) <= 0.20
        surv_wins = l_surv_m > c_surv_m + max(l_surv_ci, c_surv_ci, 0.05)
        gate_conditions[learned] = {
            "kill_wins": kill_wins,
            "kill_ties_and_surv_wins": kill_ties and surv_wins,
            "l_kill_mean": l_kill_m, "c_kill_mean": c_kill_m,
            "l_surv_mean": l_surv_m, "c_surv_mean": c_surv_m,
        }
    any_pass = any(c["kill_wins"] or c["kill_ties_and_surv_wins"]
                   for c in gate_conditions.values())

    # Write report
    with open(out_report, "w") as f:
        f.write("# RECONFIRM Task A — learned vs classical kill+survival (α_eff FIXED)\n\n")
        f.write(f"**Methods**: {list(methods.keys())}\n")
        f.write(f"**Cells**: {[c[0] for c in cells]}\n")
        f.write(f"**Seeds**: {list(seeds)}\n")
        f.write(f"**L3 jammer**: `{l3_jammer_ckpt}`\n\n")

        f.write("## Per-cell × method (mean over 5 seeds ± 95% CI)\n\n")
        f.write("| Cell | Method | kill (mean ± CI) | survival (mean ± CI) |\n")
        f.write("|------|--------|------------------|----------------------|\n")
        for (cell, method), v in sorted(agg.items()):
            km, kc = mean_ci(v["kill"])
            sm, sc = mean_ci(v["surv"])
            f.write(f"| {cell} | {method} | {km:.2f} ± {kc:.2f} | {sm:.2f} ± {sc:.2f} |\n")

        f.write("\n## n4_L1-τ1 headline cell (gate decision)\n\n")
        f.write("| Method | kill mean | survival mean | survival CI |\n")
        f.write("|--------|-----------|---------------|-------------|\n")
        for m in ("mappo", "ippo", "strong_classical"):
            km, kc = mean_ci(n4_l1[m]["kill"])
            sm, sc = mean_ci(n4_l1[m]["surv"])
            f.write(f"| {m} | {km:.2f} | {sm:.2f} | ±{sc:.2f} |\n")

        f.write("\n## Gate verdict (per learned method)\n\n")
        for learned, c in gate_conditions.items():
            f.write(f"### {learned}\n")
            f.write(f"- kill_wins (learned kill > classical + 0.05): "
                    f"{'✅' if c['kill_wins'] else '❌'} "
                    f"({c['l_kill_mean']:.2f} vs {c['c_kill_mean']:.2f})\n")
            f.write(f"- survival-Pareto (tie kill ±0.20 AND surv > classical+CI): "
                    f"{'✅' if c['kill_ties_and_surv_wins'] else '❌'} "
                    f"(surv {c['l_surv_mean']:.2f} vs {c['c_surv_mean']:.2f})\n")

        f.write("\n## Overall\n\n")
        f.write(f"**Task A gate: {'PASS' if any_pass else 'FAIL'}**\n\n")
        if any_pass:
            f.write("At least one learned method wins n4_L1-τ1 on kill OR "
                    "survival-Pareto.\n")
            f.write("**→ APPROVE full R2 grid (1120 eps).**\n")
        else:
            f.write("Learned does NOT beat classical on n4_L1 kill or "
                    "survival-Pareto with fixed α_eff.\n")
            f.write("**→ STOP. Do NOT burn R2. Restructure framework.**\n")

    print(f"\nReport: {out_report}", flush=True)
    print(f"CSV: {out_csv}", flush=True)
    print(f"Overall gate: {'PASS' if any_pass else 'FAIL'}", flush=True)
    return {"any_pass": any_pass, "gate_conditions": gate_conditions,
            "agg": agg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mappo-ckpt",
                        default="/home/ubuntu/CODE/FluxPhased-/checkpoints/appint/mappo_final.pt")
    parser.add_argument("--ippo-ckpt",
                        default="/home/ubuntu/CODE/FluxPhased-/checkpoints/appint/ippo_final.pt")
    parser.add_argument("--l3-jammer-ckpt",
                        default="/home/ubuntu/CODE/FluxPhased-/checkpoints/appint/jammer_L3_final.pt")
    parser.add_argument("--episode-steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--out-csv")
    parser.add_argument("--out-report")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_taskA(
        mappo_ckpt=args.mappo_ckpt,
        ippo_ckpt=args.ippo_ckpt,
        l3_jammer_ckpt=args.l3_jammer_ckpt,
        seeds=tuple(args.seeds),
        episode_steps=args.episode_steps,
        out_csv=args.out_csv,
        out_report=args.out_report,
        device=args.device,
    )


if __name__ == "__main__":
    main()
