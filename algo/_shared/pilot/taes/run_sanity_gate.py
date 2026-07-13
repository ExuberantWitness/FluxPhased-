"""AppInt pre-flight sanity gate driver.

4 cells × 4 methods × 3 seeds = 48 episodes.

Cells (must cover all 3 fixes per user patch ②):
  - n1_L0:        basic sanity (single target, no EW)
  - n4_L0:        anti-strawman check (RL trained, classical strong)
  - n8_L0:        Fix #2 (mixed-N training → MAPPO scales to n8)
  - n4_L3-trained: Fix #1 (trained L3 jammer; classical and RL both non-trivial)

Methods:
  - mappo:        CTDE central critic + α_eff blend (Step 4 trained)
  - ippo:         local critic only (Step 4 trained)
  - strong_classical:  TaesClassicalCommander
  - static_classical:   TaesFictitiousPlayCommander

Pass criteria (see plan Step 5 table):
  - n1_L0:   all 4 methods kill_rate ≥ 0.90
  - n4_L0:   MAPPO/IPPO kill_rate ≥ 0.70; strong_classical kill_rate ≥ 0.70
  - n8_L0:   MAPPO kill_count ≥ 0.7 × n4_MAPPO; strong_classical kill_count ≥ 6.0
  - n4_L3-trained: strong_classical kill_count ≥ 1.0; MAPPO kill_count ≥ 1.0

Outputs:
  - experiments/wp12_results/sanity_gate.csv
  - experiments/wp12_results/PREFLIGHT_GATE_REPORT.md
"""

from __future__ import annotations

import os
import sys
import csv
import argparse
import torch
from typing import Dict

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.taes.taes_env import TAESVecEnv
from env.gpu.qos_rrm.adversary import make_jammer
from algo._shared.pilot.taes.taes_actor_critic import (
    TaesCommanderActorCritic, build_privileged)
from algo._shared.pilot.taes.run_wp2 import (
    make_classical_static, make_classical_fp)
from algo._shared.baselines.taes_classical_commander import TaesClassicalCommander
from algo._shared.baselines.taes_fp_classical_commander import TaesFictitiousPlayCommander


# ----------------- RL commander wrapper -----------------

class RLCommanderWrapper:
    """Deterministic greedy action from a trained AC checkpoint."""
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


# ----------------- Eval loop -----------------

def eval_one(method_fn, n_targets: int, jammer_level: str,
             episode_steps: int, n_envs: int, seed: int,
             l3_jammer_ckpt: str = None, device: str = "cuda") -> Dict[str, float]:
    """One eval config: n_envs parallel episodes."""
    torch.manual_seed(seed)
    env = TAESVecEnv(n_envs=n_envs, n_targets=n_targets, device=device,
                     seed=seed, episode_steps=episode_steps)
    # Build jammer
    if jammer_level == "L3-trained" and l3_jammer_ckpt:
        jammer = make_jammer("L3-trained", device=device,
                             policy_path=l3_jammer_ckpt)
    elif jammer_level == "L3-trained":
        # No ckpt → fall back to L1-τ1 per R5
        jammer = make_jammer("L1-tau1", device=device)
    else:
        jammer = make_jammer(jammer_level, device=device)
    jammer.reset(env.E, 1, env.device)
    env._last_jam = torch.zeros(env.E, device=device)
    env.reset()

    ep_kills = torch.zeros(env.E, device=device)
    ep_homejam = torch.zeros(env.E, device=device)
    ep_len = torch.zeros(env.E, device=device)
    first_kill = torch.full((env.E,), float(episode_steps), device=device)
    for step in range(episode_steps):
        action = method_fn(env)
        obs, r, done, info = env.step(action, jammer=jammer)
        ep_kills += info["n_kills_step"]
        ep_homejam += info["homejam_death"]
        any_new = info["n_kills_step"] > 0
        not_yet = first_kill >= episode_steps
        first_kill = torch.where(any_new & not_yet,
                                  torch.full_like(first_kill, step + 1),
                                  first_kill)
        ep_len += (~done).float()
        if done.all():
            break

    n_actual = float(env.target_n_actual.float().mean())
    kill_count = float(ep_kills.mean())
    kill_rate = float((ep_kills >= n_actual).float().mean())
    survival = float((ep_homejam < 0.5).float().mean())
    ttk = float(first_kill[ep_kills > 0].mean()) if (ep_kills > 0).any() else float(episode_steps)
    return {
        "kill_count": kill_count,
        "kill_rate": kill_rate,
        "survival_rate": survival,
        "ttk_first": ttk,
        "ep_len": float(ep_len.mean()),
        "n_actual": n_actual,
    }


# ----------------- Main gate driver -----------------

def run_gate(mappo_ckpt: str = None, ippo_ckpt: str = None,
             l3_jammer_ckpt: str = None,
             seeds=(42, 43, 44), episode_steps: int = 600,
             out_csv: str = None, out_report: str = None,
             device: str = "cuda"):
    out_csv = out_csv or "/home/ubuntu/CODE/FluxPhased-/experiments/wp12_results/sanity_gate.csv"
    out_report = out_report or "/home/ubuntu/CODE/FluxPhased-/experiments/wp12_results/PREFLIGHT_GATE_REPORT.md"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Method builders (call factory to get the inner fn(env) → action)
    methods = {
        "strong_classical": make_classical_static(),
        "static_classical": make_classical_fp(),
    }
    if mappo_ckpt and os.path.exists(mappo_ckpt):
        wrapper = RLCommanderWrapper(mappo_ckpt, device=device)
        methods["mappo"] = lambda env, _w=wrapper: _w(env)
    if ippo_ckpt and os.path.exists(ippo_ckpt):
        wrapper = RLCommanderWrapper(ippo_ckpt, device=device)
        methods["ippo"] = lambda env, _w=wrapper: _w(env)

    # 4 cells (per user patch ②): (name, n_targets, jammer_level)
    cells = [
        ("n1_L0",          1, "L0"),
        ("n4_L0",          4, "L0"),
        ("n8_L0",          8, "L0"),
        ("n4_L3-trained",  4, "L3-trained"),
    ]

    rows = []
    print(f"Sanity gate: {len(cells)} cells × {len(methods)} methods × "
          f"{len(seeds)} seeds = {len(cells)*len(methods)*len(seeds)} eps",
          flush=True)
    for cell_name, n_t, jam_lvl in cells:
        for m_name, m_fn in methods.items():
            for seed in seeds:
                m = eval_one(m_fn, n_targets=n_t, jammer_level=jam_lvl,
                             episode_steps=episode_steps, n_envs=8, seed=seed,
                             l3_jammer_ckpt=l3_jammer_ckpt, device=device)
                row = {"cell": cell_name, "method": m_name, "seed": seed, **m}
                rows.append(row)
                print(f"  {cell_name:14s} {m_name:18s} seed={seed}: "
                      f"kill={m['kill_count']:.2f}/{m['n_actual']:.0f}  "
                      f"surv={m['survival_rate']:.2f}", flush=True)

    # Aggregate: mean over seeds
    agg = {}
    for row in rows:
        key = (row["cell"], row["method"])
        if key not in agg:
            agg[key] = {"kill_count": [], "kill_rate": [], "survival_rate": []}
        agg[key]["kill_count"].append(row["kill_count"])
        agg[key]["kill_rate"].append(row["kill_rate"])
        agg[key]["survival_rate"].append(row["survival_rate"])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "method", "kill_count_mean", "kill_count_std",
                    "kill_rate_mean", "survival_rate_mean"])
        for (cell, method), v in sorted(agg.items()):
            kc = torch.tensor(v["kill_count"]).mean().item()
            kc_std = torch.tensor(v["kill_count"]).std().item()
            kr = torch.tensor(v["kill_rate"]).mean().item()
            sr = torch.tensor(v["survival_rate"]).mean().item()
            w.writerow([cell, method, f"{kc:.3f}", f"{kc_std:.3f}",
                        f"{kr:.3f}", f"{sr:.3f}"])

    # Compute pass/fail per cell × method
    def g(cell, method, key="kill_count"):
        return torch.tensor(agg[(cell, method)][key]).mean().item()

    n4_mappo_kill = g("n4_L0", "mappo") if "mappo" in methods else 0.0
    n8_mappo_kill = g("n8_L0", "mappo") if "mappo" in methods else 0.0
    gate = {
        "n1_L0_all_killrate_0.90": all(
            g("n1_L0", m, "kill_rate") >= 0.90 for m in methods
        ),
        "n4_L0_mappo_killrate_0.70": (
            "mappo" in methods and g("n4_L0", "mappo", "kill_rate") >= 0.70
        ),
        "n4_L0_ippo_killrate_0.70": (
            "ippo" in methods and g("n4_L0", "ippo", "kill_rate") >= 0.70
        ),
        "n4_L0_classical_killrate_0.70": (
            g("n4_L0", "strong_classical", "kill_rate") >= 0.70
        ),
        "n8_L0_mappo_scales": (
            "mappo" in methods and n8_mappo_kill >= 0.7 * n4_mappo_kill
        ),
        "n8_L0_classical_kill_6": (
            g("n8_L0", "strong_classical") >= 6.0
        ),
        "n4_L3_classical_kill_1": (
            g("n4_L3-trained", "strong_classical") >= 1.0
        ),
        "n4_L3_mappo_kill_1": (
            "mappo" not in methods or
            g("n4_L3-trained", "mappo") >= 1.0
        ),
    }
    overall = all(gate.values())

    # Write report
    with open(out_report, "w") as f:
        f.write("# AppInt Pre-flight Sanity Gate Report\n\n")
        f.write(f"**Overall: {'PASS' if overall else 'FAIL'}**\n\n")
        f.write("## Gate criteria\n\n")
        for k, v in gate.items():
            f.write(f"- {'✅' if v else '❌'} `{k}`\n")
        f.write("\n## Results (mean over seeds)\n\n")
        f.write("| Cell | Method | kill_count | kill_rate | survival |\n")
        f.write("|------|--------|------------|-----------|----------|\n")
        for (cell, method), v in sorted(agg.items()):
            kc = torch.tensor(v["kill_count"]).mean().item()
            kr = torch.tensor(v["kill_rate"]).mean().item()
            sr = torch.tensor(v["survival_rate"]).mean().item()
            f.write(f"| {cell} | {method} | {kc:.2f} | {kr:.2f} | {sr:.2f} |\n")
        f.write("\n## Recommendation\n\n")
        if overall:
            f.write("**APPROVE** R2 full grid expansion.\n")
        else:
            f.write("**BLOCK** — fix list:\n")
            for k, v in gate.items():
                if not v:
                    f.write(f"- {k}\n")
    print(f"\nReport written to: {out_report}", flush=True)
    print(f"CSV: {out_csv}", flush=True)
    print(f"Overall gate: {'PASS' if overall else 'FAIL'}", flush=True)
    return {"gate": gate, "overall_pass": overall, "agg": agg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mappo-ckpt",
                        default="/home/ubuntu/CODE/FluxPhased-/checkpoints/appint/mappo_final.pt")
    parser.add_argument("--ippo-ckpt",
                        default="/home/ubuntu/CODE/FluxPhased-/checkpoints/appint/ippo_final.pt")
    parser.add_argument("--l3-jammer-ckpt",
                        default="/home/ubuntu/CODE/FluxPhased-/checkpoints/appint/jammer_L3_final.pt")
    parser.add_argument("--episode-steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--out-csv")
    parser.add_argument("--out-report")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_gate(
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
