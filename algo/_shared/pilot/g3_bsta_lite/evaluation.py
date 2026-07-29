"""Evaluation harness for G3-BSTA-lite (F2 §7 Gate 1).

Per DEBUG_CONTRACT.md §7:
  - At least 128 locked paired scenarios.
  - Stochastic policies averaged over a frozen number of action-RNG
    replicates within scenario.
  - LCB95 = mean(delta_s) - t_(0.95, S-1) * sd(delta_s) / sqrt(S).
  - Confirmation passes against every frozen baseline; no new best
    baseline selection on confirmation data.

This harness produces:
  - per-scenario mean drop_ratio for each policy
  - paired delta vs each frozen baseline
  - LCB95 of paired delta
  - oracle-vs-best-baseline gap (for reachable headroom assessment)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import torch

from env.gpu.g3_bsta_lite import (
    EnvConfig,
    G3BstaLiteVecEnv,
    generate_paired_manifest,
)
from .baselines import (
    AlwaysOff,
    Baseline,
    BudgetedBarrage,
    BudgetedRoundRobin,
    CausalReactiveOrEDF,
    FROZEN_BASELINES,
    PeriodicBlink,
    RandomFeasible,
)
from .oracle import make_clairvoyant_oracle


@dataclass
class ScenarioResult:
    seed: int
    drop_ratio: float
    n_eligible: int
    n_success: int
    n_timeout: int
    n_horizon_failure: int


@dataclass
class PolicyResult:
    name: str
    scenario_results: list[ScenarioResult] = field(default_factory=list)

    def mean_drop(self) -> float:
        ratios = [r.drop_ratio for r in self.scenario_results
                  if not math.isnan(r.drop_ratio)]
        if not ratios:
            return float("nan")
        return sum(ratios) / len(ratios)

    def macro_mean(self) -> float:
        return self.mean_drop()


def _run_one_scenario(
    *,
    policy_factory: Callable[[], Baseline],
    scenario_seed: int,
    cfg: EnvConfig,
    n_action_reps: int,
    device: str,
) -> list[ScenarioResult]:
    """Run policy on one scenario, averaged over n_action_reps replicates."""
    results: list[ScenarioResult] = []
    for rep in range(n_action_reps):
        env = G3BstaLiteVecEnv(cfg)
        policy = policy_factory()
        rep_seed = scenario_seed * 100003 + rep * 17 + 7
        env.reset(seed=scenario_seed)
        policy.reset(env, seed=rep_seed)
        for t in range(cfg.horizon):
            obs = env._build_observation()
            mask = env._compute_mask()
            actions = policy.act(obs, mask, step_idx=t)
            env.step(actions)
        c = env.counters
        results.append(ScenarioResult(
            seed=scenario_seed,
            drop_ratio=float(env.drop_ratio()[0]),
            n_eligible=int(c.n_eligible[0]),
            n_success=int(c.n_success[0]),
            n_timeout=int(c.n_timeout[0]),
            n_horizon_failure=int(c.n_horizon_failure[0]),
        ))
    return results


def evaluate_policies(
    *,
    cfg: EnvConfig | None = None,
    n_scenarios: int = 128,
    n_action_reps: int = 4,
    device: str = "cpu",
    include_oracle: bool = True,
    extra_policies: dict[str, Callable[[], Baseline]] | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run Gate 1 / Gate 4 evaluation matrix.

    Returns a dict with per-policy macro_mean, paired deltas vs each frozen
    baseline, and LCB95.
    """
    cfg = cfg or EnvConfig(n_envs=1, device=device, seed=0)

    # Generate frozen scenario manifest (128 eligible scenarios).
    print(f"[eval] generating {n_scenarios} paired scenarios...")
    manifest = generate_paired_manifest(
        base_seed=20260729,
        n_scenarios=n_scenarios,
        horizon=cfg.horizon,
        n_services=cfg.n_services,
        arrival_rate_per_service=cfg.arrival_rate_per_service,
        baseline_snr_db=cfg.baseline_snr_db,
        device=device,
    )
    print(f"[eval] manifest frozen ({len(manifest)} scenarios)")

    # Build the policy factory map.
    policy_factories: dict[str, Callable[[], Baseline]] = {
        cls.name: cls for cls in FROZEN_BASELINES
    }
    if include_oracle:
        policy_factories["clairvoyant_oracle"] = make_clairvoyant_oracle
    if extra_policies:
        policy_factories.update(extra_policies)

    print(f"[eval] policies: {list(policy_factories.keys())}")

    # Run all (policy, scenario, rep) combinations.
    results: dict[str, list[ScenarioResult]] = {name: [] for name in policy_factories}
    for s_idx, scenario in enumerate(manifest):
        if (s_idx + 1) % 16 == 0:
            print(f"  scenario {s_idx + 1}/{n_scenarios}")
        for name, factory in policy_factories.items():
            rep_results = _run_one_scenario(
                policy_factory=factory,
                scenario_seed=scenario.seed,
                cfg=cfg,
                n_action_reps=n_action_reps,
                device=device,
            )
            # Average over reps within scenario (mean drop_ratio).
            mean_drop = sum(r.drop_ratio for r in rep_results) / len(rep_results)
            r0 = rep_results[0]
            results[name].append(ScenarioResult(
                seed=scenario.seed,
                drop_ratio=mean_drop,
                n_eligible=r0.n_eligible,
                n_success=int(sum(rr.n_success for rr in rep_results) / len(rep_results)),
                n_timeout=int(sum(rr.n_timeout for rr in rep_results) / len(rep_results)),
                n_horizon_failure=int(sum(rr.n_horizon_failure for rr in rep_results) / len(rep_results)),
            ))

    # Compute macro means.
    summary = {}
    for name, res_list in results.items():
        summary[name] = {
            "macro_mean_drop": PolicyResult(name=name, scenario_results=res_list).macro_mean(),
            "n_scenarios": len(res_list),
        }

    # Compute paired deltas vs each frozen baseline (always_off, etc.).
    frozen_names = [cls.name for cls in FROZEN_BASELINES]
    paired_deltas = {}  # policy_name -> {baseline_name: {mean, lcb95, n}}
    for policy_name, policy_res in results.items():
        paired_deltas[policy_name] = {}
        for bn in frozen_names:
            if bn == policy_name:
                continue
            base_res = results[bn]
            deltas = []
            for pr, br in zip(policy_res, base_res):
                if math.isnan(pr.drop_ratio) or math.isnan(br.drop_ratio):
                    continue
                deltas.append(pr.drop_ratio - br.drop_ratio)
            if not deltas:
                paired_deltas[policy_name][bn] = {
                    "mean_delta": float("nan"), "lcb95": float("nan"), "n": 0,
                }
                continue
            n = len(deltas)
            mean_d = sum(deltas) / n
            var_d = sum((d - mean_d) ** 2 for d in deltas) / max(n - 1, 1)
            sd_d = math.sqrt(var_d)
            # t critical value at 0.95 one-sided with n-1 dof.
            # For n >= 30 use normal approx 1.645; small n uses t-table lookup.
            t_crit = _t_critical_one_sided_0p95(n)
            lcb95 = mean_d - t_crit * sd_d / math.sqrt(n)
            paired_deltas[policy_name][bn] = {
                "mean_delta": mean_d, "lcb95": lcb95, "n": n,
            }

    # Oracle headroom = oracle_drop - best_non_witness_baseline_drop.
    # Per MODIFICATION_PLAN Gate 1, the witness is the policy under test
    # (criterion 2: witness LCB > 7.5 pp vs each baseline). Oracle headroom
    # (criterion 1: ≥10 pp) is measured against the best NON-WITNESS baseline,
    # so a strong witness does not artificially shrink the apparent headroom.
    oracle_headroom = None
    if include_oracle:
        oracle_drop = summary["clairvoyant_oracle"]["macro_mean_drop"]
        witness_name = "causal_reactive_or_edf"
        non_witness = [bn for bn in frozen_names if bn != witness_name]
        nw_drops = {bn: summary[bn]["macro_mean_drop"] for bn in non_witness}
        best_baseline = max(nw_drops, key=nw_drops.get)
        best_drop = nw_drops[best_baseline]
        witness_drop = (
            summary[witness_name]["macro_mean_drop"]
            if witness_name in summary else float("nan")
        )
        oracle_headroom = {
            "oracle_drop": oracle_drop,
            "best_non_witness_baseline": best_baseline,
            "best_baseline_drop": best_drop,
            "absolute_gap_pp": (oracle_drop - best_drop) * 100.0,
            "witness_drop": witness_drop,
            "oracle_vs_witness_gap_pp": (oracle_drop - witness_drop) * 100.0
            if not math.isnan(witness_drop) else float("nan"),
        }

    output = {
        "n_scenarios": n_scenarios,
        "n_action_reps": n_action_reps,
        "policy_summary": summary,
        "paired_deltas": paired_deltas,
        "oracle_headroom": oracle_headroom,
        "frozen_baselines": frozen_names,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "BASELINE_FREEZE.json"), "w") as f:
            json.dump(_to_jsonable(output), f, indent=2)
        # Also write raw scenario-level rows.
        with open(os.path.join(output_dir, "paired_raw_rows.json"), "w") as f:
            rows = []
            for name, res_list in results.items():
                for r in res_list:
                    rows.append({
                        "policy": name, "seed": r.seed,
                        "drop_ratio": r.drop_ratio,
                        "n_eligible": r.n_eligible,
                        "n_success": r.n_success,
                        "n_timeout": r.n_timeout,
                        "n_horizon_failure": r.n_horizon_failure,
                    })
            json.dump(rows, f, indent=2)

    return output


def _t_critical_one_sided_0p95(n: int) -> float:
    """Approximate t critical value for one-sided 0.95 with n-1 dof.

    For n >= 30, normal approximation z = 1.645.
    For smaller n, lookup table (chosen conservative).
    """
    if n >= 30:
        return 1.645
    table = {
        2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 6: 2.015,
        7: 1.943, 8: 1.895, 9: 1.860, 10: 1.833,
        15: 1.761, 20: 1.725, 25: 1.711, 30: 1.699,
    }
    if n in table:
        return table[n]
    # Interpolate between nearest entries.
    keys = sorted(table.keys())
    for i in range(len(keys) - 1):
        if keys[i] < n < keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            t_lo, t_hi = table[lo], table[hi]
            return t_lo + (t_hi - t_lo) * (n - lo) / (hi - lo)
    return 1.645


def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, (int, str, bool)) or o is None:
        return o
    if isinstance(o, float):
        if math.isnan(o):
            return None
        return o
    return str(o)
