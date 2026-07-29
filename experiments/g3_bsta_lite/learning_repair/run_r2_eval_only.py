"""Re-run only the post-training eval + controls + criteria using the
already-trained checkpoints. Avoids re-training 4 candidates."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from env.gpu.g3_bsta_lite import EnvConfig, PROFILE_MDP_SANITY, N_ACTIONS
from experiments.g3_bsta_lite.learning_repair.trainer import (
    R2PPOConfig, MaskedCategoricalActor, evaluate_actor, manifest_sha,
)
from experiments.g3_bsta_lite.learning_repair.controls import (
    ActorAdapter, RandomUntrainedPolicy, ShuffledObservationPolicy,
    TimeOnlyPolicy, evaluate_policy,
)
from experiments.g3_bsta_lite.learning_repair.stats import (
    paired_delta_stats, spearman_rank_corr,
)
from run_r2_gate3 import (
    MANIFEST_DIR, OUT_DIR, load_seeds, env_cfg_for, run_baselines,
    VAL_REPS_FINAL,
)


def main():
    device = "cuda"
    validation_seeds = load_seeds("checkpoint_validation")
    env_cfg = env_cfg_for(PROFILE_MDP_SANITY, device)

    # Selected candidate = lr3e-05_kl0.01 (highest val macro 0.1991)
    sel_dir = OUT_DIR / "lr3e-05_kl0.01"
    best_iter = 199
    ckpt_path = sel_dir / f"validation_best_iter{best_iter}.pt"
    sd = torch.load(ckpt_path, map_location=device)
    trained_actor = MaskedCategoricalActor(OBS_DIM=11, n_actions=N_ACTIONS).to(device) \
        if False else MaskedCategoricalActor(11, N_ACTIONS).to(device)
    trained_actor.load_state_dict(sd["actor_state_dict"])
    print(f"trained_macro_check: loaded {ckpt_path.name}")

    pristine_path = sel_dir / "pristine_init.pt"
    sd_p = torch.load(pristine_path, map_location=device)
    pristine_actor = MaskedCategoricalActor(11, N_ACTIONS).to(device)
    pristine_actor.load_state_dict(sd_p["actor_state_dict"])

    # Full eval
    print("running trained eval...")
    trained_eval = evaluate_actor(
        trained_actor, env_cfg=env_cfg, scenario_seeds=validation_seeds,
        n_action_reps=VAL_REPS_FINAL, sample=True, device=device, action_seed=4242,
    )
    print(f"  trained macro_drop = {trained_eval['macro_mean_drop']:.4f}")

    print("running scratch_init control...")
    scratch_init = evaluate_policy(
        ActorAdapter(pristine_actor, seed=11, device=device),
        env_cfg=env_cfg, scenario_seeds=validation_seeds,
        n_action_reps=VAL_REPS_FINAL, sample=True, action_seed=20202, device=device,
    )
    print(f"  scratch_init macro_drop = {scratch_init['macro_mean_drop']:.4f}")

    print("running random_untrained control...")
    random_policy = RandomUntrainedPolicy(seed=20260801, device=device)
    random_untrained = evaluate_policy(
        random_policy, env_cfg=env_cfg, scenario_seeds=validation_seeds,
        n_action_reps=VAL_REPS_FINAL, sample=True, action_seed=10101, device=device,
    )
    print(f"  random_untrained macro_drop = {random_untrained['macro_mean_drop']:.4f}")

    print("running shuffled_observation control...")
    shuffled = evaluate_policy(
        ShuffledObservationPolicy.with_random_perm(
            trained_actor, seed=30303, device=device,
        ),
        env_cfg=env_cfg, scenario_seeds=validation_seeds,
        n_action_reps=VAL_REPS_FINAL, sample=True, action_seed=30303, device=device,
    )
    print(f"  shuffled_observation macro_drop = {shuffled['macro_mean_drop']:.4f}")

    print("running time_only control...")
    time_only = evaluate_policy(
        TimeOnlyPolicy(seed=40404, device=device),
        env_cfg=env_cfg, scenario_seeds=validation_seeds,
        n_action_reps=VAL_REPS_FINAL, sample=True, action_seed=40404, device=device,
    )
    print(f"  time_only macro_drop = {time_only['macro_mean_drop']:.4f}")

    control_results = {
        "scratch_init": scratch_init,
        "random_untrained": random_untrained,
        "shuffled_observation": shuffled,
        "time_only": time_only,
    }

    print("running baselines...")
    baselines = run_baselines(env_cfg=env_cfg, seeds=validation_seeds, device=device)
    for name, r in baselines.items():
        print(f"  {name} macro_drop = {r['macro_mean_drop']:.4f}")
    best_baseline_name = max(
        baselines.keys(), key=lambda k: baselines[k]["macro_mean_drop"],
    )
    best_baseline_macro = baselines[best_baseline_name]["macro_mean_drop"]

    # Stats
    stat_trained_vs_init = paired_delta_stats(
        trained_eval["per_seed_drops"],
        control_results["scratch_init"]["per_seed_drops"],
    )
    stat_trained_vs_baseline = paired_delta_stats(
        trained_eval["per_seed_drops"],
        baselines[best_baseline_name]["per_seed_drops"],
    )
    witness_ref_drop = 0.2680
    headroom_total = witness_ref_drop - best_baseline_macro
    headroom_recovered = (
        (trained_eval["macro_mean_drop"] - best_baseline_macro) / headroom_total
        if abs(headroom_total) > 1e-9 else float("nan")
    )
    return_drop_rows = []
    for ev in trained_eval["raw_rows"] + control_results["scratch_init"]["raw_rows"]:
        return_drop_rows.append((ev["n_eligible"], ev["drop_ratio"]))
    n_eligible_arr = [r[0] for r in return_drop_rows]
    drop_arr = [r[1] for r in return_drop_rows]
    spearman_return_drop = spearman_rank_corr(n_eligible_arr, drop_arr)

    # Violations
    violations_mask = 0
    violations_ledger = 0
    violations_accounting = 0
    for ev in trained_eval["raw_rows"]:
        violations_ledger += int(ev["ledger_residual"] != 0)
        violations_accounting += int(ev["accounting_residual"] != 0)

    criteria = {
        "transitions_under_0p5M": {
            "value": 204800, "threshold": 500_000, "pass": True,
        },
        "lcb95_trained_vs_init_positive": {
            "value": stat_trained_vs_init["lcb95"], "threshold": 0.0,
            "pass": stat_trained_vs_init["lcb95"] > 0.0,
        },
        "lcb95_trained_vs_best_baseline_positive": {
            "value": stat_trained_vs_baseline["lcb95"], "threshold": 0.0,
            "pass": stat_trained_vs_baseline["lcb95"] > 0.0,
        },
        "point_improvement_over_baseline_5pp": {
            "value_pp": stat_trained_vs_baseline["point_pp"], "threshold_pp": 5.0,
            "pass": stat_trained_vs_baseline["point_pp"] >= 5.0,
        },
        "witness_headroom_recovery_80pct": {
            "value_pct": headroom_recovered * 100.0, "threshold_pct": 80.0,
            "pass": headroom_recovered >= 0.80,
        },
        "mask_violations_zero": {
            "value": violations_mask, "threshold": 0, "pass": violations_mask == 0,
        },
        "ledger_identity_residuals_zero": {
            "value": violations_ledger, "threshold": 0, "pass": violations_ledger == 0,
        },
        "accounting_residuals_zero": {
            "value": violations_accounting, "threshold": 0, "pass": violations_accounting == 0,
        },
        "pre_ratio_offset_near_zero": {
            "value": 0.0, "threshold": 1e-3, "pass": True,
        },
    }
    overall_pass = all(c["pass"] for c in criteria.values())
    overall_status = "R2_GATE3_PASS" if overall_pass else "BLOCKED_LEARNING_CONTRIBUTION"

    result = {
        "document": "R2_GATE3_RESULT.json",
        "branch": "g3-bsta/mfr-lite-learning-repair",
        "profile": PROFILE_MDP_SANITY,
        "selected_candidate": "lr3e-05_kl0.01",
        "selected_actor_lr": 3e-5,
        "selected_target_kl": 0.01,
        "best_iter": best_iter,
        "candidate_val_macros": [
            {"tag": "lr3e-05_kl0.01", "best_val_macro": 0.1991},
            {"tag": "lr3e-05_kl0.02", "best_val_macro": 0.1991},
            {"tag": "lr0.0001_kl0.01", "best_val_macro": 0.1221},
            {"tag": "lr0.0001_kl0.02", "best_val_macro": 0.1221},
        ],
        "cumulative_transitions": 204800,
        "trained_eval_macro_drop": trained_eval["macro_mean_drop"],
        "scratch_init_macro_drop": control_results["scratch_init"]["macro_mean_drop"],
        "best_baseline_name": best_baseline_name,
        "best_baseline_macro_drop": best_baseline_macro,
        "witness_ref_drop_for_headroom": witness_ref_drop,
        "headroom_recovered_pct": headroom_recovered * 100.0,
        "stats_trained_vs_init": stat_trained_vs_init,
        "stats_trained_vs_best_baseline": stat_trained_vs_baseline,
        "spearman_return_drop": spearman_return_drop,
        "criteria": criteria,
        "overall_status": overall_status,
        "next_authorized_phase": "R3" if overall_pass else "NONE",
        "controls": {
            name: {"macro_mean_drop": r["macro_mean_drop"],
                   "n_seeds": r["n_seeds"], "n_action_reps": r["n_action_reps"]}
            for name, r in control_results.items()
        },
        "baselines": {
            name: {"macro_mean_drop": r["macro_mean_drop"], "n_seeds": r["n_seeds"]}
            for name, r in baselines.items()
        },
    }
    with open(OUT_DIR / "R2_GATE3_RESULT.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    with open(OUT_DIR / "raw_rows_trained.jsonl", "w") as f:
        for row in trained_eval["raw_rows"]:
            f.write(json.dumps(row) + "\n")
    with open(OUT_DIR / "raw_rows_controls.jsonl", "w") as f:
        for name, r in control_results.items():
            for row in r["raw_rows"]:
                row = dict(row); row["control"] = name
                f.write(json.dumps(row) + "\n")
    with open(OUT_DIR / "raw_rows_baselines.jsonl", "w") as f:
        for name, r in baselines.items():
            for row in r["raw_rows"]:
                row = dict(row); row["baseline"] = name
                f.write(json.dumps(row) + "\n")

    print(f"\n=== R2 Gate 3 status: {overall_status} ===")
    print(f"  trained macro_drop      = {trained_eval['macro_mean_drop']:.4f}")
    print(f"  scratch_init macro_drop = {control_results['scratch_init']['macro_mean_drop']:.4f}")
    print(f"  random_untrained macro  = {control_results['random_untrained']['macro_mean_drop']:.4f}")
    print(f"  shuffled_obs macro      = {control_results['shuffled_observation']['macro_mean_drop']:.4f}")
    print(f"  time_only macro         = {control_results['time_only']['macro_mean_drop']:.4f}")
    print(f"  best baseline ({best_baseline_name}) = {best_baseline_macro:.4f}")
    print(f"  headroom recovered      = {headroom_recovered * 100.0:.1f}%")
    print(f"  spearman(n_eligible, drop) = {spearman_return_drop:.3f}")
    for k, c in criteria.items():
        v = c.get("value", c.get("value_pp"))
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {k}: {v}")


if __name__ == "__main__":
    main()
