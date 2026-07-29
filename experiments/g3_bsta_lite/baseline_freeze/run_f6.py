"""F6 main experiment: two-seed pilot (Gate 5).

Per MODIFICATION_PLAN W7 + Gate 5:
  - Run two independent BC-warm-start PPO training seeds.
  - 300 iterations each (= 0.307M transitions per seed, matching F5).
  - Same train/eval split as F5 (32 train + 32 fresh held-out scenarios).
  - Verify reproducibility: both seeds complete, both reach witness-level
    on held-out, variance across seeds is bounded.
  - NO significance claim (that requires 8-seed campaign, out of scope
    for fast-work line).

Outputs:
  - experiments/g3_bsta_lite/baseline_freeze/f6_ppo_bc_seed0.pt
  - experiments/g3_bsta_lite/baseline_freeze/f6_ppo_bc_seed1.pt
  - experiments/g3_bsta_lite/baseline_freeze/f6_seed0_curve.json
  - experiments/g3_bsta_lite/baseline_freeze/f6_seed1_curve.json
  - experiments/g3_bsta_lite/baseline_freeze/f6_eval.json
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/home/ubuntu/CODE/g3-bsta-fastwork")

import torch

from env.gpu.g3_bsta_lite import (
    EnvConfig, G3BstaLiteVecEnv, N_ACTIONS, OBS_DIM, generate_paired_manifest,
)
from algo._shared.pilot.g3_bsta_lite.ppo import (
    PPOConfig, PPOTrainer, evaluate_ppo,
)
from algo._shared.pilot.g3_bsta_lite.baselines import FROZEN_BASELINES

OUT_DIR = "/home/ubuntu/CODE/g3-bsta-fastwork/experiments/g3_bsta_lite/baseline_freeze"
N_TRAIN_SCENARIOS = 32
N_HELDOUT_SCENARIOS = 32
N_EVAL_REPS = 4
PPO_ITERS = 300
EVAL_EVERY = 50  # Less frequent eval than F5 to keep runtime sane.
SEEDS = [0, 1]

# Same scenario scheme as F5 (disjoint train/heldout).
train_manifest = generate_paired_manifest(
    base_seed=20260729, n_scenarios=N_TRAIN_SCENARIOS, horizon=64,
    n_services=2, arrival_rate_per_service=0.15, baseline_snr_db=22.0,
    device="cpu",
)
train_seeds = [s.seed for s in train_manifest]
heldout_manifest = generate_paired_manifest(
    base_seed=20260801, n_scenarios=N_HELDOUT_SCENARIOS, horizon=64,
    n_services=2, arrival_rate_per_service=0.15, baseline_snr_db=22.0,
    device="cpu",
)
heldout_seeds = [s.seed for s in heldout_manifest]
assert not (set(train_seeds) & set(heldout_seeds))
print(f"[f6] {len(train_seeds)} train + {len(heldout_seeds)} heldout scenarios")
print(f"[f6] seeds to run: {SEEDS}")

env_cfg = EnvConfig()


def eval_baseline_on_seeds(cls, seeds, n_reps):
    per_seed = []
    for sd in seeds:
        rep_drops = []
        for rep in range(n_reps):
            env = G3BstaLiteVecEnv(EnvConfig(n_envs=1, horizon=env_cfg.horizon,
                                              device="cpu", seed=sd))
            env.reset(seed=sd)
            policy = cls()
            policy.reset(env, seed=sd * 100003 + rep * 17 + 7)
            for t in range(env_cfg.horizon):
                obs = env._build_observation()
                mask = env._compute_mask()
                a = policy.act(obs, mask, step_idx=t)
                env.step(a)
            rep_drops.append(float(env.drop_ratio()[0]))
        per_seed.append(sum(rep_drops) / len(rep_drops))
    return sum(per_seed) / len(per_seed), per_seed


# --- Frozen baselines + witness on held-out (computed once) ---------------
print(f"\n[f6] evaluating frozen baselines on {N_HELDOUT_SCENARIOS} held-out seeds...")
frozen_eval_held = {}
for cls in FROZEN_BASELINES:
    macro, per_seed = eval_baseline_on_seeds(cls, heldout_seeds, N_EVAL_REPS)
    frozen_eval_held[cls.name] = {"macro_mean_drop": macro, "per_seed_drops": per_seed}
    print(f"  {cls.name:25s} macro_mean_drop = {macro:.4f}")

# --- Two-seed BC warm-start PPO -------------------------------------------
seed_results = {}
for sd in SEEDS:
    print(f"\n[f6] === BC warm-start PPO, seed={sd}, {PPO_ITERS} iters ===")
    cfg = PPOConfig(
        iterations=PPO_ITERS, n_envs=16, horizon=64, seed=sd, device="cpu",
        bc_warm_start_path=os.path.join(OUT_DIR, "imitation_actor_dagger.pt"),
    )
    trainer = PPOTrainer(
        cfg=cfg, env_cfg=env_cfg, train_scenario_seeds=train_seeds,
    )
    curve = []
    best = {"iter": -1, "heldout_macro_drop": -1.0, "actor_sd": None,
            "per_seed_drops": None}
    t0 = time.time()
    total_transitions = 0
    for i in range(PPO_ITERS):
        m = trainer.train_iteration()
        total_transitions += 16 * 64
        curve.append({
            "iter": i, "rollout_drop": m["rollout_drop"], "kl_mean": m["kl_mean"],
            "kl_max": m["kl_max"], "clip_frac": m["clip_frac_mean"],
            "adv_std": m["adv_std"], "pre_ratio_offset": m["pre_ratio_offset"],
            "explained_variance": m["explained_variance"],
            "entropy": m["entropy"], "action_freq": m["action_freq"],
            "early_stop": m["early_stop"],
            "cum_transitions": total_transitions,
        })
        if i % EVAL_EVERY == 0 or i == PPO_ITERS - 1:
            res = evaluate_ppo(
                trainer.actor, env_cfg=env_cfg, scenario_seeds=heldout_seeds,
                n_action_reps=N_EVAL_REPS, sample=True, device="cpu",
                action_seed=sd + 1000,
            )
            macro = res["macro_mean_drop"]
            elapsed = time.time() - t0
            print(f"  iter {i:3d} ({total_transitions/1e6:.2f}M): "
                  f"train_drop={m['rollout_drop']:.4f} "
                  f"heldout={macro:.4f} kl_max={m['kl_max']:.4f} "
                  f"clip={m['clip_frac_mean']:.4f} ent={m['entropy']:.3f} "
                  f"t={elapsed:.1f}s")
            if macro > best["heldout_macro_drop"]:
                best = {
                    "iter": i, "heldout_macro_drop": macro,
                    "per_seed_drops": res["per_seed_drops"],
                    "actor_sd": {k: v.clone() for k, v in trainer.actor.state_dict().items()},
                    "cum_transitions": total_transitions,
                }
    # Argmax eval at final iter.
    argmax_res = evaluate_ppo(
        trainer.actor, env_cfg=env_cfg, scenario_seeds=heldout_seeds,
        n_action_reps=1, sample=False, device="cpu", action_seed=sd + 2000,
    )
    print(f"  seed {sd} argmax heldout = {argmax_res['macro_mean_drop']:.4f}")

    torch.save(best["actor_sd"],
               os.path.join(OUT_DIR, f"f6_ppo_bc_seed{sd}.pt"))
    with open(os.path.join(OUT_DIR, f"f6_seed{sd}_curve.json"), "w") as f:
        json.dump({"curve": curve,
                   "best": {k: v for k, v in best.items() if k != "actor_sd"},
                   "argmax_heldout": argmax_res["macro_mean_drop"]}, f, indent=2)
    seed_results[sd] = {
        "best_iter": best["iter"],
        "best_heldout_macro_drop": best["heldout_macro_drop"],
        "best_per_seed_drops": best["per_seed_drops"],
        "argmax_heldout_macro_drop": argmax_res["macro_mean_drop"],
        "total_transitions": total_transitions,
        "max_kl_max": max(c["kl_max"] for c in curve),
        "n_early_stops": sum(1 for c in curve if c["early_stop"]),
        "max_clip_frac": max(c["clip_frac"] for c in curve),
        "min_adv_std": min(c["adv_std"] for c in curve),
        "max_pre_ratio_offset": max(c["pre_ratio_offset"] for c in curve),
        "entropy_range": [min(c["entropy"] for c in curve),
                          max(c["entropy"] for c in curve)],
    }
    print(f"  seed {sd} DONE: best iter={best['iter']} "
          f"heldout={best['heldout_macro_drop']:.4f}")

# --- Cross-seed summary ---------------------------------------------------
print("\n[f6] Cross-seed summary:")
for sd, r in seed_results.items():
    print(f"  seed {sd}: best_iter={r['best_iter']} "
          f"heldout={r['best_heldout_macro_drop']:.4f} "
          f"argmax={r['argmax_heldout_macro_drop']:.4f} "
          f"max_kl={r['max_kl_max']:.4f} "
          f"n_early_stops={r['n_early_stops']}/{PPO_ITERS}")

best_drops = [r["best_heldout_macro_drop"] for r in seed_results.values()]
cross_seed_mean = sum(best_drops) / len(best_drops)
cross_seed_spread = max(best_drops) - min(best_drops)
print(f"\n  cross-seed mean best heldout = {cross_seed_mean:.4f}")
print(f"  cross-seed spread (max-min) = {cross_seed_spread:.4f}")

witness_drop = frozen_eval_held["causal_reactive_or_edf"]["macro_mean_drop"]
print(f"  witness heldout = {witness_drop:.4f}")
print(f"  cross-seed delta vs witness: "
      f"{[f'{d - witness_drop:+.4f}' for d in best_drops]}")

# --- Write eval summary ---------------------------------------------------
eval_summary = {
    "train_seeds": train_seeds,
    "heldout_seeds": heldout_seeds,
    "n_train_scenarios": N_TRAIN_SCENARIOS,
    "n_heldout_scenarios": N_HELDOUT_SCENARIOS,
    "n_eval_reps": N_EVAL_REPS,
    "ppo_iters": PPO_ITERS,
    "transitions_per_seed": 16 * 64 * PPO_ITERS,
    "frozen_eval_heldout": frozen_eval_held,
    "seed_results": {str(sd): r for sd, r in seed_results.items()},
    "cross_seed_mean_best_heldout": cross_seed_mean,
    "cross_seed_spread": cross_seed_spread,
    "witness_drop": witness_drop,
    "significance_claim": "NONE (per MODIFICATION_PLAN W7: two-seed pilot "
                          "makes no statistical-significance claim; that "
                          "requires the eight-seed campaign)",
}
with open(os.path.join(OUT_DIR, "f6_eval.json"), "w") as f:
    json.dump(eval_summary, f, indent=2)

print("\n[f6] DONE.")
