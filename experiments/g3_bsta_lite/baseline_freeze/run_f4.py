"""F4 main experiment: masked PPO fixed-scenario overfit (Gate 3).

Per MODIFICATION_PLAN W5 + Gate 3:
  - Train BC-warm-start PPO and scratch PPO on 8 fixed debug scenarios.
  - Eval on same 8 scenarios (overfit gate).
  - Compare to witness and frozen baselines on the same 8 scenarios.

Outputs:
  - experiments/g3_bsta_lite/baseline_freeze/f4_ppo_bc.pt  (best BC actor)
  - experiments/g3_bsta_lite/baseline_freeze/f4_ppo_scratch.pt  (final scratch actor)
  - experiments/g3_bsta_lite/baseline_freeze/f4_train_curve.json
  - experiments/g3_bsta_lite/baseline_freeze/f4_eval.json
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
N_TRAIN_SCENARIOS = 8
N_EVAL_REPS = 4
PPO_ITERS = 30
EVAL_EVERY = 5
SEED = 0

# Fixed debug suite: 8 eligible scenarios via F2 manifest path.
manifest = generate_paired_manifest(
    base_seed=20260729, n_scenarios=N_TRAIN_SCENARIOS, horizon=64, n_services=2,
    arrival_rate_per_service=0.15, baseline_snr_db=22.0, device="cpu",
)
train_seeds = [s.seed for s in manifest]
print(f"[f4] train_seeds = {train_seeds}")

env_cfg = EnvConfig()


def eval_baseline_on_seeds(cls, seeds, n_reps):
    """Macro-mean drop_ratio of a baseline class over the given seeds."""
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


# --- Frozen baselines + witness on the same 8 scenarios -------------------
print("\n[f4] evaluating frozen baselines on the 8 fixed scenarios...")
frozen_eval = {}
for cls in FROZEN_BASELINES:
    macro, per_seed = eval_baseline_on_seeds(cls, train_seeds, N_EVAL_REPS)
    frozen_eval[cls.name] = {"macro_mean_drop": macro, "per_seed_drops": per_seed}
    print(f"  {cls.name:25s} macro_mean_drop = {macro:.4f}")

# --- BC warm-start PPO ----------------------------------------------------
print(f"\n[f4] BC warm-start PPO, {PPO_ITERS} iters...")
bc_cfg = PPOConfig(
    iterations=PPO_ITERS, n_envs=16, horizon=64, seed=SEED, device="cpu",
    bc_warm_start_path=os.path.join(OUT_DIR, "imitation_actor_dagger.pt"),
)
bc_trainer = PPOTrainer(
    cfg=bc_cfg, env_cfg=env_cfg, train_scenario_seeds=train_seeds,
)
bc_curve = []
bc_best = {"iter": -1, "macro_drop": -1.0, "actor_sd": None,
           "per_seed_drops": None}
t0 = time.time()
for i in range(PPO_ITERS):
    m = bc_trainer.train_iteration()
    bc_curve.append({
        "iter": i, "rollout_drop": m["rollout_drop"], "kl_mean": m["kl_mean"],
        "kl_max": m["kl_max"], "clip_frac": m["clip_frac_mean"],
        "adv_std": m["adv_std"], "pre_ratio_offset": m["pre_ratio_offset"],
        "explained_variance": m["explained_variance"], "entropy": m["entropy"],
        "action_freq": m["action_freq"], "early_stop": m["early_stop"],
    })
    if i % EVAL_EVERY == 0 or i == PPO_ITERS - 1:
        res = evaluate_ppo(
            bc_trainer.actor, env_cfg=env_cfg, scenario_seeds=train_seeds,
            n_action_reps=N_EVAL_REPS, sample=True, device="cpu",
            action_seed=SEED + 1000,
        )
        macro = res["macro_mean_drop"]
        print(f"  iter {i:3d}: train_drop={m['rollout_drop']:.4f} "
              f"eval_macro_drop={macro:.4f} kl_max={m['kl_max']:.4f} "
              f"clip_frac={m['clip_frac_mean']:.4f} adv_std={m['adv_std']:.4f} "
              f"entropy={m['entropy']:.3f} t={time.time()-t0:.1f}s")
        if macro > bc_best["macro_drop"]:
            bc_best = {
                "iter": i, "macro_drop": macro,
                "per_seed_drops": res["per_seed_drops"],
                "actor_sd": {k: v.clone() for k, v in bc_trainer.actor.state_dict().items()},
            }

# Save best BC actor.
torch.save(bc_best["actor_sd"], os.path.join(OUT_DIR, "f4_ppo_bc.pt"))
print(f"\n[f4] BC best iter={bc_best['iter']} macro_drop={bc_best['macro_drop']:.4f}")

# --- Scratch PPO ----------------------------------------------------------
print(f"\n[f4] scratch PPO, {PPO_ITERS} iters...")
scratch_cfg = PPOConfig(
    iterations=PPO_ITERS, n_envs=16, horizon=64, seed=SEED, device="cpu",
    bc_warm_start_path=None,
)
scratch_trainer = PPOTrainer(
    cfg=scratch_cfg, env_cfg=env_cfg, train_scenario_seeds=train_seeds,
)
scratch_curve = []
scratch_best = {"iter": -1, "macro_drop": -1.0, "actor_sd": None,
                "per_seed_drops": None}
t0 = time.time()
for i in range(PPO_ITERS):
    m = scratch_trainer.train_iteration()
    scratch_curve.append({
        "iter": i, "rollout_drop": m["rollout_drop"], "kl_mean": m["kl_mean"],
        "kl_max": m["kl_max"], "clip_frac": m["clip_frac_mean"],
        "adv_std": m["adv_std"], "pre_ratio_offset": m["pre_ratio_offset"],
        "explained_variance": m["explained_variance"], "entropy": m["entropy"],
        "action_freq": m["action_freq"], "early_stop": m["early_stop"],
    })
    if i % EVAL_EVERY == 0 or i == PPO_ITERS - 1:
        res = evaluate_ppo(
            scratch_trainer.actor, env_cfg=env_cfg, scenario_seeds=train_seeds,
            n_action_reps=N_EVAL_REPS, sample=True, device="cpu",
            action_seed=SEED + 1000,
        )
        macro = res["macro_mean_drop"]
        print(f"  iter {i:3d}: train_drop={m['rollout_drop']:.4f} "
              f"eval_macro_drop={macro:.4f} kl_max={m['kl_max']:.4f} "
              f"clip_frac={m['clip_frac_mean']:.4f} adv_std={m['adv_std']:.4f} "
              f"entropy={m['entropy']:.3f} t={time.time()-t0:.1f}s")
        if macro > scratch_best["macro_drop"]:
            scratch_best = {
                "iter": i, "macro_drop": macro,
                "per_seed_drops": res["per_seed_drops"],
                "actor_sd": {k: v.clone() for k, v in scratch_trainer.actor.state_dict().items()},
            }
torch.save(scratch_best["actor_sd"] if scratch_best["actor_sd"] is not None
           else {k: v.clone() for k, v in scratch_trainer.actor.state_dict().items()},
           os.path.join(OUT_DIR, "f4_ppo_scratch.pt"))
print(f"\n[f4] scratch best iter={scratch_best['iter']} "
      f"macro_drop={scratch_best['macro_drop']:.4f}")

# --- Argmax eval (secondary) ----------------------------------------------
print("\n[f4] argmax eval (secondary)...")
bc_argmax_res = evaluate_ppo(
    bc_trainer.actor, env_cfg=env_cfg, scenario_seeds=train_seeds,
    n_action_reps=1, sample=False, device="cpu", action_seed=SEED + 2000,
)
scratch_argmax_res = evaluate_ppo(
    scratch_trainer.actor, env_cfg=env_cfg, scenario_seeds=train_seeds,
    n_action_reps=1, sample=False, device="cpu", action_seed=SEED + 2000,
)
print(f"  BC argmax macro_drop = {bc_argmax_res['macro_mean_drop']:.4f}")
print(f"  scratch argmax macro_drop = {scratch_argmax_res['macro_mean_drop']:.4f}")

# --- Write outputs --------------------------------------------------------
train_curve = {
    "train_seeds": train_seeds,
    "n_train_scenarios": N_TRAIN_SCENARIOS,
    "ppo_iters": PPO_ITERS,
    "frozen_eval_on_same_8": frozen_eval,
    "bc_curve": bc_curve,
    "scratch_curve": scratch_curve,
    "bc_best": {k: v for k, v in bc_best.items() if k != "actor_sd"},
    "scratch_best": {k: v for k, v in scratch_best.items() if k != "actor_sd"},
}
with open(os.path.join(OUT_DIR, "f4_train_curve.json"), "w") as f:
    json.dump(train_curve, f, indent=2)

eval_summary = {
    "train_seeds": train_seeds,
    "n_train_scenarios": N_TRAIN_SCENARIOS,
    "n_eval_reps": N_EVAL_REPS,
    "frozen_eval_on_same_8": frozen_eval,
    "bc_warm_start": {
        "best_iter": bc_best["iter"],
        "macro_mean_drop": bc_best["macro_drop"],
        "per_seed_drops": bc_best["per_seed_drops"],
        "argmax_macro_drop": bc_argmax_res["macro_mean_drop"],
    },
    "scratch": {
        "best_iter": scratch_best["iter"],
        "macro_mean_drop": scratch_best["macro_drop"],
        "per_seed_drops": scratch_best["per_seed_drops"],
        "argmax_macro_drop": scratch_argmax_res["macro_mean_drop"],
    },
}
with open(os.path.join(OUT_DIR, "f4_eval.json"), "w") as f:
    json.dump(eval_summary, f, indent=2)

# --- Gate 3 verdict -------------------------------------------------------
print("\n[f4] Gate 3 verdict:")
witness_drop = frozen_eval["causal_reactive_or_edf"]["macro_mean_drop"]
non_witness = {k: v["macro_mean_drop"] for k, v in frozen_eval.items()
               if k != "causal_reactive_or_edf"}
best_baseline_name = max(non_witness, key=non_witness.get)
best_baseline_drop = non_witness[best_baseline_name]
witness_headroom = witness_drop - best_baseline_drop
threshold_80pct = best_baseline_drop + 0.8 * witness_headroom
threshold_5pp_above_best = best_baseline_drop + 0.05

bc_macro = bc_best["macro_drop"]
bc_pass_headroom = bc_macro >= threshold_80pct
bc_pass_5pp = bc_macro >= threshold_5pp_above_best

print(f"  witness_drop = {witness_drop:.4f}")
print(f"  best_baseline (excl witness) = {best_baseline_name} @ {best_baseline_drop:.4f}")
print(f"  witness_headroom = {witness_headroom:.4f}")
print(f"  threshold (80% headroom) = {threshold_80pct:.4f}")
print(f"  threshold (+5pp over best baseline) = {threshold_5pp_above_best:.4f}")
print(f"  BC macro_drop = {bc_macro:.4f}")
print(f"  BC PASS headroom (>=80%) = {bc_pass_headroom}")
print(f"  BC PASS 5pp (>=+5pp) = {bc_pass_5pp}")

# Training health checks.
all_kl_max = [c["kl_max"] for c in bc_curve]
all_clip = [c["clip_frac"] for c in bc_curve]
all_adv = [c["adv_std"] for c in bc_curve]
all_pre = [c["pre_ratio_offset"] for c in bc_curve]
print(f"\n  BC training health:")
print(f"    max kl_max across iters = {max(all_kl_max):.4f} (threshold 0.05)")
print(f"    max clip_frac across iters = {max(all_clip):.4f} "
      f"(threshold persistent >0.5)")
print(f"    min adv_std across iters = {min(all_adv):.6f} (threshold >1e-3)")
print(f"    max pre_ratio_offset = {max(all_pre):.2e} (should be ~0)")

print("\n[f4] DONE.")
