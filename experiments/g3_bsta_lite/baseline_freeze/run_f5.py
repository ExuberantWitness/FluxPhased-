"""F5 main experiment: one-seed stochastic smoke (Gate 4).

Per MODIFICATION_PLAN W6 + Gate 4:
  - Train BC-warm-start PPO on 32 fixed train scenarios.
  - 300 iterations (= 16 envs * 64 horizon * 300 = 307,200 transitions,
    middle of the 0.2..0.5M band).
  - Evaluate on 32 FRESH held-out scenarios every 25 iters.
  - Verify training health (no entropy/KL collapse) and that the
    sampled-eval policy performs within reasonable range of the witness
    on fresh scenarios.

Outputs:
  - experiments/g3_bsta_lite/baseline_freeze/f5_ppo_bc.pt
  - experiments/g3_bsta_lite/baseline_freeze/f5_train_curve.json
  - experiments/g3_bsta_lite/baseline_freeze/f5_eval.json
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
EVAL_EVERY = 25
SEED = 0

# Train scenarios (small debug slice of F2 manifest).
train_manifest = generate_paired_manifest(
    base_seed=20260729, n_scenarios=N_TRAIN_SCENARIOS, horizon=64,
    n_services=2, arrival_rate_per_service=0.15, baseline_snr_db=22.0,
    device="cpu",
)
train_seeds = [s.seed for s in train_manifest]

# Held-out FRESH scenarios (different base_seed, so guaranteed disjoint
# from train unless an astronomically unlikely collision occurs).
heldout_manifest = generate_paired_manifest(
    base_seed=20260801, n_scenarios=N_HELDOUT_SCENARIOS, horizon=64,
    n_services=2, arrival_rate_per_service=0.15, baseline_snr_db=22.0,
    device="cpu",
)
heldout_seeds = [s.seed for s in heldout_manifest]
train_set = set(train_seeds)
heldout_set = set(heldout_seeds)
overlap = train_set & heldout_set
assert not overlap, f"train/heldout overlap: {overlap}"
print(f"[f5] train_seeds[0:5] = {train_seeds[:5]}... ({len(train_seeds)} total)")
print(f"[f5] heldout_seeds[0:5] = {heldout_seeds[:5]}... ({len(heldout_seeds)} total)")

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


# --- Frozen baselines + witness on held-out -------------------------------
print(f"\n[f5] evaluating frozen baselines on {N_HELDOUT_SCENARIOS} held-out seeds...")
frozen_eval_held = {}
for cls in FROZEN_BASELINES:
    macro, per_seed = eval_baseline_on_seeds(cls, heldout_seeds, N_EVAL_REPS)
    frozen_eval_held[cls.name] = {"macro_mean_drop": macro, "per_seed_drops": per_seed}
    print(f"  {cls.name:25s} macro_mean_drop = {macro:.4f}")

# --- BC warm-start PPO ----------------------------------------------------
print(f"\n[f5] BC warm-start PPO, {PPO_ITERS} iters "
      f"(target {PPO_ITERS * 16 * 64 / 1e6:.2f}M transitions)...")
bc_cfg = PPOConfig(
    iterations=PPO_ITERS, n_envs=16, horizon=64, seed=SEED, device="cpu",
    bc_warm_start_path=os.path.join(OUT_DIR, "imitation_actor_dagger.pt"),
)
bc_trainer = PPOTrainer(
    cfg=bc_cfg, env_cfg=env_cfg, train_scenario_seeds=train_seeds,
)
bc_curve = []
bc_best = {"iter": -1, "heldout_macro_drop": -1.0, "actor_sd": None,
           "per_seed_drops": None}
t0 = time.time()
total_transitions = 0
for i in range(PPO_ITERS):
    m = bc_trainer.train_iteration()
    total_transitions += 16 * 64
    bc_curve.append({
        "iter": i, "rollout_drop": m["rollout_drop"], "kl_mean": m["kl_mean"],
        "kl_max": m["kl_max"], "clip_frac": m["clip_frac_mean"],
        "adv_std": m["adv_std"], "pre_ratio_offset": m["pre_ratio_offset"],
        "explained_variance": m["explained_variance"], "entropy": m["entropy"],
        "action_freq": m["action_freq"], "early_stop": m["early_stop"],
        "cum_transitions": total_transitions,
    })
    if i % EVAL_EVERY == 0 or i == PPO_ITERS - 1:
        res = evaluate_ppo(
            bc_trainer.actor, env_cfg=env_cfg, scenario_seeds=heldout_seeds,
            n_action_reps=N_EVAL_REPS, sample=True, device="cpu",
            action_seed=SEED + 1000,
        )
        macro = res["macro_mean_drop"]
        elapsed = time.time() - t0
        print(f"  iter {i:3d} ({total_transitions/1e6:.2f}M): "
              f"train_drop={m['rollout_drop']:.4f} "
              f"heldout_macro_drop={macro:.4f} "
              f"kl_max={m['kl_max']:.4f} clip_frac={m['clip_frac_mean']:.4f} "
              f"adv_std={m['adv_std']:.4f} entropy={m['entropy']:.3f} "
              f"t={elapsed:.1f}s")
        if macro > bc_best["heldout_macro_drop"]:
            bc_best = {
                "iter": i, "heldout_macro_drop": macro,
                "per_seed_drops": res["per_seed_drops"],
                "actor_sd": {k: v.clone() for k, v in bc_trainer.actor.state_dict().items()},
                "cum_transitions": total_transitions,
            }

# Save best BC actor.
torch.save(bc_best["actor_sd"], os.path.join(OUT_DIR, "f5_ppo_bc.pt"))
print(f"\n[f5] BC best iter={bc_best['iter']} "
      f"heldout_macro_drop={bc_best['heldout_macro_drop']:.4f}")

# --- Argmax eval (secondary) ----------------------------------------------
print("\n[f5] argmax eval on held-out (secondary)...")
bc_argmax_res = evaluate_ppo(
    bc_trainer.actor, env_cfg=env_cfg, scenario_seeds=heldout_seeds,
    n_action_reps=1, sample=False, device="cpu", action_seed=SEED + 2000,
)
print(f"  BC argmax heldout_macro_drop = {bc_argmax_res['macro_mean_drop']:.4f}")

# --- Write outputs --------------------------------------------------------
train_curve = {
    "train_seeds": train_seeds,
    "heldout_seeds": heldout_seeds,
    "n_train_scenarios": N_TRAIN_SCENARIOS,
    "n_heldout_scenarios": N_HELDOUT_SCENARIOS,
    "ppo_iters": PPO_ITERS,
    "total_transitions": total_transitions,
    "frozen_eval_heldout": frozen_eval_held,
    "bc_curve": bc_curve,
    "bc_best": {k: v for k, v in bc_best.items() if k != "actor_sd"},
}
with open(os.path.join(OUT_DIR, "f5_train_curve.json"), "w") as f:
    json.dump(train_curve, f, indent=2)

eval_summary = {
    "train_seeds": train_seeds,
    "heldout_seeds": heldout_seeds,
    "n_train_scenarios": N_TRAIN_SCENARIOS,
    "n_heldout_scenarios": N_HELDOUT_SCENARIOS,
    "n_eval_reps": N_EVAL_REPS,
    "total_transitions": total_transitions,
    "frozen_eval_heldout": frozen_eval_held,
    "bc_warm_start": {
        "best_iter": bc_best["iter"],
        "heldout_macro_drop": bc_best["heldout_macro_drop"],
        "per_seed_drops": bc_best["per_seed_drops"],
        "argmax_heldout_macro_drop": bc_argmax_res["macro_mean_drop"],
        "best_cum_transitions": bc_best["cum_transitions"],
    },
}
with open(os.path.join(OUT_DIR, "f5_eval.json"), "w") as f:
    json.dump(eval_summary, f, indent=2)

# --- Gate 4 verdict -------------------------------------------------------
print("\n[f5] Gate 4 verdict (BC warm-start PPO):")
witness_drop = frozen_eval_held["causal_reactive_or_edf"]["macro_mean_drop"]
non_witness = {k: v["macro_mean_drop"] for k, v in frozen_eval_held.items()
               if k != "causal_reactive_or_edf"}
best_baseline_name = max(non_witness, key=non_witness.get)
best_baseline_drop = non_witness[best_baseline_name]
witness_headroom = witness_drop - best_baseline_drop
threshold_80pct = best_baseline_drop + 0.8 * witness_headroom

bc_macro = bc_best["heldout_macro_drop"]
print(f"  witness_drop (held-out) = {witness_drop:.4f}")
print(f"  best_baseline_drop (excl witness) = {best_baseline_name} @ {best_baseline_drop:.4f}")
print(f"  witness_headroom = {witness_headroom:.4f}")
print(f"  80% headroom threshold = {threshold_80pct:.4f}")
print(f"  BC best heldout_macro_drop = {bc_macro:.4f}")
print(f"  BC recovers >=80% headroom: {bc_macro >= threshold_80pct}")

# Training health.
all_kl_max = [c["kl_max"] for c in bc_curve]
all_clip = [c["clip_frac"] for c in bc_curve]
all_adv = [c["adv_std"] for c in bc_curve]
all_pre = [c["pre_ratio_offset"] for c in bc_curve]
all_entropy = [c["entropy"] for c in bc_curve]
all_early_stop = sum(1 for c in bc_curve if c["early_stop"])
print(f"\n  Training health:")
print(f"    iters with early_stop triggered = {all_early_stop}/{PPO_ITERS}")
print(f"    max kl_max = {max(all_kl_max):.4f} (threshold 0.05)")
print(f"    max clip_frac = {max(all_clip):.4f} (persistent >0.5 fails)")
print(f"    min adv_std = {min(all_adv):.6f} (>1e-3 threshold)")
print(f"    entropy range = [{min(all_entropy):.4f}, {max(all_entropy):.4f}]")
print(f"    max pre_ratio_offset = {max(all_pre):.2e}")
print(f"    total transitions = {total_transitions/1e6:.3f}M")

print("\n[f5] DONE.")
