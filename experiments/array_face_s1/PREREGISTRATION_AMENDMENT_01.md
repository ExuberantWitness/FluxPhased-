# Array-Face S1 Preregistration Amendment 01

**Date**: 2026-07-30
**Scope**: PPO exploration hyperparameters only

## Trigger

Initial S1 run (anneal_frac=0.3, entropy_coef_init=1e-3, original prereg)
saturated at val_drop=0.0929 by iter 100. Diagnosis:

1. Actor collapsed to "always jam_svc_1 first 16 steps" (deterministic,
   entropy=0.0013 at iter 999).
2. Same local minimum lite R2 was stuck at iter 49-129; lite broke out
   at iter ~169. S1 did not break out within 1000 iter.
3. Root cause isolated via embedded-policy verification: lite iter 2999
   weights (with beam_az columns zeroed) on S1 env achieves drop=0.2734
   ~ lite env drop=0.2628. So S1 optimal policy = lite optimal policy;
   S1 failure is exploration collapse, NOT problem difficulty.
4. Direct trigger of collapse: `entropy_anneal_frac=0.3` drove
   `entropy_coef` to 0 at iter 300, removing exploration bonus entirely.

## Change

| Hyperparameter | Original (prereg §4) | Amendment 01 |
|---|---|---|
| `entropy_coef_init` | 1e-3 | **5e-3** (5x) |
| `entropy_anneal_frac` | 0.3 | **1.0** (never anneal) |

All other hyperparameters unchanged (lr=3e-5, target_kl=0.01, clip=0.2,
n_envs=16, horizon=64, iterations=1000).

## What this is NOT

- NOT a change to physics, env, observation, action space, or manifests
- NOT a change to evaluation protocol (same 64 val seeds, same every-10-iter cadence)
- NOT a gate re-judgment (still exploratory, > 0.5M prereg cap)
- NOT a multi-seed rescue (single seed, single run, same training_seed=20260730)

## Re-run plan

1. Backup original output: `s1_ppo_output_anneal0.3_coef1e-3/` (kept frozen)
2. Fresh run with Amendment 01 hyperparameters
3. New output: `s1_ppo_output/`
4. Compare learning curves (anneal=0.3 vs anneal=1.0)
5. If still stuck at <0.15, escalate to Plan C (lr=1e-4 + entropy=3e-3) or
   Plan D (n_envs=32)
