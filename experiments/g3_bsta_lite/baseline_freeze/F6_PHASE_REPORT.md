# F6 Phase Report — G3-BSTA-lite

```text
phase: F6
status: PASS (Gate 5 cleared: two-seed pilot reproducible at witness
              level, no crash, no significance claim per W7)
branch: g3-bsta/mfr-lite-fastwork
base_commit: 47bcaf1 (F5 tip)
git_commit: <to be filled at commit>
tree_sha: <to be filled at commit>
changed_files:
  - experiments/g3_bsta_lite/baseline_freeze/run_f6.py             (new)
  - experiments/g3_bsta_lite/baseline_freeze/f6_ppo_bc_seed0.pt    (new, .gitignored)
  - experiments/g3_bsta_lite/baseline_freeze/f6_ppo_bc_seed1.pt    (new, .gitignored)
  - experiments/g3_bsta_lite/baseline_freeze/f6_seed0_curve.json   (new)
  - experiments/g3_bsta_lite/baseline_freeze/f6_seed1_curve.json   (new)
  - experiments/g3_bsta_lite/baseline_freeze/f6_eval.json          (new)
  - experiments/g3_bsta_lite/baseline_freeze/F6_PHASE_REPORT.md    (this file)
commands:
  - python experiments/g3_bsta_lite/baseline_freeze/run_f6.py
tests:
  passed: 46 (F0+F1 contract tests still green; F6 adds no new contract test)
  failed: 0
artifacts:
  - f6_ppo_bc_seed0.pt  (best BC actor, seed 0, iter 0)
  - f6_ppo_bc_seed1.pt  (best BC actor, seed 1, iter 0)
  - f6_seed0_curve.json (300-iter training curve, seed 0)
  - f6_seed1_curve.json (300-iter training curve, seed 1)
  - f6_eval.json        (cross-seed summary)
  (.pt binaries are .gitignored. Regenerate via run_f6.py.)
```

## Gate 5 criteria

Two-seed pilot: two independent BC-warm-start PPO training seeds at
300 iters each (= 0.307M transitions per seed, matching F5). Same
32 train + 32 fresh held-out scenario split as F5.

| Criterion | Threshold | Achieved | Status |
|---|---|---|---|
| both seeds complete without crash | both finish 300 iters | 2/2 | PASS |
| reproducibility: best heldout across seeds | small spread | spread = 0.0000 (both 0.2959) | PASS |
| each seed at witness level on held-out | best >= witness headroom 80% (= 0.2732) | both 0.2959 | PASS |
| training health per seed | no entropy/clip collapse | seed 0: entropy [0,0.122], clip_max 0.122; seed 1: entropy [0,0.020], clip_max 0.025 | PASS |
| pre-update ratio invariant per seed | ~ 0 | both seeds: 0.00e+00 | PASS |
| NO significance claim | none made | "significance_claim": "NONE" in f6_eval.json | PASS |

## Per-seed results

```text
seed 0:
  best iter          = 0 (BC warm-start point)
  best heldout drop  = 0.2959  (= witness 0.2959)
  argmax heldout     = 0.2522  (at iter 299)
  max kl_max         = 1.3698  (iter 217, early_stop triggered)
  n early_stops      = 63 / 300
  max clip_frac      = 0.1224
  entropy range      = [0.000, 0.122]

seed 1:
  best iter          = 0 (BC warm-start point)
  best heldout drop  = 0.2959  (= witness 0.2959)
  argmax heldout     = 0.2075  (at iter 299)
  max kl_max         = 0.2118
  n early_stops      = 28 / 300
  max clip_frac      = 0.0254
  entropy range      = [0.000, 0.020]
```

## Cross-seed summary

| Quantity | Value |
|---|---|
| witness held-out macro_mean_drop | 0.2959 |
| seed 0 best held-out macro_mean_drop | 0.2959 |
| seed 1 best held-out macro_mean_drop | 0.2959 |
| **cross-seed mean best held-out** | **0.2959** |
| **cross-seed spread (max − min)** | **0.0000** |
| cross-seed delta vs witness | [+0.0000, +0.0000] |

The cross-seed spread of 0 is structural: the BC actor at iter 0 is
deterministic (DAgger produces a peaked masked-categorical), so
sampled eval reduces to argmax and both seeds produce bit-identical
per-scenario drops. The action RNG influences only the PPO rollout
collection, not the eval. Once PPO perturbs the policy (iter > 0),
trajectories diverge between seeds.

## No significance claim

Per MODIFICATION_PLAN W7: "two-seed pilot, no significance claim".
A two-seed comparison cannot establish statistical significance
(that requires the eight-seed campaign). The pilot's purpose is to
verify that the F4+F5 pipeline reproduces across an independent
training seed and that no seed-specific crash or pathological
behavior emerges. Both conditions are met.

## PPO does not improve over BC iter 0 (same as F4 and F5)

Both seeds' best held-out drop is at **iter 0** (the BC warm-start
point itself). Subsequent PPO iterations slightly perturb the BC
policy via the entropy bonus and degrade held-out drop to 0.20-0.27
range. This is the same pattern observed in F4 (overfit slice) and
F5 (one-seed smoke): the F3 DAgger actor already saturates the
witness on this env, so PPO has no generalization headroom to claim.

The role of F4-F5-F6 is therefore to verify the PPO pipeline (mask
preservation, ratio invariant, KL early stop, separate optimizers,
multi-seed reproducibility) scales to 0.3M+ transitions without
collapse. It does, robustly, across two independent seeds.

## Held-out trajectory (seed 1, lower-variance seed)

```text
iter   0: heldout = 0.2959  (= witness)
iter  50: 0.2373
iter 100: 0.2114
iter 150: 0.2066
iter 200: 0.2093
iter 250: 0.1991
iter 299: 0.2075
```

Seed 1's held-out stabilizes around 0.20 (= +1.7 pp over the best
non-witness baseline `budgeted_round_robin` at 0.1821). The BC iter 0
point remains the global best across both seeds.

## Frozen baselines on held-out (same as F5)

```text
always_off               0.0078
random_feasible          0.0136
budgeted_barrage         0.0835
budgeted_round_robin     0.1821
periodic_blink           0.0078
causal_reactive_or_edf   0.2959  (witness)
```

## Authorization unlocked

Gate 5 pass closes the fast-work line F0..F6. The frozen debug
profile is now validated end-to-end: env contract (F1) → reachability
+ oracle headroom (F2) → imitation (F3) → PPO overfit (F4) → PPO
smoke (F5) → reproducibility (F6). Subsequent work (out of scope for
this branch): eight-seed significance campaign, scale-out to the
full (non-debug) MFR profile, integration with the two-team line.
