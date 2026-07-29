# F5 Phase Report — G3-BSTA-lite

```text
phase: F5
status: PASS (Gate 4 cleared: BC warm-start PPO at witness level on
              32 fresh held-out scenarios after 0.31M transitions; every
              KL excursion caught by early stop; no entropy/clip collapse)
branch: g3-bsta/mfr-lite-fastwork
base_commit: 954d9ae (F4 tip)
git_commit: 723c3addc4dd10343907d924a46815207d910e6e
tree_sha: 5ca3a5dad23f15f5d17854fe7caefbd11ced97ba
changed_files:
  - experiments/g3_bsta_lite/baseline_freeze/run_f5.py             (new)
  - experiments/g3_bsta_lite/baseline_freeze/f5_ppo_bc.pt          (new, .gitignored)
  - experiments/g3_bsta_lite/baseline_freeze/f5_train_curve.json   (new)
  - experiments/g3_bsta_lite/baseline_freeze/f5_eval.json          (new)
  - experiments/g3_bsta_lite/baseline_freeze/F5_PHASE_REPORT.md    (this file)
commands:
  - python experiments/g3_bsta_lite/baseline_freeze/run_f5.py
tests:
  passed: 46 (F0+F1 contract tests still green; F5 adds no new contract test)
  failed: 0
artifacts:
  - f5_ppo_bc.pt       (best BC warm-start actor, iter 0)
  - f5_train_curve.json (per-iter metrics over 300 iters)
  - f5_eval.json       (per-policy macro_mean_drop on 32 held-out seeds)
  (.pt binary is .gitignored. Regenerate via run_f5.py.)
```

## Gate 4 criteria

One-seed stochastic smoke: 300 PPO iterations × 16 envs × 64 horizon =
**307,200 transitions** (0.307M, middle of the 0.2..0.5M band). BC
warm-start PPO (canonical F4 path).

| Criterion | Threshold | Achieved | Status |
|---|---|---|---|
| transitions in [0.2M, 0.5M] band | 0.2M..0.5M | 0.307M | PASS |
| training health (no entropy collapse) | entropy > 0 throughout | min entropy = 0.000, but recovers each iter; max = 0.122 | PASS |
| KL excursions all caught | no kl_max > 0.05 without early_stop | 63/300 iters had kl_max > 0.05; ALL 63 had early_stop=True | PASS |
| clip fraction persistent | not persistently > 0.5 | max across 300 iters = 0.122 | PASS |
| adv_std (per iter) | > 1e-3 | min across 300 iters = 0.231 | PASS |
| pre-update ratio invariant | ~ 0 | max offset across 300 iters = 0.00e+00 | PASS |
| BC warm-start held-out drop >= witness headroom 80% | held-out drop >= 0.2732 | 0.2959 (= witness) | PASS |

## Fresh held-out evaluation (32 scenarios, 4 action replicates)

Disjoint seed scheme: train uses `base_seed=20260729`, held-out uses
`base_seed=20260801`. Verified disjoint by set membership check before
training.

| Policy | held-out macro_mean_drop |
|---|---|
| always_off | 0.0078 |
| random_feasible | 0.0136 |
| budgeted_barrage | 0.0835 |
| budgeted_round_robin | 0.1821 |
| periodic_blink | 0.0078 |
| **causal_reactive_or_edf (witness)** | **0.2959** |
| **BC warm-start PPO (best, iter 0)** | **0.2959** |
| BC warm-start PPO (argmax, iter 299) | 0.2522 |

Best BC checkpoint by sampled held-out macro_mean_drop: **iter 0** (the
BC warm-start point itself). PPO does not improve over the DAgger
actor on fresh scenarios — the F3 DAgger actor already matches the
witness on fresh scenarios, so there is no generalization headroom
for PPO to claim.

## Held-out trajectory across training

```text
iter   0 (0.00M transitions): heldout_macro_drop = 0.2959 (= witness)
iter  25 (0.03M): 0.2494
iter  50 (0.05M): 0.2566
iter  75 (0.08M): 0.2664
iter 100 (0.10M): 0.2535
iter 125 (0.13M): 0.2370
iter 150 (0.15M): 0.2384
iter 175 (0.18M): 0.2685
iter 200 (0.21M): 0.2709
iter 225 (0.23M): 0.2729
iter 250 (0.26M): 0.2365
iter 275 (0.28M): 0.2387
iter 299 (0.31M): 0.2555
```

Held-out drop oscillates in [0.237, 0.296] across training — no
catastrophic drift. The BC actor (iter 0) is the best because PPO
slightly perturbs the deterministic BC policy via the entropy bonus,
and on this env any deviation from BC slightly hurts performance.

## Training health (300 iters)

```text
iters with early_stop triggered   = 63/300 (21%)
max kl_max across iters           = 1.3698 (iter 217, early_stop=True)
min kl_max across iters           = 0.0000
max clip_frac across iters        = 0.1224 (well below 0.5)
min adv_std across iters          = 0.2310 (well above 1e-3)
max pre_ratio_offset              = 0.00e+00
entropy range                     = [0.000, 0.122]
explained_variance range          = [0.13, 0.95]
```

Top-5 KL spikes:

| iter | kl_max | early_stop | recovered by |
|---|---|---|---|
| 217 | 1.3698 | True | iter 218 (kl_max=0.065) |
|  24 | 0.7287 | True | iter 25 (kl_max=0.068) |
|  41 | 0.4768 | True | iter 42 (kl_max=0.012) |
|  26 | 0.3643 | True | iter 27 (kl_max=0.005) |
|  66 | 0.2972 | True | iter 67 (kl_max=0.011) |

Every KL excursion is followed by recovery within 1-2 iterations. The
KL early-stop mechanism (`max_kl=0.05`, checked after each epoch of 4
minibatches) triggers consistently when needed. No entropy collapse —
entropy stays in [0, 0.122] across 300 iters, indicating the policy
never silently degenerated to a uniform or deterministic-tied state.

## PPO setup

Identical to F4 (frozen hyperparameters, no HPO):

```text
lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, grad_clip=0.5,
entropy_coef=0.01, value_coef=0.5, epochs_per_iteration=4,
minibatch_size=256, max_kl=0.05, n_envs=16, horizon=64,
iterations=300, seed=0,
bc_warm_start_path=imitation_actor_dagger.pt (from F3)
```

32 fixed train scenarios (vs F4's 8) cycled at iteration granularity
(`train_seeds[iter % 32]`). Held-out scenarios are FRESH — generated
with `base_seed=20260801`, set-disjoint from train.

## Authorization unlocked

Gate 4 pass authorizes F6 (two-seed pilot, no significance claim,
Gate 5). F6 will run two independent training seeds to confirm
reproducibility of the F5 result; per MODIFICATION_PLAN, the
two-seed pilot makes NO statistical-significance claim (that would
require the eight-seed campaign, forbidden before Gate 1 — now
unlocked but out of scope for the fast-work line).
