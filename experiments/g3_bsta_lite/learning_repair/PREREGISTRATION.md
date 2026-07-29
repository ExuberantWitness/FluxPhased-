# Preregistration — G3-BSTA-lite Learning-Repair Branch

```text
document:   PREREGISTRATION.md
branch:     g3-bsta/mfr-lite-learning-repair
base:       01ea4283f422cd01fec5fb15037c5012543adf7a
registered: 2026-07-29 (before any new training on this branch)
```

This preregistration binds the experimental design **before** any new
training run on this branch. It may only be amended by an explicit
`PREREGISTRATION_AMENDMENT_*.md` document that names what changed and
why. Thresholds listed here are **frozen**: a result that fails a
threshold is reported as a failure, not retuned.

## 1. Primary research question

> Does PPO, trained from scratch on `mdp_sanity_v1` and selected only
> by a frozen validation set, produce a policy that on a held-out,
> never-touched locked-test set is strictly better than (a) its own
> pre-training initialization and (b) the strongest non-learning causal
> baseline, with scenario-level paired LCB95 > 0?

The same question is then asked for `pomdp_v1` in R4, conditional on
R2/R3 PASS.

## 2. Profiles

Two profiles are introduced and must not be conflated:

### 2.1 `mdp_sanity_v1`

A genuine fully-observed MDP. The observation explicitly exposes the
current queue / load bucket and the current radar service. Purpose:
prove the PPO machinery can learn **at all** on this env. **Not** a
POMDP and **not** a paper claim about active perception.

### 2.2 `pomdp_v1`

A genuine POMDP. The actor / critic / witness share the same
information set, which excludes:
- the true MissionTracker pending count,
- the radar's hidden phase / mode,
- the scenario's future arrivals.

Activity / urgency must come from a jammer-side delayed / noisy
emission / intercept estimator with `delay ≥ 1` and a non-invertible
proxy. If a recurrent actor is used it must be GRU / LSTM with hidden
state reset on episode boundary.

## 3. Scenario manifests (four disjoint sets)

Generated **once**, committed as JSON, and asserted disjoint in code
and unit tests:

| manifest | size (eligible scenarios) | base_seed | purpose |
|---|---|---|---|
| `dagger_train.json` | 128 | 21000101 | F3-style DAgger initial + aggregation rounds |
| `ppo_train.json` | 64 | 21000201 | PPO rollout scenarios, cycled per iter |
| `checkpoint_validation.json` | 64 | 21000301 | checkpoint selection (validation only) |
| `locked_test.json` | 128 | 21000401 | untouched test, run **once** at R3 |

All DAgger aggregation rounds' seeds are appended to `dagger_train.json`
and included in the overlap audit.

Pairwise intersection must be empty. The audit is recorded in
`manifests/MANIFEST_AUDIT.json`. The "old" `20260801..` seed range is
explicitly excluded from the locked test.

Each manifest entry stores the **concrete scenario seed** (not just
`base_seed`) plus the generating config, generator commit SHA, and a
per-scenario SHA-256 of the arrivals table.

## 4. Checkpoint semantics

- `iteration = -1`: **pristine init**, evaluated and snapshotted before
  any optimizer update;
- `iteration = 0`: after the first PPO outer update;
- `iteration = k ≥ 0`: after the (k+1)-th outer update.

Every artifact records: `update_count`, `cumulative_transitions`,
`checkpoint_origin`, `training_seed`, `config_sha`, `manifest_sha`.

## 5. PPO update safety

- KL is estimated after **every minibatch**;
- if `KL(post) > target_kl`, the actor update is rolled back to the
  pre-outer-update snapshot and the iteration ends;
- `target_kl ∈ {0.01, 0.02}`; the value is chosen by validation only
  from this preregistered set;
- actor learning rate `∈ {3e-5, 1e-4}` chosen by validation only;
- entropy coefficient annealed `1e-3 → 0` over the first 30 % of
  training;
- locked-test set is **never** queried for tuning.

## 6. R2 Gate 3 (scratch PPO on `mdp_sanity_v1`)

Preregistered PASS criteria (all required):

- training completes within 0.5 M transitions;
- scenario-level paired LCB95(scratch_trained − scratch_init) > 0 on
  `checkpoint_validation`;
- scenario-level paired LCB95(scratch_trained − best_non_witness_baseline)
  > 0 on `checkpoint_validation`;
- point improvement over best non-witness baseline ≥ 5 pp;
- witness-headroom recovery ≥ 80 %;
- mask violations = 0, requested=executed, energy never negative;
- pre-update ratio offset ≈ 0 (mask-replay invariant).

If any sub-criterion fails, R2 status is
`BLOCKED_LEARNING_CONTRIBUTION` and the next authorised phase is
`NONE`. No BC warm-start substitution is permitted at this gate.

## 7. R3 corrected two-seed pilot

Run only if R2 PASS. Two independent training seeds, ≤ 0.5 M
transitions each. Checkpoint selected by `checkpoint_validation` only.
After both seeds' configs and checkpoints are frozen, the
`locked_test` set is run **exactly once**.

Preregistered PASS criteria (all required, per seed):

- training completes;
- checkpoint used is the **validation-selected** one, not init;
- paired LCB95(trained − pristine_init) > 0 on locked_test;
- paired LCB95(trained − best_non_witness_baseline) > 0 on locked_test;
- point improvement over best non-witness baseline ≥ 2 pp;
- shuffled-observation control and time-only control each ≥ 3 pp
  below trained policy;
- return/drop Spearman ≥ 0.8 across (scenario, seed, rep) rows;
- per-episode mask / energy / requested-executed violations all 0;
- last-iter checkpoint also reported (not just the validation-best).

If either seed fails any sub-criterion, R3 status is
`BLOCKED_TWO_SEED_LEARNING`, the next authorised phase is `NONE`, and
no third seed is run.

## 8. R4 POMDP pilot

Run only if R3 PASS. `pomdp_v1` profile with a recurrent masked PPO.
The actor / critic / witness must share the same information set. The
prior `clairvoyant_oracle` is renamed `privileged_greedy_heuristic`
throughout; it is **not** an upper bound and is not deployable.

POMDP pilot PASS requires directionally:
- learned policy above the strongest causal heuristic baseline;
- shuffled-history and zero-history controls drop by ≥ 3 pp;
- recurrent hidden state carries radar-mode belief (action distribution
  conditioned on history differs from history-blind baseline);
- direction consistent across ≥ 4 preregistered neighbor cells
  (budget / tau / SNR / arrival-rate);
- does not hold on a single deterministic radar pattern only.

## 9. Statistics — required for every claimed metric

- policy-independent denominator (pre-generated arrivals table);
- within-scenario aggregation across action reps first;
- paired delta across policies on the same scenario;
- mean, SE, LCB95 / CI95 reported;
- zero-denominator handling stated explicitly;
- replicate-level raw rows saved to `raw_rows.jsonl`;
- run command, commit, tree, config SHA, manifest SHA recorded;
- no test-set checkpoint selection;
- no pseudo-replication (scenarios × reps are not training replicates).

## 10. Controls that must exist before any PASS claim

| control | definition |
|---|---|
| `random_untrained` | uniform-over-mask sampled actions, fixed seed |
| `scratch_init` | iteration = -1 snapshot of the scratch PPO actor |
| `scratch_trained` | validation-selected checkpoint of scratch PPO |
| `pristine_bc` | F3 DAgger actor, no PPO fine-tuning |
| `residual_ppo` | stop-gradient BC logits + α · residual, α grown 0 → 1, with KL-to-BC constraint |
| `shuffled_observation` | trained actor with observation channels shuffled by a fixed permutation |
| `time_only` | actor sees only `[step_idx / horizon]` (no queue, no urgency, no prev action) |
| `no_update` | initial policy evaluated under the same evaluation harness |

## 11. Forbidden

- substituting any control's number for the scratch PPO number in Gate 3;
- reporting only the validation-best checkpoint without the last-iter checkpoint;
- running `locked_test` more than once;
- retuning thresholds after seeing locked-test results;
- declaring statistical significance from two seeds;
-echoing credentials in any artifact;
- re-using quarantined orphan MFR archive bytes.
