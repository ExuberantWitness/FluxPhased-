# Post-Audit Correction — G3-BSTA-lite Fast-Work Line

```text
document: POST_AUDIT_CORRECTION.md
branch:    g3-bsta/mfr-lite-learning-repair
base:      01ea4283f422cd01fec5fb15037c5012543adf7a (g3-bsta/mfr-lite-fastwork tip)
base_tree: e1e502e5ac7089ee9e8a0649d46307731780e8be
issued:    2026-07-29
status:    DEBUG_VERTICAL_SLICE_COMPLETE / PPO_LEARNING_NOT_ESTABLISHED
```

## 1. Purpose

This document is an audit-driven correction of the prior
`g3-bsta/mfr-lite-fastwork` line's claims. It does **not** modify or
rewrite history on `g3-bsta/mfr-lite-fastwork`; the prior
`LINE_REPORT.md` is preserved as a historical engineering record. This
new branch (`g3-bsta/mfr-lite-learning-repair`) carries the correction
and all subsequent repair work.

A separate note on naming: the prior `LINE_REPORT.md` describes `f2ef2da`
as the branch tip, but the actual summary commit on
`g3-bsta/mfr-lite-fastwork` is **`01ea4283f422cd01fec5fb15037c5012543adf7a`**
(tree `e1e502e5ac7089ee9e8a0649d46307731780e8be`). The new repair branch
is created from this exact commit.

## 2. What the audit confirms

The following claims from the prior line are accepted as **correct** and
remain standing:

1. **46/46 contract tests pass** on `tests/g3_bsta_lite/` — verified by
   an independent re-run on this branch:
   ```
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD \
     pytest -q tests/g3_bsta_lite
   → 46 passed
   ```
   These tests prove the **debug vertical slice** (action/mask/transition
   invariants, vector isolation, basic PPO math primitives). They do
   not prove that PPO learns a better policy.

2. **Gate 1 headroom is real** (128 paired scenarios × 4 action reps):
   - privileged clairvoyant heuristic macro drop = **0.3313**
   - causal witness (CausalReactiveOrEDF) macro drop = **0.2680**
   - best non-witness frozen baseline (budgeted_round_robin) = **0.1601**
   - privileged-vs-round_robin gap = **+17.12 pp** (criterion ≥10 pp: PASS)
   - witness LCB95 vs every non-witness baseline ≥ **+9.12 pp**
     (criterion >7.5 pp: PASS)
   - neighbor sweep min LCB ≥ **+7.50 pp** (criterion >5 pp: PASS)

3. **Gate 2 imitation feasibility is real** (F3 DAgger):
   - held-out static top-1 = 100%
   - held-out rollout gap recovery = 101.5%
   This proves an MLP can express the witness from the same
   observation. It does **not** prove PPO learning.

## 3. What the audit invalidates

### 3.1 Data leak: F5/F6 "fresh held-out" overlaps F3 DAgger train

- F3 DAgger initial training scenarios start at `base_seed = 20260729`
  and produce 128 eligible scenarios covering seeds roughly in
  `20260729..20260856` (eligible filter skips some seeds).
- F5/F6 "fresh held-out" used `base_seed = 20260801, n_scenarios = 32`,
  i.e. seeds `20260801..20260832`. **All 32 of these are inside the
  F3 DAgger training seed range.**
- The same "held-out" set was also used as the **checkpoint-selection
  validation set** in F5/F6.
- Consequence: the F5/F6 held-out numbers are **validation**, not
  untouched test. They cannot support any generalization claim.

### 3.2 Gate 3 scratch PPO actually FAILS the original criterion

Reported numbers on the 8 fixed debug scenarios (4 action reps):

| policy | macro drop |
|---|---|
| scratch PPO (best iter, 30 iters) | 0.16891 |
| budgeted_round_robin | 0.13109 |
| causal witness | 0.25112 |

Derived:
- witness-headroom recovery by scratch PPO ≈ **31.5 %**
  ((0.16891 − 0.13109) / (0.25112 − 0.13109))
- scratch PPO vs round_robin = **+3.78 pp**
- scenario-level one-sided LCB95(scratch − round_robin) ≈ **negative**

Original Gate 3 criterion required scratch PPO to recover ≥ 80 % witness
headroom and to beat every frozen baseline by ≥ 5 pp on the fixed
slice. Scratch PPO **fails both sub-criteria**.

The prior phase report used **BC warm-start PPO** to claim Gate 3 PASS.
This is incorrect: BC warm-start is the F3 imitation actor carried
forward, not evidence of PPO learning. The "best checkpoint = iter 0"
pattern across F4 / F5 / F6 reflects that BC is already at witness
level, and PPO never improves over it on this profile.

### 3.3 F5/F6 best_iter = 0 is not a pristine BC init

In the prior trainer, the eval at `iter 0` was performed **after** the
first `train_iteration()` had run, which collects a rollout of 16 envs ×
64 horizon = 1024 transitions and applies one PPO update across 4
epochs × 4 minibatches. So "iter 0" already reflects a non-trivial
PPO update, not a pristine BC actor. The cross-seed spread of 0 on
this "iter 0" therefore reflects a common BC starting point + a tiny
deterministic perturbation, **not** a reproducibility result for PPO
training.

### 3.4 PPO-trained final policy is worse than witness

On the 32 "held-out" scenarios used by F5/F6:

| policy | macro drop |
|---|---|
| causal witness | 0.2959 |
| seed 0 PPO argmax (final iter) | 0.2522 |
| seed 1 PPO argmax (final iter) | 0.2075 |

PPO training **degrades** the BC-warm-started policy on this slice.
The "best = iter 0" pattern is therefore an artefact of of the
checkpoint selection rule picking the least-degraded point, not
evidence of learning.

## 4. Environment / contract issues found

### 4.1 Observation leaks pending queue count

The current urgency channel in the actor observation is computed as
`n_pending_per_service / tau_window` with `obs_delay_steps = 0`. With
delay 0, this is a **direct, invertible readout of the true pending
count** (multiply by tau_window). This is a fully-observed MDP sneak
path through the "causal" observation, not a POMDP.

### 4.2 Radar schedule is deterministic from step index

`FrozenRuleRadar.service_at_step(step) = step % 2`. Combined with the
observation containing the exact remaining-time channel, the next
radar service is predictable from the observation alone. There is no
hidden radar phase to infer.

### 4.3 Potential-based shaping uses wrong temporal order

Current code computes both `Phi_before` and `Phi_after` **after** the
environment transition completes; between them only `step_idx += 1`
changes. This is not the telescoping form `γ·Φ(s_{t+1}) − Φ(s_t)`. The
theoretical policy-invariance guarantee of potential-based shaping is
therefore not actually established by the implementation.

### 4.4 PPOTrainer overwrites the caller's EnvConfig

`PPOTrainer.__init__` constructs a fresh `EnvConfig(n_envs=...,
horizon=..., device=..., seed=...)`, dropping every other field
(arrival rate, detect thresholds, physics constants, mission
parameters). Train and eval envs can therefore silently diverge.

### 4.5 KL early-stop does not roll back large excursions

The current KL check fires at the end of each PPO epoch, after the
minibatch updates have already been applied. A single minibatch can
move the policy by KL ≈ 1.37 (recorded max in F5); the early-stop
flag is set but the parameter damage is not rolled back. This violates
the trust-region intent.

### 4.6 Energy uses floats; rounding can silently skip an action

Energy is stored as `float32` and decremented by `P_jam_W * dt`. With
non-integer `P_jam_W * dt`, repeated additions can leave a residual
that allows or disallows one extra action depending on floating-point
rounding. The mask is computed from `energy >= cost`, so a 1e-7
negative residual changes the legal-action set.

### 4.7 Missing Gate 4/5 controls and statistics

The prior Gate 4/5 reports lack every required control and statistic:

- no pristine / frozen-init control
- no shuffled-observation control
- no random / untrained control
- no time-only actor control
- no return/drop Spearman correlation
- no action-vs-causal-state dependence check
- no scenario-level paired LCB95 (only macro means reported)
- no per-episode mask / energy / requested-vs-executed violation counts

## 5. Status revision

The prior summary's "F0–F6 all learning gates PASS" claim is **retracted**.
The correct status is:

```text
DEBUG_VERTICAL_SLICE_COMPLETE
PPO_LEARNING_NOT_ESTABLISHED
```

The fast-work line is a usable **engineering vertical slice** (env,
imitation pipeline, PPO pipeline, evaluation harness). It is **not**
evidence that PPO produces a learning contribution on this profile.

## 6. Authorised repair scope

This branch will:

- create four disjoint scenario manifests (DAgger train, PPO train,
  checkpoint validation, locked test) and assert pairwise disjointness
  in code and unit tests;
- split the env into `mdp_sanity_v1` (explicit fully-observed sanity
  profile, **not** for POMDP claims) and `pomdp_v1` (genuine partial
  observation with hidden radar phase and non-invertible activity
  proxy);
- fix potential-based shaping temporal order and add telescoping /
  terminal-potential tests;
- add a real per-mission event ledger with the disposition identity
  `eligible = success + timeout + admission_reject + horizon_failure`;
- fix `PPOTrainer` to preserve the caller's full `EnvConfig`;
- move energy to integer tokens;
- add per-minibatch KL rollback in the PPO update;
- distinguish `iteration = -1` pristine init checkpoints from
  `iteration = 0` post-first-update checkpoints;
- require **scratch PPO** to clear Gate 3 on `mdp_sanity_v1`
  (BC warm-start / residual PPO allowed only as a secondary path);
- run controls (random / shuffled-obs / time-only / no-update) and
  report return/drop Spearman, scenario-level paired LCB95, and
  per-episode violation counts;
- only after R2 PASS, run a corrected two-seed pilot with the locked
  test set frozen and evaluated once;
- only after R3 PASS, attempt the POMDP pilot on `pomdp_v1`.

## 7. Forbidden in this branch

- 8-seed campaign;
- full MFR scale-out;
- MAPPO / two-team integration;
- modifying `main`, `g3-bsta/mfr-lite-fastwork`, `docs/pro9000-forensic-output`,
  or any historical handoff / evidence branch;
- renaming or re-tuning Gate thresholds post-hoc to manufacture PASS;
- substituting BC warm-start for scratch PPO in Gate 3;
- deleting or squashing failed-seed evidence;
- re-introducing quarantined orphan MFR archive bytes.
