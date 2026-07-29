# G3-BSTA Fast-Work Modification Plan

**Date:** 2026-07-29

**Status:** implementation guidance

**Dominant route:** `DEBUG_THEN_MINIMAL_CAUSAL_MFR_LITE`

**Evidence base:** `docs/pro6000-forensic-output` at `aad49e6a9f5cc6158656c6207c59623fee0980d1`

**Implementation base commit:** `80769974cb41fd86e2f80bc2a8992955fb228058`

**Implementation base tree:** `5db1294431db8315ca2f75e59ca8ef2ba45bb66b`

**Historical G2'a status:** remains `FAIL`; this plan does not relabel it

## 1. Executive decision

Do not spend more compute tuning the current M7 PPO. The fastest credible route is:

```text
runtime/MDP contract repair
→ minimal causal MFR-lite
→ oracle/headroom
→ supervised imitation
→ single-scenario PPO overfit
→ one-seed smoke
→ two-seed pilot
→ only then full G3-BSTA expansion
```

No stochastic RL method can be guaranteed to converge. This workflow instead guarantees fail-fast localization: a failed gate identifies whether the active defect is in the runtime/API closure, transition semantics, action/observation contract, physical response, reward, optimizer implementation, or generalization.

## 2. Scope and authorization interpretation

This document defines a **fresh implementation route**. It does not promote the quarantined orphan Python files into authoritative source code and does not authorize direct copying of those bytes.

Implementation should use a new namespace:

```text
env/gpu/g3_bsta_lite/
algo/_shared/pilot/g3_bsta_lite/
tests/g3_bsta_lite/
```

The quarantined package may be used only as static defect evidence. The clean implementation must be written from this specification and the currently verified repository dependencies.

The first MFR-lite profile is explicitly:

```text
DEBUG_ONLY_NOT_A_PHYSICAL_CLAIM
```

Its purpose is to establish a valid learning pipeline quickly. Final publication claims still require physical parameter binding, robust operating-point sweeps, frozen baselines, and independent training seeds.

## 3. Candidate ranking

| Candidate | Time to real learning evidence | Main risk | Decision |
|---|---:|---|---|
| Direct PPO tuning on current M7 | poor | all-on dominance and broken semantics can create false success | reject |
| Change evaluation to sampled and accept parity | very fast | cannot satisfy superiority; only a diagnostic | retain as historical diagnostic |
| Minimal causal MFR-lite | fastest credible | bounded implementation work, each layer testable | promote |
| Full clean successor immediately | slower | too many simultaneous failure surfaces | defer until MFR-lite passes |

The promotion cap is one line: **minimal causal MFR-lite**.

## 4. Static evidence: why the current line must not be tuned

The following findings come from read-only static inspection of the quarantined package and comparison with the currently visible repository dependency interfaces. They must be independently rechecked against the executor's exact base commit.

### 4.1 Runtime and API closure

1. `mfr_env.py:313-318` calls `IqInterference.compute_jnr_matrix(..., P_tx_override_W=...)`.
2. The currently visible `env/gpu/twoteam/iq_interference.py` interface has no such argument.
3. The visible IQ implementation hard-codes four nodes and reshapes measurement state to `2×2`, while the orphan MFR path constructs `K + J + N` nodes.
4. `mfr_env.py` contains a host-specific `sys.path` insertion.

Consequence: the implementation cannot be presumed runnable or semantically compatible on a different base merely because its AST parses.

### 4.2 Transition and action identity

`mfr_env.py:491-516` advances the target pool, creates arrivals, expires tasks, and rebuilds the exposed queue before resolving the task-slot action sampled from the previous observation.

Consequence: action slot `i` may execute a different task from the task shown in slot `i` to the actor.

The new MFR-lite action does not address task slots or task UIDs. It addresses two stable physical service/channel IDs. `step(action)` must execute that service action against the state represented by the current observation before applying the next exogenous transition. Log an observation/state version and decision mask so this order is testable.

### 4.3 Allocation and physical-effect defects

1. `mfr_env.py:573` scatters allocation on the subarray dimension rather than the allocation-mode dimension.
2. Pool jammer nodes are configured with emission disabled, so their power override may never enter the active-interferer path.
3. Antijam modifies the returned `jnr_at_sub` but does not consistently modify the JNR consumed by detection/tracking.
4. A worst-subarray measurement sigma is broadcast to every target.
5. The legacy sigma-progress rule uses a saturating floor:

   ```text
   clamp(1 / sqrt(1 + JNR), 0.1, 1.0)
   ```

   This makes many different actions outcome-equivalent after saturation.

Consequence: actions can appear valid in telemetry while having no correct or selective downstream effect.

### 4.4 Reset and randomness defects

1. `MfrVecEnv.reset()` reseeds the environment with the same seed on every call.
2. Jammer state is not fully reset with the episode.
3. Evaluation's seed offset is overwritten by the following reset.
4. Detector random draws and stochastic policy actions use global Torch randomness.

Consequence: training iterations replay the same exogenous trace; episode state can leak across resets; and paired evaluation can be confounded because changing policy actions changes the detector RNG stream.

### 4.5 Observation, vector isolation, and metrics

1. Exact queue depth and exact average progress enter jammer observation.
2. True slot IDs and true jammer state appear in other observation paths.
3. `n_completed.sum()` aggregates all vector environments and is broadcast back to every environment.
4. Counters are not maintained per environment.
5. Queue-overflow arrivals enter the denominator without a corresponding rejection/drop event.
6. EW-generated cue tasks are mixed into the main arrival/drop accounting.

Consequence: the policy is not demonstrably causal, vector environments are not statistically isolated, and `drop_ratio` is not a stable primary estimand.

### 4.6 Objective and optimizer defects

1. Jammer reward is the negative of radar shaped reward.
2. Radar reward includes idle and infeasible-action penalties that are not the paper's raw drop objective.
3. Independent Bernoulli actions have no energy or concurrency constraint; all-on is a dominating action.
4. Actor and critic share optimizer-level gradient clipping, so a large value gradient can shrink the actor update.
5. Training uses stochastic Bernoulli actions while the final path thresholds logits greedily at 0.5.

Consequence: current training can match an always-on-equivalent distribution without learning a state-dependent policy, while greedy evaluation reports an unrelated degradation.

## 5. Minimal causal MFR-lite contract

### 5.1 Environment profile

```text
n_jammers: 1
n_services: 2
service_semantics: two physically addressable receiver/frequency channels
horizon: 64 decision steps
active_budget_steps: 16
duty_budget: 25%
jammer_power: fixed within the debug profile
radar_opponent: one frozen rule policy
pool_jammer_fraction: 0
self_play: disabled
```

The two services must correspond to an explicit frequency-overlap or receiver-selection path. They must not be disguised true task/target IDs.

### 5.2 Action contract

```text
0 = idle
1 = jam_service_0
2 = jam_service_1
```

The policy distribution is one masked categorical distribution.

Required invariants:

- idle is always legal;
- both physical service actions remain legal whenever own energy and public hardware availability permit them;
- the mask must not reveal whether a service is currently active or valuable;
- at most one service is jammed per step;
- fixed-power energy is deducted from the executed action;
- legal actions satisfy `requested_action == executed_action`;
- no silent replacement of an illegal action;
- every transition logs requested action, executed action, mask, energy before/after, and selected service.

### 5.3 Resource dynamics

```text
E[t+1] = E[t] - 1(executed_action != idle) * P_fixed * dt
E[t] >= 0
```

The debug profile must make always-on infeasible:

```text
E[0] < P_fixed * dt * horizon
```

Use a hard resource constraint, not a learner-only activation penalty.

### 5.4 Causal actor observation

Allowed:

- remaining energy / initial energy;
- remaining time / horizon;
- delayed and noisy service activity;
- delayed urgency proxy;
- intercept confidence and age;
- previous executed action.

Forbidden:

- exact internal queue length;
- exact task progress or exact deadline;
- true target slot/ID;
- future arrivals;
- post-action detector outcome;
- next rule-radar action;
- environment RNG state.

The action mask must be a deterministic function of actor-visible observation/history and own resource state.
For the two-service debug profile, service activity is not a legality condition: the policy must infer which service is worth jamming from delayed/noisy observation.

### 5.5 Physical response

The primary response chain must be:

```text
executed service action
→ jammer power/frequency
→ frequency overlap and path/receiver gain
→ service-specific receiver JNR/SINR
→ detector or measurement quality
→ mission task outcome
```

Required counterfactual properties:

- increasing jammer dose on a selected service cannot improve that service in the calibrated monotonic regime;
- the selected service is measurably affected;
- the unselected service is not damaged by the same amount;
- antijam changes the JNR actually consumed by detection/tracking, not telemetry only;
- no direct write to hidden task progress is allowed in the primary mode.

### 5.6 Primary metric and reward

Separate mission tasks from EW-defense cue tasks.

```text
mission_drop_ratio =
(
  mission_timeout
  + mission_admission_reject
  + preregistered_horizon_failures
)
/
eligible_mission_arrivals
```

`eligible_mission_arrivals` is the pre-generated, policy-independent set of exogenous mission arrivals that pass a frozen eligibility rule before any jammer action is applied. The event accounting identity is:

```text
eligible_mission_arrivals
= mission_success
  + mission_timeout
  + mission_admission_reject
  + mission_horizon_failure
```

Construct evaluation manifests so every included scenario has at least one eligible arrival. A zero-denominator scenario is marked `NA` and excluded by the frozen manifest before any policy is run; report the excluded count. The primary aggregation is the macro mean of scenario-level drop ratios. Report the pooled numerator/denominator ratio as secondary only.

Training reward:

```text
r'[t] =
  newly_dropped_mission_tasks[t]
  + gamma * Phi(causal_belief[t+1])
  - Phi(causal_belief[t])
```

Terminal potential must be zero. Energy remains a hard constraint. Log raw reward and shaping components separately.

## 6. Implementation work packages

### W0 — Base and dependency closure

1. Record hostname, worktree, branch, exact 40-hex base commit, and tree SHA.
2. Use exactly commit `80769974cb41fd86e2f80bc2a8992955fb228058`, tree `5db1294431db8315ca2f75e59ca8ef2ba45bb66b`. Do not substitute another local commit.
3. Create `g3-bsta/mfr-lite-fastwork`; do not modify `main`, `twoteam/bc-ppo`, or evidence branches.
4. Verify the IQ, detection, and tracker interfaces.
5. Implement either:

   - a backward-compatible dynamic-node extension to the shared IQ kernel, with legacy bit-regression; or
   - an isolated G3-BSTA-lite IQ adapter.

6. Remove all host-specific import paths.

Exit criterion: clean import plus CPU shape smoke tests. CUDA shape smoke is mandatory only when `torch.cuda.is_available()`; otherwise record `NOT_RUN/ENV_UNAVAILABLE` and continue the CPU-capable F0-F3 path.

### W1 — Test-first contracts

Create at least:

```text
tests/g3_bsta_lite/test_runtime_contract.py
tests/g3_bsta_lite/test_transition_order.py
tests/g3_bsta_lite/test_vector_isolation.py
tests/g3_bsta_lite/test_resource_contract.py
tests/g3_bsta_lite/test_causal_observation.py
tests/g3_bsta_lite/test_counterfactual_physics.py
tests/g3_bsta_lite/test_metric_accounting.py
tests/g3_bsta_lite/test_ppo_math.py
```

All tests must fail for the intended reason before the corresponding implementation and pass afterward.

### W2 — Environment vertical slice

Implement:

- one frozen rule-radar opponent;
- two addressable frequency/receiver services;
- one budgeted jammer;
- stable service IDs and action-before-next-exogenous-transition ordering;
- separate environment, detector, and action RNG streams;
- per-environment counters;
- causal observation;
- raw mission accounting;
- event log sufficient to recompute every aggregate metric.

### W3 — Baselines and headroom

Implement and freeze:

- always-off;
- random-feasible;
- budgeted barrage;
- budgeted round-robin;
- periodic blink;
- causal reactive/EDF.

On a reduced `H=32/64` state, keep three concepts separate:

- an exact DP or executable clairvoyant policy, used to demonstrate reachable headroom;
- a certified/admissible upper bound, used only as an upper-bound diagnostic and never reported as an executable policy return;
- a same-observation executable causal planner/witness, used for the learnability gate.

Use paired, pre-generated exogenous scenarios for every policy.

### W4 — Supervised imitation

Generate approximately:

```text
10,000 planner-development examples
2,000 held-out validation examples
```

Labels may use only actions available to the causal witness. The actor must first demonstrate that the observation/action representation can express the witness.

### W5 — PPO

Use:

```text
actor: 2 x 128 Tanh
distribution: masked categorical
lr: 3e-4
gamma: 0.99
GAE lambda: 0.95
clip: 0.2
grad clip: 0.5, actor and critic separately
```

Do not start hyperparameter search.

Required trainer behavior:

- save the exact rollout mask;
- recompute log probability with the same mask;
- verify pre-update probability ratio is one;
- unit-test terminal and time-limit bootstrapping;
- use separate actor and critic optimizers/clipping;
- actor and critic use the same causal observation in the debug profile; any future asymmetric/privileged critic is a separately registered experiment variable;
- log KL, clip fraction, advantage standard deviation, explained variance, gradient norms, action frequencies, energy usage, and return/drop correlation;
- pre-register sampled stochastic evaluation as primary;
- use a separate action RNG and multiple action replicates;
- report deterministic argmax only as a secondary diagnostic.

### W6 — Controlled expansion

Only after the two-seed pilot passes:

1. expand the scenario distribution;
2. add physically bound power/energy values;
3. add additional service-selectivity mechanisms;
4. add more emitters only when the single-emitter contract remains verified;
5. run robust headroom cells;
6. perform power analysis;
7. run at least eight independent training seeds.

## 7. Fail-fast gates

### Gate 0 — Contract gate

Required:

- all contract tests pass;
- no NaN/Inf;
- no mask violation;
- no requested/executed mismatch;
- energy never negative;
- same explicit seed/state/action reproduces the same transition;
- changing env 0 cannot alter env 1;
- hidden-truth perturbations do not change actor observation;
- event rows exactly reproduce aggregate metrics;
- before any optimizer update, recomputing log probability with identical observation, action, mask, and parameters differs from stored old log probability by less than `1e-6`;
- pre-update PPO ratio differs from one by less than `1e-6`.

Any failure blocks training.

### Gate 1 — Reachability/headroom

Evaluate at least 128 locked paired scenarios. For every stochastic policy, first average its raw scenario metric over a frozen number of independent action-RNG replicates. Then form a scenario-level paired delta against each frozen baseline.

Use a one-sided 95% Student-t lower confidence bound over scenario-level paired deltas:

```text
LCB95 = mean(delta_s) - t_(0.95, S-1) * sd(delta_s) / sqrt(S)
```

Baseline-family configuration and tie-breaking are frozen on planner-development scenarios. Confirmation requires the inequality against **every** frozen baseline (intersection-union rule); do not select a new “best” baseline on confirmation data.

```text
delta = mission_drop_ratio(causal_witness)
        - mission_drop_ratio(best_frozen_baseline)
```

Pass:

- exact DP or executable clairvoyant-policy mean gap at least `10 pp`;
- causal witness `LCB95(delta) > 7.5 pp`;
- neighboring energy and detector settings `LCB95(delta) > 5 pp`.

Routes:

- executable-oracle gap below `5 pp`: redesign action/resource/physical mechanism; do not tune PPO;
- executable-oracle gap in `[5 pp, 10 pp)`: Gate 1 still fails; adjust only the debug profile's action, resource, or physical-selectivity mechanism and rerun;
- oracle has gap but causal witness does not: repair causal information/history;
- causal witness passes: authorize imitation and PPO.

### Gate 2 — Imitation

Pass:

- mask-valid actions: 100%;
- tie-aware top-1 accuracy: at least 90%;
- normalized oracle regret: at most 10%;
- held-out rollouts recover at least 90% of witness-versus-random gap.

Failure indicates observation/action encoding or model-expression defects.

### Gate 3 — Fixed-scenario PPO overfit

Use `H=64`, 16 vector environments, and at most roughly 100k transitions initially. Set initial energy explicitly as `E0 = 16 * P_fixed * dt`. Run scratch PPO and BC-warm-start PPO.

Pass:

- `adv_std > 1e-3` during effective updates;
- no KL excursion above 0.05 without early stop;
- clip fraction is not persistently above 0.5;
- scratch PPO recovers at least 80% of causal-witness headroom;
- primary sampled evaluation's paired mean point estimate beats every frozen baseline by at least `5 pp` on the fixed debug suite.

Gate 3 is a debugging gate, not an inferential claim; report paired scenario rows and uncertainty but route on the frozen paired mean point estimate.

Routes:

- BC fails: observation/action/model defect;
- BC succeeds but warm-start PPO degrades: GAE, reward sign, done, old log-prob, mask replay, or optimizer defect;
- warm-start succeeds but scratch fails: exploration/sparsity issue; use causal curriculum or valid potential shaping before any broad HPO.

### Gate 4 — One-seed stochastic smoke

Budget: approximately 0.2–0.5 million transitions.

Pass:

- held-out paired mean advantage over every frozen baseline at least `5 pp`, with scenario-level `LCB95 > 0`;
- recover at least 60% of causal-witness headroom;
- trained exceeds frozen-init, random, and shuffled-observation by at least `3 pp`;
- Spearman correlation between episodic return and raw mission drop at least 0.8;
- action depends on causal state rather than collapsing to fixed duty or fixed service.

### Gate 5 — Two-seed pilot

Use two independent training seeds and the same locked paired evaluation suite. Do not make significance claims from two seeds.

Go:

- each seed's paired scenario mean beats every frozen baseline by at least `5 pp`;
- mean advantage at least `7.5 pp`;
- both recover at least 60% of causal-witness headroom;
- both beat frozen-init and shuffled-observation by at least `5 pp`;
- all constraint, mask, and requested/executed violation counts are zero.

If one seed passes and one fails, permit only one evidence-based correction, then rerun with two new training seeds.

Two training seeds are a go/no-go pilot only. Do not compute or claim a training-seed significance test. Report per-seed paired scenario distributions and the two-seed arithmetic mean.

## 8. Explicitly forbidden before Gate 1 passes

- PPO hyperparameter sweep;
- full eight-seed campaign;
- MAPPO or self-play;
- redefining G2'a to manufacture a pass;
- post-hoc switching between sampled and greedy evaluation;
- direct adoption or execution of quarantined source files;
- adding hidden truth to make imitation or PPO pass;
- claiming the debug profile is a physically validated benchmark.

## 9. Commit and report discipline

Recommended commit sequence:

1. `docs(g3-bsta-lite): freeze debug contracts`
2. `test(g3-bsta-lite): add runtime and MDP invariants`
3. `feat(g3-bsta-lite): implement causal budgeted environment`
4. `feat(g3-bsta-lite): add frozen baselines and oracle`
5. `feat(g3-bsta-lite): add imitation and masked PPO`
6. `exp(g3-bsta-lite): record smoke and pilot evidence`

Every phase report must include:

```text
phase
status
branch
commit_sha
tree_sha
changed_files
commands
tests_passed
tests_failed
artifacts
metrics
invariants
stop_reason
next_authorized_phase
```

Raw event rows, configs, seed manifests, and environment hashes must be committed or stored as immutable artifacts with SHA-256.

Before every commit and push, scan:

```text
staged diff
tracked and newly generated text logs/configs
artifact manifests
archive member names and text payloads
```

for access tokens, private keys, passwords, authorization headers, and credential-bearing URLs. If any candidate secret is found, stop the push, report only the affected path and secret class, and never echo the secret value.

## 10. Publication path

The debug profile is not the final contribution. After controlled expansion, the defensible contribution is:

> A resource-constrained, causal-observation, service-selective MFR interference benchmark with oracle-first reachability tests and learning-verification controls.

The contribution is not “novel PPO.” Historical G2'a remains a documented negative result. A new G3-BSTA result must be reported under a new benchmark name and its own frozen evaluation protocol.

## 11. Primary methodological references

- Schulman et al., *Proximal Policy Optimization Algorithms*: https://arxiv.org/abs/1707.06347
- Huang and Ontañón, *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*: https://arxiv.org/abs/2006.14171
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games*: https://arxiv.org/abs/2103.01955
- Ng, Harada, and Russell, *Policy Invariance Under Reward Transformations*: https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html
