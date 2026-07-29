# PRO6000 Fast-Work Execution Prompt

Copy the complete prompt below into the PRO6000 agent session.

---

You are the implementation and experiment owner for a new clean line in:

```text
https://github.com/ExuberantWitness/FluxPhased-
```

Your objective is to make the jammer-learning pipeline produce **real, state-dependent, held-out learning evidence as quickly as possible**, while preserving causal observations, raw-metric integrity, and reproducibility.

## 1. Authority and scope

The user explicitly authorizes you to:

- create a new implementation branch;
- write a fresh G3-BSTA-lite implementation;
- modify verified shared dependencies only through backward-compatible changes with regression tests;
- run bounded CPU/CUDA tests, oracle calculations, supervised imitation, PPO smoke tests, and a two-seed pilot;
- commit and push phase-separated changes and evidence on the new branch.

Create:

```text
g3-bsta/mfr-lite-fastwork
```

Do not modify:

```text
main
twoteam/bc-ppo
docs/g3-bsta-pro6000-handoff
docs/pro6000-forensic-output
agent/g3-bsta-fastwork-guidance-20260729
```

This authorization resolves the previous “no implementation authority” stop condition for the **fresh clean line**. Do not stop again on historical M7 provenance.

However:

- do not execute or directly copy quarantined orphan Python files;
- do not call the new code recovered/original M7;
- do not relabel historical G2'a from FAIL to PASS;
- do not expose credentials or include tokens in logs/artifacts;
- use a new namespace:

  ```text
  env/gpu/g3_bsta_lite/
  algo/_shared/pilot/g3_bsta_lite/
  tests/g3_bsta_lite/
  ```

The first profile is:

```text
DEBUG_ONLY_NOT_A_PHYSICAL_CLAIM
```

Do not wait for RF-owner signoff to run this debug profile. Final publication experiments still require physical binding later.

## 2. Mandatory reading

Read these files completely before editing:

1. `evidence/fastwork_guidance/G3_BSTA_FASTWORK_MODIFICATION_PLAN.md`
2. `evidence/route_b_plus_eligibility/G3_BSTA_CLEAN_SUCCESSOR_SPEC.draft.md`
3. `evidence/route_b_plus_eligibility/SOURCE_CLOSURE_GAP.md`
4. `evidence/route_b_plus_eligibility/FINAL_VERDICT.md`
5. `evidence/PRO6000_SESSION_REPORT_20260729.md`

Source branch:

```text
agent/g3-bsta-fastwork-guidance-20260729
```

Immutable plan anchor:

```text
PLAN_COMMIT: f2047fc45b1ceb01a5b79c800cd681c9c824f24f
PLAN_TREE: c925689afb8f7d9e29ddd3d51604455398285a1f
PLAN_SHA256: 5365715d46e393adceb68651459ca425b363a67e98f3114626bdc5279bd33b74
```

The fast-work plan is the controlling implementation document when it narrows the older clean-successor draft for the debug vertical slice. User instructions override older P0 stop language for this fresh line.

After reading, report:

```text
MANDATED_DOCS_READ: 5/5
FASTWORK_PLAN_SHA256: <sha256>
```

Require `FASTWORK_PLAN_SHA256 == PLAN_SHA256` above. Read the plan from `PLAN_COMMIT`, not only from the mutable branch tip.

## 3. Dominant route

Use exactly one dominant route:

```text
DEBUG_THEN_MINIMAL_CAUSAL_MFR_LITE
```

Do not:

- tune the existing M7 PPO;
- start with the full MFR environment;
- run MAPPO or self-play;
- run an eight-seed campaign;
- change the superiority threshold to manufacture success;
- use hidden state to make the learner pass;
- treat sampled/greedy evaluation as interchangeable.

The required sequence is:

```text
F0 runtime/MDP contracts
→ F1 minimal causal environment
→ F2 baselines + oracle/headroom
→ F3 supervised imitation
→ F4 fixed-scenario PPO overfit
→ F5 one-seed stochastic smoke
→ F6 two-seed pilot
```

Do not skip a gate.

## 4. F0 — Establish exact base and runtime closure

1. Record:

   ```text
   hostname
   whoami
   repo path
   current branch
   exact 40-hex base commit
   tree SHA
   git status --porcelain=v2
   ```

2. Use exactly:

   ```text
   BASE_COMMIT = 80769974cb41fd86e2f80bc2a8992955fb228058
   BASE_TREE = 5db1294431db8315ca2f75e59ca8ef2ba45bb66b
   ```

3. A GitHub network failure does not block work if this exact commit and tree are already available locally. Do not substitute a different commit. If the exact object is absent and cannot be obtained, report `BLOCK_BASE_OBJECT_ABSENT`.
4. Create `g3-bsta/mfr-lite-fastwork`.
5. Verify the exact interfaces of:

   ```text
   env/gpu/twoteam/iq_interference.py
   env/gpu/twoteam/detection.py
   env/gpu/twoteam/tracker.py
   ```

6. If the shared IQ implementation is hard-coded to four nodes or lacks explicit jammer power:

   - either extend it backward-compatibly to dynamic node shapes and explicit per-node power;
   - or create an isolated G3-BSTA-lite adapter.

7. Add a legacy four-node bit-regression test before altering shared semantics.
8. Remove host-specific `sys.path` insertion from all new code.

F0 pass:

- clean imports;
- CPU shape smoke passes;
- CUDA shape smoke passes when `torch.cuda.is_available()`;
- otherwise CUDA is `NOT_RUN/ENV_UNAVAILABLE`, which does not block the CPU-capable F0-F3 path;
- old twoteam regression remains unchanged;
- no orphan file is imported or executed.

If F0 fails, fix only runtime/API closure. Do not start environment design or PPO.

## 5. F1 — Implement minimal causal MFR-lite

Freeze this debug config:

```text
n_jammers = 1
n_services = 2
service_semantics = two frequency/receiver channels
horizon = 64
active_budget_steps = 16
duty_budget = 0.25
jammer_power = one fixed debug value
pool_jammer_fraction = 0
radar_opponent = frozen rule radar
self_play = false
```

Action:

```text
0 = idle
1 = jam_service_0
2 = jam_service_1
```

Use one masked categorical distribution, not independent Bernoulli outputs.

Resource dynamics:

```text
E[t+1] = E[t]
         - 1(executed_action != idle) * P_fixed * dt
E[t] >= 0
```

Required:

- idle always legal;
- both service actions remain legal whenever own energy and public hardware availability permit them;
- the mask must not reveal current channel activity or hidden value;
- at most one service active;
- legal `requested_action == executed_action`;
- illegal actions produce an explicit contract violation, never silent substitution;
- always-on is infeasible by construction;
- every step logs mask, requested/executed action, selected service, energy before/after.

### Decision ordering

The actor must act on the state it observed.

The two actions address stable physical `service_id/channel_id`, never task UIDs. At observation construction, log:

```text
observation_state_version
decision_mask
```

`step(action)` must execute the selected service action against that state before applying the next exogenous arrival/transition. Add a test that changes the next arrival table and proves it cannot change the meaning of the current service action.

### Reset and RNG

Implement:

```text
reset(seed=None, reset_metrics=False)
```

- explicit seed reproduces;
- ordinary reset advances episode seed;
- jammer state, energy, cooldowns, counters, and step state reset correctly;
- environment-event RNG, detector RNG, and policy-action RNG are separate;
- stochastic action sampling cannot advance detector randomness;
- paired policies consume the same pre-generated exogenous event table.

### Causal observation

Actor may see only:

```text
remaining_energy / initial_energy
remaining_time / horizon
delayed/noisy service activity
delayed urgency proxy
intercept confidence and age
previous executed action
```

Actor may not see:

```text
exact queue length
exact progress/deadline
true target slot/id
future arrivals
post-action detector outcome
next radar action
environment RNG state
```

The action mask must be derived only from actor-visible history and the jammer's own resource state.
In this two-service debug profile, hidden service activity is never a mask input. The policy must infer value from delayed/noisy observation.

### Physical effect

Use this path:

```text
executed service
→ frequency/power
→ overlap/path/receiver gain
→ service-specific JNR/SINR
→ detection or measurement quality
→ mission task outcome
```

Do not directly write hidden task progress in the primary mode.

Counterfactual tests must show:

- selected-service degradation is monotone over a calibrated dose range;
- selected service is affected;
- unselected service is not equally affected;
- antijam changes the JNR consumed downstream;
- telemetry and task outcomes agree.

### Metrics and reward

Maintain per-environment, per-task counters.

Separate:

```text
mission arrivals/drops
EW cue arrivals/drops
admission rejection
timeout
unfinished-at-horizon
```

Freeze:

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

`eligible_mission_arrivals` is the pre-generated, policy-independent exogenous set admitted by a frozen eligibility rule before any jammer action.

Freeze the accounting identity:

```text
eligible_mission_arrivals
= mission_success
  + mission_timeout
  + mission_admission_reject
  + mission_horizon_failure
```

Pre-filter the locked manifest so included scenarios have at least one eligible arrival. Mark zero-denominator cases `NA`, exclude them before any policy executes, and report their count. The primary aggregation is the macro mean of scenario-level ratios; pooled ratio is secondary.

Training reward:

```text
r'[t] =
  newly_dropped_mission_tasks[t]
  + gamma * Phi(causal_belief[t+1])
  - Phi(causal_belief[t])
```

Terminal potential is zero. Log raw and shaping reward separately. Do not use the negative radar-shaped reward.

## 6. F1 contract tests

Create:

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

Gate 0 requires 100% pass:

- no NaN/Inf;
- no invalid mask;
- no requested/executed mismatch;
- energy never negative;
- same seed/state/action reproduces;
- env-0 perturbation cannot affect env-1;
- hidden-truth perturbation cannot change actor observation;
- event rows exactly recompute aggregate metrics;
- before optimizer update, `max_abs(recomputed_logp - stored_old_logp) < 1e-6` using identical observation, action, mask, and parameters;
- pre-update PPO ratio differs from one by less than `1e-6`;
- hand-computed GAE terminal and time-limit examples pass.

No training before Gate 0 passes.

Commit after Gate 0:

```text
docs(g3-bsta-lite): freeze debug contracts
test(g3-bsta-lite): add runtime and MDP invariants
feat(g3-bsta-lite): implement causal budgeted environment
```

## 7. F2 — Baselines and oracle/headroom

Implement and freeze:

```text
always_off
random_feasible
budgeted_barrage
budgeted_round_robin
periodic_blink
causal_reactive_or_edf
```

All use the same action feasibility and resource path.

Keep these concepts separate:

- exact DP or an executable clairvoyant policy on a reduced `H=32/64` problem, used to establish reachable headroom;
- a certified/admissible upper bound, used only as a diagnostic upper bound and never as an executable policy return;
- a same-observation executable causal planner/witness.

Evaluate at least 128 locked paired scenarios. Average each stochastic policy over a frozen number of action-RNG replicates within scenario, then compute paired scenario deltas against every frozen baseline.

Use:

```text
LCB95 = mean(delta_s)
        - t_(0.95, S-1) * sd(delta_s) / sqrt(S)
```

Freeze baseline family selection and tie-breaking on planner-development scenarios. Confirmation must pass against every frozen baseline; do not select a new best baseline on confirmation data.

```text
delta = mission_drop_ratio(causal_witness)
        - mission_drop_ratio(best_frozen_baseline)
```

Gate 1 pass:

```text
exact/executable-oracle mean gap >= 10 pp
causal witness LCB95(delta) > 7.5 pp
neighboring energy/detector cells LCB95(delta) > 5 pp
```

Failure routes:

- executable-oracle gap `<5 pp`: redesign only action/resource/physical selectivity; PPO remains forbidden;
- executable-oracle gap `[5 pp, 10 pp)`: Gate 1 still fails; adjust only debug action/resource/selectivity and rerun;
- oracle has gap but causal witness does not: repair causal observation/history;
- causal witness passes: freeze baseline and planner configs, then continue.

Produce:

```text
BASELINE_FREEZE.json
ORACLE_HEADROOM_REPORT.json
ORACLE_HEADROOM_REPORT.md
paired raw rows
scenario manifest
```

## 8. F3 — Supervised imitation

Generate approximately:

```text
10,000 planner-development samples
2,000 held-out validation samples
```

Labels may use only causal-witness information.

Gate 2 pass:

```text
mask-valid actions = 100%
tie-aware top-1 accuracy >= 90%
normalized oracle regret <= 10%
held-out rollout recovers >= 90% witness-vs-random gap
```

If imitation fails, fix representation, normalization, history, slot alignment, or model capacity. Do not add hidden truth and do not start PPO.

## 9. F4 — PPO implementation and fixed-scenario overfit

Initial PPO:

```text
actor = 2x128 Tanh
distribution = masked categorical
lr = 3e-4
gamma = 0.99
GAE lambda = 0.95
clip = 0.2
grad clip = 0.5 independently for actor and critic
```

Do not perform HPO.

Required:

- save rollout masks;
- use the same masks during update;
- separate actor and critic optimizer/clipping;
- actor and critic use the same causal observation in this debug profile;
- any later asymmetric/privileged critic is a separately registered experiment variable;
- correct terminal and time-limit bootstrap;
- separate action RNG;
- sampled stochastic evaluation is the primary preregistered protocol;
- use multiple action replicates;
- deterministic argmax is secondary only;
- log KL, clip fraction, advantage standard deviation, explained variance, gradient norms, action frequencies, energy, raw drop, reward components, and return/drop correlation.

Run:

```text
H = 64
vector_envs = 16
E0 = 16 * P_fixed * dt
maximum initial budget ≈ 100k transitions
scratch PPO
BC-warm-start PPO
```

Gate 3 pass:

```text
adv_std > 1e-3 during effective updates
KL > 0.05 triggers early stop
clip fraction not persistently > 0.5
scratch PPO recovers >= 80% causal-witness headroom
sampled evaluation paired mean beats every frozen baseline >= 5 pp
```

Gate 3 is a debugging gate and routes on the frozen paired mean point estimate; report uncertainty but make no inferential claim.

Failure routes:

- BC fails: observation/action/model defect;
- BC succeeds but warm-start PPO degrades: inspect GAE, reward sign, done, old log-prob, mask replay, optimizer;
- warm-start succeeds and scratch fails: exploration/sparsity issue; use causal curriculum or valid potential shaping, not broad HPO.

## 10. F5 — One-seed stochastic smoke

Budget:

```text
0.2–0.5 million transitions
```

Gate 4 pass:

```text
held-out paired mean advantage over every frozen baseline >= 5 pp
scenario-level LCB95 against every baseline > 0
recover >= 60% causal-witness headroom
trained - frozen-init/random/shuffled-observation >= 3 pp
Spearman(return, raw mission drop) >= 0.8
state-dependent action test passes
all violation counts = 0
```

If return rises but raw drop does not, freeze PPO and repair reward. If train passes but held-out fails, widen only the scenario curriculum.

## 11. F6 — Two-seed pilot

Use two independent training seeds and the same locked paired evaluation suite. Do not claim statistical significance from two seeds.

Go:

```text
each seed's paired scenario mean beats every frozen baseline >= 5 pp
mean advantage >= 7.5 pp
both recover >= 60% causal-witness headroom
both beat frozen-init and shuffled-observation >= 5 pp
constraint/mask/requested-executed violations = 0
```

If one seed passes and one fails, allow one evidence-based correction only, then use two new seeds.

Do not compute or claim significance across two training seeds. Report each seed's paired scenario distribution and the arithmetic mean across the two seeds.

If both pass, stop the fast-work phase and propose the controlled expansion plan. Do not automatically launch the eight-seed full experiment.

## 12. Commit sequence

Use bounded commits:

```text
docs(g3-bsta-lite): freeze debug contracts
test(g3-bsta-lite): add runtime and MDP invariants
feat(g3-bsta-lite): implement causal budgeted environment
feat(g3-bsta-lite): add frozen baselines and oracle
feat(g3-bsta-lite): add imitation and masked PPO
exp(g3-bsta-lite): record smoke and pilot evidence
```

Push only the new implementation branch. Never force-push existing branches.

Before every commit and push, scan the staged diff, generated logs/configs, artifact manifests, and archive text payloads for tokens, private keys, passwords, authorization headers, and credential-bearing URLs. If a candidate secret is found:

- stop the push;
- report only the affected path and secret class;
- never echo the secret value.

## 13. Required phase response

After every phase, respond exactly with:

```text
phase:
status:
branch:
base_commit:
git_commit:
tree_sha:
changed_files:
commands:
tests:
  passed:
  failed:
artifacts:
metrics:
invariants:
  requested_equals_executed:
  energy_violations:
  mask_violations:
  vector_isolation:
  causal_observation:
claims_made:
constraint_violations:
stop_reason:
next_authorized_phase:
```

For long-running work, continue autonomously from one passed phase to the next. Stop only when:

- the current phase's explicit gate fails and its prescribed repair route has been exhausted;
- required local dependencies are physically absent and no clean adapter can be written;
- continuing would require modifying a protected branch;
- two-seed pilot passes and the fast-work objective is complete.

Do not stop because historical M7 provenance is unavailable. This is a newly authorized clean implementation line.

Begin with mandatory reading and F0. Do not begin with PPO.

---
