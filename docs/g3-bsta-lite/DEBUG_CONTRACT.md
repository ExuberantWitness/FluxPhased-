# G3-BSTA-lite Debug Contract (Frozen)

**Branch**: `g3-bsta/mfr-lite-fastwork`
**Base commit**: `80769974cb41fd86e2f80bc2a8992955fb228058`
**Base tree**: `5db1294431db8315ca2f75e59ca8ef2ba45bb66b`
**Status**: FROZEN for the F0..F6 fast-work gates
**Profile tag**: `DEBUG_ONLY_NOT_A_PHYSICAL_CLAIM`
**Controlling plan**: `evidence/fastwork_guidance/G3_BSTA_FASTWORK_MODIFICATION_PLAN.md` (SHA-256 `5365715d46e393adceb68651459ca425b363a67e98f3114626bdc5279bd33b74`)
**Dominant route**: `DEBUG_THEN_MINIMAL_CAUSAL_MFR_LITE`

## 1. Invariants (do not deviate)

1. No quarantined orphan Python file is imported, copied or executed by any
   module in `env/gpu/g3_bsta_lite/`, `algo/_shared/pilot/g3_bsta_lite/`, or
   `tests/g3_bsta_lite/`.
2. The shared `env/gpu/twoteam/{iq_interference,detection,tracker}.py`
   modules are not modified in this line. An isolated IQ adapter lives under
   `env/gpu/g3_bsta_lite/` instead (see §6).
3. No host-specific `sys.path` insertion in any new module. Imports use the
   project's package root only.
4. Historical G2'a remains `FAIL`. Nothing in this line relabels it.
5. The debug profile below is not a physically validated benchmark claim.
   Final publication experiments require physical binding, robust operating-
   point sweeps, frozen baselines, and independent training seeds (W6).

## 2. Debug environment profile (FROZEN for F1..F5)

```text
n_jammers: 1
n_services: 2
service_semantics: two physically addressable receiver/frequency channels
horizon: 64 decision steps
active_budget_steps: 16
duty_budget: 0.25
jammer_power: one fixed debug value (calibrated in F1; hard constraint E0 < P_fixed*dt*horizon)
pool_jammer_fraction: 0
radar_opponent: one frozen rule radar
self_play: false
```

The two services correspond to explicit frequency-overlap or receiver-
selection paths. They are not disguised task UIDs or true target slots.

## 3. Action contract

```text
0 = idle (always legal)
1 = jam_service_0
2 = jam_service_1
```

- One masked categorical distribution over `{0, 1, 2}`, not independent
  Bernoullis.
- `idle` is always legal.
- Service actions remain legal whenever own energy and public hardware
  availability permit them; the mask must not reveal whether a service is
  currently active or valuable.
- At most one service is jammed per step.
- Fixed power is deducted only on executed non-idle action.
- `requested_action == executed_action` for legal actions; illegal actions
  are an explicit contract violation and never silently substituted.
- Always-on is infeasible by construction: `E0 < P_fixed * dt * horizon`.
- Every transition logs: mask, requested action, executed action, selected
  service, energy before, energy after, observation_state_version.

### 3.1 Decision ordering

`step(action)` applies the selected service action against the state
represented by the current observation **before** applying the next exogenous
arrival/transition. Slot identity is the stable physical `service_id`, never
a task UID. A test must prove that changing the next-arrival table cannot
change the meaning of the current service action.

## 4. Resource dynamics

```text
E[t+1] = E[t] - 1(executed_action != idle) * P_fixed * dt
E[t]   >= 0
```

Hard resource constraint, not a learner-only activation penalty.

## 5. Reset and RNG

```text
reset(seed=None, reset_metrics=False)
```

- Explicit seed reproduces the trajectory bit-for-bit.
- Ordinary reset advances the episode seed.
- Jammer state, energy, cooldowns, counters and step state all reset.
- Three RNG streams: environment-event RNG, detector RNG, action RNG.
- Stochastic policy action sampling cannot advance the detector RNG.
- Paired policies consume the same pre-generated exogenous event table.

## 6. Causal actor observation

Allowed (during the F1..F5 debug profile):

- `remaining_energy / initial_energy`
- `remaining_time / horizon`
- delayed and noisy service activity
- delayed urgency proxy
- intercept confidence and age
- previous executed action

Forbidden:

- exact internal queue length
- exact task progress or exact deadline
- true target slot/ID
- future arrivals
- post-action detector outcome
- next rule-radar action
- environment RNG state

The action mask is a deterministic function of actor-visible
observation/history and own resource state. For the two-service debug
profile, service activity is not a mask input: the policy must infer value
from delayed/noisy observation.

## 7. Physical response

```text
executed service action
  -> jammer power/frequency
  -> frequency overlap and path/receiver gain
  -> service-specific receiver JNR/SINR
  -> detector or measurement quality
  -> mission task outcome
```

Counterfactual properties that must hold in the calibrated regime:

- increasing jammer dose on a selected service cannot improve that service;
- the selected service is measurably affected;
- the unselected service is not equally affected;
- antijam changes the JNR consumed by detection/tracking, not telemetry only;
- no direct write to hidden task progress in the primary mode.

## 8. Primary metric (raw)

```text
mission_drop_ratio =
  (mission_timeout + mission_admission_reject + preregistered_horizon_failures)
  / eligible_mission_arrivals
```

`eligible_mission_arrivals` is the pre-generated, policy-independent exogenous
set admitted by a frozen eligibility rule before any jammer action is applied.
Accounting identity:

```text
eligible_mission_arrivals
  = mission_success
  + mission_timeout
  + mission_admission_reject
  + mission_horizon_failure
```

Scenarios with zero eligible arrivals are marked `NA` and excluded by the
frozen manifest before any policy executes; their count is reported. Primary
aggregation: macro mean of scenario-level drop ratios. Pooled numerator/
denominator ratio is secondary only.

## 9. Training reward (potential-based shaping only)

```text
r'[t] =
  newly_dropped_mission_tasks[t]
  + gamma * Phi(causal_belief[t+1])
  - Phi(causal_belief[t])
```

Terminal potential is zero. Energy remains a hard constraint. Raw reward and
shaping components are logged separately. The negative radar-shaped reward is
not used.

## 10. F0..F6 gate order (must not skip)

| Gate | Content | Block |
|---|---|---|
| F0 | runtime/MDP contract: base commit, interface verification, legacy regression, namespace, DEBUG_CONTRACT.md | contract tests + smoke |
| F1 | minimal causal MFR-lite env + 8 contract test files | Gate 0 |
| F2 | 6 baselines + executable clairvoyant oracle + 128 paired scenarios | Gate 1 |
| F3 | supervised imitation 10k dev + 2k held-out | Gate 2 |
| F4 | masked PPO fixed-scenario overfit | Gate 3 |
| F5 | one-seed stochastic smoke 0.2..0.5M transitions | Gate 4 |
| F6 | two-seed pilot, no significance claim | Gate 5 |

## 11. Forbidden before Gate 1 passes

- PPO hyperparameter sweep;
- full eight-seed campaign;
- MAPPO or self-play;
- relabeling historical G2'a from FAIL to PASS;
- post-hoc switching between sampled and greedy evaluation;
- direct adoption or execution of quarantined orphan source files;
- adding hidden truth to make imitation or PPO pass;
- claiming the debug profile is a physically validated benchmark.

## 12. Phase report template

After every phase:

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

## 13. Secret scan rule (every commit and push)

Before each commit/push, scan:

- staged diff;
- tracked and newly generated text logs/configs;
- artifact manifests;
- archive member names and text payloads.

For: access tokens, private keys, passwords, authorization headers, and
credential-bearing URLs. If a candidate secret is found, stop the push,
report only the affected path and secret class, and never echo the secret
value.
