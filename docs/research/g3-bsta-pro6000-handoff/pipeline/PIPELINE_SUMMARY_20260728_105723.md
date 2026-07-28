# Research-Pipeline Summary：FluxPhased MFR-IQ G2'a Redesign

## 1. Executive decision

**Do not let PRO6000 patch or train yet.**

The current audited workspace does not contain the M7/MFR-IQ source claimed by the attached report. The next authorized action is only:

```text
P0 source handoff → symbol resolution → physical binding packet → owner approval
```

Current verdict:

```text
BLOCK_PPO_PROVENANCE
```

The recommended new benchmark is:

```text
G3-BSTA
Budgeted Service-Target Allocation
```

It is a new environment and a new gate. Original G2'a remains:

```text
FAIL / unverified reachability
```

No code was modified in the target FluxPhased repository, no PPO was trained, and no positive result is claimed.

## 2. Why the original report cannot drive a direct patch

The earlier experiment audit found:

1. the report compared PPO against blink `0.450`, while the visible strongest scripted result was noise `0.520`;
2. PPO argmax was `0.491/0.475`, so it trailed noise by `2.9pp/4.5pp`, and missed a 5pp superiority gate by `7.9pp/9.5pp`;
3. the tested null was effectively `delta=0`; the gate requires `H0: delta <= .05`;
4. eight evaluation RNG seeds came from one training checkpoint and therefore did not estimate training-seed variance;
5. untrained, feasible-random and causally shuffled controls were absent;
6. the cited MFR source, raw rows, checkpoint/config hashes and `/tmp` gate scripts were absent;
7. a five-policy sweep did not prove universal structural impossibility;
8. the report contained an arithmetic contradiction (`.491 < .489`);
9. the hard progress floor is a credible saturation mechanism, but final 50%-vs-100% duty parity still requires measurement.

Therefore the correct response is not “tune PPO harder.” It is to make the scientific question auditable and falsifiable before spending training compute.

## 3. Research result and novelty boundary

### Selected core

Candidate A, revised after independent review:

- explicit physical emitter set;
- action `idle` or `(emitter, physically-addressable service)`;
- fixed calibrated active power;
- per-emitter peak power, beam count and episode energy;
- team pooling only if a platform owner proves it exists;
- masked categorical recurrent actor;
- causal ESM-derived service slots;
- calibrated emitter-to-receiver/service coupling and cross-talk;
- task-specific detector/tracker transition;
- learned, scripted and planner policies share one action/resource/physics path.

### Mandatory entry

Candidate E, oracle-first reachability and stop-RL, is mandatory Phase 0. PPO is authorized only after:

- source provenance passes;
- resource/selectivity binding passes;
- held-out physics validation passes;
- a reduced exact decision certificate shows a material causal Q-value gap;
- an untouched same-observation causal witness has robust all-script headroom.

### Novelty conclusion

Algorithm novelty is low. Recent work already combines jammer target selection, jamming type and power control with MARL, including a 2025 many-jammer/many-radar formulation whose action contains target, type and power. That paper itself describes its model as functional-level rather than an IQ-calibrated benchmark. [Cai et al., 2025](https://link.springer.com/article/10.1007/s43684-025-00090-4)

The defensible contribution, if experiments succeed, is:

> benchmark pathology diagnosis + IQ-calibrated repair + causal observation contract + oracle-first admissibility protocol

It is not:

> a first or novel PPO target/beam/power allocation algorithm

## 4. Non-negotiable physical contract

Before implementing target selection, PRO6000 must establish:

1. which physical emitter transmits;
2. which radar receiver/dwell constitutes a service;
3. how the emitter can select that service through beam direction, receiver identity, range/Doppler/angle gate, frequency/waveform, time-aligned dwell or explicit deception;
4. how much interference leaks into non-selected services;
5. where `S/N/J/C` are measured and in which units;
6. whether task semantics are detection, dynamic tracking or another operation.

An internal task ID is not a physical RF selector. If a single receiver-wide broadband noise jammer affects every internal task identically, target-local JNR is undefined and the executor must stop:

```text
BLOCK_TARGET_LOCALITY
```

Radar detection probability depends on target/clutter statistics, receiver processing and false-alarm assumptions, so a one-dimensional JNR lookup is insufficient. [Shnidman, 1995](https://ieeexplore.ieee.org/abstract/document/395246)

For dynamic tracking, a standalone `HᵀR⁻¹H` increment is also insufficient. The implementation must include state prediction, process noise, missed detection, association, measurement update and a frozen mapping from covariance/information to task completion.

## 5. Implementation phases for PRO6000

| Phase | Authorized change | Required evidence | Stop condition |
|---|---|---|---|
| P0 | Verify source handoff, resolve symbols, restore available provenance | `SOURCE_HANDOFF.json`, `SYMBOL_MAP.md`, hashes, dirty-state report | missing/hash mismatch → `BLOCK_SOURCE_HANDOFF` |
| P0-B | Freeze emitter/resource/service/transition/objective contracts | binding packet and owner approval | no selectivity → `BLOCK_TARGET_LOCALITY`; no semantics → `BLOCK_PHYSICS_SEMANTICS` |
| P1 | Instrument legacy behavior without changing it | metric tests, event-key RNG, duty/untrained/sampled-vs-argmax diagnostics | legacy mismatch → original claim remains unverified |
| P2 | Add feature-flagged calibrated detector/tracker path | IQ fit and untouched validation, units and no-jam regression | calibration/error-budget failure |
| P3 | Add one emitter×service resource/action adapter | per-emitter conservation, selectivity/cross-talk and policy-path parity tests | resource/action invariant failure |
| P4 | Implement causal observation/slot contract | field provenance/latency, mask=f(obs), no hidden-state leakage | observation or mask leakage |
| P5 | Replace Bernoulli with recurrent masked categorical | stored mask/log-prob/hidden-state replay, requested=executed | any fallback/mismatch |
| P6 | Freeze one reward and metric contract | numerator/denominator/NA/unfinished-task/macro-micro tests | objective hash or alignment failure |
| P7 | Freeze baselines and oracle/planner | exact reduced certificate, causal planner, separate development/confirmation splits | open-loop equivalence or no headroom |
| P8 | Learnability smoke, variance pilot, power analysis, full training | two smoke seeds; 6–8 full-budget variance seeds; noncentral-t N | no learning signal or insufficient power budget |
| P9 | Run locked all-script gate | raw rows, hashes, all-script IUT, conditional and crossed analyses | any script margin LCB `<=.05` |

Every phase produces a small commit and phase report. A failed STOP gate prevents all later GPU work.

## 6. Exact action and PPO contract

### Feasible action

\[
a_t \in \{0\}\cup\{(e,q): e\text{ can physically address service }q\}.
\]

Per emitter:

\[
0\le P_{e,t}\le P_{e,\max},\qquad
E_{e,t+1}=E_{e,t}-\Delta tP_{e,t}\ge0.
\]

If the mission contract intends to make fixed-power always-on infeasible:

\[
E_{e,0}<P_{\mathrm{fixed},e}T_{\mathrm{active,max},e}.
\]

Do not compare energy against team peak power.

### On-policy invariant

During training and scientific evaluation:

```text
requested_action == executed_action
```

Any mismatch invalidates the rollout/run. Silently projecting an invalid request to idle would combine the log-probability of one action with the transition/reward of another and is not a valid PPO sample.

### Mask invariant

```text
mask_t = f(actor_visible_observation_or_history_t, static_template_table)
```

The mask cannot read true alive state, true task ID, hidden geometry, queue/progress or a future radar mode.

### RNG invariant

All exogenous draws use stateless event keys:

```text
(protocol, split, scenario, episode, time, entity, event_type, draw_index)
```

Action RNG is separate. Changing a policy branch must not shift future arrival, geometry or detector draws.

## 7. Physics validation

Fit and validation data must be disjoint. The grid covers:

- baseline SNR/SCR, range and RCS;
- receiver-band JNR;
- noise and clutter distributions/correlation;
- waveform, receiver/filter and pulse-compression processing;
- look count and integration/correlation mode;
- `Pfa`;
- target/Swerling model;
- geometry/antenna gain;
- spectral overlap, on-target/off-target and cross-service coupling.

Natural `Pd→Pfa` or `Pd→1` asymptotes are acceptable; arbitrary clamps, lookup truncation and unrecorded quantization plateaus are not.

The precise acceptance tolerance, such as an initial `.03` `Pd` error candidate, must come from a preregistered error budget. It is not a universal literature constant.

## 8. Baselines and oracle

Freeze one finalist from every competent script family using baseline-tuning data only:

- off;
- random feasible;
- budgeted barrage;
- round robin;
- periodic blink duty/phase family;
- EDF/threat-first;
- reactive service follower;
- marginal information/drop per joule;
- short-horizon assignment/knapsack.

For reduced cases, exact DP/B&B supplies an actual certificate. MFR scheduling literature demonstrates the appropriate separation between exact branch-and-bound and approximate MCTS/policy methods. [Shaghaghi et al., 2018](https://arxiv.org/abs/1805.07069)

For the full environment:

- same-observation causal planner = headroom witness;
- full-state planner = privileged diagnostic;
- clairvoyant planner = optimistic ceiling;
- approximate MCTS/beam/CEM = lower-bound witness, not an upper bound.

Planner development and headroom confirmation use mutually exclusive scenario splits.

## 9. Headroom and final statistical gates

Let `B` be all frozen script finalists.

### Pre-PPO headroom

On untouched confirmation scenarios:

\[
\min_{b\in B}\operatorname{LCB}_{95}
(D_{\mathrm{causal\ witness}}-D_b) > 0.075.
\]

On every preregistered energy × detector-envelope × scenario-shift cell:

\[
\min_{c,b}\operatorname{LCB}_{95}
(D_{\mathrm{causal\ witness},c}-D_{b,c}) > 0.05.
\]

The `.075` buffer is a project design choice and must be justified by calibration, planner-to-learner and evaluation uncertainty.

### Final gate

For training seed `i` and frozen script `b`:

\[
d_{i,b}=\operatorname{mean}_{s,a,e}
(D^L_{i,s,a,e}-D^b_{s,a,e}).
\]

Pass only if:

\[
\min_{b\in B}
\left[
\bar d_b-t_{0.95,K-1}\frac{s_b}{\sqrt K}
\right] > 0.05.
\]

This is an intersection-union gate: every frozen script must be beaten by more than 5pp. A calibration-selected winner or a per-scenario post-hoc script maximum cannot replace it.

Two short seeds are smoke tests only. Use external variance or at least 6–8 full-config/full-budget variance seeds, preregister `mu_alt>.05`, a conservative SD, power `>=.8` and `N_max`, and solve `N` with a noncentral-t calculation. Eight is only an engineering floor. RL methodology literature strongly cautions against conclusions from point estimates and a few runs without uncertainty. [Agarwal et al., 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)

The primary claim is conditional on the locked scenario suite. A scenario-population claim requires a separate training-seed × scenario crossed analysis.

## 10. Minimum acceptance tests

```text
test_source_handoff_hashes
test_binding_approval_hashes
test_event_key_rng_branch_invariance
test_detector_fit_heldout
test_tracker_information_recursion
test_fixed_power_always_on_is_infeasible
test_target_selectivity_and_crosstalk
test_mask_is_function_of_obs
test_slot_identity_no_leak
test_requested_equals_executed
test_saved_mask_logprob_replay
test_recurrent_state_reset_and_truncation
test_policy_adapter_path_parity
test_reward_metric_alignment
test_all_script_iut
test_gate_null_coverage
```

## 11. Claim boundary

Only after every gate passes may the project state:

> On the preregistered, physically bound G3-BSTA simulation benchmark, fixed rule-radar opponent, causal observation contract, frozen resource constraints and locked scenario suite, PPO improves raw drop ratio by more than 5pp relative to every frozen competent script.

It must not state:

- original G2'a passed;
- PPO or target/beam/power allocation is novel;
- a planner witness is an exact upper bound;
- headroom proves PPO learnability;
- simulation results generalize to real systems or all MFR scenarios.

## 12. Handoff package

Primary executor artifacts:

- `PRO6000_EXECUTION_PROMPT_20260728_105723.md` — copy-paste directive;
- `PRO6000_AGENT_IMPLEMENTATION_SPEC_20260728_023103.md` — detailed code/task specification;
- `refine-logs/EXPERIMENT_PLAN_20260728_023103.md` — split, gate and statistics protocol;
- `refine-logs/EXPERIMENT_TRACKER_20260728_023103.md` — phase checklist;
- `SOURCE_HANDOFF.template.json` — required external handoff;
- `SOURCE_PROVENANCE_AUDIT_20260728_105723.md` — current source blocker;
- `review-traces/REVIEW_RESPONSE_20260728_105723.md` — independent-review amendments.

Supporting research:

- `LITERATURE_REPORT_20260728_023103.md`;
- `idea-stage/IDEA_REPORT_20260728_023103.md`;
- `idea-stage/JURY_DECISION_20260728_023103.md`;
- `review-traces/NOVELTY_REVIEW_20260728_105723.md`;
- `review-traces/RESEARCH_REVIEW_20260728_105723.md`;
- `review-traces/STATS_REVIEW_20260728_105723.md`.

All review verdicts are `same-family / provisional`; they are independent work threads, not external peer review or experimental validation.

