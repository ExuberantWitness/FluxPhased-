# Experiment Tracker：MFR-IQ `G3-BSTA`

**Owner**：PRO6000 agent  
**Initial status**：`BLOCKED_AT_P0_SOURCE_NOT_PRESENT_ON_AUDIT_HOST`  
**Rule**：每个 checkbox 后附 artifact path、command、commit SHA 和 verdict；不得只写“完成”。

## P0 — Provenance and Symbol Resolution

- [ ] `SOURCE_HANDOFF.json` commit/tree/archive hashes independently verified:
- [ ] Verified M7 commit/branch:
- [ ] Clean/dirty state recorded:
- [ ] `SYMBOL_MAP.md` complete:
- [ ] `drop_ratio` definition traced:
- [ ] Old `/tmp` scripts restored under repository:
- [ ] Raw episode/seed records restored:
- [ ] Checkpoint/config hashes recorded:
- [ ] `SPLIT_MANIFEST.json` created with first-access timestamps:
- [ ] Physics fit/validation, baseline-tune, planner-dev, headroom-confirm, PPO-validation, locked-test IDs non-overlapping:
- [ ] `RNG_KEY_SCHEMA.md` records event-key and action-key namespaces:
- [ ] One legacy scripted run reproduced:
- [ ] One legacy checkpoint evaluation reproduced:
- [ ] `P0_PROVENANCE_REPORT.md` verdict:
- [ ] Missing legacy artifacts, if any, labeled `LEGACY_CLAIM_UNVERIFIED` rather than blocking a separately preregistered new benchmark:

## P0-B — Resource and Selectivity Binding

- [ ] Actual emitter count and per-emitter power/energy/beam caps owner-approved:
- [ ] Team pooling either physically justified or disabled:
- [ ] Every service slot maps to a radar receiver/dwell and causal ESM association:
- [ ] Selectivity mechanism identified and calibrated:
- [ ] Cross-service/off-target coupling model frozen:
- [ ] Authoritative detect/track/estimate task semantics frozen:
- [ ] `RESOURCE_AND_SELECTIVITY_CONTRACT.md` hash:
- [ ] Human binding approval and timestamp:

**STOP reason if failed**：

## P1 — Legacy Diagnostics

- [ ] RNG streams separated and event-key CRN implemented:
- [ ] Event-key CRN is invariant to policy-dependent RNG call count:
- [ ] Fixed-duty sweep complete:
- [ ] Untrained actor evaluated:
- [ ] Sampled vs argmax evaluated under frozen protocol:
- [ ] JNR/floor-hit/drop-cause logs complete:
- [ ] Zero denominator is recorded as NaN, never 0:
- [ ] Legacy regression tests pass:

**Verdict**：

## E1 — Current-Environment Reachability

- [ ] Comparator tune uses only current-env tune split:
- [ ] `b_cur*` code/config/selection/tie-break frozen before bound confirm:
- [ ] Bound-confirm split untouched before freeze:
- [ ] Every `U_s,r` has exact/admissible pathwise certificate or B&B gap:
- [ ] Upper bound and comparator use paired event-key exogenous RNG:
- [ ] Paired `g_s=U_s-B_s` rows saved:
- [ ] `UCB95(E[g])` uses paired scenario-level t upper bound:
- [ ] No `UCB(U)-UCB(B)` subtraction:
- [ ] Approximate planner labeled lower-bound witness only:

**Verdict / STOP reason**：

## P2 — IQ-Calibrated Progress

- [ ] Task semantics classified as detect/track/other:
- [ ] Receiver-band `S/N/J/SINR` units mapped:
- [ ] Calibration grid covers baseline SNR/SCR, clutter, waveform/filter, integration/correlation, geometry and target model:
- [ ] IQ fit and untouched validation grids/seeds preregistered separately:
- [ ] Raw IQ fit Monte Carlo complete:
- [ ] Fast detector/information fit and envelope frozen before validation:
- [ ] Untouched IQ validation run hash:
- [ ] Validation `max_abs_error(Pd) <= .03`:
- [ ] Validation interpolation/off-target controls pass:
- [ ] Validation no-jam completion regression passes:
- [ ] Validation monotonicity/no-hard-floor tests pass:
- [ ] Calibration uncertainty envelope saved:
- [ ] Dynamic-track mode, if used, passes F/P/Q, miss, association, measurement-update and completion-mapping tests:
- [ ] No tuning continued on a failed validation split:

**STOP reason if failed**：

## P3 — Physically Bound Resource Allocator

- [ ] `(emitter, physically-addressable service)` and fixed-power core implemented:
- [ ] Per-emitter peak-power/beam invariants pass:
- [ ] `K_team=1`, if used, has approved RF/mission basis:
- [ ] Episode-energy invariant `<=1e-6` passes:
- [ ] `E_j < P_fixed,j × T_active,max` excludes fixed-power always-on when required:
- [ ] Idle always valid:
- [ ] Stable target-slot contract passes:
- [ ] Slot disappearance/reappearance and generation-ID tests pass:
- [ ] Target-local JNR test passes:
- [ ] Off-target cross-talk/leakage test passes:
- [ ] Learned/script/planner path parity passes:
- [ ] Valid action `requested == executed` exact test passes:
- [ ] Invalid-action policy recorded and strict debug raise passes:
- [ ] Confirmation mismatch/invalid/fallback count is zero:

**Verdict**：

## P4 — Causal Observation

- [ ] `OBSERVATION_SPEC.md` complete:
- [ ] Every field has source/latency/unit:
- [ ] Pre-action construction test passes:
- [ ] Future-arrival leakage test passes:
- [ ] Hidden-progress leakage test passes:
- [ ] Target permutation test passes:
- [ ] Every mask bit is a deterministic function of actor-visible observation/history:
- [ ] Tracker latency/identity/age/confidence source test passes:
- [ ] Tracker dropout/reacquisition test passes:
- [ ] Tracker hidden-truth leak test passes:

**STOP reason if failed**：

## P5 — Masked Categorical PPO

- [ ] Actor output schema versioned:
- [ ] Rollout stores action mask:
- [ ] PPO update reuses stored mask:
- [ ] Stored/recomputed log-prob test passes:
- [ ] Invalid action probability is zero:
- [ ] Only-idle row has no NaN:
- [ ] Action RNG isolation passes:
- [ ] Recurrent hidden/done mask saved and replayed, or `NOT_APPLICABLE` justified:
- [ ] Vector-env done/reset hidden-state isolation passes:
- [ ] Sequence boundary/slot alignment test passes:
- [ ] CPU/GPU PPO smoke passes:
- [ ] `EVAL_PROTOCOL.md` frozen before training:

**Verdict**：

## P6 — Reward/Metric Alignment

- [ ] Raw drop numerator/denominator logged:
- [ ] Cell ratio pools counts before division:
- [ ] Denominator zero yields NaN and triggers preregistered common-rep/cap rule:
- [ ] Policy-specific denominator mismatch test passes:
- [ ] NaN/zero-denominator counts appear in summary:
- [ ] Incremental drop and terminal drop logged:
- [ ] Return/drop relationship analyzed:
- [ ] Any shaping is potential-based and telescopes:
- [ ] No learner-only active-cost:

**Verdict**：

## P7 — Baselines and Oracle

- [ ] Required baseline registry complete:
- [ ] Baseline-tune scenarios fixed and isolated:
- [ ] Per-family finite hyperparameter grids/objective/tie-break registered:
- [ ] One finalist per required family frozen:
- [ ] `BASELINE_FREEZE.json` contains full frozen set B and hashes:
- [ ] Reduced exact DP cross-checked by enumeration:
- [ ] Causal/full-state/clairvoyant labels correct:
- [ ] Same `(time,energy)`, different causal history → different optimal action set:
- [ ] Selected-vs-runner-up Q-value gap exceeds `epsilon_Q`, solver tolerance and Monte Carlo uncertainty:
- [ ] Causal DP vs open-loop `LCB95 > .02`:
- [ ] Planner development uses only planner-dev split:
- [ ] Planner/operating point frozen before headroom-confirm first access:
- [ ] Headroom-confirm split first-access audit passes:

**STOP reason if failed**：

## A4/A5 — Untouched Robust Headroom

- [ ] At least 32 untouched headroom-confirm scenarios:
- [ ] Complete scenario × episode × action-rep grid:
- [ ] Paired event-key CRN across witness and every frozen script:
- [ ] Primary one-sided scenario-level t-LCB computed for every `b in B`:
- [ ] IUT primary: `min_b LCB95(h_b) > .075`:
- [ ] Full Cartesian sensitivity grid frozen before first run:
- [ ] Energy `{low,primary,high}` complete:
- [ ] Detector `{lower,central,upper}` complete:
- [ ] Shift `{nominal,arrival,geometry}` complete:
- [ ] Every `cell × b` row present:
- [ ] IUT robustness: `min_(cell,b) LCB95 > .05`:
- [ ] No per-cell script/planner retuning:
- [ ] Any simultaneous per-cell claims use Holm/Bonferroni/max-t:

**STOP reason if failed**：

## L0 — Learnability Pilot

- [ ] Train seed 9000 complete:
- [ ] Train seed 9001 complete:
- [ ] Only PPO-training/PPO-validation splits used:
- [ ] Headroom-confirm and locked-test access count remains zero:
- [ ] Frozen-init control:
- [ ] Random-feasible control:
- [ ] Candidate-feature shuffle:
- [ ] History shuffle:
- [ ] State-dependent action signal above permutation null:
- [ ] Constraint violations zero:
- [ ] No constant-action/energy-dump exploit:
- [ ] Full-budget variance-pilot GO decision:
- [ ] L0 explicitly not used as variance/power evidence:

## Full-Budget Variance Pilot and Power

- [ ] `K0=6–8` full-budget final-config variance seeds complete:
- [ ] Variance pilot evaluated on PPO-validation only:
- [ ] Sample-size decision does not use observed locked mean:
- [ ] Superiority margin `m=.05` frozen:
- [ ] Design alternative `mu1` and `delta*=mu1-m` frozen:
- [ ] Conservative variance upper bound `sigma_U` computed:
- [ ] Noncentral-t power curve saved:
- [ ] Minimum `N_power` has one-sided alpha `.05` and power `>=.8`:
- [ ] `N_max` and incomplete-run rule frozen:
- [ ] `N_train=max(8,N_power)` and 8 is labeled only a floor:
- [ ] If `N_train>N_max`, STOP rather than weakening margin/variance:

**Power verdict / artifact**：

## Confirmatory Full Training

- [ ] Config/checkpoint-selection rule frozen:
- [ ] PPO validation only used for checkpoint selection:
- [ ] All training manifests complete:
- [ ] Exactly `N_train` independent training-seed rows planned:
- [ ] Every training seed completes the same locked `scenario × episode × action-rep` grid:
- [ ] No locked-test tuning:

## Locked G3-BSTA

- [ ] Locked scenarios untouched before final:
- [ ] Locked split first-access timestamp/hash recorded:
- [ ] All raw episode rows saved:
- [ ] Event-key exogenous RNG paired across all policies:
- [ ] Action RNG isolated and every stochastic policy completes the full grid:
- [ ] Eligible/drop counts pooled before cell ratio:
- [ ] Zero denominator stored as NaN, never 0:
- [ ] Common additional-replicate/cap rule applied identically:
- [ ] No policy-specific denominator mismatch:
- [ ] Requested/executed mismatch, invalid and fallback counts all zero:
- [ ] One row per independent training seed aggregate:
- [ ] One `d_i,b` column for every frozen finalist `b in B`:
- [ ] No calibration winner or test-realized maximum substitutes for full IUT:
- [ ] Correct `H0: delta <= .05`:
- [ ] One-sample training-seed t statistic/SE/one-sided p/LCB for every script:
- [ ] p-value labeled one-sample t, not exact:
- [ ] IUT: `min_b LCB95_b > .05`:
- [ ] Training seed is the only primary inferential df:
- [ ] Primary estimand labeled conditional on locked suite:
- [ ] Crossed `train_seed × scenario` model/sensitivity artifact:
- [ ] Scenario-population claim withheld unless crossed inference passes:
- [ ] No iid episode/action bootstrap:
- [ ] Any bootstrap is clustered by training seed; scenario resampled separately when applicable:
- [ ] Optional sign-flip result labeled symmetry-dependent sensitivity:
- [ ] Primary/secondary evaluation semantics unchanged:
- [ ] Full frozen sensitivity-grid rows reported without retuning:
- [ ] NaN/zero-denominator and RNG/mismatch counters reported:
- [ ] `summary.json` hash:
- [ ] Final verdict:

## Claim Control

- [ ] Original G2'a remains labeled FAIL:
- [ ] G3-BSTA described as a new environment/gate:
- [ ] Planner not mislabeled as PPO:
- [ ] Evaluation seeds not mislabeled as training seeds:
- [ ] Approximate planner not called exact upper bound:
- [ ] Single calibration winner not called “best scripted”:
- [ ] Conditional locked-suite claim distinguished from scenario-population claim:
- [ ] “8 seeds” not presented as sufficient without power evidence:
- [ ] t-test p-value not called exact:
- [ ] Simulation-only scope stated:
- [ ] No general “RL beats scripts” claim:
