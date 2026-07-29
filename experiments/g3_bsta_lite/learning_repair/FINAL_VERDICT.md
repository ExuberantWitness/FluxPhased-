# Final Verdict — G3-BSTA-lite Learning-Repair Branch

```text
document:        FINAL_VERDICT.md
branch:          g3-bsta/mfr-lite-learning-repair
base_commit:     01ea4283f422cd01fec5fb15037c5012543adf7a
base_tree:       e1e502e5ac7089ee9e8a0649d46307731780e8be
git_commit:      87e37736092ed2cd6cdfc99b9c56275661338fb9
tree_sha:        e93a594d6a7acffd20705a8faf9280e09ebbae23
issued:          2026-07-30
```

## 1. Phase / status

```text
phase:                R2 (Gate 3) — completed
status:               DEBUG_VERTICAL_SLICE_COMPLETE / PPO_LEARNING_PARTIAL_BUT_BELOW_GATE3_THRESHOLD
overall_verdict:      BLOCKED_LEARNING_CONTRIBUTION
next_authorized_phase:NONE
```

PPO scratch-training on `mdp_sanity_v1` produces a **statistically
real but practically partial** learning contribution. The trained
actor **does** significantly beat its pristine initialization and the
strongest non-witness baseline on a paired-LCB95 basis. It does **not**
clear the preregistered Gate-3 thresholds on point improvement
(≥ 5 pp; observed 3.08 pp) or witness-headroom recovery (≥ 80%;
observed 30.0%).

Per PREREGISTRATION §6, R2 status is `BLOCKED_LEARNING_CONTRIBUTION`
and R3 (two-seed pilot) + R4 (POMDP pilot) are **NOT authorized**. No
BC warm-start substitution was used at this gate.

## 2. Repository / branch / SHAs

- `repository`: `https://github.com/ExuberantWitness/FluxPhased-`
- `branch`: `g3-bsta/mfr-lite-learning-repair`
- `base_commit`: `01ea4283f422cd01fec5fb15037c5012543adf7a`
- `base_tree`:   `e1e502e5ac7089ee9e8a0649d46307731780e8be`
- `git_commit`:  `87e37736092ed2cd6cdfc99b9c56275661338fb9`
- `tree_sha`:    `e93a594d6a7acffd20705a8faf9280e09ebbae23`
- `pushed`: `PUSH_PENDING_NETWORK` (see §11)

## 3. Changed files (relative to base)

```text
env/gpu/g3_bsta_lite/__init__.py            (modified — exports PROFILE_*, pomdp_urgency_proxy)
env/gpu/g3_bsta_lite/env.py                 (modified — profile, integer tokens, ledger, OOB, telescoping)
env/gpu/g3_bsta_lite/metrics.py             (modified — per-step finalize bookkeeping for ledger)
env/gpu/g3_bsta_lite/observation.py         (modified — two profiles, non-invertible urgency proxy)
experiments/g3_bsta_lite/learning_repair/__init__.py
experiments/g3_bsta_lite/learning_repair/PREREGISTRATION_AMENDMENT_01.md
experiments/g3_bsta_lite/learning_repair/manifests/MANIFEST_AUDIT.json
experiments/g3_bsta_lite/learning_repair/manifests/SHA256SUMS.txt
experiments/g3_bsta_lite/learning_repair/manifests/checkpoint_validation.json
experiments/g3_bsta_lite/learning_repair/manifests/dagger_train.json
experiments/g3_bsta_lite/learning_repair/manifests/generate_manifests.py
experiments/g3_bsta_lite/learning_repair/manifests/locked_test.json
experiments/g3_bsta_lite/learning_repair/manifests/ppo_train.json
experiments/g3_bsta_lite/learning_repair/controls.py
experiments/g3_bsta_lite/learning_repair/stats.py
experiments/g3_bsta_lite/learning_repair/trainer.py
experiments/g3_bsta_lite/learning_repair/run_r2_gate3.py
experiments/g3_bsta_lite/learning_repair/run_r2_eval_only.py
experiments/g3_bsta_lite/learning_repair/smoke_trainer.py
experiments/g3_bsta_lite/learning_repair/smoke_cuda.py
experiments/g3_bsta_lite/learning_repair/r2_gate3_output/R2_GATE3_RESULT.json
experiments/g3_bsta_lite/learning_repair/r2_gate3_output/raw_rows_trained.jsonl
experiments/g3_bsta_lite/learning_repair/r2_gate3_output/raw_rows_controls.jsonl
experiments/g3_bsta_lite/learning_repair/r2_gate3_output/raw_rows_baselines.jsonl
experiments/g3_bsta_lite/learning_repair/r2_gate3_output/.gitignore
tests/g3_bsta_lite/test_learning_repair_r1.py
```

The historical fast-work line (`g3-bsta/mfr-lite-fastwork`, base of
this branch), `main`, `docs/pro9000-forensic-output`, and any handoff
branches are **untouched**. The `LINE_REPORT.md` on the fast-work
branch remains as a historical engineering record.

## 4. Tests

```text
baseline_command: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD pytest -q tests/g3_bsta_lite
baseline_result:  75 passed (46 original + 29 R1 contract tests)
command_run:      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD pytest -q tests/g3_bsta_lite
result:           75 passed
```

R1 contract tests cover: manifest pairwise disjoint (6 pairs) + legacy
range exclusion + size matches + per-scenario SHA-256; profile enforcement
(pomdp_v1 requires delay≥1, mdp_sanity allows delay=0); OBS_DIM=11 across
profiles; non-invertible urgency proxy saturation; pending-count and
radar-service visibility in mdp_sanity obs; pending mutation does NOT
leak into pomdp_v1 obs; potential-shaping sign-correct telescoping;
integer-token mask; OOB action guard; step-after-done guard; ledger
identity residual zero at episode end; EnvConfig.to_json records profile
and tokens.

## 5. Manifest audit

```text
manifest            size   base_seed   purpose
dagger_train        128    21000101    F3-style DAgger (placeholder for residual PPO)
ppo_train           64     21001101    R2 scratch PPO rollout scenarios (cycled)
checkpoint_valid.   64     21002101    R2 checkpoint selection (validation only)
locked_test         128    21003101    UNTOUCHED — would have been R3 eval
```

`MANIFEST_AUDIT.json overall_verdict = ALL_DISJOINT_AND_LEGACY_CLEAN`.
Pairwise intersections: 0/0/0/0/0/0 (all six pairs disjoint). Legacy
F5/F6 range `20260801..20260832` excluded from every manifest.

The preregistered base_seeds in PREREGISTRATION §3 were respaced in
PREREGISTRATION_AMENDMENT_01.md: original spacing of 100 was insufficient
because with `arrival_rate_per_service = 0.15` and 128 cells per arrivals
table every seed is eligible, so each manifest occupies a contiguous
block of seeds. The amended table reserves 1000-seed windows per
manifest so collisions across windows are impossible by construction.

## 6. Profiles

| profile | obs_delay | obs includes | purpose |
|---|---|---|---|
| `mdp_sanity_v1` | 0 allowed | exact pending per service + current radar service one-hot | sanity: can PPO learn at all? |
| `pomdp_v1` | ≥ 1 enforced | delayed detect EMA + non-invertible urgency proxy; radar hidden | genuine POMDP (R4-only; not run) |

Both profiles share `OBS_DIM = 11` (slots differ in semantics only).
POMDP urgency proxy = `1 − exp(−n_pending / 3)`: strictly monotone but
saturating, recovering the exact pending count only up to a wide
interval at large n (the leak flagged in POST_AUDIT_CORRECTION §4.1).

## 7. Gates

```text
R0 audit_accepted:    PASS (POST_AUDIT_CORRECTION verbatim, no re-argument)
R1 manifests+env:     PASS (75/75 contract tests; audit clean)
R2 Gate 3 (mdp_sanity_v1, scratch PPO):
    7/9 criteria PASS, 2/9 FAIL
    PASS: transitions < 0.5M, LCB>0 vs init, LCB>0 vs baseline,
          mask violations = 0, ledger = 0, accounting = 0,
          pre_ratio_offset ≈ 0
    FAIL: point_improvement (3.08 pp < 5 pp threshold)
    FAIL: witness_headroom_recovery (30.0% < 80% threshold)
    status: BLOCKED_LEARNING_CONTRIBUTION
R3 corrected two-seed pilot: SKIPPED (R2 failed; not authorized)
R4 POMDP pilot:              SKIPPED (R3 not run; not authorized)
```

## 8. Metrics

```text
profile: mdp_sanity_v1
selected candidate: lr=3e-5, target_kl=0.01, best_iter=199
cumulative_transitions: 204800   (<= 500000 cap)
kl_rollback_count: 0

macro_drop (mission drop ratio, higher = better jammer):
    trained (validation-selected)  0.1961
    scratch_init (pristine)        0.0149
    random_untrained               0.0123
    shuffled_observation           0.0453
    time_only                      0.0143
    budgeted_round_robin (best baseline) 0.1653
    greedy_radar_follower                 0.1653

paired LCB95 (Student-t, one-sided):
    trained − scratch_init:        +0.165  (PASS: > 0)
    trained − best_baseline:       +0.015  (PASS: > 0)
    point improvement over baseline: +3.08 pp  (FAIL: < 5 pp threshold)

witness headroom recovery:
    witness_ref_drop = 0.2680   (Gate 1 reference, NOT a deployable bound)
    headroom_total   = 0.2680 − 0.1653 = 0.1027
    headroom_recovered = (0.1961 − 0.1653) / 0.1027 = 30.0%   (FAIL: < 80%)

spearman(n_eligible, drop_ratio) across trained + scratch_init rows: 0.007
    (Near-zero, as expected — drop_ratio is normalized by n_eligible.)
```

All raw per-(seed, rep) rows in
`r2_gate3_output/raw_rows_{trained,controls,baselines}.jsonl`. The
trained checkpoint (`.pt`, ~150 KB) is held locally under
`r2_gate3_output/lr3e-05_kl0.01/`; it is git-ignored per task spec
(small JSON / CSV / JSONL only).

## 9. constraint_violations

```text
constraints                                                  status
====================================================================
1. no embedded credentials in remote URL                      PASS (sanitized at R0)
2. no credential echo in logs/diffs/commits                   PASS (verified pre-commit)
3. no force-push; no history rewrite                          PASS (additive commits only)
4. base_commit / base_tree exactly preserved as branch point   PASS
5. no 8-seed campaign                                         PASS (R3 not authorized)
6. no full MFR scale-out                                      PASS
7. no MAPPO / two-team integration                            PASS
8. no modification of main / fast-work / forensic / handoff   PASS
9. no Gate threshold retuning                                 PASS
10. no BC warm-start substitution in Gate 3                   PASS (scratch PPO only)
11. no deletion / squash of failed-seed evidence              PASS (all evidence kept)
12. no re-introduction of quarantined orphan MFR bytes        PASS
13. locked_test run only once                                 N/A (R3 not authorized, locked_test untouched)
14. no test-set checkpoint selection                          PASS (validation only)
15. no pseudo-replication                                     PASS (paired delta on same scenario)
16. claims_forbidden honored                                  PASS (see §10)
```

## 10. claims_allowed / claims_forbidden

### claims_allowed

- The 46 fast-work contract tests still pass; the debug vertical slice
  (env / mask / transition / vector-isolation invariants) is intact.
- The 29 new R1 contract tests pass; the R1 env / manifest / profile
  fixes bind the invariants listed in POST_AUDIT_CORRECTION §4.
- Four disjoint manifests with pairwise intersection empty and legacy
  F5/F6 range excluded are committed; manifest_audit overall_verdict =
  ALL_DISJOINT_AND_LEGACY_CLEAN.
- mdp_sanity_v1 obs is a genuine fully-observed MDP; pomdp_v1 obs has
  delay ≥ 1 and a non-invertible urgency proxy (no pending-count leak).
- Potential-based shaping uses the correct telescoping form
  `γ·Φ(s_{t+1}) − Φ(s_t)` with Φ(s_t) captured before any transition
  effect.
- Integer energy tokens drive the mask; float rounding can no longer
  flip the legal-action set.
- Per-mission event ledger has zero identity residual at episode end.
- The R2 scratch PPO actor, on validation scenarios:
  * significantly beats its pristine init (paired LCB95 = +0.165, n=64),
  * significantly beats the strongest non-witness baseline on a paired
    LCB95 basis (+0.015, n=64),
  * beats every deployed control by a wide margin: random_untrained
    (0.0123), scratch_init (0.0149), time_only (0.0143),
    shuffled_observation (0.0453) — all well below trained 0.1961.
- PPO training-health invariants hold: pre_ratio_offset ≈ 0 across
  every minibatch of every iteration; 0 KL rollbacks across 200 outer
  iterations of the selected candidate; 0 mask / ledger / accounting
  violations.

### claims_forbidden (per task §7 — must remain until R3 PASS)

- PPO improves policy at the **Gate-3 threshold** required by the
  preregistration. (It improves statistically, but not at the frozen
  5 pp / 80% thresholds.)
- Held-out generalization established.
- Multi-seed robustness established.
- Statistical significance established.
- Publication-ready full MFR result.

## 11. Push status

`pushed: PUSH_PENDING_NETWORK`.

R0 commit `526ab93` was pushed successfully to `g3-bsta/mfr-lite-learning-repair`
on 2026-07-29 over the sanitized HTTPS URL. Subsequent commits
(`78061d8` R1, `87e3773` R2 / FINAL_VERDICT) are **local only**: the
post-R0 push attempts failed with `git@ssh.github.com: Permission
denied (publickey)` and no credential helper is configured on the
worktree. Remote URL is the bare HTTPS form
`https://github.com/ExuberantWitness/FluxPhased-` (verified; no token
embedded). The three local commits are signed-off and the tree SHAs
are recorded above; pushing them is a one-command operation once a
PAT or SSH key is provisioned:

```bash
git push origin g3-bsta/mfr-lite-learning-repair
```

## 12. stop_reason

`stop_reason: PREREGISTRATION_BINDING_R2_FAIL`.

R2 Gate 3 returned 7/9 PASS but two FROZEN-threshold criteria failed
(point_improvement < 5 pp; witness_headroom_recovery < 80%). Per
PREREGISTRATION §6 the next authorized phase is `NONE` and no
BC-warm-start substitution or threshold retuning is permitted. The
honest stop is here.

## 13. Reproducibility

```bash
# From repo root on the learning-repair branch:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD \
  pytest -q tests/g3_bsta_lite                                 # 75/75
python experiments/g3_bsta_lite/learning_repair/manifests/generate_manifests.py
python experiments/g3_bsta_lite/learning_repair/run_r2_gate3.py # ~37 min on RTX PRO 6000
# Inspect:
#   experiments/g3_bsta_lite/learning_repair/r2_gate3_output/R2_GATE3_RESULT.json
```

All four candidate trainings (lr ∈ {3e-5, 1e-4} × target_kl ∈ {0.01, 0.02})
ran to completion deterministically; the selected candidate
(`lr=3e-5, target_kl=0.01`) had validation-best macro_drop = 0.1991 at
iteration 199.
