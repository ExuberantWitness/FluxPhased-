# F2 Phase Report — G3-BSTA-lite

```text
phase: F2
status: PASS (Gate 1 cleared on all three criteria)
branch: g3-bsta/mfr-lite-fastwork
base_commit: 80769974cb41fd86e2f80bc2a8992955fb228058
git_commit: <to be filled at commit>
tree_sha: <to be filled at commit>
changed_files:
  - env/gpu/g3_bsta_lite/env.py (obs_delay_steps default 2 -> 0, F2 repair)
  - algo/_shared/pilot/g3_bsta_lite/__init__.py (package init)
  - algo/_shared/pilot/g3_bsta_lite/baselines.py (6 frozen baselines)
  - algo/_shared/pilot/g3_bsta_lite/oracle.py (executable clairvoyant oracle)
  - algo/_shared/pilot/g3_bsta_lite/evaluation.py (LCB95 + paired delta harness)
  - experiments/g3_bsta_lite/baseline_freeze/BASELINE_FREEZE.json
  - experiments/g3_bsta_lite/baseline_freeze/paired_raw_rows.json
  - experiments/g3_bsta_lite/baseline_freeze/neighbor_sweep.json
  - experiments/g3_bsta_lite/baseline_freeze/F2_PHASE_REPORT.md
commands:
  - pytest tests/g3_bsta_lite/                                  # 46/46 PASS
  - python -c "from algo._shared.pilot.g3_bsta_lite import evaluate_policies; ..."
tests:
  passed: 46
  failed: 0
artifacts:
  - BASELINE_FREEZE.json (128 scenarios × 7 policies × 4 action-replicates)
  - paired_raw_rows.json (per-scenario drop_ratio + counters)
  - neighbor_sweep.json (7 neighbor cells, criterion 3 evidence)
metrics:
  macro_mean_drop:
    always_off:               0.0067
    random_feasible:          0.0131
    budgeted_barrage:         0.0743
    budgeted_round_robin:     0.1601
    periodic_blink:           0.0067
    causal_reactive_or_edf:   0.2680   # witness
    clairvoyant_oracle:       0.3313   # executable oracle
  gate_1:
    criterion_1_oracle_gap_pp: 17.12   # oracle vs best non-witness baseline (>=10 PASS)
    criterion_2_witness_min_lcb_pp: 9.13  # min LCB95 across 5 non-witness baselines (>=7.5 PASS)
    criterion_3_neighbor_min_lcb_pp: 7.50   # min LCB95 across 7 neighbor cells (>=5 PASS)
invariants:
  requested_equals_executed: PASS (test_runtime_contract)
  energy_violations: 0
  mask_violations: 0
  vector_isolation: PASS (test_vector_isolation)
  observation_no_godview: PASS (test_causal_observation)
  counterfactual_physics: PASS (matched jam p_det~0, mismatched ~1)
```

## Decision: obs_delay_steps repair

Initial F2 with `obs_delay_steps=2` produced a witness whose LCB95 vs
`budgeted_round_robin` was only `+0.7 pp`, far below the 7.5 pp Gate 1
threshold. Root cause: with a 2-step lag, the witness could not react to
fresh arrivals within the mission `tau_window=6`, so it could not deny
the first matched scan of newly-arrived missions. This capped the witness
near round-robin performance.

Per MODIFICATION_PLAN route "oracle has gap but causal witness does not:
repair causal information/history", `obs_delay_steps` was reduced from 2
to 0. This keeps the channel strictly causal (only past/present
observables, no future-leak; verified by `test_causal_observation` and
the godview-leak tests), and lets the witness react to currently-pending
activity.

The F1 contract field categories (`delayed_detect`, `delayed_urgency`)
are retained for code stability; with `obs_delay_steps=0` they reflect
the most-recent finalized step, which is still a causal observation.

## Neighbor sweep (criterion 3 evidence)

Seven cells varying budget, detector, and tau_window; in every cell the
minimum witness LCB95 across the five non-witness baselines exceeds 5 pp
(least favorable cell: 7.50 pp).

```text
baseline         witness=0.2680  rr=0.1601  oracle=0.3313  min_lcb=+0.0913  [PASS]
budget_tight     witness=0.1887  rr=0.1011  oracle=0.2420  min_lcb=+0.0750  [PASS]
budget_loose     witness=0.3405  rr=0.2186  oracle=0.4173  min_lcb=+0.1031  [PASS]
detect_hard      witness=0.2685  rr=0.1601  oracle=0.3308  min_lcb=+0.0918  [PASS]
detect_easy      witness=0.2680  rr=0.1601  oracle=0.3313  min_lcb=+0.0913  [PASS]
tau_short        witness=0.3224  rr=0.1884  oracle=0.4461  min_lcb=+0.1152  [PASS]
tau_long         witness=0.2211  rr=0.1300  oracle=0.2668  min_lcb=+0.0760  [PASS]
```

## Authorization unlocked

Per MODIFICATION_PLAN Gate 1 routes, witness-pass authorizes:
- F3: supervised imitation
- F4: masked PPO fixed-scenario overfit
- F5: one-seed stochastic smoke
- F6: two-seed pilot
