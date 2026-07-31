---
name: g3-bsta-lite-f2-gate1-pass
description: 2026-07-29 G3-BSTA-lite F2 PASS — Gate 1 cleared on all 3 criteria after obs_delay_steps repair (2→0). Branch g3-bsta/mfr-lite-fastwork commit 9873697.
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c76cde4-1c23-4df6-94f5-ee29ee81afbc
---

G3-BSTA-lite fast-work milestone F2 finished 2026-07-29. Gate 1 PASS on
all three criteria.

**Branch / commit**: `g3-bsta/mfr-lite-fastwork` @ `9873697` on
`https://github.com/ExuberantWitness/FluxPhased-.git`. Base commit
`80769974cb41fd86e2f80bc2a8992955fb228058`.

**Why**: F2 is the reachability/headroom gate per MODIFICATION_PLAN.
Passing it authorizes F3 (imitation) → F4 (PPO overfit) → F5 (smoke) →
F6 (pilot). See [[g3-bsta-pro6000-p0-blocked]] for prior quarantine
context that this clean line sidesteps.

**How to apply**: Before recommending F3/F4 work or quoting baseline
numbers, verify the artifacts under
`experiments/g3_bsta_lite/baseline_freeze/` (BASELINE_FREEZE.json +
paired_raw_rows.json + neighbor_sweep.json + F2_PHASE_REPORT.md) and
that branch tip still matches `9873697`.

**Key numbers** (128 paired scenarios × 4 action-replicates):
- Oracle drop = 0.3313, witness drop = 0.2680, best non-witness baseline
  (budgeted_round_robin) = 0.1601.
- Criterion 1 oracle headroom = 17.12 pp (≥10 pp).
- Criterion 2 witness min LCB95 = 9.13 pp vs every non-witness baseline
  (≥7.5 pp).
- Criterion 3 neighbor sweep: 7 cells, min LCB 7.50 pp (≥5 pp).

**Critical repair**: `EnvConfig.obs_delay_steps` default changed 2 → 0.
Why: with delay=2 the witness could not react to fresh arrivals inside
`mission_tau_window=6` (always missed the first matched scan of newly-
admitted missions), structurally capping it near `budgeted_round_robin`
and failing Gate 1 criterion 2 (LCB only 0.7 pp). The MODIFICATION_PLAN
explicitly lists "oracle has gap but causal witness does not: repair
causal information/history" as the remedy. delay=0 keeps the channel
causal — only past/present observables, no future-leak — verified by
`test_causal_observation` and the godview-leak tests. All 46 F0+F1
contract tests still pass.

**Interpretation note**: Witness is one of the 6 frozen baselines per
contract §10, but for Gate 1 evaluation it is treated as the policy
under test. `evaluation.py` computes oracle headroom against the best
NON-WITNESS baseline so a strong witness doesn't artificially shrink
the apparent ML headroom. The oracle-vs-witness gap itself is 6.32 pp,
which is the headroom remaining for a learned policy above the witness.
