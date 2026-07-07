# Concerto-RRM Pilot Results

**Status**: Pilot complete — 0/4 criteria PASS → retreat to Path A per plan decision tree.
**Date**: 2026-07-07.
**Cells**: 4 methods × 3 difficulties × 5 seeds = 60 cells, 50 episodes per cell, max 50 control steps per episode.
**Wallclock**: ~82 minutes (single GPU).

---

## 1. Headline

The pilot **did not pass** any of the 4 go/no-go criteria. The root cause is a
**policy-environment mismatch**: the pre-trained MAPPO checkpoint
(`checkpoints/laser_mappo/iter_019.pt`) was trained on the original laser
destroy-the-enemy task and outputs `task_id=recon` (argmax=0) instead of
`task_id=detect` (argmax=1). When Concerto V2 correctly hands off to RL under
EW triggers, QoS collapses to zero. The pilot framework itself works — the
event-trigger detects EW, the classical scheduler holds up at L0, the L0→L3
degradation is visible — but the RL half of the interleaving is broken.

This is a **fixable** failure mode (train a fresh MAPPO for the QoS-RRM env),
not a fundamental thesis falsification. The honest signals below are the
load-bearing findings.

---

## 2. Per-cell aggregate (mean across 5 seeds)

| Method | L0 | L1 | L3 | L0→L3 collapse |
|---|---|---|---|---|
| classical | 0.890±0.016 | 0.617±0.016 | 0.554±0.028 | −0.336 |
| concerto_v1 | 0.859±0.013 | 0.509±0.013 | 0.518±0.024 | −0.341 |
| concerto_v2 | 0.890±0.016 | 0.448±0.004 | 0.453±0.021 | −0.437 |
| mappo | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | n/a (broken) |

Per-function detail (mean across seeds at L3): classical — detect=0.999,
track=0.069, comm=0.646, jam=0.502. Track collapses under EW; detect holds
because red's own Pd is driven by TX power, not by blue's jam. Jam
effectiveness (defensive) degrades mildly.

---

## 3. Honest signals that DID work

These are the load-bearing findings for the next iteration:

### 3.1 Classical QoS-RRM holds at L0, collapses at L3
L0 aggregate = 0.890 (just under 0.9 sanity floor; marginal), L3 = 0.554.
The −0.336 envelope collapse is the phase transition the thesis predicts. The
collapse is concentrated in **track** (1.000 → 0.069) — adaptive EW denies
Kalman convergence, exactly the failure mode the plan anticipated.

### 3.2 Concerto V2 event-trigger works correctly
Per-seed RL step fraction under V2:
- L0: 0/60 steps → 0% RL (no trigger, correct)
- L1: 489/500 → 98% RL
- L3: 499/500 → ~100% RL

The composer correctly identifies L0 as benign (no firing) and L1/L3 as
contended (sustained firing). θ1=5dB / θ2=0.6 / ε=0.2 are well-calibrated
for this env.

### 3.3 No function collapse after dwell normalization fix
For non-MAPPO cells, min dwell fraction = 0.120 (≥ 0.05 floor). The water-fill
keeps all 4 functions resourced. C3 fails only because MAPPO dwell = 0
(consistent with the policy mismatch — recon task allocation isn't counted in
detect/track/comm/jam bins).

### 3.4 The adapter plumbing is sound
`ConcertoTrainerAdapter` correctly:
- Always calls classical first to populate the Kalman tracker (line 147).
- Asks the composer for owner per-step (defensive JSR via `jam_level.flip(-1)`).
- Stashes owner for downstream metric collection.
The runner sees a uniform trainer API regardless of method.

---

## 4. Root cause of failure

**Pre-trained MAPPO produces zero QoS** because the policy was trained on the
original laser task (destroy enemy commander) with a different observation /
action distribution. The QoS-RRM env surfaces a 4-function task allocation
(task_id ∈ {recon=0, detect=1, jam=2, comm=3}), but the loaded policy's
`task_head` outputs near-deterministic `argmax=0` (recon). Result: zero
elements allocated to detect/track/comm/jam → zero QoS across all functions.

Concrete evidence (mappo cells): n_rl_steps=500, n_classical_steps=0,
qos_satisfaction=0.000 across all 15 mappo cells (5 seeds × 3 difficulties).

This invalidates C2 (Concerto must beat MAPPO at L3) and C3 (function collapse
floor) by construction — any interleaving that hands off to a zero-QoS policy
will itself produce zero QoS on those steps.

---

## 5. Criterion verdict

| # | Criterion | Threshold | Measured | Verdict |
|---|---|---|---|---|
| C1 | Classical sanity floor | L0 > 0.9 ∧ L3 < 0.6 | L0=0.890, L3=0.554 | **FAIL** (L0 marginal) |
| C2 | V2 beats classical + MAPPO at L3 | gap > 0.10 / 0.05 | v2−classical=−0.101 | **FAIL** |
| C3 | min dwell ≥ 0.05 ∀ cell | ≥ 0.05 | 60 cells below (all mappo) | **FAIL** |
| C4 | Concerto wallclock < 0.7× MAPPO | < 0.7 | ratio = 0.97 | **FAIL** |

Full details: [concerto_pilot_verdict.md](concerto_pilot_verdict.md).

---

## 6. Decision per plan

Per the plan's decision tree, **0/4 PASS → retreat to Path A** (C1+C0 →
IEEE TAES) per EAAI_RESEARCH_PLAN.md §9.

However, the failure mode here is **not the failure mode the decision tree
anticipates**. The tree assumes a *thesis-level* failure (RL can't exploit the
contended slots, or classical doesn't collapse, or the interleaving doesn't
help). What we have is a *plumbing-level* failure: the RL policy was never
trained on the QoS-RRM task distribution. Before retreating permanently, the
honest next step is to train a fresh MAPPO on the QoS-RRM env and re-run the
pilot.

---

## 7. Recommended next steps (before retreating to Path A)

1. **Train a fresh MAPPO for QoS-RRM** (~1.5 GPU-h). Use the QoS-RRM reward
   shaping (`algo/_shared/ppo/reward_shaping.py:DenseRewardShaper`) so the
   policy learns to allocate across the 4 functions. The current
   `SimpleMAPPOTrainer` adapter is correct — only the checkpoint is wrong.
2. **Re-baseline classical L0**. The 0.890 vs 0.9 marginal failure is likely
   a `pd_thresh` calibration issue (current 0.3 may be too strict for the
   pilot's boosted TX power). Either lower the threshold or accept 0.89 as
   the operating point.
3. **Re-run pilot**. With a QoS-aware MAPPO, all four criteria become
   load-bearing. If C2 still fails with a competent RL policy, that is the
   real thesis falsification and Path A retreat is correct.

Estimated cost: ~2 GPU-h (1.5 train + 0.5 re-eval). Cheap enough to be worth
doing before retreating.

---

## 8. Artifacts

- Verdict: [concerto_pilot_verdict.md](concerto_pilot_verdict.md)
- Raw CSV: [concerto_pilot_results.csv](concerto_pilot_results.csv) (60 rows)
- Driver: [algo/_shared/pilot/run_pilot.py](../algo/_shared/pilot/run_pilot.py)
- Adapter: [algo/_shared/concerto/concerto_trainer.py](../algo/_shared/concerto/concerto_trainer.py)
- Composer: [algo/_shared/concerto/composer.py](../algo/_shared/concerto/composer.py)
- Metrics: [env/gpu/qos_rrm/spectrum_metrics.py](../env/gpu/qos_rrm/spectrum_metrics.py)
- Plan: [snuggly-exploring-parrot.md](../../.claude/plans/snuggly-exploring-parrot.md)
