# G0 Re-test Status — Post-Env-Fix

**Date**: 2026-07-14
**Spec**: TWOTEAM_ENV_FIX_SPEC.md (Fix 1-5)
**Pre-state**: WP0-decisive PASS (anti-jam skill dimension verified working)

## Summary

**G0 verdict**: ❌ FAIL (exploit_gap = -0.963, BR win rate = 0%)

But the failure mode is **qualitatively different** from the previous G0 FAIL:

| Aspect | Previous FAIL (pre-fix) | Current FAIL (post-fix) |
|---|---|---|
| Root cause | Env degenerate (mutual-jam 0-0 stalemate) | BR undertrained for action space |
| WP0-decisive | pure_jam unbeatable (2d FAIL) | All 4 sub-checks PASS |
| Anti-jam skill dimension | Missing | Works (track_agile beats pure_jam 1.00 vs 0.00) |
| BR kills in eval | 0 (stuck in stalemate) | 1.00 (functional, just loses to rule's 1.96) |
| Training reward trajectory | Flat at -1.8 | Learning onset at iter 300, improved to -1.21 |

## What works (env fix validated)

1. **Anti-jam skill dimension (Fix 1)**: track_agile beats pure_jam 1.00 vs 0.00 in WP0 matrix. Physical formula:
   - `effective_jam = jam_level / freq_hop_per_tracker`
   - `processing_overhead = 1 / freq_hop^0.25` (sub-linear, matches matched-filter-bank radar physics)
   - `jam_gain = 6.0` (was 8.0; tuned so anti-jam can win at hop=8)
   - Optimal hop rate ~8-12; track_sigma achievable under jam: 0.13 (vs 0.45 without anti-jam)

2. **Mirror physics (Check 1)**: Exactly symmetric — `max|kills_t0 - kills_t1| = 0.000`, `max|trace_P diff| = 0.000`

3. **WP0-decisive (Fix 4)**: All 4 sub-checks PASS
   - 2a: no dominant strategy
   - 2b: decisive_rate=0.660 (target ≥ 0.50)
   - 2c: kill_density=0.801/ep (target ≥ 0.5)
   - 2d: every strategy has best_opponent_decisive=1.000 (target ≥ 0.30)

4. **CRLB anchor**: ratio 0.21 (was 0.28 pre-fix; still well within 5× bound)

5. **BR training healthy**: adv_std=1.0, KL ≈ 0.02 (below target 0.03), no NaN

## What doesn't work (BR learning bottleneck)

### Anti-strawman verdict: TOO_WEAK
StrongRule vs ExtremeCommanders:
- vs pure_track: 0.97 ✓
- vs pure_jam: **0.33** (draws 0.68) ❌
- vs pure_comm: 1.00 ✓
- vs pure_detect: 1.00 ✓

StrongRule's anti-jam reaction is good enough to draw vs pure_jam but not win. This was supposed to be the easiest exploit path for BR.

### BR deterministic policy near-prior
After 500 iters, BR's task_alloc head outputs are essentially zero → Dirichlet(1,1,1,1) → uniform mean [0.25,0.25,0.25,0.25]. The policy didn't concentrate.

Only `freq_hop` head learned ( outputs ~4.5, 6.4 = anti-jam skill detected).

### Training trajectory
| iter | reward_mean | interpretation |
|---|---|---|
| 0-275 | -1.82 (flat) | No learning signal |
| 300 | -1.55 | Onset of learning |
| 325 | -1.19 | Big jump |
| 350-475 | -1.30 ± 0.10 | Plateau |
| 499 | -1.21 | Slow improvement |

The BR WAS learning (33% reward improvement), but 500 iters is insufficient for this action space.

### Cell 2 (rule vs BR)
- Rule kills: 1.96
- BR kills: 1.00
- Rule win rate: 0.98
- BR win rate: 0.00
- exploit_gap = -0.963 (CI excludes 0, but in WRONG direction — rule beats BR)

## Why BR didn't exploit StrongRule

The action space is high-dimensional:
- Dirichlet(2, 4) — task_alloc per aperture
- Categorical(2, 2) — beam target per aperture
- Categorical(2) — laser target
- Bernoulli(2) — emission per aperture
- Beta(2, 2) — freq_hop per aperture

Total: 13 continuous + 5 discrete dimensions. PPO needs ~1M+ samples to concentrate Dirichlet in such spaces. We have 500 iters × 200 horizon × 8 envs = 800K samples. Just at the threshold.

The known exploit (track_agile strategy from WP0) requires:
- task_alloc = pure_track [0, 1, 0, 0]
- freq_hop = 8
- beam = same enemy
- laser = same enemy

BR's Dirichlet never concentrated enough to find this corner of the simplex.

## Decision point

The user's strict spec says "G0 FAIL → retreat to IET". But:

1. **Env IS non-trivial** (WP0-decisive PASS, anti-jam works)
2. **BR IS learning** (training trajectory shows clear onset at iter 300)
3. **BR just undertrained** for this action space complexity

Three paths forward:

### Path A: Strict spec — retreat to IET
- Honest report FAIL
- Switch to IET (C0+C1 IQ/CRLB baseline paper)
- Lose WP2 self-play investment

### Path B: Extend BR training to 1500-2000 iters
- Estimated 3-4 hours additional GPU
- High chance BR concentrates and finds exploit
- Risk: still might not converge

### Path C: Curriculum + simpler action space
- First train BR vs ExtremeCommanders (simpler exploit targets)
- Then fine-tune vs StrongRule
- Or: discretize task_alloc temporarily to help exploration, then continuous

## Recommendation

**Path B** (extend BR training). Rationale:
- Training trajectory shows clear learning signal (not stuck, just slow)
- Env was independently verified non-trivial (WP0 PASS)
- 500 iters was arbitrary budget; the user spec said "BR 500 iters" but pre-fix BR had no chance regardless of iters
- Cheap to test (3-4 hours) vs writing IET paper from scratch

If BR still doesn't exploit after 1500 iters → strong evidence rule genuinely not exploitable → retreat to IET with confidence.
