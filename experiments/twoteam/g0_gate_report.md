# G0 Exploitability Gate Report — Two-team WP1

**Date**: 2026-07-13
**Spec**: TWOTEAM_MULTIFUNCTION_PLAN.md §WP1.2
**Elapsed**: 6.8 min
**Overall**: ❌ FAIL — diagnose before WP2

## Anti-strawman check (StrongRule vs ExtremeCommanders)

Verdict: **TOO_WEAK**

| opponent | win_rate | draw_rate | kills_rule | kills_opp |
|---|---|---|---|---|
| pure_track | 1.00 | 0.00 | 1.98 | 0.97 |
| pure_jam | 0.15 | 0.85 | 0.15 | 0.00 |
| pure_comm | 1.00 | 0.00 | 2.00 | 0.00 |
| pure_detect | 1.00 | 0.00 | 2.00 | 0.00 |
| balanced | 0.20 | 0.80 | 0.20 | 0.00 |
| balanced_jam_heavy | 0.25 | 0.75 | 0.25 | 0.00 |

## Cell 1: π_rule vs π_rule (mirror)

- n_episodes × n_envs = 30 × 8 = 240
- kills_t0 mean: 0.000
- kills_t1 mean: 0.000
- margin (t0−t1): +0.000 (95% CI [+0.000, +0.000])
- winner dist: t0=0.00, t1=0.00, draw=1.00

## BR training

- iters: 200, horizon: 200, n_envs: 8
- lr_actor=0.0003, lr_critic=0.001, entropy_coef=0.01
- final reward_mean: 0.7390772700309753
- final adv_std: 0.9999999403953552
- final entropy: -0.7609228968620301
- final approx_kl: 0.06744117502123118
- checkpoint: `/home/ubuntu/CODE/FluxPhased-/checkpoints/twoteam/br_vs_strong_rule_final.pt`
- training log: `g0_br_training_log.csv`

## Cell 2: π_rule vs BR(π_rule)

- n_episodes × n_envs = 30 × 8 = 240
- kills_rule mean: 1.096
- kills_BR mean: 1.000
- margin (rule POV): +0.096 (95% CI [+0.058, +0.133])
- winner dist: rule=0.23, BR=0.03, draw=0.74

## Exploitability

exploit_gap = mean(mirror_margin) − mean(br_margin)

- **exploit_gap = -0.096** (95% CI [-0.133, -0.058])
- BR win rate vs rule: 0.03
- BR healthy: True

## Verdict

| check | threshold | actual | pass |
|---|---|---|---|
| exploit_gap ≥ 0.5 | 0.500 | -0.096 | ❌ |
| CI excludes 0 | > 0 | -0.133 | ❌ |
| BR win rate ≥ 0.55 | 0.55 | 0.03 | ❌ |
| BR healthy | adv_std∈[0.1,100], no NaN | 0.9999999403953552 | ✅ |

❌ **G0 FAIL** — StrongRule is NOT exploitable (or BR undertrained).
**Diagnose before any WP2 self-play burn:**
- CI includes 0 → exploit not statistically significant. Try more episodes or longer BR training.
- BR win rate 0.03 < 0.55 → BR didn't learn to beat rule. Check BR training curves for collapse / undertraining.

**If diagnosis confirms rule genuinely not exploitable** → root A present.
**→ Recommend retreat to IET (C0+C1 IQ/CRLB baseline paper).**
