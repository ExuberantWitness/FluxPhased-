# G0 Exploitability Gate Report — Two-team WP1

**Date**: 2026-07-13
**Spec**: TWOTEAM_MULTIFUNCTION_PLAN.md §WP1.2
**Elapsed**: 13.1 min
**Overall**: ❌ FAIL — diagnose before WP2

## Anti-strawman check (StrongRule vs ExtremeCommanders)

Verdict: **TOO_WEAK**

| opponent | win_rate | draw_rate | kills_rule | kills_opp |
|---|---|---|---|---|
| pure_track | 0.97 | 0.03 | 1.98 | 1.00 |
| pure_jam | 0.33 | 0.68 | 0.33 | 0.00 |
| pure_comm | 1.00 | 0.00 | 2.00 | 0.00 |
| pure_detect | 1.00 | 0.00 | 1.95 | 0.00 |
| balanced | 1.00 | 0.00 | 2.00 | 0.97 |
| balanced_jam_heavy | 0.95 | 0.05 | 1.95 | 1.00 |

## Cell 1: π_rule vs π_rule (mirror)

- n_episodes × n_envs = 30 × 8 = 240
- kills_t0 mean: 1.946
- kills_t1 mean: 1.946
- margin (t0−t1): +0.000 (95% CI [+0.000, +0.000])
- winner dist: t0=0.00, t1=0.00, draw=1.00

## BC pretrain (AlphaStar SL → RL paradigm)

- samples collected: 50000
- epochs: 15, batch_size: 256, lr: 0.001
- final train_loss: -28.921
- final val_loss:   -29.012
- checkpoint: `/home/ubuntu/CODE/FluxPhased-/checkpoints/twoteam/bc_pretrained.pt`

| epoch | train_loss | val_loss |
|---|---|---|
| 1 | -16.296 | -20.652 |
| 2 | -22.175 | -23.242 |
| 3 | -23.881 | -24.388 |
| 4 | -24.833 | -25.214 |
| 5 | -25.509 | -25.797 |
| 6 | -26.072 | -26.296 |
| 7 | -26.531 | -26.733 |
| 8 | -26.940 | -27.119 |
| 9 | -27.302 | -27.470 |
| 10 | -27.633 | -27.787 |
| 11 | -27.927 | -28.067 |
| 12 | -28.198 | -28.332 |
| 13 | -28.457 | -28.572 |
| 14 | -28.694 | -28.801 |
| 15 | -28.921 | -29.012 |

## BR training

- iters: 500, horizon: 200, n_envs: 8
- lr_actor=0.0001, lr_critic=0.001, entropy_coef=0.01
- final reward_mean: -0.049099817872047424
- final adv_std: 0.9999998807907104
- final entropy: -2.3740459156036375
- final approx_kl: 0.09463058352470398
- checkpoint: `/home/ubuntu/CODE/FluxPhased-/checkpoints/twoteam/br_vs_strong_rule_final.pt`
- training log: `g0_br_training_log.csv`

## Cell 2: π_rule vs BR(π_rule)

- n_episodes × n_envs = 30 × 8 = 240
- kills_rule mean: 1.950
- kills_BR mean: 1.933
- margin (rule POV): +0.017 (95% CI [-0.017, +0.050])
- winner dist: rule=0.05, BR=0.03, draw=0.93

## Exploitability

exploit_gap = mean(mirror_margin) − mean(br_margin)

- **exploit_gap = -0.016** (95% CI [-0.050, +0.017])
- BR win rate vs rule: 0.03
- BR healthy: True

## Verdict

| check | threshold | actual | pass |
|---|---|---|---|
| exploit_gap ≥ 0.5 | 0.500 | -0.016 | ❌ |
| CI excludes 0 | > 0 | -0.050 | ❌ |
| BR win rate ≥ 0.55 | 0.55 | 0.03 | ❌ |
| BR healthy | adv_std∈[0.1,100], no NaN | 0.9999998807907104 | ✅ |

❌ **G0 FAIL** — StrongRule is NOT exploitable (or BR undertrained).
**Diagnose before any WP2 self-play burn:**
- CI includes 0 → exploit not statistically significant. Try more episodes or longer BR training.
- BR win rate 0.03 < 0.55 → BR didn't learn to beat rule. Check BR training curves for collapse / undertraining.

**If diagnosis confirms rule genuinely not exploitable** → root A present.
**→ Recommend retreat to IET (C0+C1 IQ/CRLB baseline paper).**
