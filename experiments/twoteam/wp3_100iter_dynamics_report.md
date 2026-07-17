# WP2 League Training Report

**Started**: 2026-07-17 08:49:38

## Setup

- n_iters: 100
- snapshot_every: 25
- pfsp_hardness: 1.0
- bc_samples: 20000, bc_epochs: 8
- ppo_lr_actor: 0.0001, entropy_coef: 0.01
- horizon: 200, n_envs: 64
- ckpt_dir: `checkpoints/blind/wp3_100iter_dynamics`

## Training health

- final reward: -0.024  (initial: -0.001)
- final entropy: -1.383
- final adv_std: 1.000  (healthy range [0.1, 100])
- final pool EMA variance: 0.074  (low → PFSP may be stuck)
- total elapsed: 50.4 min

## Per-iter log (every 5 iters)

| iter | opp | reward | adv_std | entropy | kl | wr_vs_opp | ema_var | t(min) |
|---|---|---|---|---|---|---|---|---|
| 0 | `exploit/hard_jam_focus` | -0.001 | 1.00 | -1.594 | 0.0467 | 0.50 | 0.000 | 0.3 |
| 5 | `self/iter000_bc` | -0.001 | 1.00 | -1.577 | 0.0402 | 0.50 | 0.000 | 3.2 |
| 10 | `extreme/pure_track` | +0.003 | 1.00 | -1.577 | 0.0130 | 0.56 | 0.083 | 6.1 |
| 15 | `exploit/jam_spread` | -0.005 | 1.00 | -1.579 | 0.0232 | 0.50 | 0.118 | 8.6 |
| 20 | `blind_classical` | -0.060 | 1.00 | -1.555 | 0.0259 | 0.47 | 0.109 | 11.1 |
| 25 | `exploit/jam_spread` | -0.004 | 1.00 | -1.551 | 0.0366 | 0.50 | 0.104 | 13.6 |
| 30 | `exploit/jam_spread` | -0.008 | 1.00 | -1.550 | 0.0500 | 0.44 | 0.094 | 16.2 |
| 35 | `blind_classical` | -0.038 | 1.00 | -1.515 | 0.0237 | 0.47 | 0.098 | 18.7 |
| 40 | `exploit/jam_spread` | -0.006 | 1.00 | -1.559 | 0.0676 | 0.50 | 0.076 | 21.2 |
| 45 | `blind_classical` | -0.003 | 1.00 | -1.488 | 0.0411 | 0.38 | 0.077 | 23.7 |
| 50 | `exploit/jam_spread` | -0.006 | 1.00 | -1.544 | 0.0173 | 0.50 | 0.079 | 26.2 |
| 55 | `exploit/jam_spread` | -0.003 | 1.00 | -1.547 | 0.0176 | 0.50 | 0.054 | 28.8 |
| 60 | `blind_classical` | -0.063 | 1.00 | -1.409 | 0.0472 | 0.53 | 0.050 | 31.3 |
| 65 | `exploit/jam_spread` | -0.004 | 1.00 | -1.526 | 0.0272 | 0.47 | 0.055 | 33.8 |
| 70 | `blind_classical` | -0.055 | 1.00 | -1.370 | 0.0544 | 0.50 | 0.051 | 36.2 |
| 75 | `exploit/jam_spread` | -0.002 | 1.00 | -1.504 | 0.0248 | 0.47 | 0.064 | 38.7 |
| 80 | `exploit/jam_spread` | -0.003 | 1.00 | -1.507 | 0.0218 | 0.47 | 0.068 | 41.1 |
| 85 | `blind_classical` | -0.039 | 1.00 | -1.382 | 0.0568 | 0.38 | 0.069 | 43.5 |
| 90 | `exploit/jam_spread` | -0.009 | 1.00 | -1.510 | 0.0192 | 0.41 | 0.078 | 46.0 |
| 95 | `blind_classical` | -0.024 | 1.00 | -1.391 | 0.0797 | 0.44 | 0.067 | 48.4 |

## Pool final state

| name | kind | is_self | win_rate_vs_current | games |
|---|---|---|---|---|
| `strong_rule` | rule | False | 1.000 | 1 |
| `blind_classical` | rule | False | 0.149 | 43 |
| `extreme/pure_track` | extreme | False | 1.000 | 1 |
| `extreme/pure_jam` | extreme | False | 1.000 | 1 |
| `extreme/pure_comm` | extreme | False | 1.000 | 1 |
| `extreme/pure_detect` | extreme | False | 1.000 | 1 |
| `extreme/balanced` | extreme | False | 1.000 | 1 |
| `extreme/balanced_jam_heavy` | extreme | False | 1.000 | 1 |
| `extreme/track_agile` | extreme | False | 1.000 | 1 |
| `exploit/jam_spread` | script | False | 0.201 | 43 |
| `exploit/hard_jam_focus` | script | False | 1.000 | 1 |
| `exploit/track_heavy_agile` | script | False | 1.000 | 1 |
| `self/iter000_bc` | checkpoint | True | 1.000 | 1 |
| `self/iter025` | checkpoint | True | 1.000 | 1 |
| `self/iter050` | checkpoint | True | 1.000 | 1 |
| `self/iter075` | checkpoint | True | 1.000 | 1 |
| `self/iter100` | checkpoint | True | — | 0 |

## Notes

- Pool win_rate_vs_current = EMA of how often the *current* training AC beats this opponent (draw counts as 0.5). Low values → hard opponents → PFSP samples them more.
- self/iterNNN snapshots are added to the pool as the league evolves, enabling population-level diversity.
- Pool EMA variance near 0 → PFSP degenerating toward uniform; investigate.
- Run `run_wp2_crossplay.py` next for cross-play Elo + non-transitivity detection.
