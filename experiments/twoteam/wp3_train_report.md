# WP2 League Training Report

**Started**: 2026-07-18 09:08:03

## Setup

- n_iters: 100
- snapshot_every: 50
- pfsp_hardness: 1.0
- bc_samples: 20000, bc_epochs: 8
- ppo_lr_actor: 0.0001, entropy_coef: 0.01
- horizon: 300, n_envs: 64
- ckpt_dir: `checkpoints/blind/wp3_20260718_090802`

## Training health

- final reward: +0.067  (initial: -0.000)
- final entropy: -1.387
- final adv_std: 1.000  (healthy range [0.1, 100])
- final pool EMA variance: 0.073  (low → PFSP may be stuck)
- total elapsed: 138.9 min

## Per-iter log (every 10 iters)

| iter | opp | reward | adv_std | entropy | kl | wr_vs_opp | ema_var | t(min) |
|---|---|---|---|---|---|---|---|---|
| 0 | `exploit/hard_jam_focus` | -0.000 | 1.00 | -1.629 | 0.1253 | 0.50 | 0.000 | 0.4 |
| 10 | `extreme/track_agile` | +0.003 | 1.00 | -1.591 | 0.0499 | 0.51 | 0.188 | 14.3 |
| 20 | `blind_classical` | -0.053 | 1.00 | -1.519 | 0.4409 | 0.35 | 0.130 | 28.1 |
| 30 | `blind_classical` | -0.098 | 1.00 | -1.478 | 0.2202 | 0.36 | 0.130 | 41.9 |
| 40 | `exploit/jam_spread` | -0.188 | 1.00 | -1.441 | 0.0604 | 0.61 | 0.094 | 55.8 |
| 50 | `self/iter050` | +0.029 | 1.00 | -1.413 | 0.0643 | 0.51 | 0.083 | 69.8 |
| 60 | `extreme/track_agile` | -0.165 | 1.00 | -1.341 | 0.0162 | 0.59 | 0.083 | 83.8 |
| 70 | `blind_classical` | +0.105 | 1.00 | -1.389 | 0.0283 | 0.38 | 0.083 | 97.5 |
| 80 | `blind_classical` | +0.063 | 1.00 | -1.383 | 0.0797 | 0.35 | 0.082 | 111.5 |
| 90 | `self/iter050` | +0.135 | 1.00 | -1.387 | 0.0252 | 0.53 | 0.076 | 125.7 |

## Pool final state

| name | kind | is_self | win_rate_vs_current | games |
|---|---|---|---|---|
| `strong_rule` | rule | False | 1.000 | 2 |
| `blind_classical` | rule | False | 0.000 | 32 |
| `extreme/pure_track` | extreme | False | 1.000 | 6 |
| `extreme/pure_jam` | extreme | False | 1.000 | 1 |
| `extreme/pure_comm` | extreme | False | 1.000 | 4 |
| `extreme/pure_detect` | extreme | False | 1.000 | 3 |
| `extreme/balanced` | extreme | False | 1.000 | 2 |
| `extreme/balanced_jam_heavy` | extreme | False | 1.000 | 2 |
| `extreme/track_agile` | extreme | False | 1.000 | 3 |
| `exploit/jam_spread` | script | False | 0.570 | 13 |
| `exploit/hard_jam_focus` | script | False | 1.000 | 5 |
| `exploit/track_heavy_agile` | script | False | 1.000 | 4 |
| `self/iter000_bc` | checkpoint | True | 1.000 | 1 |
| `self/iter050` | checkpoint | True | 0.919 | 4 |
| `self/iter100` | checkpoint | True | — | 0 |

## Notes

- Pool win_rate_vs_current = EMA of how often the *current* training AC beats this opponent (draw counts as 0.5). Low values → hard opponents → PFSP samples them more.
- self/iterNNN snapshots are added to the pool as the league evolves, enabling population-level diversity.
- Pool EMA variance near 0 → PFSP degenerating toward uniform; investigate.
- Run `run_wp2_crossplay.py` next for cross-play Elo + non-transitivity detection.
