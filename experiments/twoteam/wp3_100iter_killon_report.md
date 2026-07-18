# WP2 League Training Report

**Started**: 2026-07-18 00:28:01

## Setup

- n_iters: 100
- snapshot_every: 50
- pfsp_hardness: 1.0
- bc_samples: 20000, bc_epochs: 8
- ppo_lr_actor: 0.0001, entropy_coef: 0.01
- horizon: 300, n_envs: 64
- ckpt_dir: `checkpoints/blind/wp3_100iter_killon_20260718_002800`

## Training health

- final reward: -0.020  (initial: +0.000)
- final entropy: -1.518
- final adv_std: 1.000  (healthy range [0.1, 100])
- final pool EMA variance: 0.066  (low → PFSP may be stuck)
- total elapsed: 136.0 min

## Per-iter log (every 10 iters)

| iter | opp | reward | adv_std | entropy | kl | wr_vs_opp | ema_var | t(min) |
|---|---|---|---|---|---|---|---|---|
| 0 | `exploit/hard_jam_focus` | +0.000 | 1.00 | -1.632 | 0.2197 | 0.50 | 0.000 | 0.4 |
| 10 | `extreme/pure_track` | +0.004 | 1.00 | -1.585 | 0.0513 | 0.53 | 0.145 | 13.7 |
| 20 | `exploit/jam_spread` | -0.007 | 1.00 | -1.565 | 0.0525 | 0.62 | 0.169 | 27.3 |
| 30 | `extreme/balanced` | +0.002 | 1.00 | -1.518 | 0.0303 | 0.61 | 0.120 | 41.1 |
| 40 | `blind_classical` | -0.051 | 1.00 | -1.476 | 0.0537 | 0.39 | 0.101 | 54.9 |
| 50 | `blind_classical` | -0.075 | 1.00 | -1.522 | 0.0633 | 0.34 | 0.086 | 68.6 |
| 60 | `exploit/jam_spread` | -0.042 | 1.00 | -1.512 | 0.0304 | 0.59 | 0.075 | 82.6 |
| 70 | `strong_rule` | +1.401 | 1.00 | -1.547 | 0.0781 | 0.56 | 0.071 | 96.4 |
| 80 | `exploit/jam_spread` | -0.014 | 1.00 | -1.527 | 0.0309 | 0.65 | 0.069 | 110.2 |
| 90 | `strong_rule` | +1.359 | 1.00 | -1.529 | 0.1005 | 0.62 | 0.067 | 123.8 |

## Pool final state

| name | kind | is_self | win_rate_vs_current | games |
|---|---|---|---|---|
| `strong_rule` | rule | False | 0.865 | 20 |
| `blind_classical` | rule | False | 0.000 | 33 |
| `extreme/pure_track` | extreme | False | 1.000 | 1 |
| `extreme/pure_jam` | extreme | False | 1.000 | 1 |
| `extreme/pure_comm` | extreme | False | 1.000 | 1 |
| `extreme/pure_detect` | extreme | False | 1.000 | 1 |
| `extreme/balanced` | extreme | False | 0.850 | 19 |
| `extreme/balanced_jam_heavy` | extreme | False | 1.000 | 1 |
| `extreme/track_agile` | extreme | False | 1.000 | 1 |
| `exploit/jam_spread` | script | False | 0.794 | 18 |
| `exploit/hard_jam_focus` | script | False | 1.000 | 1 |
| `exploit/track_heavy_agile` | script | False | 1.000 | 1 |
| `self/iter000_bc` | checkpoint | True | 1.000 | 1 |
| `self/iter050` | checkpoint | True | 1.000 | 1 |
| `self/iter100` | checkpoint | True | — | 0 |

## Notes

- Pool win_rate_vs_current = EMA of how often the *current* training AC beats this opponent (draw counts as 0.5). Low values → hard opponents → PFSP samples them more.
- self/iterNNN snapshots are added to the pool as the league evolves, enabling population-level diversity.
- Pool EMA variance near 0 → PFSP degenerating toward uniform; investigate.
- Run `run_wp2_crossplay.py` next for cross-play Elo + non-transitivity detection.
