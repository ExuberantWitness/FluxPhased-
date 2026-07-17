# WP2 League Training Report

**Started**: 2026-07-17 10:53:53

## Setup

- n_iters: 100
- snapshot_every: 25
- pfsp_hardness: 1.0
- bc_samples: 20000, bc_epochs: 8
- ppo_lr_actor: 0.0001, entropy_coef: 0.01
- horizon: 200, n_envs: 64
- ckpt_dir: `checkpoints/blind/wp3_100iter_shaped`

## Training health

- final reward: -0.144  (initial: -0.114)
- final entropy: -1.391
- final adv_std: 1.000  (healthy range [0.1, 100])
- final pool EMA variance: 0.052  (low → PFSP may be stuck)
- total elapsed: 57.7 min

## Per-iter log (every 5 iters)

| iter | opp | reward | adv_std | entropy | kl | wr_vs_opp | ema_var | t(min) |
|---|---|---|---|---|---|---|---|---|
| 0 | `exploit/hard_jam_focus` | -0.114 | 1.00 | -1.601 | 0.0900 | 0.50 | 0.000 | 0.3 |
| 5 | `self/iter000_bc` | -0.101 | 1.00 | -1.475 | 0.0184 | 0.50 | 0.000 | 3.1 |
| 10 | `extreme/pure_track` | -0.100 | 1.00 | -1.442 | 0.0121 | 0.53 | 0.083 | 5.8 |
| 15 | `blind_classical` | -0.133 | 1.00 | -1.398 | 0.0222 | 0.47 | 0.071 | 8.5 |
| 20 | `exploit/hard_jam_focus` | -0.249 | 1.00 | -1.389 | 0.0119 | 0.50 | 0.071 | 11.5 |
| 25 | `blind_classical` | -0.122 | 1.00 | -1.376 | 0.0472 | 0.41 | 0.071 | 14.3 |
| 30 | `exploit/hard_jam_focus` | -0.245 | 1.00 | -1.370 | 0.0144 | 0.50 | 0.066 | 17.3 |
| 35 | `blind_classical` | -0.112 | 1.00 | -1.378 | 0.0251 | 0.41 | 0.066 | 20.1 |
| 40 | `extreme/balanced` | -0.189 | 1.00 | -1.409 | 0.0199 | 0.47 | 0.066 | 23.0 |
| 45 | `blind_classical` | -0.161 | 1.00 | -1.405 | 0.0330 | 0.38 | 0.067 | 26.0 |
| 50 | `self/iter050` | -0.093 | 1.00 | -1.415 | 0.0191 | 0.50 | 0.062 | 28.9 |
| 55 | `blind_classical` | -0.139 | 1.00 | -1.439 | 0.0643 | 0.41 | 0.062 | 31.8 |
| 60 | `extreme/balanced` | -0.121 | 1.00 | -1.411 | 0.0188 | 0.50 | 0.062 | 34.7 |
| 65 | `blind_classical` | -0.114 | 1.00 | -1.383 | 0.0241 | 0.38 | 0.062 | 37.7 |
| 70 | `extreme/balanced` | -0.175 | 1.00 | -1.396 | 0.0178 | 0.50 | 0.054 | 40.6 |
| 75 | `blind_classical` | -0.143 | 1.00 | -1.390 | 0.0255 | 0.44 | 0.049 | 43.6 |
| 80 | `extreme/balanced` | -0.139 | 1.00 | -1.390 | 0.0162 | 0.50 | 0.050 | 46.5 |
| 85 | `blind_classical` | -0.170 | 1.00 | -1.408 | 0.0389 | 0.44 | 0.042 | 49.4 |
| 90 | `extreme/balanced` | -0.100 | 1.00 | -1.416 | 0.0172 | 0.44 | 0.037 | 52.4 |
| 95 | `blind_classical` | -0.138 | 1.00 | -1.376 | 0.0347 | 0.47 | 0.048 | 55.3 |

## Pool final state

| name | kind | is_self | win_rate_vs_current | games |
|---|---|---|---|---|
| `strong_rule` | rule | False | 1.000 | 1 |
| `blind_classical` | rule | False | 0.146 | 45 |
| `extreme/pure_track` | extreme | False | 1.000 | 2 |
| `extreme/pure_jam` | extreme | False | 1.000 | 3 |
| `extreme/pure_comm` | extreme | False | 1.000 | 1 |
| `extreme/pure_detect` | extreme | False | 1.000 | 2 |
| `extreme/balanced` | extreme | False | 0.541 | 30 |
| `extreme/balanced_jam_heavy` | extreme | False | 1.000 | 3 |
| `extreme/track_agile` | extreme | False | 1.000 | 1 |
| `exploit/jam_spread` | script | False | 1.000 | 1 |
| `exploit/hard_jam_focus` | script | False | 1.000 | 5 |
| `exploit/track_heavy_agile` | script | False | 1.000 | 1 |
| `self/iter000_bc` | checkpoint | True | 1.000 | 2 |
| `self/iter025` | checkpoint | True | 1.000 | 1 |
| `self/iter050` | checkpoint | True | 1.000 | 1 |
| `self/iter075` | checkpoint | True | 1.000 | 1 |
| `self/iter100` | checkpoint | True | — | 0 |

## Notes

- Pool win_rate_vs_current = EMA of how often the *current* training AC beats this opponent (draw counts as 0.5). Low values → hard opponents → PFSP samples them more.
- self/iterNNN snapshots are added to the pool as the league evolves, enabling population-level diversity.
- Pool EMA variance near 0 → PFSP degenerating toward uniform; investigate.
- Run `run_wp2_crossplay.py` next for cross-play Elo + non-transitivity detection.
