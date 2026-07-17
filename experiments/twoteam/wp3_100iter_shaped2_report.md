# WP2 League Training Report

**Started**: 2026-07-17 13:39:29

## Setup

- n_iters: 100
- snapshot_every: 25
- pfsp_hardness: 1.0
- bc_samples: 20000, bc_epochs: 8
- ppo_lr_actor: 0.0001, entropy_coef: 0.005
- horizon: 200, n_envs: 64
- ckpt_dir: `checkpoints/blind/wp3_100iter_shaped2`

## Training health

- final reward: -0.180  (initial: -0.226)
- final entropy: -1.381
- final adv_std: 1.000  (healthy range [0.1, 100])
- final pool EMA variance: 0.037  (low → PFSP may be stuck)
- total elapsed: 59.8 min

## Per-iter log (every 5 iters)

| iter | opp | reward | adv_std | entropy | kl | wr_vs_opp | ema_var | t(min) |
|---|---|---|---|---|---|---|---|---|
| 0 | `exploit/hard_jam_focus` | -0.226 | 1.00 | -1.591 | 0.0977 | 0.50 | 0.000 | 0.3 |
| 5 | `self/iter000_bc` | -0.210 | 1.00 | -1.455 | 0.0113 | 0.50 | 0.000 | 3.1 |
| 10 | `extreme/pure_track` | -0.223 | 1.00 | -1.467 | 0.0107 | 0.50 | 0.083 | 6.1 |
| 15 | `blind_classical` | -0.229 | 1.00 | -1.444 | 0.0120 | 0.41 | 0.071 | 9.0 |
| 20 | `exploit/hard_jam_focus` | -0.287 | 1.00 | -1.420 | 0.0096 | 0.50 | 0.059 | 12.0 |
| 25 | `blind_classical` | -0.278 | 1.00 | -1.400 | 0.0269 | 0.47 | 0.062 | 14.9 |
| 30 | `exploit/hard_jam_focus` | -0.354 | 1.00 | -1.401 | 0.0089 | 0.50 | 0.059 | 17.9 |
| 35 | `blind_classical` | -0.220 | 1.00 | -1.382 | 0.0216 | 0.47 | 0.061 | 20.9 |
| 40 | `extreme/balanced` | -0.289 | 1.00 | -1.404 | 0.0106 | 0.50 | 0.062 | 23.8 |
| 45 | `blind_classical` | -0.239 | 1.00 | -1.370 | 0.0466 | 0.44 | 0.053 | 26.7 |
| 50 | `self/iter050` | -0.185 | 1.00 | -1.385 | 0.0111 | 0.50 | 0.042 | 29.7 |
| 55 | `blind_classical` | -0.184 | 1.00 | -1.376 | 0.0145 | 0.47 | 0.047 | 32.8 |
| 60 | `self/iter000_bc` | -0.189 | 1.00 | -1.380 | 0.0106 | 0.50 | 0.050 | 35.8 |
| 65 | `blind_classical` | -0.271 | 1.00 | -1.376 | 0.0338 | 0.38 | 0.044 | 38.9 |
| 70 | `self/iter000_bc` | -0.175 | 1.00 | -1.389 | 0.0102 | 0.50 | 0.047 | 42.0 |
| 75 | `blind_classical` | -0.189 | 1.00 | -1.370 | 0.0150 | 0.47 | 0.051 | 45.0 |
| 80 | `self/iter000_bc` | -0.172 | 1.00 | -1.388 | 0.0105 | 0.47 | 0.050 | 48.1 |
| 85 | `blind_classical` | -0.193 | 1.00 | -1.380 | 0.0139 | 0.47 | 0.043 | 51.2 |
| 90 | `self/iter000_bc` | -0.142 | 1.00 | -1.403 | 0.0111 | 0.53 | 0.036 | 54.2 |
| 95 | `blind_classical` | -0.214 | 1.00 | -1.381 | 0.0336 | 0.53 | 0.032 | 57.3 |

## Pool final state

| name | kind | is_self | win_rate_vs_current | games |
|---|---|---|---|---|
| `strong_rule` | rule | False | 1.000 | 1 |
| `blind_classical` | rule | False | 0.209 | 45 |
| `extreme/pure_track` | extreme | False | 1.000 | 2 |
| `extreme/pure_jam` | extreme | False | 1.000 | 3 |
| `extreme/pure_comm` | extreme | False | 1.000 | 1 |
| `extreme/pure_detect` | extreme | False | 1.000 | 2 |
| `extreme/balanced` | extreme | False | 1.000 | 4 |
| `extreme/balanced_jam_heavy` | extreme | False | 1.000 | 3 |
| `extreme/track_agile` | extreme | False | 1.000 | 1 |
| `exploit/jam_spread` | script | False | 1.000 | 2 |
| `exploit/hard_jam_focus` | script | False | 1.000 | 6 |
| `exploit/track_heavy_agile` | script | False | 1.000 | 2 |
| `self/iter000_bc` | checkpoint | True | 0.861 | 25 |
| `self/iter025` | checkpoint | True | 1.000 | 1 |
| `self/iter050` | checkpoint | True | 1.000 | 1 |
| `self/iter075` | checkpoint | True | 1.000 | 1 |
| `self/iter100` | checkpoint | True | — | 0 |

## Notes

- Pool win_rate_vs_current = EMA of how often the *current* training AC beats this opponent (draw counts as 0.5). Low values → hard opponents → PFSP samples them more.
- self/iterNNN snapshots are added to the pool as the league evolves, enabling population-level diversity.
- Pool EMA variance near 0 → PFSP degenerating toward uniform; investigate.
- Run `run_wp2_crossplay.py` next for cross-play Elo + non-transitivity detection.
