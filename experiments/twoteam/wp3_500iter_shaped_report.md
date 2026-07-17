# WP2 League Training Report

**Started**: 2026-07-17 14:58:12

## Setup

- n_iters: 500
- snapshot_every: 50
- pfsp_hardness: 1.0
- bc_samples: 20000, bc_epochs: 8
- ppo_lr_actor: 0.0001, entropy_coef: 0.01
- horizon: 300, n_envs: 64
- ckpt_dir: `checkpoints/blind/wp3_500iter_shaped`

## Training health

- final reward: -0.230  (initial: -0.170)
- final entropy: -1.695
- final adv_std: 1.000  (healthy range [0.1, 100])
- final pool EMA variance: 0.037  (low → PFSP may be stuck)
- total elapsed: 379.2 min

## Per-iter log (every 25 iters)

| iter | opp | reward | adv_std | entropy | kl | wr_vs_opp | ema_var | t(min) |
|---|---|---|---|---|---|---|---|---|
| 0 | `exploit/hard_jam_focus` | -0.170 | 1.00 | -1.619 | 0.1577 | 0.50 | 0.000 | 0.4 |
| 25 | `blind_classical` | -0.276 | 1.00 | -1.365 | 0.2829 | 0.41 | 0.158 | 19.1 |
| 50 | `extreme/balanced` | -0.354 | 1.00 | -1.436 | 0.0106 | 0.47 | 0.131 | 37.6 |
| 75 | `blind_classical` | -0.234 | 1.00 | -1.436 | 0.0320 | 0.28 | 0.101 | 56.8 |
| 100 | `self/iter100` | -0.111 | 1.00 | -1.415 | 0.0156 | 0.50 | 0.095 | 76.0 |
| 125 | `exploit/jam_spread` | -0.145 | 1.00 | -1.430 | 0.0129 | 0.50 | 0.066 | 94.8 |
| 150 | `self/iter150` | -0.145 | 1.00 | -1.447 | 0.0225 | 0.50 | 0.044 | 113.5 |
| 175 | `extreme/balanced` | -0.147 | 1.00 | -1.462 | 0.0260 | 0.72 | 0.053 | 132.5 |
| 200 | `self/iter200` | -0.117 | 1.00 | -1.468 | 0.0327 | 0.50 | 0.054 | 151.6 |
| 225 | `exploit/jam_spread` | -0.133 | 1.00 | -1.482 | 0.0272 | 0.53 | 0.055 | 170.7 |
| 250 | `self/iter250` | -0.173 | 1.00 | -1.534 | 0.0361 | 0.50 | 0.043 | 189.6 |
| 275 | `blind_classical` | -0.238 | 1.00 | -1.491 | 0.0468 | 0.22 | 0.045 | 208.3 |
| 300 | `self/iter300` | -0.154 | 1.00 | -1.530 | 0.0360 | 0.50 | 0.048 | 226.8 |
| 325 | `blind_classical` | -0.252 | 1.00 | -1.542 | 0.0719 | 0.34 | 0.049 | 244.9 |
| 350 | `self/iter350` | -0.140 | 1.00 | -1.633 | 0.0437 | 0.50 | 0.041 | 262.9 |
| 375 | `blind_classical` | -0.205 | 1.00 | -1.576 | 0.0698 | 0.44 | 0.042 | 280.7 |
| 400 | `self/iter400` | -0.156 | 1.00 | -1.662 | 0.0477 | 0.53 | 0.044 | 300.6 |
| 425 | `blind_classical` | -0.204 | 1.00 | -1.635 | 0.0678 | 0.44 | 0.038 | 320.5 |
| 450 | `self/iter450` | -0.156 | 1.00 | -1.726 | 0.0476 | 0.53 | 0.041 | 340.4 |
| 475 | `blind_classical` | -0.147 | 1.00 | -1.658 | 0.0762 | 0.41 | 0.043 | 360.2 |

## Pool final state

| name | kind | is_self | win_rate_vs_current | games |
|---|---|---|---|---|
| `strong_rule` | rule | False | 1.000 | 1 |
| `blind_classical` | rule | False | 0.075 | 230 |
| `extreme/pure_track` | extreme | False | 1.000 | 1 |
| `extreme/pure_jam` | extreme | False | 1.000 | 1 |
| `extreme/pure_comm` | extreme | False | 1.000 | 1 |
| `extreme/pure_detect` | extreme | False | 1.000 | 1 |
| `extreme/balanced` | extreme | False | 1.000 | 107 |
| `extreme/balanced_jam_heavy` | extreme | False | 1.000 | 1 |
| `extreme/track_agile` | extreme | False | 1.000 | 1 |
| `exploit/jam_spread` | script | False | 0.917 | 144 |
| `exploit/hard_jam_focus` | script | False | 1.000 | 1 |
| `exploit/track_heavy_agile` | script | False | 1.000 | 1 |
| `self/iter000_bc` | checkpoint | True | 1.000 | 1 |
| `self/iter050` | checkpoint | True | 1.000 | 1 |
| `self/iter100` | checkpoint | True | 1.000 | 1 |
| `self/iter150` | checkpoint | True | 1.000 | 1 |
| `self/iter200` | checkpoint | True | 1.000 | 1 |
| `self/iter250` | checkpoint | True | 1.000 | 1 |
| `self/iter300` | checkpoint | True | 1.000 | 1 |
| `self/iter350` | checkpoint | True | 1.000 | 1 |
| `self/iter400` | checkpoint | True | 1.000 | 1 |
| `self/iter450` | checkpoint | True | 1.000 | 1 |
| `self/iter500` | checkpoint | True | — | 0 |

## Notes

- Pool win_rate_vs_current = EMA of how often the *current* training AC beats this opponent (draw counts as 0.5). Low values → hard opponents → PFSP samples them more.
- self/iterNNN snapshots are added to the pool as the league evolves, enabling population-level diversity.
- Pool EMA variance near 0 → PFSP degenerating toward uniform; investigate.
- Run `run_wp2_crossplay.py` next for cross-play Elo + non-transitivity detection.
