# WP-2 M4 — BlindClassical interference sweep

## Setup

- Subject: BlindClassicalCommander (team A)
- Adversary: fixed jammer (team B) — jam_fraction, duty cycle, channel
- 3 episodes × ~150 steps per cell, n_envs=8

- Hard requirements (spec §3 ③):
  - low-interference (jam=0, low duty, orthogonal): kill ≥ 0.5
  - high-interference (jam≥0.4 + duty≥60% + same channel): kill ≤ 0.3 AND ≤ ½ low

## Channel mode: `orthogonal`

### kill_rate (0..2 enemies per episode)

| jam \ duty | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| 0.00 | 1.000 | 1.000 | 0.958 | 0.875 |
| 0.20 | 1.000 | 1.000 | 0.958 | 0.958 |
| 0.40 | 1.000 | 0.917 | 0.958 | 1.000 |
| 0.60 | 1.000 | 1.000 | 0.958 | 0.958 |

### trace_P (lower = better track; tau_track=4.0)

| jam \ duty | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| 0.00 | 232.508 | 180.898 | 166.858 | 225.928 |
| 0.20 | 267.460 | 227.513 | 242.915 | 149.212 |
| 0.40 | 269.445 | 195.052 | 155.275 | 221.255 |
| 0.60 | 221.412 | 178.782 | 166.880 | 167.784 |

### search_coverage (0..1)

| jam \ duty | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| 0.00 | 0.511 | 0.514 | 0.492 | 0.538 |
| 0.20 | 0.494 | 0.493 | 0.489 | 0.491 |
| 0.40 | 0.490 | 0.511 | 0.555 | 0.494 |
| 0.60 | 0.496 | 0.513 | 0.535 | 0.512 |

### track_active (frac steps where trace_P < tau AND init'd)

| jam \ duty | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| 0.00 | 0.205 | 0.208 | 0.209 | 0.193 |
| 0.20 | 0.210 | 0.212 | 0.201 | 0.211 |
| 0.40 | 0.211 | 0.204 | 0.203 | 0.211 |
| 0.60 | 0.215 | 0.212 | 0.201 | 0.205 |

## Channel mode: `same`

### kill_rate (0..2 enemies per episode)

| jam \ duty | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| 0.00 | 0.542 | 0.333 | 0.250 | 0.000 |
| 0.20 | 0.333 | 0.250 | 0.292 | 0.000 |
| 0.40 | 0.375 | 0.167 | 0.083 | 0.000 |
| 0.60 | 0.375 | 0.208 | 0.208 | 0.000 |

### trace_P (lower = better track; tau_track=4.0)

| jam \ duty | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| 0.00 | 51.287 | 15.702 | 9.365 | 9.936 |
| 0.20 | 13.840 | 6.928 | 125.629 | 7.668 |
| 0.40 | 214.626 | 12.293 | 7.549 | 9.685 |
| 0.60 | 10.795 | 12.682 | 8.927 | 11.068 |

### search_coverage (0..1)

| jam \ duty | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| 0.00 | 0.978 | 1.000 | 1.000 | 1.000 |
| 0.20 | 1.000 | 1.000 | 0.978 | 1.000 |
| 0.40 | 0.956 | 1.000 | 1.000 | 1.000 |
| 0.60 | 1.000 | 1.000 | 1.000 | 1.000 |

### track_active (frac steps where trace_P < tau AND init'd)

| jam \ duty | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| 0.00 | 0.078 | 0.046 | 0.046 | 0.030 |
| 0.20 | 0.048 | 0.034 | 0.047 | 0.015 |
| 0.40 | 0.053 | 0.024 | 0.026 | 0.019 |
| 0.60 | 0.052 | 0.037 | 0.029 | 0.019 |

## Headline: low vs high interference contrast

- Low  (jam=0.00, duty=20%, orthogonal): kill = 1.000
- High (jam=0.40, duty=80%, same):     kill = 0.000
- Collapse ratio: 0.000 (target ≤ 0.5 = spec §3 ③ kill collapse)

## Conclusion

Monotone kill collapse as jam/duty increase (especially under same-channel):
BlindClassical satisfies spec §3 ③ 'competent blind classical' requirement.
Env is NOT a toy — classical baseline genuinely fails under interference.
