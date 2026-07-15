# Bet B Step 0 — Rule Sensitivity Sweep (Cheap Gate)

**Generated**: 2026-07-14 23:49:13

## Setup

- Players: TwoTeamStrongRuleCommander vs `pure_track` baseline
- Episodes per direction: 100 → total per grid: 200
- Horizon: 200, n_envs: 8
- Nominal WR (rule vs pure_track): **0.983**
- Cliff definition: WR drop > 0.15 from nominal
- Total grid points: 14 (1 nominal + 13 variants)

## Sweep results

| Config | Rule WR | 95% CI | Δ from nominal | Draw | Rule kills | PT kills | Cliff? |
|---|---|---|---|---|---|---|---|
| `nominal` ← NOMINAL | **0.983** | [0.978, 0.987] | +0.000 | 0.03 | 1.95 | 0.97 |  |
| `jam_gain=3.0` | **0.980** | [0.975, 0.984] | -0.003 | 0.04 | 1.95 | 0.99 |  |
| `jam_gain=9.0` | **0.998** | [0.997, 1.000] | +0.016 | 0.00 | 1.95 | 0.93 |  |
| `range_sigma=0.02` | **0.982** | [0.978, 0.987] | -0.001 | 0.04 | 1.95 | 0.99 |  |
| `range_sigma=0.10` | **0.999** | [0.998, 1.000] | +0.016 | 0.00 | 1.94 | 0.92 |  |
| `sigma_q=1.0` | **0.976** | [0.971, 0.981] | -0.007 | 0.05 | 1.94 | 0.96 |  |
| `sigma_q=4.0` | **1.000** | [0.999, 1.000] | +0.017 | 0.00 | 1.95 | 0.92 |  |
| `exposure=100` | **0.991** | [0.988, 0.994] | +0.008 | 0.02 | 1.97 | 0.98 |  |
| `exposure=400` | **0.964** | [0.957, 0.970] | -0.019 | 0.07 | 1.90 | 0.93 |  |
| `radar_sep=1000` | **0.981** | [0.976, 0.986] | -0.002 | 0.04 | 1.94 | 0.96 |  |
| `radar_sep=2000` | **0.978** | [0.972, 0.983] | -0.005 | 0.04 | 1.95 | 0.96 |  |
| `map_size=6000` | **0.982** | [0.978, 0.987] | -0.001 | 0.04 | 1.95 | 0.97 |  |
| `map_size=10000` | **0.990** | [0.986, 0.993] | +0.007 | 0.02 | 1.97 | 0.97 |  |
| `geometry=RANDOM` | **0.978** | [0.973, 0.983] | -0.004 | 0.04 | 1.95 | 0.96 |  |

## Verdict

- **Cliff count** (WR drop > 0.15): **0 / 13** (brittleness score = 0.00)
- **Rule loses to pure_track** (WR < 0.5): **0 / 13**
- **Verdict**: ❌ rule 全程稳健 — Bet B 前提死 → 退 IET 地板

## Decision tree

→ **Hard stop**. Bet B premise dead. **Pivot to IET floor**.
  - 3-line near-Nash evidence (G0 #3 + V1 exploits + rule design) is strong.
  - Sensitivity sweep adds 4th line: 'rule is also robust to off-nominal physics.'
  - AppInt pivot dead. IET is the right venue.

## Per-axis reading

- **jam_gain**: Δ range [-0.003, +0.016] (2 variants)
- **range_sigma**: Δ range [-0.001, +0.016] (2 variants)
- **sigma_q**: Δ range [-0.007, +0.017] (2 variants)
- **exposure**: Δ range [-0.019, +0.008] (2 variants)
- **radar_sep**: Δ range [-0.005, -0.002] (2 variants)
- **map_size**: Δ range [-0.001, +0.007] (2 variants)
- **geometry**: Δ range [-0.004, -0.004] (1 variants)