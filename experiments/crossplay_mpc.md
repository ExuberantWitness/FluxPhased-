# Phase 1.5 Cross-Play: NN finals vs ClassicalMPC (Exp B supplement)

Each NN final plays ClassicalMPC in BOTH directions (NN_red vs MPC_blue + MPC_red vs NN_blue) averaged to remove red/blue asymmetry. ClassicalMPC = rule-based beam-steer to fused enemy anchor + always-fire (no learning, no waveform agility).

Games per direction: 36 (total per row ≈ 2 × n_per_direction)


| NN final | NN wins | MPC wins | draws | NN win rate |
|---|---|---|---|---|
| mappo | 34 | 38 | 0 | **0.472** |
| ippo | 12 | 60 | 0 | **0.167** |
| pspfix | 31 | 41 | 0 | **0.431** |

## Verdict

- NN win rate > 0.5 → RL beats the classical engineering baseline.
- All three arms should beat ClassicalMPC; if any loses, the sensing frontend (not learning) is doing the work (EAAI: 'AI beats classical').
