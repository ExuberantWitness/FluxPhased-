# Phase 1.5 Cross-Play Tournament (Exp B)

Each unordered pair plays BOTH directions (A_red vs B_blue + B_red vs A_blue) averaged to remove red/blue starting-position asymmetry.

Games per direction: 36 (total per cell = 2 × n_per_direction, ±env.num_envs rounding)


## Round-robin (finals vs finals)

| | mappo | ippo | pspfix | full_league | mean |
|---|---|---|---|---|---|
| mappo | — | 0.750 | 0.764 | 0.500 | **0.671** |
| ippo | 0.250 | — | 0.236 | 0.208 | **0.231** |
| pspfix | 0.236 | 0.764 | — | 0.194 | **0.398** |
| full_league | 0.500 | 0.792 | 0.806 | — | **0.699** |

## Held-out (finals vs held-out opponents)

Held-out set: {} (classical_mpc + each arm's iter_010 snapshot).

| final |  | mean |
|---|---|

## Verdict

- Round-robin mean > 0.5 against all other finals → that arm is the strongest policy.
- Held-out mean > 0.5 against classical_mpc → RL beats the engineering baseline.
- If two arms are within ±0.05 win rate of each other, the difference is noise (binomial stderr ≈ √(p(1-p)/n) ≈ 0.05 for n=72).
