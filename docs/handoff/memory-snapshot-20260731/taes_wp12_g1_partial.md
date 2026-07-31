---
name: taes-wp12-g1-partial
description: "TAES WP1+WP2 results 2026-07-09 — G1 PARTIAL: RL ties FP classical on L3 kill, beats on survival; L3 curriculum phase collapsed due to random-init jammer"
metadata: 
  node_type: memory
  type: project
  originSessionId: 902a7f7f-2d60-4a53-b927-2a75af5c8fc4
---

TAES mainline WP1+WP2 finished 2026-07-09 with **G1 PARTIAL** verdict.

**Files** (in `/home/ubuntu/CODE/FluxPhased-/`):
- `algo/_shared/pilot/taes/taes_actor_critic.py` — joint commander AC (95-dim obs → 4 heads)
- `algo/_shared/pilot/taes/taes_ppo.py` — PPO trainer with GAE + CTDE privileged critic
- `algo/_shared/pilot/taes/run_wp2.py` — train+eval+exploitability driver
- `algo/_shared/baselines/taes_fp_classical_commander.py` — fictitious-play classical (G1 target)
- `experiments/wp12_results/WP2_VERDICT.md` — full verdict report
- `checkpoints/taes_mainline/taes_rl_phase1_L1.pt` — RL checkpoint (used for eval)

**Eval results (5 seeds × 6 cells, kill/surv)**:
| cell | static | FP | RL |
|---|---|---|---|
| n1_L0 | 1.00/0.97 | 1.00/0.97 | 1.00/0.97 |
| n4_L0 | 3.98/0.85 | 4.00/0.82 | **4.00/0.93** |
| n4_L1 | 3.73/0.40 | 3.48/0.40 | **3.98/0.82** (RL wins both) |
| n4_L3 | 1.75/0.40 | 1.40/0.40 | 1.43/**0.65** (RL ties FP kill, wins surv) |
| n8_L0 | 8.00/0.80 | 7.70/0.53 | 3.80/0.60 (RL fails to scale) |
| n8_L3 | 2.52/0.40 | 1.57/0.40 | 1.02/0.65 |

**Exploitability** (3 seeds, BR-jammer 80 iters, mean ± std):
- rl_commander: 29.84 ± 19.4
- static_classical: 33.21 ± 24.3
- fp_classical: 21.23 ± 24.9
High variance; no method approaches Nash equilibrium.

**Critical caveat — L3 curriculum collapse**:
The LearnedJammer in `env/gpu/qos_rrm/adversary.py` is *random-init* (untrained MLP, sigmoid→~0.5 = constant heavy jam). When training entered L3 phase after L0+L1, policy collapsed to near-uniform (entropy 6.0, kill 0). The reported RL checkpoint is **phase1_L1** (after 80 L0 + 150 L1 iters), NOT a fully curriculum-trained policy. L3 cells are reported as *generalization* results.

**Why**: training was single-sided PPO against fixed jammers; no league/PSRO. The user's plan calls for league training which would likely close the gap.

**How to apply**: For follow-up, the strongest lever is implementing a real red/blue league (PSRO-style) — the n4_L1 result (+0.50 kill, +0.42 surv vs FP) is too strong to ignore and suggests RL can clearly beat classical with proper self-play. Do NOT re-run the L0→L1→L3 sequential curriculum as-is; the L3 collapse is a pipeline bug, not a capability ceiling.

Related: [[taes-wp0-results]], [[taes-mainline-plan]]
