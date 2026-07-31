---
name: taes-wp0-results
description: TAES WP0 env validation PASS — hard regime established with empirical tau_track=0.04
metadata: 
  node_type: memory
  type: project
  originSessionId: 902a7f7f-2d60-4a53-b927-2a75af5c8fc4
---

TAES WP0 (env testbed) finished 2026-07-09 with PASS verdict.

**Files** (all in `/home/ubuntu/CODE/FluxPhased-/`):
- `env/gpu/taes/taes_env.py` — purpose-built asymmetric env (2 radars + commander + laser vs adaptive jammer + N targets); does NOT reuse IQ-level MFARVecEnv
- `algo/_shared/laser/crlb.py` — single-time CRLB + recursive PCRLBTracker
- `algo/_shared/baselines/taes_classical_commander.py` — strong modular classical (Q-RAM-lite + shoot-look-shoot + emission control) + greedy strawman
- `algo/_shared/pilot/taes/run_wp0_validation.py` — validation harness
- `experiments/wp0_validation/WP0_VERDICT.md` — verdict report

**Key env params** (after tuning):
- sigma_q=2.0 m/s², jam_gain=8.0, exposure_gain=50.0
- tau_track=0.04 (FIXED after auto-compute proved unstable — Riccati overestimates 10-50×)
- e_kill=2.0, dwell_rate=1.0 → time-to-kill = 20 control steps (dt=0.1s)
- emit_power_per_subarray=0.005 (lowered from 0.02 — homejam was too aggressive)
- use_range_bearing=True (measurement type)
- range_sigma_m=0.05, bearing_sigma_rad=1e-4, crossrange_factor=7.4e-5

**WP0 validation grid** (5 seeds × 4 envs × 600 steps):
| Cell | kill | ttk | surv | track_loss | P/PCRLB |
|---|---|---|---|---|---|
| n1_l0 | 1.00 | 20 | 1.00 | 0.00 | 1.77 |
| n4_l0 | 1.00 | 20 | 1.00 | 0.00 | 3.53 |
| n4_l1 | 0.89 | 27 | 0.85 | 0.59 | 48.4 |
| n4_l3 | 0.39 | 315 | 0.65 | 0.82 | 55.9 |
| n8_l0 | 0.97 | 20 | 0.90 | 0.00 | 4.01 |
| n8_l3 | 0.23 | 267 | 0.55 | 0.93 | 56.4 |

**Sanity gate PASS**: n1_l0 kill=1.0, ttk=20 (=e_kill/dwell_rate/dt), P/PCRLB ∈ [0.5, 3.0].

**Hard regime PASS**: kill drop 0.61 (>0.20 target), ttk blow-up 20→315 (>2× target).

**How to apply**: WP1 (Stone Soup IMM-PDAF + fictitious-play) and WP2 (learned commander + G1) build on this env. Tuning philosophy: tau_track must be a FIXED value chosen empirically so no-jam trace_P (~0.005-0.03) is well below and jammed trace_P (~0.07-0.15) is well above. Per-target tau based on CRLB fails because CRLB scales with jam_mul², defeating the threshold's purpose.

Related: [[taes-mainline-plan]]
