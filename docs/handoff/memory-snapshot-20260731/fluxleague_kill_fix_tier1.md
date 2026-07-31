---
name: fluxleague-kill-fix-tier1
description: "FluxLeague laser kill-learning fix (Tier 1) — beam_az threading, fire-commitment reward, fire_init_logit bias. Training launched 2026-06-21."
metadata: 
  node_type: memory
  type: project
  originSessionId: f140dbce-c539-4be4-9cf7-6a1b64fdc269
---

Tier 1 algorithmic fixes to FluxLeague laser kill-learning, applied 2026-06-21.

**Why:** Policy was timing out in training episodes (avg_r=16447, wr=0.00) despite win_rate=0.50 fix being applied. Diagnosed as fire head commitment failure: 50% Bernoulli init → 0.5^4 = 6% chance of 4 consecutive fire_on needed for 2ms dwell-to-kill.

**How to apply:** When tuning laser kill rate or fire head behavior, check:
1. `fire_init_logit` in yaml (default 1.0 → 73% fire prob at init)
2. `fire_commitment_weight/cap/exp` in reward_shaping (superlinear streak bonus)
3. `beam_az/beam_el` threaded through `env.step()` via `LaserEpisodeRunner._last_beam_az/el`

**Training launched:** 2026-06-21 19:31, PID 1417143, config `configs/laser_25x25_pro6000_league.yaml`, log `logs/laser_league_tier1_kill_fix_20260621_193136.log`. Expected ~5 days for 24 PSRO iter. NO parallel runs (GPU OOM risk per user constraint).

**Early signal (iter 0):** avg_r 17220→19483 (+13%), payoff eval 50× faster (5-6s vs 28-591s/pair → kills happening in deterministic eval), training still shows timeout because PPO exploration entropy (rad ent=322) masks kills.

Related: [[fluxleague-paper-framing]]
