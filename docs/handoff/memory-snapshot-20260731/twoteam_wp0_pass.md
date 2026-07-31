---
name: twoteam-wp0-pass
description: "Two-team symmetric multifunction env WP0 PASS 2026-07-13 — mirror physics exactly symmetric, no dominant extreme strategy, CRLB anchor validated. Ready for WP1 BR training."
metadata: 
  node_type: memory
  type: project
  originSessionId: 902a7f7f-2d60-4a53-b927-2a75af5c8fc4
---

Two-team symmetric multifunction env WP0 verification **PASS** 2026-07-13.

**Why:** Hard gate per TWOTEAM_MULTIFUNCTION_PLAN.md — must clear mirror unbiased + 4-fn tradeoff real + CRLB anchor before any WP1 BR training or self-play burn. Failing → root A (calm sea) or env bug, don't pass go.

**How to apply:** When user asks "are we clear for WP1?" or similar — yes, this is the gate. The env is real (non-trivial game, no hidden asymmetry). BR training / G0 exploitability is the next milestone.

## Results (all 3 checks PASS)

**Check 1 (mirror physics symmetry, 20 episodes, MIRROR_GEOMETRY, identical actions both teams):**
- max |kills_t0 − kills_t1| = 0.0000
- max |exposure_t0 − exposure_t1| = 0.0000
- max |mean_trace_P_t0 − mean_trace_P_t1| = 0.0000
- mean |reward_t0| (zero-sum check) = 0.0000
- → physics EXACTLY mirror-symmetric. Required fix: home-on-jam RNG broadcast across team axis (`torch.rand(E,1,R).expand(E,T,R)`) — without this, independent team rolls → asymmetric deaths → kills/exposure/trace_P diverge on RNG, not on real bias.

**Check 2 (4-function tradeoff matrix, 6 strategies × 6 strategies, RANDOM_GEOMETRY):**
- **No dominant strategy** (PASS — no strategy wins > 90% vs ALL).
- Structure: pure_track beats pure_comm/pure_detect (100%, kill chain works) but LOSES to balanced/balanced_jam_heavy (0%, single-target limitation exposed). balanced beats pure_track (97%) + pure_comm/detect (100%). balanced vs balanced_jam_heavy → 0% / 0% (mutual jam deadlock, no kills). pure_jam/pure_comm/pure_detect all draw among themselves (no kill capability).
- → root A (calm sea) NOT present. The game is non-trivial. G0 has a real chance to be meaningful.

**Check 3 (CRLB anchor):**
- Theoretical CRLB trace_P = 0.0303 (σ_r=0.05, baseline 1500m, range 2500m, bistatic 33.4°)
- Achieved trace_P = 0.00858 (split-beam pure_track, 80 steps, each aperture → different enemy)
- Ratio = 0.28 (below theoretical — Kalman smoothing beats single-shot CRLB; PASS, tolerance was < 5)
- → multi-baseline tracking fusion anchor works.

## Critical implementation details (don't regress)

- `priv[:, 4]` = `trace_P.clamp(0, 1).mean() / tau_track` (per [[twoteam-multifunction-pivot]] α_eff bug warning). Assert `priv_4_max < 100` in `get_obs()`. Initial value = 25 (trace_P=1.0/tau_track=0.04).
- `_alive_mean_trace_P()` masks dead enemy radars (their P grows via Q unboundedly — without masking, mean ≈ 36, looks like tracker broken).
- Home-on-jam RNG MUST be broadcast across team axis for mirror symmetry.
- For CRLB check: use beam_strategy="split" (aperture 0 → enemy 0, aperture 1 → enemy 1), NOT pure_track's default "same_as_laser" (only tracks 1 enemy).

## Files

- `env/gpu/twoteam/twoteam_env.py` — TwoTeamVecEnv (E, 2 teams, 2 radars, soft 4-fn alloc)
- `algo/_shared/pilot/twoteam/extreme_commanders.py` — 6 fixed-strategy commanders
- `algo/_shared/pilot/twoteam/run_wp0_check.py` — WP0 driver (mirror + tradeoff + CRLB)
- `experiments/twoteam/wp0_check_report.md` — auto-generated report

Next: WP1 BR training + G0 exploitability gate `U(π_rule vs mirror) − U(π_rule vs BR(π_rule))`.
