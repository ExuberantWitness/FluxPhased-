---
name: twoteam-multifunction-pivot
description: "2026-07-13 framework pivot — AppInt paused, two-team symmetric multifunction adversarial is now the live direction; TAES primary, IET fallback"
metadata: 
  node_type: memory
  type: project
  originSessionId: 902a7f7f-2d60-4a53-b927-2a75af5c8fc4
---

**Pivot (2026-07-13, commit 4329bae — TWOTEAM_MULTIFUNCTION_PLAN.md)**: AppInt single-team direction paused. New direction: **two-team symmetric multifunction adversarial** (red + blue, each = 2 phased arrays + 1 commander + 1 laser; per-aperture time-shared 4-function: detect/track/jam/comm).

**Why**: User diagnosed two root causes that killed us twice:
- **Root A (calm sea)**: Phase1.5 / AppInt had no EW stress → good sensing solves task → classical near-optimal → league zero-gain. Sensing must be **contested**.
- **Root B (1-bit opponent)**: External scalar jammer has nothing to adapt to → policy collapses to constant. Confirmed empirically 2026-07-11 in `train_jammer_league` — 200 iters, output literally 0.4755 across N∈{1,2,4,8}.

Two-team fixes both: opponent is a full symmetric commander (not scalar), and sensing is contested via per-aperture jam function.

**Why now**: AppInt RECONFIRM Task A passed (IPPO n4_L1 kill 3.85 > classical 3.62 + survival 0.78 > 0.40), but the headline's "learned commander beats classical" is one cell; n8 classical dominates 8.0 vs 4.65. Two-team restores TAES-grade problem (Li'22/Xiong'23/Dolinger'25 lineage).

**How to apply**:
- Live plan: `TWOTEAM_MULTIFUNCTION_PLAN.md` (commit 4329bae). Read it before any two-team work.
- Branch TBD (likely from current `appint/data-preflight` or fresh from `a8a96e0`).
- Venue: TAES primary, IET fallback, AppInt slice备用.
- Reusable from AppInt: `env/gpu/taes/taes_env.py` (single-team — needs symmetric two-team rewrite), `algo/_shared/pilot/taes/taes_actor_critic.py` (AC + α_eff fix + local critic), `algo/_shared/pilot/taes/taes_ppo.py` (PPO + GAE + IPPO ablation), `algo/_shared/laser/{sensing,crlb}.py`, `algo/_shared/self_play/opponent_pool.py` (PFSP fixed).

See [[twoteam-g0-discipline]] for the gate-first rule. See [[appint-preflight-paused]] for paused AppInt artifacts.
