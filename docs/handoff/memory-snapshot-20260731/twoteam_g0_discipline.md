---
name: twoteam-g0-discipline
description: Hard gate-first rule — G0 exploitability must PASS before any self-play/league burn; calm-sea discipline bought at cost of two crashes
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 902a7f7f-2d60-4a53-b927-2a75af5c8fc4
---

**Rule**: Two-team work must run WP0 (testbed verification) + WP1 G0 (strong rule-based exploitable?) BEFORE any self-play/league training. G0 FAIL → honest retreat to IET, do NOT burn self-play.

**Why**: User diagnosed two framework crashes (Phase1.5 + AppInt) where self-play was launched on a "calm sea" domain — sensing solved the task, classical was near-optimal, league/PSRO had nothing to exploit. The lesson bought at real cost: **"G0 未过前一切是假设"**. Per `TWOTEAM_MULTIFUNCTION_PLAN.md` commit 4329bae §WP1: G0 = `exploitability(π_rule) = U(π_rule vs 镜像 π_rule) − U(π_rule vs BR(π_rule))`. Significant gap → non-trivial game → WP2. Gap≈0 → calm sea → retreat.

**How to apply**:
- WP0 deliverables before any training: mirror self-play symmetry + four-function tradeoff realism (no dominant single strategy) + CRLB anchor + NaN-free + adv_std ∈ [3,14].
- WP1 deliverables before WP2: a strong rule-based multifunction commander (NOT strawman "aim+fire") AND a measured exploitability number with CI.
- The G0 BR team is a SINGLE learned commander trained specifically to exploit π_rule — not a league.
- Do NOT propose or launch self-play/league runs until G0 is reported and APPROVED.
- Honest framing: if G0 fails, that is a valid finding ("this domain is not RL-friendly"); clean retreat to IET (C1 multi-baseline CRLB + C0 IQ baseline).

Related: [[twoteam-multifunction-pivot]], [[fluxleague-anchor-root-cause]] (same discipline pattern — diagnose root cause before stacking fixes).
