---
name: appint-preflight-paused
description: AppInt single-team pre-flight branch is paused (not deleted) at RECONFIRM_FINAL_REPORT.md; checkpoints preserved for revival if two-team G0 fails
metadata: 
  node_type: memory
  type: project
  originSessionId: 902a7f7f-2d60-4a53-b927-2a75af5c8fc4
---

**State (2026-07-13)**: AppInt single-team direction on branch `appint/data-preflight` is **paused, not deleted**. The RECONFIRM gate passed (IPPO n4_L1 kill 3.85 vs classical 3.62, survival 0.78 vs 0.40), but the user pivoted to two-team before authorizing full R2 grid burn.

**Why paused, not killed**:
- AppInt Task A PASSED — there is a real result there ("learned commander beats strong modular classical on n4 mid-EW").
- If two-team G0 fails (calm sea), AppInt slice becomes the fallback paper (user explicitly listed AppInt as "slice备选" in `TWOTEAM_MULTIFUNCTION_PLAN.md`).
- All artifacts preserved: `checkpoints/appint/{jammer_L3,mappo,ippo}_final.pt` + `experiments/wp12_results/RECONFIRM_FINAL_REPORT.md` + working code in `algo/_shared/pilot/taes/`.

**How to apply**:
- Treat AppInt branch as a frozen snapshot. Don't add new work there.
- If user asks to revive AppInt (e.g. after G0 fail), the resume point is: "run R2 full grid (56 cells × 5 seeds × 4 methods = 1120 eps)" — that's the unstarted step.
- Honest framing already locked in for any AppInt revival: headline = "learned commander (IPPO) Pareto-dominates strong classical on n4 mid-EW"; honest limitation = "n8 classical wins; L3 = trained worst-case near-constant, NOT adaptive".
- Reusable code for two-team: AC + α_eff fix + local critic + IPPO ablation in `taes_actor_critic.py` / `taes_ppo.py`; PPO trainer kernel + log_std bounds in `run_wp2.py::JammerPPOTrainer`; PFSP-fixed `OpponentPool`.

Related: [[twoteam-multifunction-pivot]], [[taes-wp12-g1-partial]].
