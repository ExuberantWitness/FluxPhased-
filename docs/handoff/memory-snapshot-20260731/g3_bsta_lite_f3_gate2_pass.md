---
name: g3-bsta-lite-f3-gate2-pass
description: 2026-07-29 G3-BSTA-lite F3 PASS — Gate 2 cleared on all 4 criteria via DAgger (held-out top-1 100%, gap recovery 101.5%). Branch g3-bsta/mfr-lite-fastwork commit 15c56e2.
metadata:
  node_type: memory
  type: project
  originSessionId: 0c76cde4-1c23-4df6-94f5-ee29ee81afbc
---

G3-BSTA-lite fast-work milestone F3 finished 2026-07-29. Gate 2 PASS on
all four criteria.

**Branch / commit**: `g3-bsta/mfr-lite-fastwork` @ `15c56e2`. Base
commit `9873697` (F2 tip).

**Why**: F3 is the supervised imitation gate per MODIFICATION_PLAN W4.
Passing it authorizes F4 (PPO overfit). See
[[g3-bsta-lite-f2-gate1-pass]] for prior gate.

**How to apply**: Before recommending F4 work, verify the artifacts
under `experiments/g3_bsta_lite/baseline_freeze/` (F3_PHASE_REPORT.md +
imitation_*.pt, the latter .gitignored).

**Key numbers** (held-out 32 fresh scenarios × 64 steps × 4 reps):
- mask_valid = 100% (PASS, threshold 100%)
- tie-aware top-1 = 100% (PASS, threshold ≥90%)
- normalized witness regret ≈ -0.015 (PASS, threshold ≤10%); actor
  rollout slightly EXCEEDS witness
- held-out rollout gap recovery = 101.5% (PASS, threshold ≥90%)

**Why DAgger (not plain supervised)**: Initial supervised-only training
reached 100% top-1 on static held-out but rollout gap recovery was only
61.7%. Classic covariate shift: model trained on witness-trajectory obs
but encounters different obs under its own (slightly different) policy;
errors compound. DAgger fix = 3 rounds of (1) roll out current model +
witness 50/50 mixed on 64 fresh scenarios, (2) label every visited
(obs, mask) with witness action, (3) retrain on aggregated set. Final
training set 20,480 samples.

**Labels = witness (CausalReactiveOrEDF), NOT clairvoyant oracle**:
The oracle has privileged info (pending queue) the actor cannot see.
Per W4 "labels may use only actions available to the causal witness".
Using witness as supervisor makes Gate 2 well-posed: can a small MLP
express the witness from the same observation?

**Model spec**: ImitationActor(obs_dim=11, n_actions=3, hidden=128),
2x128 Tanh, masked-categorical output (illegal actions get -inf logits
before softmax/argmax). Architecture matches W5 PPO actor spec exactly
so state_dict loads directly into PPO at F4.

**ImitationActor state_dict**: `imitation_actor_dagger.pt` at
`experiments/g3_bsta_lite/baseline_freeze/`. .gitignored. Regenerate
via `train_imitation_dagger` from
`algo/_shared/pilot/g3_bsta_lite/imitation.py`.
