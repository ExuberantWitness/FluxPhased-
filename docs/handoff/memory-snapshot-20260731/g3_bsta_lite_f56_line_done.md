---
name: g3-bsta-lite-f5-f6-line-done
description: 2026-07-29 G3-BSTA-lite fast-work line F0..F6 complete. F5 @47bcaf1 (Gate 4 PASS, 0.307M transitions, BC=witness on fresh held-out). F6 @f2ef2da (Gate 5 PASS, two-seed reproducible spread=0). PPO doesn't improve over DAgger BC on this env (witness saturates).
metadata:
  node_type: memory
  type: project
  originSessionId: 0c76cde4-1c23-4df6-94f5-ee29ee81afbc
---

G3-BSTA-lite fast-work line **F0..F6 complete** 2026-07-29. All gates
0-5 PASS.

**Branch tip**: `g3-bsta/mfr-lite-fastwork` @ `f2ef2da` on
`https://github.com/ExuberantWitness/FluxPhased-.git`.

| Phase | Commit | Gate | Result |
|---|---|---|---|
| F0 | aa142f4 | (none, docs+regression) | base+namespace+regression+smoke |
| F1 | 005e6c3 / db76216 | 0 | env + 8 contract tests, 46/46 green |
| F2 | 9873697 | 1 | oracle gap 17.12pp, witness LCB 9.13pp, neighbors >=7.5pp |
| F3 | 15c56e2 | 2 | DAgger top-1 100%, gap recovery 101.5% |
| F4 | 954d9ae | 3 | BC warm-start PPO macro_drop 0.2511=witness on 8 fixed |
| F5 | 47bcaf1 | 4 | 0.307M transitions, BC=witness on 32 fresh held-out |
| F6 | f2ef2da | 5 | two-seed reproducible, spread=0.0000 |

**Why this matters**: This is the clean-line reboot after the
PRO6000 P0 BLOCKED quarantine ([[g3-bsta-pro6000-p0-blocked]]). All
work is independent of the quarantined orphan files. The frozen debug
profile (2 services, horizon=64, duty_budget=0.25) is validated
end-to-end: env contract → reachability/headroom → imitation → PPO
overfit → PPO smoke → reproducibility.

**How to apply**: Before extending this line (eight-seed significance
campaign, scale-out to non-debug profile, integration with two-team),
read the latest phase report at
`experiments/g3_bsta_lite/baseline_freeze/F6_PHASE_REPORT.md` and
verify branch tip matches `f2ef2da`. The F1 contract test suite
(46 tests under `tests/g3_bsta_lite/`) is the regression backbone —
re-run before any merge.

**Key structural finding (F4-F5-F6)**: PPO does NOT improve over the
F3 DAgger actor on this env. The DAgger BC actor saturates the
witness (drop = 0.296 = witness on fresh held-out). PPO's role at
F4-F6 is purely the training-pipeline invariants (mask preservation,
ratio=1 verification, KL early stop, multi-seed reproducibility).
Best checkpoint is always iter 0 (BC warm-start point). This is
consistent across F4 (8 fixed), F5 (32 fresh held-out, 1 seed), and
F6 (32 fresh held-out, 2 seeds).

**PPO training-health robustness (F5+F6)**: 21% of iters trigger
KL early_stop (max kl_max = 1.37 in seed 0 iter 217; 0.21 in seed
1). Every KL excursion > 0.05 had early_stop=True (Gate 4 strict
criterion satisfied). No entropy collapse; pre_ratio_offset = 0
across all 600 PPO iters.

**Frozen hyperparameters (no HPO, per spec)**: lr=3e-4 (Adam),
gamma=0.99, gae_lambda=0.95, clip=0.2, grad_clip=0.5,
entropy_coef=0.01, value_coef=0.5, epochs=4, minibatch=256,
max_kl=0.05 (early stop). Actor + critic = 2x128 Tanh.

**NO significance claim** at F6 (per MODIFICATION_PLAN W7): two-seed
pilot cannot establish statistical significance; that requires the
eight-seed campaign which is out of scope for the fast-work line.

See [[g3-bsta-lite-f2-gate1-pass]] / [[g3-bsta-lite-f3-gate2-pass]]
/ [[g3-bsta-lite-f4-gate3-pass]] for per-phase details.
