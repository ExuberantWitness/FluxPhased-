---
name: g3-bsta-lite-f4-gate3-pass
description: 2026-07-29 G3-BSTA-lite F4 PASS — Gate 3 cleared on BC warm-start PPO (macro_drop=0.2511, witness=0.2511 on 8 fixed debug scenarios). Branch g3-bsta/mfr-lite-fastwork commit 954d9ae.
metadata:
  node_type: memory
  type: project
  originSessionId: 0c76cde4-1c23-4df6-94f5-ee29ee81afbc
---

G3-BSTA-lite fast-work milestone F4 finished 2026-07-29. Gate 3 PASS on
BC warm-start path.

**Branch / commit**: `g3-bsta/mfr-lite-fastwork` @ `954d9ae` on
`https://github.com/ExuberantWitness/FluxPhased-.git`. Base commit
`15c56e2` (F3 tip).

**Why**: F4 is the masked-PPO fixed-scenario overfit gate per
MODIFICATION_PLAN W5. Passing it authorizes F5 (one-seed stochastic
smoke) → F6 (two-seed pilot). See [[g3-bsta-lite-f2-gate1-pass]] and
[[g3-bsta-lite-f3]] for prior gates.

**How to apply**: Before quoting F4 numbers or recommending F5 work,
verify the artifacts under
`experiments/g3_bsta_lite/baseline_freeze/` (f4_train_curve.json +
f4_eval.json + F4_PHASE_REPORT.md) and that branch tip matches
`954d9ae`.

**Fixed debug suite** (8 scenarios, same seed scheme as F2 manifest):
- train_seeds = [20260729..20260736], 4 action replicates per seed
- witness drop = 0.2511, best non-witness baseline (round_robin) = 0.1311
- BC warm-start PPO macro_drop = 0.2511 (= witness, +12.0pp over
  round_robin, satisfies both >=80% headroom threshold 0.2271 and
  +5pp threshold 0.1811)
- scratch PPO secondary: 0.1689 at iter 29, monotone convergence
  (+3.78pp over round_robin, below +5pp threshold in 30 iters; not
  gated — canonical F4 path is BC warm-start)

**Training health (BC warm-start, 30 iters)**:
- max kl_max = 0.102 (iter 25) with early_stop=True (no excursion
  without early stop)
- max clip_frac = 0.038 (well below 0.5)
- min adv_std = 0.372 (well above 1e-3)
- max pre_ratio_offset = 0 (mask replay invariant holds bit-for-bit)
- entropy stays near 0 (BC actor is near-deterministic; PPO doesn't
  collapse it)

**PPO hyperparameters (frozen per spec)**: lr=3e-4 (Adam, separate
actor/critic optimizers), gamma=0.99, gae_lambda=0.95, clip=0.2,
grad_clip=0.5, entropy_coef=0.01, value_coef=0.5, epochs=4,
minibatch=256, max_kl=0.05 (early stop). Actor/critic = 2x128 Tanh.

**Caveat — PPO does not improve over BC at iter 0**: DAgger already
saturates witness on these 8 scenarios (F3 macro_drop recovery 101.5%).
PPO's role at F4 is the training-pipeline invariants (mask preservation,
ratio=1 verification, KL early stop, separate optimizers). The best
checkpoint is iter 0 (the BC warm-start point itself). This is
consistent with the overfit interpretation; F5 introduces FRESH
scenarios where the comparison is meaningful.

**Three RNG streams honored**: env-event, detector, action all use
separate torch.Generator. Inverse-CDF action sampling with the action
RNG (seed = cfg.seed + 7). Bit-for-bit reproducible.
