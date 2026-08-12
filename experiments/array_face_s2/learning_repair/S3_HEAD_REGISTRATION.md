# S3 Head Registration Guide

This document describes how to extend the S2 v2 trainer (`trainer_v2.py` +
`actor_heads.py`) to S3 by registering a third `Bernoulli(5)` cell-binding head.
The N-head framework was designed so that S3 requires **zero trainer-side code
changes** — only an env-side extension and a one-line head-spec registration.

## S3 vs S2 (per HANDOFF §11.1)

| Aspect | S2 | S3 |
|---|---|---|
| Action space | `MultiDiscrete([3, 5])` = base + beam | `Bernoulli(5) + Cat(3) + Cat(5)` = **cell + base + beam** |
| Physics | N² coherent gain over fixed 5 cells | N² gain over `N_active = sum(cell_mask)` (dynamic) |
| Observation | 21 dims (unchanged) | 21 dims (unchanged) |
| New risk | — | Bernoulli(5) may initialize all-zero (reward=0, grad=0) |
| Mitigation | — | Higher entropy coef + lr on cell head |

## What the v2 trainer already provides

The `MultiHeadActor` (`actor_heads.py`) and `S2PPOTrainerV2` (`trainer_v2.py`)
were built generic-first. The following are **already implemented and tested**
for mixed categorical + bernoulli heads:

- `HeadSpec(name, kind, n_actions)` with `kind ∈ {"categorical", "bernoulli"}`
- `MultiHeadActor`: shared trunk → per-head Linear layers, dispatched by kind
- `distribution()`: returns `Categorical` or `Bernoulli` per head
- `joint_log_prob / joint_entropy`: sum over heads (bernoulli per-cell values
  summed to a per-env scalar, so every head contributes `[E]`)
- `categorical_kl` + `bernoulli_kl` + `joint_kl_multihead`: dispatched by kind
- `sample_multihead`: inverse-CDF for categorical, threshold for bernoulli
  (one `torch.rand` per head per env, preserving the RNG contract)
- Per-head entropy coefficients (`_entropy_coef_for_head`)
- Equivalence verified: 2-categorical-head config is bit-exact to `MultiDiscreteActor`
- Bernoulli smoke verified: logits/log_prob/entropy/kl/sampling all finite

The `bernoulli_kl` and bernoulli sampling paths were validated in
`_verify_actor_equiv.py` (the `[bernoulli head smoke]` and `[bernoulli sample]`
checks), so S3 can use them without writing new math.

## S3 implementation checklist (env-side only)

The trainer is ready. S3 work is entirely in `env/gpu/array_face_s3/` (new)
plus a one-line head-spec registration in the driver.

### 1. New module `env/gpu/array_face_s3/`

Mirror the S2 module structure:

```
env/gpu/array_face_s3/
  __init__.py          # exports: EnvConfig, ArrayFaceS3VecEnv, RadarULAConfig,
                       #           JammerULAConfig, OBS_DIM_S3, N_ACTIONS_BASE,
                       #           N_ACTIONS_BEAM, N_CELLS
  action_contract.py   # add N_CELLS=5, cell mask validation, BernoulliTransitionTrace
  array_factor.py      # reuse S2's (same 5-cell ULA geometry)
  observation.py       # reuse S2's 21-dim obs (HANDOFF says obs unchanged)
  physics.py           # MODIFIED: N_active = sum(cell_mask) instead of fixed N
  env.py               # MODIFIED: step() takes (action_base, action_beam, action_cell)
```

### 2. Physics change (`physics.py`)

S2 computes peak EIRP as `P_cell_dBm + 20*log10(N)` with fixed `N = jammer.n_cells = 5`.
S3 must use the **active cell count**:

```python
# S2 (current):
N = float(jammer.n_cells)                                    # fixed 5
P_peak_dBm = P_cell_dBm + 20.0 * torch.log10(torch.tensor(N, device=device))

# S3 (planned):
N_active = cell_mask.sum(dim=-1).clamp(min=1).float()        # [E], dynamic 1..5
# Per-cell power stays P_cell; coherent gain scales with N_active.
# Cells that are off contribute 0 to the coherent sum.
P_peak_dBm = P_cell_dBm + 20.0 * torch.log10(N_active.clamp(min=1e-12))
# Also: cells that are off don't radiate, so the per-cell power term should
# only count active cells. The cleanest formulation:
#   P_total_radiated_dBm = 10*log10(N_active * P_cell_W * 1000)
#   P_peak_dBm = P_total_radiated_dBm + 10*log10(N_active)   # array gain
# (net: same as S2 when all 5 active; scales down as cells turn off)
```

**Important**: HANDOFF §11.1 says "cell weights `w_mn ∈ {0, 1}` replace S2's
all-1s". The exact physics formula should be cross-checked against the S1/S2
JNR baseline (67.5 dB when all cells on, both AFs at broadside) — S3 with all
cells on must reproduce S2's JNR exactly (this is the M0 physical gate).

### 3. Env `step()` signature change

S2's `step(action_base, action_beam)` becomes S3's
`step(action_base, action_beam, action_cell)` where `action_cell` is
`[E, N_CELLS]` float in {0., 1.}. The mask check adds: `action_cell` must be
binary; no energy constraint on individual cells (the energy budget is still
on the base action — jamming at all consumes 1 token regardless of how many
cells are active, per S2 semantics — **confirm this with HANDOFF §11.1 before
implementing**, as it affects whether S3 can "concentrate power by turning
off cells to save energy", which is the core research question).

### 4. Cold-start mitigation (HANDOFF §11.1 risk)

Bernoulli(5) initialized randomly will be ~50% on per cell, but if the policy
quickly learns "all-off saves energy / all-on is simplest", it may collapse
to a degenerate mode. HANDOFF specifies:

> **Mitigation**: entropy bonus + cell head higher lr.

In `trainer_v2.py`, the per-head entropy is already supported. For S3, set:

```python
S2PPOConfigV2(
    per_head_entropy=True,
    entropy_coef_per_head={"base": 5e-3, "beam": 5e-3, "cell": 1e-2},   # cell higher
    entropy_anneal_frac_per_head={"base": 0.5, "beam": 0.3, "cell": 0.5}, # cell slower anneal
    ...
)
```

For **per-head learning rate**, the current trainer uses a single `actor_opt`
for the whole actor (shared trunk + all heads). To give the cell head a
different lr, either:
- (a) Use a separate `torch.optim.Adam` for `actor.heads["cell"].parameters()`
      with a higher lr — requires a small `__init__` change in `S2PPOTrainerV2`.
- (b) Accept uniform lr initially and rely on the higher entropy coef alone.
      Option (b) is the lower-risk first attempt; add (a) only if the cell
      head shows signs of collapse.

### 5. Driver registration (one line)

In `run_s2_ppo_v2.py` (or a new `run_s3_ppo.py`), the only change is the
head-spec list passed to the trainer:

```python
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec

head_specs = [
    HeadSpec("base", "categorical", N_ACTIONS_BASE),
    HeadSpec("beam", "categorical", N_ACTIONS_BEAM),
    HeadSpec("cell", "bernoulli",   N_CELLS),   # <-- S3 adds this line
]
trainer = S2PPOTrainerV2(..., head_specs=head_specs)
```

The trainer's `collect_rollout` currently unpacks `actions["base"]` and
`actions["beam"]` to call `env.step(...)`. For S3, this becomes:

```python
step_out = self.env.step(actions["base"], actions["beam"], actions["cell"])
```

This is the **only** trainer-side change needed (one line in `collect_rollout`,
plus the S3-specific evaluate function mirrors it). Everything else — KL,
ratio, entropy loss, KL-rollback — works unchanged because they iterate over
`self.head_specs` generically.

## S3 physical gate (M0, before training)

Before any S3 PPO run, verify (mirroring S2's M0 check):

```python
# All cells on, both AFs broadside:
JNR_peak(cell_mask=all_1s) ≈ 67.48 dB   # must match S2 baseline
# 1 cell on:
JNR(cell_mask=[1,0,0,0,0]) ≈ 67.48 - 20*log10(5) ≈ 53.46 dB
# AF spread unchanged (depends only on geometry, not cell count):
spread ≈ 19.88 dB
```

If all-cells-on JNR ≠ S2's 67.48 dB, the physics refactor is wrong — do not
proceed to training.

## Summary

| Component | S3 work needed |
|---|---|
| `actor_heads.py` (HeadSpec, MultiHeadActor, KL, sampling) | **None** — already supports bernoulli |
| `trainer_v2.py` (PPO update, per-head entropy) | **None** — already iterates head_specs |
| `run_*_v2.py` driver | 1 line: add `HeadSpec("cell","bernoulli",5)` + pass `actions["cell"]` to `env.step` |
| `env/gpu/array_face_s3/physics.py` | New: `N_active = sum(cell_mask)` in EIRP formula |
| `env/gpu/array_face_s3/env.py` | New: `step()` accepts `action_cell` |
| `env/gpu/array_face_s3/action_contract.py` | New: `N_CELLS`, cell mask validation |
| Tests | New: `tests/array_face/test_array_factor_s3.py`, `test_array_face_s3.py` |

The trainer-side investment is complete. S3 is an env-side task.
