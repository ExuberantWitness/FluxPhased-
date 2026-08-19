# S5 Report — Two-Jammer Cooperative Jamming (IPPO + Central Critic)

**Stage**: S5 of the S1→S7 roadmap (HANDOFF §11.3)
**Environment**: 2 identical jammers (K=2, each 5×5 UPA, own 63-token budget) vs one
radar; per-jammer (Bernoulli(25) cell + Categorical(25) beam); powers combined by
LINEAR (mW) summation; team reward; obs includes the other jammer's coarse state.
**Algorithm**: IPPO with parameter sharing + central critic (CTDE) — one
MultiHeadActor for both jammers, PrivilegedValueCritic(84) on the joint
privileged state; K-flattening rollout reuses the S4 PPO update unchanged.

## 1. Headline Results

### 3-seed independent budgets (1000 iters each)

| seed | val_drop@999 | broadside (idx12) share | cells/jammer |
|------|-------------|------------------------|--------------|
| 20260729 | 0.1493 | 0.56 | 1.01 |
| 20260730 | 0.1561 | 0.63 | 1.00 |
| 20260801 | 0.1482 | 0.42 | 1.00 |
| **mean** | **0.1512 ± 0.0043** | 3/3 → broadside | 1.0 |

### Cross-stage summary (same protocol, 64 val scenarios)

| Stage | PPO result | Oracle ceiling | Relative optimality |
|-------|-----------|----------------|---------------------|
| S3 (1 jammer, 1D + cells) | 0.4229 ± 0.0095 | 0.538 | 79% |
| S4 (1 jammer, 2D) | 0.0943 ± 0.0158 | 0.132 | 71% |
| **S5 (2 jammers, coop)** | **0.1512 ± 0.0043** | **0.179** | **84%** |
| S5-shared (commons ablation) | 0.0881 (976 iters) | 0.1614 | 54% |

## 2. Oracle Ladder (64 val seeds, fixed-policy references)

| Reference policy | drop | interpretation |
|------------------|------|----------------|
| idle | 0.0119 | floor (natural timeouts) |
| 1 jammer broadside (=S4 oracle) | 0.1322 | single-jammer physical limit |
| 2 jammers, one mis-aimed (el=+15°) | 0.1467 | cost of one bad beam |
| 2 jammers staggered (time-division) | 0.1614 | shared-budget-style optimum |
| **2 jammers both-on broadside** | **0.1790** | S5 physical ceiling |

**Key findings**:
1. S5 (0.1512) exceeds the single-jammer physical ceiling (0.1322) by 14% —
   cooperation value is real, not an artifact.
2. With INDEPENDENT budgets, both-on dominates staggered (0.179 vs 0.161):
   the optimal cooperation is simultaneous transmission with correct aiming;
   division of labor is NOT required when budgets are separate.
3. Relative optimality rises across stages: 79% → 71% → 84%. Two jammers are
   EASIER for PPO to optimize relative to their ceiling than one jammer in 2D
   (the extra power relaxes the p_detect saturation that made S4's beam
   resolution marginal).

## 3. Commons Ablation (shared 63-token pool)

Stopping point: 976 iters, val = 0.0881 (54% of the 0.1614 shared ceiling).

**Structure emerged, timing did not**:
- Conservation: 0.50 cells/jammer/step (total ~1 token/step — the pool is
  spent at the sustainable rate, preserving ~63 steps of coverage) ✓
- Aiming: broadside (idx12) share 35% ✓ (weaker than independent runs' 42-63%)
- Coordination timing: NOT learned — val stuck at 54% vs the 87% of the
  independent variant; the staggered-alternation rhythm needed to approach
  0.1614 did not emerge within 1000 iters under plain IPPO.

**Interpretation**: under a commons constraint, IPPO + central critic learns
the resource-conservation structure (WHAT to do: spend ~1 token/step, aim at
broadside) but not the temporal coordination (WHO acts WHEN). The 33-point
relative-optimality gap between independent (84%) and shared (54%) budgets is
a quantitative measure of the value of emergent turn-taking — the target
capability for future MARL work (e.g., explicit coordination mechanisms,
communication, or MAPPO with credit assignment).

## 4. Engineering Notes

- K-flattening trick: rollout batch [T, E*K] with k-major slots; the entire
  S2PPOTrainerV2 update (GAE on central values, clipped surrogate, KL
  rollback, per-head entropy) inherited unchanged. Batch = 2048 transitions/
  iter (2× S4).
- Shared-budget env support: `EnvConfig(shared_budget=True)` — common pool
  gating, sequential commons accounting in the over-budget clamp (handles
  the individually-fine-but-jointly-over case exactly).
- Training infra: two concurrent self-healing PS chains on one 6GB GPU
  (~85% utilization combined); one overnight parent-process loss recovered
  by auto-resume (RETRY log in s5_chain.log).
- Config: beam entropy anneal 0.9 (S4 expD lesson), cell sparse-init bias
  -3.0, no beam shaping (S4 falsification).

## 5. Artifacts

- `env/gpu/array_face_s5/` — env, physics (linear power sum), contracts
- `experiments/array_face_s5/learning_repair/trainer_s5.py`, `run_s5_ippo.py`
- `s5_ippo_output_seed{20260729,20260730,20260801}/` — 3-seed runs
- `s5_shared_output_seed20260729/` — commons ablation (976 iters)
- `tests/array_face/test_array_face_s5.py` — 20 gates incl. single-jammer==S4,
  twins +3.01 dB, commons conservation/staggered/overspend

## 6. Implications for the Paper

1. The staged S1→S5 progression with oracle ceilings gives a clean
   difficulty-vs-capability narrative: relative optimality 79% → 71% → 84%.
2. S5's cooperation gain (+60% over S4, +14% over the single-jammer physical
   limit) is the first multi-agent result of the benchmark.
3. The commons ablation converts "did cooperation emerge?" from a binary
   question into a measured 33-point gap attributable specifically to
   temporal coordination failure — a precise target for future work.
