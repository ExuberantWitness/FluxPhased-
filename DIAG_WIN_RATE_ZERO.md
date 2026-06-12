# Diagnostic Report: PSRO League Training — win_rate = 0%

**Date:** 2026-06-12
**GPU:** NVIDIA RTX PRO 6000 Blackwell (98 GB VRAM, sm_120, CUDA 13.2)
**PyTorch:** 2.12.0+cu132 | **Warp:** 1.10.1
**Config:** 25×25 = 625 elements, 2 radars, fft_size=32768, streaming mode (~3 GB VRAM)

---

## 1. What was done

Ran full 4-phase curriculum pipeline across three ablation cells:

| Cell | Meta-solver | Elo-band | Description |
|------|-------------|----------|-------------|
| R0 | Nash | No | Baseline |
| R1 | TC-DAMS (λ=0.3) | No | Task-diversity meta-solver |
| R3 | TC-DAMS (λ=0.3) | Yes | TC-DAMS + Elo-band PFSP |

Each cell ran Phase A (single-task pre-training: recon/detect/jam, 50 eps each) → Phase B (multi-task integration, 30 eps) → Phase C (3 PSRO iterations, 30 eps/iter) → Phase D (league exploiter refinement + final eval, 100 games).

### Bug fixes applied during this run

1. **`vec_battlefield.py`**: Commander obs offset for enemy pos / velocity / missile status was hardcoded at indices 68–75, which only works when `num_input_length=32`. Changed to dynamic offset `4 + 2*N_in + ...` so it works for any `num_input_length`.
2. **`train_league.py`**: `CommanderActorCritic(obs_dim=68)` → `obs_dim=76` to match the actual 76-dim commander observation (4 pos + 2×32 latent + 8 enemy/missile).
3. **`training/verify_training.py`**: Same `obs_dim=68→76` fix, plus `get_action` / `evaluate_actions` return 4 values (action, logp, value, privileged_value) but code was unpacking 3.
4. **`radar_sim/gpu/test_mfar.py`**: Per-element steering test used `N=25` (grid dimension) instead of `N=625` (total elements) for 25×25 array.
5. **`radar_sim/gpu/test_missile_env.py`**: Commander obs shape assertion expected `4+2*N_in` but actual dimension is `4+2*N_in+8`.

### Verification results (all passed after fixes)

| Test Suite | Result |
|---|---|
| `training.verify_training` (PPO gradients, 2×2) | PASS — gradients flow, loss decreases |
| `radar_sim/gpu/test_gpu_pipeline` (5 modules) | PASS — array, channel, receiver, interference, pipeline |
| `radar_sim/gpu/test_vec_env` (3/3) | PASS — consistency, benchmark |
| `radar_sim/gpu/test_2env_25` (5/5) | PASS — equivalence, independence, memory, sanity, reset |
| `radar_sim/gpu/test_mfar` (6/6) | PASS — default, mixed, BPSK, FFT, waveforms, per-element |
| `radar_sim/gpu/test_missile_env` (8/8) | PASS |
| `radar_sim/pz_gpu/test_pettingzoo` (28/28) | PASS |
| `radar_sim/evaluation/test_evaluation` (13/13) | PASS |
| `smoke_tcdams_25.py` | PASS — TC-DAMS solver, payoff matrix, diag JSON |

---

## 2. Observed results

### Phase A — Single-task pre-training

```
Task     Episode 0 policy_loss   Episode 15 policy_loss
recon    28.05  →                0.21
detect    0.09  →                0.06
jam      21.34  →                0.36
```

Policy loss converges well for all three tasks. The radar networks learn to produce valid actions.

### Phase B — Multi-task integration

```
Team 0 win_rate = 0.000
Team 1 win_rate = 0.000
```

### Phase C — PSRO population training

| Iteration | Pool | Team 0 sigma | Team 1 sigma | NashConv | H_task | effK |
|-----------|------|-------------|-------------|----------|--------|------|
| 0 | 2 policies | [1.0] | [1.0] | 0.0000 | -0.000 | 1.00 |
| 1 | 4 policies | [1.0, 0.0] | [1.0, 0.0] | 0.0000 | -0.000 | 1.00 |
| 2 | 8 policies | [1.0, 0.0, 0.0, 0.0] | [1.0, 0.0, 0.0, 0.0] | 0.0000 | -0.000 | 1.00 |

All three cells (R0/R1/R3) show identical behavior:
- **sigma always concentrates on the first policy** (weight = 1.0, rest = 0.0)
- **NashConv = 0** — the payoff matrix is flat (all entries equal)
- **Task entropy = 0** — no task diversity in population
- **effective K = 1** — no meaningful population diversity

### Phase D — Final evaluation

```
R0: Team 0 win_rate = 0/100 (0.00%)    Team 1 win_rate = 0/100 (0.00%)
R1: Team 0 win_rate = 0/100 (0.00%)    Team 1 win_rate = 0/100 (0.00%)
R3: Team 0 win_rate = 0/100 (0.00%)    Team 1 win_rate = 0/100 (0.00%)
```

No kills occur in any evaluation game. All episodes end via truncation (timeout).

---

## 3. Root cause analysis

### Primary cause: Episode too short for missile kill

The production config has:

```python
LEAGUE_DEFAULTS = {
    "max_steps_per_episode": 50,        # ← too short
    "episodes_per_training": 50,
}
```

The missile physics requires ~600 steps for a cruise missile to travel from one side of the 20 km × 20 km battlefield to the other (missile speed = 62,500 m/s in env, but each CPI step corresponds to a large time increment with fft_size=32768). With `max_steps_per_episode=50`, **no missile ever reaches its target** within an episode.

Consequences:
1. **No kill events** → reward signal for the most important strategic action (launch + guide missile to kill) is never triggered
2. **Flat payoff matrix** → all policy matchups result in the same outcome (timeout, no kill), so NashConv = 0 and sigma concentrates on one policy
3. **No gradient signal for missile guidance** → the commander's launch decision and the radar's beam steering for missile support never receive meaningful feedback
4. **PSRO adds policies but they are indistinguishable** → new policies can't exploit existing ones because there's no win/loss differentiation

### Secondary causes

| Factor | Current | Issue |
|--------|---------|-------|
| `max_steps_per_episode` | 50 | Too short for any combat resolution (~600 steps needed) |
| `episodes_per_training` | 50 | Low sample efficiency for 25×25 array (163,783-dim obs, 13,753-dim action) |
| `n_radars=2` | 2 radars (1v1) | Minimal team interaction; reduced strategic complexity |
| Reward sparsity | Kill bonus = 10.0 | Only awarded on actual kill; with no kills, shaped reward dominates but is weak |
| Commander training | Separate PPO | Commander needs to coordinate launch timing + target selection, but with 50-step episodes it never sees the outcome of its launch decision |

### Why policy_loss still converges in Phase A

Phase A pre-trains on single tasks (recon/detect/jam) using shaped rewards (SNR, coverage, etc.) that are achievable within a few steps. The radar learns to steer beams and generate waveforms. This is **purely perceptual** learning — no combat strategy involved. The loss convergence here is expected and does not imply combat capability.

---

## 4. Recommended fixes

### Critical (must fix for non-zero win rate)

```python
LEAGUE_DEFAULTS = {
    "max_steps_per_episode": 1000,      # 50 → 1000: allow missile flight (~600 steps)
    "episodes_per_training": 500,       # 50 → 500: more data for 13,753-dim action space
}
```

### Important (improve learning efficiency)

```python
CURRICULUM_DEFAULTS = {
    "phase_a_episodes": 200,            # 50 → 200: stronger single-task base
    "phase_b_episodes": 100,            # 30 → 100: more multi-task experience
    "phase_c_iterations": 5,            # 3 → 5: more PSRO iterations for diversity
    "phase_c_episodes_per_iter": 100,   # 30 → 100: more games per iteration
    "phase_d_episodes": 100,            # 40 → 100: more exploiter training
}
```

### Nice to have

- Increase `n_radars` from 2 to 4 (standard 2v2 adversarial setup) for richer team interaction
- Use `num_envs=2-4` instead of 1 for better GPU utilization and sample diversity
- Add curriculum on `max_steps_per_episode`: start at 200 in Phase A, ramp to 1000 by Phase C
- Tune reward shaping: increase `missile_guidance_weight` and add intermediate rewards for missile proximity to target

### Estimated training time with recommended settings

With `max_steps=1000` and `episodes=500`:
- Phase A: ~4 hours (3 tasks × 200 eps × ~4 min/ep at 1000 steps)
- Phase B: ~2 hours
- Phase C: ~10 hours (5 iters × 100 eps × ~4 min/ep)
- Phase D: ~4 hours
- **Total: ~20 hours per cell, ~60 hours for all 3 cells**

With the RTX PRO 6000's 98 GB VRAM, `num_envs=4` would cut this to ~15 hours total by batching 4 environments per step.

---

## 5. Hardware & resource summary

| Resource | Used | Available |
|----------|------|-----------|
| GPU VRAM | 14 GB | 98 GB |
| GPU utilization | 68-74% | 100% |
| System RAM | 13 GB | 91 GB |
| Disk (checkpoints) | ~500 MB | ample |
| Training time per cell | ~2 hours | — |
| Total (3 cells) | ~6 hours | — |

No OOM issues. The streaming mode (`cpi_preallocate=False`) keeps VRAM at ~14 GB, leaving ample headroom for `num_envs=4` or larger batch sizes.

---

## 6. Checkpoint inventory

```
checkpoints/
├── league_R0_seed42/     (14 files, ~180 MB) — Nash baseline, Phase A→D complete
├── league_R1_seed42/     (14 files, ~180 MB) — TC-DAMS, Phase A→D complete
└── league_R3_seed42/     (14 files, ~180 MB) — TC-DAMS+Elo, Phase A→D complete
    └── pool/pool_metadata.json  (16 policies each)
```

Each cell produced 16 policies (2 teams × (1 main + 1 main_exploiter + 1 league_exploiter + PSRO additions)). All checkpoints are valid and loadable for resumed training with updated hyperparameters.
