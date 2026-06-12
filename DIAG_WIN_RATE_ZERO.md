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

### Primary cause: Ran stale monolithic `train_league.py` instead of modular `training/` package

> **Update (2026-06-13):** `train_league.py` has been deleted. The correct training system is the modular `training/` package invoked via `python -m training.train --config configs/league_25x25_configA.yaml`.

The experiment was run using the monolithic `train_league.py` (last updated 2025-05-22), which had **fatal parameter errors** compared to the correct modular config `configs/league_25x25_configA.yaml`:

| Parameter | `train_league.py` (wrong) | `configA.yaml` (correct) | Impact |
|-----------|--------------------------|--------------------------|--------|
| `pulses_per_cpi` | 1 | 4 | 4× CPI resolution loss |
| `speed_ms` | not set (default 244.4) | 62,500 | **3 orders of magnitude** — missile flies 2.44 cm/step instead of 25 m/step |
| `max_steps_per_episode` | 50 | 2000 | Episode too short even with correct speed |
| `n_radars` | 2 | 4 | 1v1 instead of 2v2 |
| `num_envs` | 1 | 4 | Under-utilizes 98 GB VRAM |

**Time-scale mismatch**: With `speed_ms=244.4` m/s and `dt≈0.1 ms` (implied by `pulses_per_cpi=1`), each step moves the missile **2.44 cm**. A 15 km intercept requires ~615,000 steps, but episodes are capped at 50. The missile never reaches its target.

With `configA.yaml`: `speed_ms=62500` m/s × `dt≈0.4 ms` = **25 m/step**. A 15 km intercept takes ~600 steps, well within the 2000-step episode limit.

### Secondary causes

| Factor | `train_league.py` | `configA.yaml` | Issue |
|--------|-------------------|----------------|-------|
| `episodes_per_training` | 50 | 500 | Low sample efficiency for 625-element array |
| `n_radars` | 2 (1v1) | 4 (2v2) | Minimal team interaction |
| Reward sparsity | Kill bonus = 10.0 | Kill bonus = 10.0 | Only awarded on actual kill — with correct physics, this becomes reachable |
| Commander training | Separate PPO | Separate PPO | With correct physics, launch decision has measurable outcome |

### Why policy_loss still converges in Phase A

Phase A pre-trains on single tasks (recon/detect/jam) using shaped rewards (SNR, coverage, etc.) that are achievable within a few steps. The radar learns to steer beams and generate waveforms. This is **purely perceptual** learning — no combat strategy involved. The loss convergence here is expected and does not imply combat capability.

---

## 4. Resolution

**2026-06-13 fix:**
1. Deleted stale `train_league.py` from repository
2. All dependent test files (`tests/minimal_detect_test.py`, `tests/selfplay_detect_test.py`) migrated to use modular `training/` package imports
3. Correct training invocation: `python -m training.train --config configs/league_25x25_configA.yaml`
4. `configA.yaml` already has correct parameters: `speed_ms=62500`, `pulses_per_cpi=4`, `max_steps=2000`, `n_radars=4`, `num_envs=4`

### Expected behavior with correct config

With `speed_ms=62500` and `max_steps=2000`, a missile can traverse ~50 km — far exceeding the 20 km battlefield. Kills should occur within the first few hundred steps of each episode, providing the reward signal needed for PSRO to differentiate policies (non-zero NashConv, effK > 1, win_rate > 0).

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
