# Postmortem: 4-Way Critic Architecture Comparison for Laser Drone Training

**Date**: 2026-06-15
**Project**: FluxPhased- Laser Drone Commander Training
**Goal**: Achieve first laser kill (0.2m kill radius × 2ms continuous illumination)

## TL;DR

Tested four critic architectures (IPPO+fix, IPPO+no-decay, CTDE, MAPPO). **None achieved kills.** The diagnosis was wrong: the failure is not a critic-architecture problem, it is a **precision ceiling problem**. The policy's tanh-Gaussian noise floor (~25m at 3km engagement) is 100× larger than the 0.2m kill radius. Changing critic only reshapes advantage estimates — it cannot fix the precision gap.

## Setup

- **Env**: 25×25 phased array, 4 radars × 2 teams, 500 pulses/episode (50ms sim), kill requires 2ms continuous illumination within 0.2m radius
- **Actor**: `CommanderActorCritic` (76-dim obs → 35-dim tanh-squashed action, log_std annealed -1.0 → -6.0)
- **Reward**: dense `(1/r²)×t⁴` beam shaping + 100pt kill bonus −10pt death penalty
- **BC aux loss**: supervise `action_mean[1:3]` to match enemy_xy from obs
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell, 98GB VRAM

## Variants Tested

### A: IPPO + Critic Fix (initial plan v2)

- **What**: Orthogonal init + value_head bias=2.8 + vf_clip_range=10 + log_std config
- **BC**: 1.0 → 0.1 over 10 iters (decay)
- **Critic**: per-agent value head, trained via shared trunk
- **Result**: Best eval=144m (iter 1), regressed to **6396m** (iter 20)
- **Why failed**: When BC decays, PPO's noisy advantages drift action_head away from BC target. PPO was never strong enough to take over from BC.

### A-noDecay: IPPO + Critic Fix + BC=1.0

- **What**: Same as A, but BC weight stays at 1.0 throughout (no decay)
- **Result**: Best eval=**151m (iter 4)** — peak of all variants
- **Final**: regressed to 1338m (iter 20)
- **Why failed**: Even with BC anchor, PPO gradient still pollutes action_head. BC's target (action_mean ≈ enemy_xy in [-1,1]) is too coarse: bc_loss=0.03 → ~1700m per-axis error.

### B-CTDE: Privileged Critic for Commander

- **What**: CTDE — privileged critic sees `beam_hit_time + enemy dists/alive` (5 extra dims) during training, regular value head during deployment
- **Result**: Same crash pattern as A. Privileged info didn't change advantage estimates enough.
- **Why failed**: CTDE addresses information asymmetry, not credit assignment horizon. The kill bonus (sparse, +100) is still 100 episodes away from typical reward signal.

### MAPPO: Centralized Team Critic

- **What**: `TeamCritic` (104-dim input: commander_obs + task_fingerprint + alive + missile state) shared across team. Commander actor uses local 76-dim obs.
- **Result**: Best eval=**329m (iter 6)**, killed at iter 8 (eval=703m)
- **Comparison**: ~2× worse than A-noDecay at every checkpoint
- **Why failed**:
  1. Removing value_loss gradient from commander's shared trunk weakened representation learning
  2. Team critic adds 28 extra input dims that need their own convergence time
  3. Even if team critic's advantages were perfect, the precision ceiling (log_std floor → 25m noise) is unchanged

## Quantitative Comparison

| Variant | Iter 1 eval | Iter 4 eval | Best eval | Iter 20 eval |
|---|---|---|---|---|
| A (IPPO+fix) | 144m | 1116m | 144m (i=1) | 6396m |
| A-noDecay | 449m | **151m** | **151m (i=4)** | 1338m |
| B-CTDE | ~similar to A | — | ~150m | crashed |
| MAPPO | 399m | 650m | 329m (i=6) | killed @ iter 8 |

## Root Cause Analysis

The original diagnosis (from `diagnose_grad.py`):
- Critic `values` mean=-0.034, std=0.016 → cold-started at 0
- `value_loss=8.38` → critic completely wrong
- `policy_loss=1e-6` → PPO wasn't moving the policy
- Conclusion: "fix the critic"

**This was correct but incomplete.** Fixing the critic (Steps 1-4 of the v2 plan) did get `values.mean > 1.0` and `value_loss < 1.0`. But kills still didn't happen.

The real ceiling:

```
Required kill precision: 0.2m @ 3km altitude → angular precision 6.7e-5 rad
Action space:           [-1, 1]²  → physical [±10000m]²
log_std floor:          -6.0  → std = 2.5e-3 in [-1,1]
Physical noise at 3km:  2.5e-3 × 10000m × (3km/3km) = 25m
Precision gap:          25m / 0.2m = 125× too coarse
```

The kill bonus (+100) requires sustained 0.2m precision for 2ms = 20 pulses. PPO's stochastic policy will **never** hold aim within 0.2m by accident — exploration noise alone is 125× too large. The `(1/r²)×t⁴` reward shaping teaches "aim close" but not "aim precisely enough to kill".

## Why the Diagnosis Was Wrong

1. **`policy_loss=1e-6` is normal early PPO behavior**, not a bug. Ratios≈1 means the policy hasn't moved yet, which is expected in iter 1-3.
2. **`value_loss=8.38` is a symptom, not a cause**. Bias init fixed the cold-start, but kills still didn't happen — because value function accuracy is not the binding constraint.
3. **BC loss decreasing doesn't imply aim precision**. `bc_loss=0.03` in normalized [-1,1] space corresponds to ~1700m physical error. BC was teaching the right thing at the wrong scale.
4. **MAPPO/CTDE don't help with continuous-control precision**. They help with multi-agent credit assignment, which is not the binding constraint here.

## Recommended Next Directions

### 1. Curriculum Learning (highest priority)

Anneal `kill_radius` from 500m → 0.2m over training. Policy first learns "hit big target" (achievable), then refines. Directly addresses precision ceiling.

```python
# In training loop
kill_radius = max(0.2, 500.0 * (1.0 - psro_iter / 20))
env.battlefield.drone.kill_radius_m = kill_radius
```

### 2. Reward Shaping Rework

Current `(1/r²)×t⁴` is too dense — policy optimizes for "close enough". Replace with potential-based shaping (Ng et al. 1999) that gives reward only when within tight radius:

```python
# Old: dense reward everywhere
reward = beam_reward_weight * (1/r²) * t⁴

# New: bonus only when truly close, scaled to make 0.2m target reachable
within_close = (r < 50.0).float()
reward = beam_reward_weight * within_close * (1/r²) * t⁴
```

### 3. SAC (Soft Actor-Critic)

Entropy-regularized off-policy. Doesn't need precise value function. 5-10× more sample-efficient than PPO for continuous control. Especially good when reward is sparse.

### 4. Hard-Mining Replay

Prioritize training on the rare episodes where aim_dist < 100m. Currently buffer is uniform-sampled, drowning the rare "close" experiences.

## What Not to Do

- **Don't try more critic variants** (QMIX, MADDPG, etc.). The critic is not the binding constraint.
- **Don't anneal BC faster**. BC is the only thing keeping eval below 1000m.
- **Don't reduce log_std floor further**. Already at the edge of tanh saturation; lower floor kills gradient flow.
- **Don't add more reward shaping terms**. The reward signal is already complex; adding terms won't fix precision.

## Artifacts

- Configs: [`configs/laser_25x25_train.yaml`](configs/laser_25x25_train.yaml), [`configs/laser_25x25_train_nodecay.yaml`](configs/laser_25x25_train_nodecay.yaml), [`configs/laser_25x25_train_ctde.yaml`](configs/laser_25x25_train_ctde.yaml), [`configs/laser_25x25_train_mappo.yaml`](configs/laser_25x25_train_mappo.yaml)
- Training logs: `/tmp/laser_25x25_v9_{A,AnoDecay,BCTDE,MAPPO}.log`
- Diagnostic: [`training/diagnose_grad.py`](training/diagnose_grad.py)
- Implementation commits: `94f4c0e` (critic fix + CTDE), `9096533` (MAPPO)
