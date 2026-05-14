# FluxPhased Multi-Agent Adversarial Training: Research Report

**Date**: 2026-05-14
**Topic**: League Training / PSRO / NFSP for Radar EW Adversarial Game
**Status**: Research complete, implementation roadmap ready

---

## 1. Problem Analysis: Why Standard Self-Play Fails

### 1.1 Non-Transitive Game Structure

FluxPhased 的对抗本质是一个**非传递博弈 (non-transitive game)**：

```
Detection-heavy → beats → Comm-heavy (finds enemy, guides missile)
Comm-heavy      → beats → Jam-heavy  (maintains missile guidance)
Jam-heavy       → beats → Detect-heavy (degrades detection SNR)
Recon-heavy     → beats → Jam-heavy  (exploits jam emissions for ELINT)
```

这种石头剪刀布结构意味着：
- **纯自我对弈 (naive self-play) 会策略循环**：今天学到的克制策略，明天就被反克制
- **单一最优策略不存在**：任何纯策略都有可利用的弱点
- **需要维持策略多样性**：找到混合策略 Nash 均衡而非单一策略

### 1.2 Game-Theoretic Properties

| Property | FluxPhased Status |
|----------|-------------------|
| Zero-sum | Approximately zero-sum (kill/death symmetric ±10, emission/urgency create non-zero-sum drift) |
| Continuous action space | Radar: 13,753 dims (625 elements × 22 params), Commander: 35 dims |
| Partial observability | Each radar sees its own spectrum, not enemy's intentions |
| Temporal structure | ~32s episode, missile flight time creates timing game |
| Win condition | First blood (destroy any enemy radar) |
| Team structure | 2 teams, each: 2 radars + 1 commander, cooperative within team |

### 1.3 Training Challenges Summary

1. **Sparse reward**: Only kill/death bonuses (+/-10) are meaningful; emission cost (-0.001) and urgency (-0.01) provide negligible gradient
2. **Credit assignment**: 625 elements × 4 radars = 2500 element-level decisions contribute to one binary outcome
3. **13,753-dim continuous action**: Standard exploration noise has negligible coverage
4. **Mixed discrete-continuous**: Task selection is argmax (discrete), beam steering is continuous
5. **Two-level hierarchy**: Commander→Radar information bottleneck (16-dim instruction vs 400k-dim observation)

---

## 2. Algorithm Comparison & Selection

### 2.1 Candidate Algorithms

| Algorithm | Core Idea | Pros | Cons | Applicability |
|-----------|-----------|------|------|---------------|
| **Naive Self-Play** | Train against latest self | Simple, single policy | Strategy cycling, no diversity guarantee | ❌ Not suitable |
| **NFSP** | Best response + average policy via fictitious play | Convergence guarantee (FP → NE), well-studied | Designed for discrete actions, two networks per agent, scaling issues | ⚠️ Partially applicable (continuous action extension needed) |
| **PSRO** | Population + meta-game Nash + best response oracle | Handles non-transitivity, modular, any RL as oracle | Population grows linearly, payoff matrix evaluation cost O(N²) | ✅ Best fit for structured populations |
| **League Training (AlphaStar)** | 3 agent roles + PFSP opponent sampling | Battle-tested at scale, maintains diversity | Extremely heavy (AlphaStar: 900 agents, 44 days × 32 TPUs) | ⚠️ Simplified version applicable |
| **P2SRO (Pipeline PSRO)** | Parallel PSRO with pipelined best response | Scalable, converges faster with more workers | Requires multiple GPU workers | ✅ Good for GPU-accelerated env |

### 2.2 Recommended Approach: Hierarchical PSRO with Full League Roles

We propose a **layered architecture** combining the best elements:

```
Layer 1: PSRO Framework (population management + meta-game)
Layer 2: Full League Roles (main agent + main exploiter + league exploiter)
Layer 3: Hierarchical PPO (commander + radar per team)
Layer 4: Reward Shaping (dense intermediate rewards)
```

**Why this combination:**
- **PSRO** provides the theoretical foundation for handling non-transitive games
- **Full 3-role league** (not simplified) — at FluxPhased's scale (6 active policies on single GPU), the complete AlphaStar mechanism is affordable and provides the "strength→champion repair→population blindspot" triple defense chain that non-transitive games specifically need
- **Hierarchical PPO** matches the existing commander-radar architecture
- **Reward shaping** is critical for making the 13,753-dim radar policy learnable at all

---

## 3. Proposed Architecture: FluxLeague

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FluxLeague Manager                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Payoff      │  │  Meta-Strat  │  │  Opponent       │ │
│  │  Matrix      │  │  Solver      │  │  Sampler (PFSP) │ │
│  │  [K × K]     │  │  (Nash/Rect) │  │  (prioritized)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│              Population Pool (K policies)                │
│  ┌─ Main Agents ──┐  ┌─ Main Exploiters ─┐  ┌─ League Exploiters ─┐│
│  │ π_red_main      │  │ π_red_main_exp    │  │ π_red_league_exp    ││
│  │ π_blue_main     │  │ π_blue_main_exp   │  │ π_blue_league_exp   ││
│  └─────────────────┘  └────────────────────┘  └─────────────────────┘│
├─────────────────────────────────────────────────────────┤
│          Per-Policy Training (Hierarchical PPO)          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Commander PPO (high-level):                      │   │
│  │    obs: positions + radar latents (68-dim)        │   │
│  │    action: launch + target + instructions (35-dim)│   │
│  │    reward: kill_bonus + urgency_shaping           │   │
│  ├──────────────────────────────────────────────────┤   │
│  │  Radar PPO (low-level) × 2 per team:             │   │
│  │    obs: spectrum + state + cmd_instruction        │   │
│  │    action: per-element task + beam + waveform     │   │
│  │    reward: detection_quality + jam_effectiveness  │   │
│  │           + comm_reliability + team_outcome       │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│              GPU Simulation (MFARVecEnv)                 │
│  ┌─────┐  ┌──────┐  ┌──────┐  ┌───────┐  ┌────────┐  │
│  │E=2-8│  │R=4   │  │N=25  │  │P=4-32 │  │VecIntf │  │
│  │envs │  │radars│  │elems │  │pulses │  │+Chnl   │  │
│  └─────┘  └──────┘  └──────┘  └───────┘  └────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Agent Roles (Full League — 3 Roles)

AlphaStar 的三角色分工对应不同层次的策略缺陷检测。在 FluxPhased 的非传递博弈中，三角色各有不可替代的作用：

**Main Agent (per team, 1×2=2 total)**:
- **对手**：通过 PFSP 采样整个对手种群的所有历史策略
- **目标**：追求对全种群的综合最强胜率
- **不可重置** — 代表该队的"冠军"策略，持续进化
- **为什么需要**：追求整体强度，作为最终部署策略的来源

**Main Exploiter (per team, 1×2=2 total)**:
- **对手**：仅与当前对方 Main Agent 对战
- **目标**：暴露 Main Agent 的特定弱点
- **可重置**到早期 checkpoint — 允许探索不同的利用方向
- **为什么需要**：快速发现冠军策略的即时漏洞（比如 Main Agent 对 jam-heavy 策略没有防御），迫使 Main Agent 修补

**League Exploiter (per team, 1×2=2 total)**:
- **对手**：与整个对手种群的所有历史策略对战
- **目标**：暴露**整个种群**的集体盲区
- **可重置**到早期 checkpoint
- **为什么关键**：防止种群整体陷入策略坍缩。例如，如果所有策略都偏向 detection-heavy 而
  没人探索 recon-heavy，League Exploiter 会发现 recon-heavy 策略能克制整个种群——
  这正是非传递博弈中最需要的能力。Main Exploiter 看不到这个问题，因为它只盯着 Main Agent。

**为什么使用完整三角色而非简化版本**：

在 FluxPhased 的规模下（2 队 × 3 角色 = 6 活跃策略，种群上限 K=20），三角色相比两角色
仅增加约 30% 计算开销（多 2 个 PPO 训练实例），但提供了完整的"追求强度→修补冠军→防止种群盲区"
三重防御链路。AlphaStar 需要 900 个 agent 和 TPU 集群是因为 StarCraft 有 3 个种族、
巨大状态空间；FluxPhased 在单卡 RTX 2060 上完全可以支撑完整联赛。

**Total population**: starts at K=6 (2 main + 2 main_exploiter + 2 league_exploiter), grows as past checkpoints are preserved up to K_max.

### 3.3 PFSP Opponent Sampling

Instead of uniform sampling over opponent population, use **win-rate-weighted** sampling:

```python
def pfsp_sampling(win_rates: np.ndarray) -> int:
    """Select opponent index with probability proportional to (1 - win_rate).

    Preferentially samples opponents the current agent performs poorly against.
    win_rates[i] = current agent's win rate against opponent i.
    """
    loss_rates = 1.0 - win_rates
    probs = softmax(loss_rates / temperature)
    return np.random.choice(len(probs), p=probs)
```

### 3.4 Meta-Strategy Solver

The **meta-game** is defined by the payoff matrix `U[i,j]` = expected win rate of policy i against policy j.

**Solver options** (complexity-ordered):

1. **Uniform** (baseline): σ_i = 1/K for all i. Simple but ignores payoff structure.
2. **Nash Equilibrium** (recommended): Solve the 2-player zero-sum meta-game via linear programming. Provides game-theoretic optimality.
3. **Rectified Nash** (PSRO variant): Zero out probabilities below threshold, renormalize. Improves diversity.

For FluxPhased's scale (K ≤ 20 policies), Nash via LP is tractable (scipy.optimize.linprog, <1ms for K=20).

### 3.5 Best Response Oracle: Hierarchical PPO

Each policy in the population is a **team policy** = (commander_policy, radar_policy_shared).

**Commander PPO**:
- Input: `[4 + 2×32] = 68` dims (radar positions + encoded latents)
- Output: `[3 + 2×16] = 35` dims (launch + target + instructions)
- Network: MLP [68 → 256 → 256 → 35], separate value head
- Reward: kill_bonus × 10 + death_penalty × 10 + urgency × (-0.01) + launch_timeliness_bonus

**Radar PPO** (shared across 2 radars per team):
- Input: spectrum_encoder(state) ≈ `[256]` + vehicle `[5]` + missile `[12]` + cmd_instruction `[16]` = ~289 dims
- The existing `SpectrumEncoder` in `openevolve_search.py` already handles `[N, P, n_bins]` → `[256]`
- Output: `[13,753]` dims (per-element task + beam + waveform + vehicle)
- Network: SpectrumEncoder(CNN/Attention) + MLP → 13,753, separate value head
- Reward: dense task-specific rewards (see §3.6) + sparse team outcome

**Key: parameter sharing within team** — both radars on the same team share the same policy network, reducing parameters by half and improving sample efficiency through shared experience.

### 3.6 Dense Reward Shaping (Critical)

The current sparse reward structure (+/-10 on kill/death) makes learning impossible for 13,753-dim radar actions. We propose **per-task dense intermediate rewards**:

| Reward Component | Target | Formula | Weight |
|------------------|--------|---------|--------|
| **Detection SNR** | Radar (detect elements) | `max(0, SNR_dB - SNR_threshold_dB) / 20` | 0.1 |
| **Detection coverage** | Radar (detect elements) | `n_cells_above_threshold / n_total_cells` | 0.05 |
| **Jam effectiveness** | Radar (jam elements) | `enemy_SNR_degradation_dB / max_degradation` | 0.1 |
| **Comm reliability** | Radar (comm elements) | `CRC_pass_rate` for own team's missile guidance | 0.05 |
| **Recon intelligence** | Radar (recon elements) | `n_enemy_emissions_detected / n_possible` | 0.03 |
| **Beam accuracy** | Radar | `cos(angle_error)` toward assigned target | 0.02 |
| **Kill bonus** | Commander + Radar | +10.0 on enemy kill | 1.0 |
| **Death penalty** | Commander + Radar | -10.0 on own death | 1.0 |
| **Emission cost** | Radar | -0.001 × active_elements / N | 0.001 |

These can be computed from existing simulation outputs (CFAR detections, interference power, BPSK CRC results) with minimal overhead.

### 3.7 Action Space Decomposition

The 13,753-dim action space is too large for standard PPO. We decompose:

**Hierarchical decomposition (per element)**:
1. **Task selection head**: 4-dim softmax → argmax → {recon, detect, jam, comm}
2. **Per-task parameter head**: Only the 8-dim parameters for the selected task are active
   - Detect: [beam_az, beam_el, carrier_freq, bw, pulse_width] + 3 unused = 8
   - Jam: [beam_az, beam_el, bw, power, freq_shift] + 3 unused = 8
   - Comm: [beam_az, beam_el, carrier_freq, sym_rate, data_x, data_y] + 2 unused = 8
   - Recon: [beam_az, beam_el] + 6 unused = 8
3. **Vehicle head**: 3-dim (speed, heading, rotation)

This reduces the effective decision from 22 to ~12 dims per element (4 task logits + 8 task-specific params).

**Implementation via masked forward pass**:
```python
# Action heads per element
task_logits = network.task_head(features)         # [B, N, 4]
task_params = network.param_head(features)         # [B, N, 8]
vehicle_cmd = network.vehicle_head(global_features) # [B, 3]

# During env step: select active params based on task
task_id = task_logits.argmax(dim=-1)               # [B, N]
active_params = gather_task_params(task_params, task_id)  # [B, N, 5-6]
```

### 3.8 Curriculum: Phased Training

Training proceeds in 4 phases:

**Phase A: Single-Task Pre-Training (5K episodes)**
- Fix task assignment (all elements detect OR all recon OR all jam)
- Train each radar policy in isolation
- Purpose: bootstrap basic spectrum understanding and beam steering

**Phase B: Multi-Task Integration (10K episodes)**
- Enable full action space with task selection
- Use dense reward shaping heavily
- Self-play against random opponents

**Phase C: PSRO Population Training (20K episodes)**
- Initialize population with Phase B policies
- Run PSRO iterations: evaluate payoff matrix → compute meta-Nash → train best response
- Population grows from K=4 to K≈12

**Phase D: League Exploiter Refinement (10K episodes)**
- Add exploiter agents targeting main agent weaknesses
- PFSP opponent sampling
- Final agent = main agent's meta-strategy

---

## 4. Implementation Roadmap

### 4.1 File Structure

```
FluxPhased/
├── training/
│   ├── __init__.py
│   ├── flux_league.py          # FluxLeague manager: population, payoff, meta-solver
│   ├── ppo/
│   │   ├── __init__.py
│   │   ├── actor_critic.py     # Actor-Critic networks (commander + radar)
│   │   ├── ppo_trainer.py      # PPO training loop with GAE
│   │   ├── buffer.py           # Rollout buffer with advantage computation
│   │   └── reward_shaping.py   # Dense reward computation from env outputs
│   ├── self_play/
│   │   ├── __init__.py
│   │   ├── opponent_pool.py    # Policy checkpoint storage + PFSP sampling
│   │   ├── payoff_matrix.py    # Win rate matrix computation
│   │   └── meta_solver.py      # Nash equilibrium via LP
│   ├── curriculum/
│   │   ├── __init__.py
│   │   └── phased_trainer.py   # Phase A→D curriculum orchestration
│   └── train.py                # CLI entry point
├── configs/
│   ├── physics.yaml            # (existing)
│   ├── algorithm.yaml          # (existing, extend)
│   └── league.yaml             # NEW: league training config
```

### 4.2 Implementation Priority

| Priority | Module | Dependencies | Est. LOC |
|----------|--------|-------------|----------|
| **P0** | `reward_shaping.py` | None (reads env outputs) | ~200 |
| **P1** | `actor_critic.py` | Existing `SpectrumEncoder` | ~300 |
| **P2** | `ppo_trainer.py` + `buffer.py` | P0, P1 | ~500 |
| **P3** | `opponent_pool.py` | P2 | ~150 |
| **P4** | `payoff_matrix.py` + `meta_solver.py` | P3 | ~200 |
| **P5** | `flux_league.py` | P3, P4 | ~300 |
| **P6** | `phased_trainer.py` | P5 | ~200 |
| **P7** | `train.py` + `league.yaml` | P6 | ~100 |

**Total estimated**: ~1,950 lines of new code.

### 4.3 Key Implementation Details

#### P0: Reward Shaping

```python
class DenseRewardShaper:
    """Compute dense intermediate rewards from MFARVecEnv step outputs."""

    def __call__(self, env_outputs: dict) -> dict:
        # Detection quality: use CFAR output (already computed in vec_receiver.py)
        detect_mask = (task_ids == TASK_DETECT)  # [E, R, N]
        detect_elements = detect_mask.sum(dim=-1).clamp(min=1)  # [E, R]
        snr_reward = (cfar_detections.float().sum(dim=-1) / detect_elements) * 0.1

        # Jam effectiveness: compare interference power at enemy vs baseline
        jam_mask = (task_ids == TASK_JAM)
        jam_elements = jam_mask.sum(dim=-1).clamp(min=1)
        enemy_intf_power = compute_enemy_interference(env_outputs)
        jam_reward = (enemy_intf_power / jam_elements) * 0.1

        # Comm reliability: CRC pass rate
        crc_ok_rate = comm_crc_ok.float().mean(dim=-1)  # [E, R]
        comm_reward = crc_ok_rate * 0.05

        return {
            "snr_reward": snr_reward,
            "jam_reward": jam_reward,
            "comm_reward": comm_reward,
            "total_shaped": snr_reward + jam_reward + comm_reward,
        }
```

#### P1: Actor-Critic Architecture

```python
class RadarActorCritic(nn.Module):
    def __init__(self, spectrum_encoder, state_dim, action_dim, hidden=256):
        super().__init__()
        self.encoder = spectrum_encoder  # Existing SpectrumEncoder
        self.shared = nn.Sequential(
            nn.Linear(state_dim - spectrum_flat + 256, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # Action heads
        self.task_head = nn.Linear(hidden, 625 * 4)     # per-element task logits
        self.param_head = nn.Linear(hidden, 625 * 8)    # per-element params
        self.vehicle_head = nn.Linear(hidden, 3)         # vehicle command
        # Value head
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        spectrum = obs[..., :spectrum_flat]
        other = obs[..., spectrum_flat:]
        encoded = self.encoder(spectrum)
        features = self.shared(torch.cat([encoded, other], dim=-1))
        value = self.value_head(features)
        task_logits = self.task_head(features).reshape(-1, 625, 4)
        params = torch.sigmoid(self.param_head(features)).reshape(-1, 625, 8)
        vehicle = torch.tanh(self.vehicle_head(features))
        return task_logits, params, vehicle, value


class CommanderActorCritic(nn.Module):
    def __init__(self, obs_dim=68, act_dim=35, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.action_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        features = self.shared(obs)
        action = torch.tanh(self.action_head(features))
        value = self.value_head(features)
        return action, value
```

#### P5: FluxLeague Manager

```python
class FluxLeague:
    """Simplified AlphaStar-style league for 2-team radar combat."""

    def __init__(self, env_config, league_config):
        self.population = {}           # {policy_id: (team, role, checkpoint_path)}
        self.payoff_matrix = {}        # {(i,j): win_rate}
        self.meta_strategy = {}        # {team: np.array of mixture weights}

    def psro_iteration(self):
        """One PSRO iteration: evaluate → solve → train best response."""
        # 1. Evaluate payoff matrix (play all pairs)
        self.evaluate_payoffs(n_games=50)

        # 2. Compute meta-Nash for each team
        for team in [0, 1]:
            opponent_policies = [p for p in self.population if p.team != team]
            payoff_sub = self.get_submatrix(team, opponent_policies)
            self.meta_strategy[team] = solve_nash(payoff_sub)

        # 3. Sample opponents via PFSP for each training policy
        for policy_id, policy in self.population.items():
            opponents = self.sample_opponents_pfsp(policy_id)
            self.train_best_response(policy_id, opponents, n_episodes=1000)

        # 4. Add new best responses to population
        self.grow_population()

    def sample_opponents_pfsp(self, policy_id, temperature=1.0):
        """PFSP: sample opponents weighted by current agent's loss rate."""
        team = self.population[policy_id].team
        win_rates = np.array([
            self.payoff_matrix.get((policy_id, opp_id), 0.5)
            for opp_id in self.population if self.population[opp_id].team != team
        ])
        loss_rates = 1.0 - win_rates
        probs = softmax(loss_rates / temperature)
        return np.random.choice(opponent_ids, size=1, p=probs)[0]
```

### 4.4 Hyperparameter Configuration

```yaml
# configs/league.yaml
league:
  population_cap: 20          # max policies in population (including historical checkpoints)
  n_main_agents: 2            # 1 per team
  n_main_exploiters: 2        # 1 per team
  n_league_exploiters: 2      # 1 per team
  payoff_eval_games: 50       # games per payoff matrix entry
  meta_solver: "nash"         # "nash" | "uniform" | "rectified_nash"
  pfsp_temperature: 1.0       # PFSP sampling temperature
  exploiter_reset_prob: 0.1   # probability of resetting exploiter to earlier checkpoint

ppo:
  commander:
    lr: 3e-4
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    entropy_coef: 0.01
    n_steps: 128
    batch_size: 32
    n_epochs: 10
  radar:
    lr: 1e-4                  # lower LR for large action space
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.1           # tighter clip for stability
    entropy_coef: 0.02        # higher entropy for exploration
    n_steps: 128
    batch_size: 16
    n_epochs: 5

curriculum:
  phase_a_episodes: 5000      # single-task pre-training
  phase_b_episodes: 10000     # multi-task integration
  phase_c_iterations: 20      # PSRO iterations
  phase_c_episodes_per_iter: 1000
  phase_d_episodes: 10000     # league exploiter refinement

reward_shaping:
  detect_snr_weight: 0.1
  detect_coverage_weight: 0.05
  jam_effectiveness_weight: 0.1
  comm_reliability_weight: 0.05
  recon_intel_weight: 0.03
  beam_accuracy_weight: 0.02
```

### 4.5 GPU Utilization Strategy

Leverage the existing `MFARVecEnv` vectorization for parallel rollout collection:

```python
# Parallel rollout collection
env = MFARVecEnv(num_envs=8, ...)  # 8 parallel envs on 1 GPU
# Each env runs a different opponent matchup
# → 8× throughput for payoff matrix evaluation
# → 8× throughput for PPO rollouts

# For PSRO payoff evaluation:
# K=12 policies → 12×12=144 pairs, but only ~K/2 cross-team pairs matter
# With E=8 envs, can evaluate 8 matchups simultaneously
# 144/8 = 18 batches × 50 games = 900 games in ~2 minutes
```

---

## 5. Expected Outcomes & Metrics

### 5.1 Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Win rate vs random** | > 90% | Main agent vs random policy, 100 games |
| **Win rate vs self-play baseline** | > 60% | Main agent vs naive self-play policy |
| **Nash_conv (exploitability)** | < 0.3 | Max regret of meta-strategy in meta-game |
| **Population diversity** | > 5 effective strategies | Effective population size metric |
| **Training stability** | Monotonic improvement | Win rate trend over PSRO iterations |
| **Cross-scenario generalization** | < 20% drop | Win rate across ScenarioGenerator configurations |

### 5.2 Computational Budget Estimate

| Phase | Episodes | GPU-hours (RTX 2060) | Time |
|-------|----------|---------------------|------|
| Phase A (pre-train) | 5K × 4 tasks | ~2h × 4 | 8h |
| Phase B (multi-task) | 10K | ~4h | 4h |
| Phase C (PSRO × 20 iter) | 20K + evaluation | ~20h | 20h |
| Phase D (league) | 10K | ~8h | 8h |
| **Total** | ~45K episodes | ~40h | ~40h |

This is feasible on a single RTX 2060 over 2-3 days of continuous training.

---

## 6. Literature References

### Core Algorithms

1. **AlphaStar League Training**: Vinyals et al. (2019). "Grandmaster level in StarCraft II using multi-agent reinforcement learning." Nature.
   - [DeepMind Blog](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
   - [Original Paper (PDF)](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)

2. **PSRO**: Lanctot et al. (2017). "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning." NeurIPS.
   - [IJCAI 2024 Survey](https://www.ijcai.org/proceedings/2024/0880.pdf)
   - [arXiv HTML](https://arxiv.org/html/2403.02227v1)

3. **NFSP**: Heinrich & Silver (2016). "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games." arXiv:1603.01121.
   - [arXiv](https://arxiv.org/abs/1603.01121)

4. **Pipeline PSRO**: McAleer et al. (2020). "Pipeline PSRO: A Scalable Approach for Finding Approximate Nash Equilibria in Large Games." arXiv:2006.08555.
   - [arXiv PDF](https://arxiv.org/pdf/2006.08555)

### Radar EW + MARL

5. **IRJ-MARL** (2025). "Intelligent Radar and Jammer Zero-Sum Waveform Design." [IEEE Xplore](https://ieeexplore.ieee.org/document/10867888/)

6. **Enhanced Radar Anti-Jamming with MADRL + NFSP** (2024). [ResearchGate](https://www.researchgate.net/publication/385640320)

7. **Hierarchical MARL for Radar Collaborative Anti-Jamming** (2024). [Beijing Institute of Technology](https://pure.bit.edu.cn/en/publications/)

8. **Cooperative Jamming Decision-Making Based on MARL** (2025). [Springer](https://link.springer.com/article/10.1007/s43684-025-00090-4)

### Diversity & Non-Transitivity

9. **Policy Space Diversity for Non-Transitive Games** (2023). [arXiv](https://arxiv.org/html/2306.16884v1)

10. **Self-Play Survey** (2024). "A Survey on Self-play Methods in Reinforcement Learning." [arXiv](https://arxiv.org/html/2408.01072v1)

11. **A-PSRO** (ICML 2025). "A Unified Strategy Learning Method with Advantage Metric." [ICML](https://icml.cc/virtual/2025/poster/43705)

---

## 7. Risk Analysis & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 13,753-dim action too large for PPO | High | Critical | Action decomposition (§3.7), reduce to ~7,500 effective dims |
| Sparse reward prevents learning | High | Critical | Dense reward shaping (§3.6) is prerequisite |
| PSRO payoff evaluation too slow | Medium | High | Vectorized env (E=8), batch evaluation |
| Strategy collapse despite league | Medium | Medium | Exploiter reset, diversity metrics monitoring |
| GPU memory during training | Low | Medium | Use 5×5 config for training, 25×25 for eval only |
| HRL training instability | Medium | Medium | Phase A pre-training, lower commander LR |

---

## 8. Conclusion & Next Steps

The FluxPhased radar EW system's non-transitive adversarial structure (detect→jam→recon→detect cycle) makes it an ideal candidate for PSRO-based multi-agent training. The proposed **FluxLeague** architecture combines:

1. **PSRO framework** for population management and Nash equilibrium seeking
2. **Full 3-role league** (main agent + main exploiter + league exploiter) directly following AlphaStar's design, providing complete "strength→champion repair→population blindspot" defense chain
3. **Hierarchical PPO** matching the existing commander-radar architecture
4. **Dense reward shaping** to make the 13,753-dim radar policy learnable
5. **4-phase curriculum** for stable training progression

**Recommended first step**: Implement P0 (reward shaping) and P1 (actor-critic), validate with naive self-play before building the full PSRO infrastructure.

---

*Report generated: 2026-05-14*
*Based on: FluxPhased codebase analysis + AlphaStar/PSRO/NFSP literature survey*
