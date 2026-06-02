# FluxPhased 零击杀修复 — 最终计划 v4

## 用户确认的所有设计决策

| 决策 | 结论 | 原因 |
|------|------|------|
| Commander 维度 | 保持 35 维，[3:35] 归零 | 保留架构，未来可扩展 |
| CRC 门控 | **禁止** | 干净分离：rule-based 仅在预训练，联赛纯 RL |
| 分场景演示 | **不需要** | HPEDF 自然切换场景，混合数据已覆盖 |
| 对抗性对手 | 双方都用 HPEDF rule-based，绝不训练 | 数据收集阶段完全基于规则 |
| BC (Behavior Cloning) | **两组实验对比** | A: BC→Critic→PPO（文献验证），B: Critic-only→PPO（我们的方案）。恢复 Radar BC，新增 Commander BC for Config A |
| KL penalty | **Config A 保留** | BC→PPO 需要 KL penalty 防止分布漂移。Config B 无 BC，不需要 KL |
| TeamCritic | **三组并行实验** | Config A: α-blend (per-agent + team), Config B: MAPPO (单一共享), Config C: AlphaStar (per-agent 全局输入) |
| 实施策略 | **全部同时实现** | urgency + N-step + γ/λ + Commander critic pretrain + 增强数据 |
| 实验设计 | **三组 A/B/C 对比** | 每组跑 3 个 PSRO 迭代，对比 wr、收敛速度、稳定性 |
| α 调度 | **多组实验** | 测试线性/对数/adaptive，最终由实验确定 |
| N-step returns | **修复并激活** | Agent: N≈400, TeamCritic: N=800 (更长视野) |
| γ / λ | 0.999 / 0.99 | 长程 credit assignment |

## Context

25×25 训练管线 wr=0.00。审计发现 35 个问题。核心矛盾：Commander 从随机初始化开始，通过稀疏 PPO 奖励（kill_bonus +10, 600+ 步延迟）学习发射——在合理的迭代次数内不可能。

## 修正后的三阶段管线

```
Phase A: 示教数据生成（HPEDF rule-based，双方对抗）
   ├─ 混合场景（HPEDF 自然切换 recon→detect→jam→comm）
   ├─ 双方都使用 HPEDF（主动对抗，非被动侦察）
   ├─ 增广：信道噪声 + 目标位置扰动 + SNR 阈值变化 + W_TASK/配额噪声
   ├─ 50+ 集/队（从 20 增加到 50）
   └─ 显式覆盖率检查 → 不达标继续收集
        ↓
Phase B: Critic 预训练（仅 MSE on MC returns，无 BC）
   ├─ Radar Value Head (已有，保留)
   ├─ Commander Value Head (新增)
   └─ 删除所有 BC (actor pretraining) 和 KL penalty 代码
        ↓
Phase C: PSRO 联赛训练（纯 RL）
   ├─ 无 force-launch，无 CRC 门控
   ├─ TeamCritic 活跃（α: 0→1 递增）
   ├─ N-step returns N=400（替代 GAE）
   ├─ γ=0.999, λ=0.99 (长程 credit assignment)
   └─ Commander 从 critic-pretrained value head 学习发射
```

---

## 实验设计：Critic 架构三组对比

### 公共基础（三组相同）
- 增强示教数据（HPEDF 双方对抗 + 增广 + 覆盖率检查）
- N-step returns (agent N=400, team N=800)
- γ=0.999, λ=0.99
- urgency_penalty 提高（-0.01 → -0.05/step）
- max_steps=2000

### 三组实验

| 维度 | A: BC→Critic→PPO | B: Critic-only→PPO | C: BC→Critic 无 TeamCritic |
|------|-------------------|--------------------|----------------------------|
| **预训练** | BC(Radar+Commander) + rollout → Critic pretrain | 仅 Critic pretrain（当前方案） | 同 A |
| **Actor 初始化** | BC-pretrained | 随机 | BC-pretrained |
| **Critic 架构** | α-blend (agent+team) | α-blend (agent+team) | 仅 agent critic（无 TeamCritic） |
| **KL penalty** | 有 (β: 0.1→0) | 无 | 有 |
| **文献** | Kernbach 2026 +86% efficiency | 新颖（无直接支撑） | 消融实验 |
| **目的** | 文献验证基线 | 测试 critic-only 可行性 | 测试 TeamCritic 必要性 |

---

## 实施步骤

### Step 1: 删除 BC 和 KL penalty 死代码

**删除**:
- `flux_league.py::pretrain_actor_bc()` (lines 776-881)
- `phased_trainer.py::_run_bc_pretrain()` (lines 239-287)
- `ppo_trainer.py::PPOTrainer.update()` 中的 KL penalty 分支 (lines 109-113)
- `buffer.py` 中任何 KL/pretrain_log_probs 占位代码
- config 中 `bc_pretrain_epochs`, `bc_pretrain_batch_size`

**影响**: ~120 行删除。管线简化为：data collection → critic pretraining → PSRO

### Step 2: 增强示教数据生成

**2a: 对抗性对手**
- 修改 `data_collector.py`：不再给对手传 passive recon，而是双方都调用 `hpedf_radar_policy` + `hpedf_commander_policy`
- 双方团队都用完整的 HPEDF 策略（场景分类 + 动态配额 + CRC 发射）

**2b: 增广**
- 在 `_hpedf_scheduler.__call__()` 中添加可选的噪声：
  - `W_TASK` 权重 ±10% 随机扰动
  - `SCENE_QUOTAS` 配额 ±5% 随机扰动  
- 在 `data_collector.py` 中添加：
  - 目标初始位置 ±2000m 高斯扰动
  - 信道 noise_std × 随机因子 [0.5, 2.0]

**2c: 集数 + 覆盖率**
- `critic_pretrain_episodes`: 20 → 50
- 收集后检查：4 种场景各 ≥5 次、任务分配波动 >10%、≥30% episode 有导弹发射
- 不达标自动增加集数

### Step 3: Commander Critic 预训练

**新增** `flux_league.py::pretrain_commander_critic()`:
- 输入: `commander_obs` [T, 68], `commander_returns` [T]
- 训练: `ac.shared(obs) → ac.value_head(features)`, MSE loss
- 每个 active trainer 独立训练，50 epochs

**修改** `phased_trainer.py::_run_critic_pretrain()`:
- 在 `pretrain_critic()` 之后调用 `pretrain_commander_critic()`

### Step 4: 激活 N-step Returns

**4a**: 修改 `TeamPPOTrainer.update()`：
- 调用 `buffer.compute_n_step_returns(N=400)` 替代 `buffer.compute_returns()` (GAE)

**4b**: 修改 config：
- `gamma: 0.99 → 0.999`
- `gae_lambda: 0.95 → 0.99`

### Step 5: 激活 TeamCritic（CTDE 架构）

**架构**: TeamCritic 为 3 个 agent（2 radar + 1 commander）提供全局视野的 value estimate，与各 agent 的局部 critic 通过 α 融合。

```
TeamCritic(全局状态 88 维) → V_team(s_global)
  ├─ commander_obs [68]  — 战场全局视图
  ├─ task_fingerprint [8] — 两队任务分配
  ├─ avg_SNR [4]          — 平均信噪比
  ├─ alive [2]            — 两队存活状态
  └─ missile_state [6]    — 导弹位置/速度/目标

每个 Agent 的局部 critic:
  RadarCritic(state_radar) → V_radar
  CommanderCritic(cmd_obs) → V_cmd

分层 advantage（PPO update 中）:
  A_agent = R + γV_agent(s') - V_agent(s)     # 局部 advantage (GAE/N-step)
  A_team  = R_team + γV_team(s') - V_team(s)  # 全局 advantage
  A_final = (1-α)*A_agent + α*A_team          # α: 0→1 递增调度

TeamReward:
  R_team = 0.1 * Σ(all_radar_rewards) + 1.0 * (kill_bonus - death_penalty)
```

**为什么 TeamCritic 能解决 Commander 发射学习**:
- TeamCritic 看到 `missile_state`（导弹飞行状态）→ 能预测发射后的 kill_bonus
- Commander 的局部 critic 只有 68 维 `cmd_obs`，看不到导弹状态细节
- TeamCritic 的全局视野填补了这个信息缺口
- α 递增确保 Commander 先靠局部信号训练基础，后期 trust team signal

**5a**: 修复 `PPOTrainer.update()`:
```python
# 在 advantages 标准化之前添加
if team_critic is not None and alpha > 0:
    team_state = build_team_state(batch, env_info)  # [B, 88]
    team_value = team_critic(team_state)              # [B, 1]
    team_advantage = compute_team_advantage(batch, team_value, gamma)
    advantages = (1 - alpha) * advantages + alpha * team_advantage
```

**5b**: `RolloutBuffer` 添加 `team_rewards` 字段
**5c**: `store_transition()` 中计算 R_team
**5d**: TeamCritic 独立更新（MSE on team_returns）

### Step 6: Bug 修复（审计发现）

**6a**: 修复 CRC 跨队共享计数器
- `env._hpedf_crc_counter`: `[E]` → `[E, n_teams]`

**6b**: 修复 `fill_fraction`（已在之前修复）

**6c**: 修复 `periodic_update` 不一致
- 第 553 行的 `trainer.update()` 也要传 `team_critic, alpha, beta_kl`

---

## 修改文件清单

| 文件 | 改动 | 类型 |
|------|------|------|
| `flux_league.py` | 新增 `pretrain_commander_critic()`、TeamCritic 更新逻辑、alpha 调度修复、删除 BC、修复 CRC gate | 核心 |
| `ppo_trainer.py` | N-step 切换、TeamCritic + alpha blend in loss、删除 KL | 核心 |
| `ppo_buffer.py` | `team_returns` 字段、删除 pretrain_log_probs | 核心 |
| `phased_trainer.py` | 调用 Commander critic pretrain、删除 BC、对抗性数据收集 | 管线 |
| `scripted_policy.py` | CRC 计数器 `[n_teams]` 修复、增广噪声参数 | 修复 |
| `data_collector.py` | 对抗性对手、增广、覆盖率检查 | 增强 |
| `ppo_actor_critic.py` | TeamCritic 可能优化 | 已有 |
| `configs/league_25x25_12env.yaml` | γ/λ/N-step/config 更新、删除 BC 项 | 配置 |

## 配置变更

```yaml
ppo.shared:
  gamma: 0.999              # 0.99→0.999
  gae_lambda: 0.99          # 0.95→0.99
  n_step_returns: 400       # 新增

training:
  critic_pretrain_episodes: 50    # 20→50
  critic_pretrain_epochs: 50
  commander_critic_pretrain_epochs: 50  # 新增
  max_steps_per_episode: 2000
  # bc_pretrain_epochs — 已删除

reward_shaping:
  team_reward_weight: 0.1         # 新增: w1 for radar sum
  team_kill_weight: 1.0           # 新增: w2 for kill
```

## 验证方案

1. **示教数据**: 覆盖率检查通过（4 场景 + 任务波动 + 发射率 ≥30%）
2. **Critic 预训练**: Radar + Commander value_loss 均 < 0.1
3. **TeamCritic**: team_value_loss 在训练中收敛
4. **N-step returns**: 确认使用 `compute_n_step_returns(N=400)`
5. **PSRO Iteration 0**: wr > 0.0 within first 100 episodes
6. **无 BC/KL**: 确认所有 BC 和 KL 代码已删除
