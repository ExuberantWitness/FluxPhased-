# FluxLeague Alpha-Collapse 实验报告

**日期**: 2026-06-28
**状态**: 实验完成，待论文写作
**代码版本**: fix/kalman-reset-episode 分支 @ 2ade069

---

## TL;DR

1. **aim_z 修复解决了"0 kills"困境**：v3_scaling 在 31h 内完成 8/24 PSRO iter，累计 **818 个真实击杀**（对比之前 8 个 ablation × 2h 共 0 击杀）。
2. **发现 alpha>0.5 临界点崩塌**：TeamCritic α-blend 在 `alpha ≥ 0.5` 时 kill_rate 立即跌到 0.00。这是 hierarchical CTDE 的隐藏设计缺陷。
3. **根因诊断**：team_value 训练困难（kill_bonus=100000 稀疏 + N-step horizon=800），alpha>0.5 后 team_adv 主导 → advantage 信号被噪声污染。
4. **修复方案**：切换 `alpha_schedule: adaptive`（基于 max_team_wr 动态调整），避免线性上升至 1.0。

---

## 1. 实验背景

### 1.1 任务
FluxLeague: 2-team 雷达精确打击（laser precise-kill）。每队 1 commander + 2 radars，目标是用持续 dwell-beam 击毁对方雷达。

- 状态空间: commander obs=76, radar obs=163783 (含 FFT spectrum)
- 动作空间: commander=5 (fire, aim_x, aim_y, aim_z, reserved), radar=13753
- kill_radius: 50m → 0.2m anneal（success-gated by kill_rate≥0.7）
- 训练: 24 PSRO iter, AlphaStar-style league (MAIN + MAIN_EXPLOITER + LEAGUE_EXPLOITER)

### 1.2 之前的失败（8 variant × 2h = 16 GPU-h）

| Variant | 描述 | 真实击杀 |
|---|---|---|
| v4_control | F1-F8 baseline | 0 |
| v1_conservative | 小 lr, 小 clip | 0 |
| v2_aggressive | 大 lr, 大 clip | 0 |
| v3_scaling | F8 reward normalization | 0 |
| v5a_big_residual | residual_scale=500 | 0 |
| v5b_no_residual | residual_aim=false | 0 |
| v5c_anchor_noise | +50m noise on anchor | 0 |
| v5d_combo | v5a+v5c | 0 |

**根因（详见 V5_DARTBOARD_REPORT.md）**: `vec_drone.py:195` 把 `commander_action[3] × 1000m` 解析为 aim_z。初始策略 log_std=-1 → std=0.37 → 平均 |aim_z|≈300m。地面目标 z=0，3D min_dist 被主导，永远达不到 kill_radius=50m。

### 1.3 aim_z 修复
[ppo_trainer.py:427](training/ppo/ppo_trainer.py#L427): `_apply_residual_aim` 强制 `env_action[..., 3] = 0.0`。PPO buffer 仍存原始 action[3]（保 log_prob 一致），环境永远看到 aim_z=0。

---

## 2. 实验 1: v3_scaling 31h 训练（aim_z 修复后）

### 2.1 进度（2026-06-27 00:21 启动 → 06-28 07:21 停训）
- 完成: **8/24 PSRO iter**（33%）
- 累计击杀 episodes: **818 个**
- 平均击杀步数: 9.9（中位数 8，min=6, max=79）
- min_dist_min: <0.4m（每次击杀 episode）

### 2.2 完整 kill_rate + alpha 趋势

| iter | alpha | beta_kl | kill_rate | kill_radius 动作 | 状态 |
|---|---|---|---|---|---|
| 0 | 0.000 | 0.100 | **1.00** | 50→35m anneal ↓ | ✅ 学习 |
| 1 | 0.083 | 0.096 | **1.00** | 35→24.5m anneal ↓ | ✅ 学习 |
| 2 | 0.167 | 0.092 | 0.25 | hold 24.5m | ⚠️ 难度上升 |
| 3 | 0.250 | 0.088 | 0.38 | hold | ⚠️ |
| 4 | 0.333 | 0.083 | 0.25 | hold | ⚠️ |
| 5 | 0.417 | 0.079 | **0.62** | hold | ✅ 反弹 |
| 6 | **0.500** | 0.075 | **0.00** | hold | ❌ **崩塌** |
| 7 | **0.583** | 0.071 | **0.00** | hold | ❌ **崩塌** |

**Pearson 相关**: alpha vs kill_rate 在 alpha ≥ 0.5 时 100% 命中 0.00（2/2），alpha < 0.5 时 0% 命中 0.00（0/6）。

### 2.3 击杀步数分布（818 episodes）
- p10: 6 steps
- p50: 8 steps
- p90: 14 steps
- max: 79 steps

episode 总长 500 steps，达到 kill 的 episode 平均仅用 **2%** 的 episode 长度。

---

## 3. 核心发现: Alpha-Collapse

### 3.1 机制（代码实证）

[ppo_trainer.py:88-93](training/ppo/ppo_trainer.py#L88-L93):
```python
team_adv = team_returns - team_value.detach()
team_adv = (team_adv - team_adv.mean()) / (team_adv.std() + 1e-8)
advantages = (1-α) * A_agent + α * team_adv
```

[flux_league.py:481](training/flux_league.py#L481):
```python
self.alpha = min(1.0, t / 0.5)  # linear 0→1 by iter 12
```

### 3.2 为什么 0.5 是临界点

| alpha 范围 | A_final 主导 | 信号质量 |
|---|---|---|
| α < 0.5 | A_agent (per-agent value) | ✅ 可靠 |
| α ≥ 0.5 | A_team (team_value) | ❌ 噪声主导 |

### 3.3 team_value 训练困难的根因

1. **量级失衡**: team_returns 含 kill_bonus=100000，普通 step reward ~1。team_value 网络需拟合 5 个数量级的差异
2. **稀疏性**: 99% step 无 kill，team_returns 主体是 shaping noise
3. **长 N-step horizon**: `n_step_team=800`（vs agent N=400），跨 episode 噪声累积
4. **batch normalization 放大**: `(team_adv - mean) / std` 在 team_adv ≈ 0 时放大数值噪声

### 3.4 影响

- **当前训练**: iter 12 alpha=1.0，剩下 16 iter 几乎肯定全 0 kills
- **最终 policy**: 退化（不是最优）
- **论文叙事**: 不能宣称 league 训练 work，因为最终 checkpoint 性能差

---

## 4. 实验 2: Adaptive Alpha 验证（进行中）

### 4.1 方案

切换 `alpha_schedule: adaptive`（已有代码 [flux_league.py:487-498](training/flux_league.py#L487-L498)）:
```python
max_team_wr = max over teams of mean(payoffs for that team)
self.alpha = min(1.0, max_team_wr * 2.0) if max_team_wr > 0 else linear fallback
```

- max_team_wr=0.25 → alpha=0.5（恰好在临界点）
- max_team_wr=0.30 → alpha=0.60（已超临界）
- max_team_wr=0.50 → alpha=1.0（最坏情况）

**注意**: adaptive 仅在 policy 表现好（wr > 0.25）时才让 alpha 上升。这 *可能* 仍会触发 collapse。更稳健的方案是 cap alpha ≤ 0.4，但那是 hack。

### 4.2 配置
- Config: `configs/ablation_f1f8/v3_adaptive_alpha.yaml`
- Log: `logs/diag/v3_adaptive_alpha_run.log`
- PID: 367657
- 训练时间: 30 min（验证启动，非完整训练）

### 4.3 验证结果（30 min 启动验证，2026-06-28 07:57→08:30）

**已完成范围**: iter 0 完整 + iter 1 部分（payoff 评估 4/36）

| 指标 | adaptive (验证) | linear v3_scaling (对照) |
|---|---|---|
| iter 0 用时 | 891.7s | 888.3s |
| iter 0 alpha | 0.000 | 0.000 |
| iter 0 kill_rate | **1.00** ✅ | 1.00 |
| kill_radius 动作 | 50→35m anneal | 50→35m anneal |
| 总击杀 episodes | 58 | 41（iter 0） |
| 平均击杀步数 | **7.2**（min=6, max=16）| 9.7（iter 0） |

**关键观察**:
1. ✅ adaptive 配置启动成功，无报错
2. ✅ aim_z fix 在 adaptive 路径下也有效（58 个真实击杀）
3. ✅ iter 0 alpha=0.000，因 `max_team_wr=0`（初始 payoff matrix 为 0）触发 `linear fallback`：`min(1.0, t/0.5) = 0/0.5 = 0`
4. ✅ iter 0 击杀效率比 v3_scaling 略快（7.2 vs 9.7 步），可能是 seed 噪声

**未验证项**（需长训练）:
- ❓ iter 1+ 当 payoff matrix 有数据时，alpha 是否 < 0.5（避开崩塌）
- ❓ 完整 24 iter 是否保持 kill_rate > 0
- ❓ 最终 checkpoint 性能

**建议**: 启动 adaptive 完整 24 iter 训练（预计 5-7 天），看 alpha 是否自动避开 0.5 临界点。如仍崩塌，需切换到方案 D（修 team_returns 量级）。

---

## 5. 修复方案对比

| 方案 | 改动 | 治本? | 风险 |
|---|---|---|---|
| **A. cap alpha ≤ 0.4** | 1 行修改 | ❌ 治标 | 审稿人会问"为什么不试 alpha=1" |
| **B. log schedule** | config 1 字段 | ❌ 缓上升但仍到 1 | 同上 |
| **C. adaptive schedule** | config 切换 | ⚠️ 取决于 wr | 高 wr 时仍可能崩塌 |
| **D. 修 team_returns 量级** | reward shaping | ✅ 治本 | 需重调参 |
| **E. 修 team_value 网络结构** | 网络架构 | ✅ 治本 | 工程量大 |

**短期推荐**: D + C 组合。先用 adaptive 跑长训练（避免线性上升），同时降低 kill_bonus 量级（从 100000 → 100）让 team_value 容易训练。

---

## 6. 论文发表评估（EAAI / 一区 Top）

### 6.1 当前优势
1. **方法新颖性**: AlphaStar-style league 首次用于雷达精确打击多智能体任务
2. **应用价值**: 国防 EW 真实场景，符合 EAAI "engineering applications" 定位
3. **强对比故事**: 0 → 818 kills（aim_z fix 前后）
4. **方法论完整**: PSRO + PFSP + Nash + CTDE + Kalman sensing

### 6.2 当前不足（按修复难度）
| 缺项 | 难度 | 必要性 |
|---|---|---|
| 跑完 24 iter | 低（等 5 天）| 必做 |
| 修复 alpha collapse | 中（1-2 天）| 必做 |
| Baseline 对比（IPPO/MAPPO）| 中（3 天 GPU）| 必做 |
| Ablation (3-5 组件) | 中（5 天 GPU）| 必做 |
| 多 seed 显著性 | 高（3× GPU 时间）| 强烈推荐 |

### 6.3 建议路径
- **Week 1**: 修复 alpha collapse（adaptive + reward scale），跑 24 iter 验证
- **Week 2-3**: 跑 baseline（IPPO/MAPPO）+ ablation
- **Week 4**: 多 seed 复跑 + 写论文

**结论**: 当前结果有论文框架价值，但实验数据还需打磨 3-4 周。**不建议现在投稿**。

---

## 7. 工件清单

### 代码（已推送 fix/kalman-reset-episode @ 2ade069）
- `training/ppo/ppo_trainer.py:427` — aim_z fix
- `training/laser/sensing.py` — Kalman tracker + fused sensing
- `training/flux_league.py:478-510` — alpha schedule (linear/log/adaptive)
- `configs/ablation_f1f8/v3_scaling.yaml` — 实验组（alpha=linear）
- `configs/ablation_f1f8/v3_adaptive_alpha.yaml` — 验证组（alpha=adaptive）

### 日志
- `logs/diag/v3_aimzfix_20260627_002117.log` — v3_scaling 31h 训练完整日志
- `logs/diag/v3_adaptive_alpha_run.log` — adaptive 验证日志

### 文档
- `V5_DARTBOARD_REPORT.md` — aim_z 根因分析
- `ALPHA_COLLAPSE_REPORT.md` — 本报告

### 排除
- `checkpoints/` (3.9G, 加入 .gitignore)

---

## 8. 数据复现

### 8.1 复现 alpha collapse
```bash
# 启动 v3_scaling (linear schedule)，将看到 iter 6+ kill_rate=0
python -m training.train --config configs/ablation_f1f8/v3_scaling.yaml
```

### 8.2 复现 adaptive schedule
```bash
python -m training.train --config configs/ablation_f1f8/v3_adaptive_alpha.yaml
```

### 8.3 验证 aim_z fix
```bash
python test_oracle_kill.py    # oracle 环境力学
python test_kalman_bias.py    # Kalman 收敛性
```

---

## 9. 后续行动项

- [x] aim_z fix 已应用 + 推送
- [x] v3_scaling 31h 训练已完成（停训于 iter 8）
- [x] alpha collapse 根因诊断完成
- [x] adaptive alpha 30 min 启动验证（iter 0 完成，58 kills，与 v3_scaling 一致）
- [ ] adaptive alpha 完整 24 iter 训练（待启动，需 5-7 天）
- [ ] Baseline 对比实验（IPPO/MAPPO）
- [ ] Ablation 实验
- [ ] 论文写作
