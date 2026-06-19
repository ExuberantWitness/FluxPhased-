# 实验设计：FluxLeague 多智能体激光精确击杀 — Q1 论文级

**Date:** 2026-06-19
**Target:** Q1 期刊（IEEE TPAMI / TNNLS / ICML / NeurIPS）
**Contribution angle:** 应用主导（首个 alpha-star 风格 3-role 联赛用于多智能体激光精确击杀） + 理论主导（TC-DAMS 收敛性、kill_radius curriculum 最优退火、CTDE credit assignment）

---

## 1. 核心假设（论文要证明的）

| 编号 | 假设 | 反例（要排除） |
|---|---|---|
| **H1** | FluxLeague 3-role + Nash + TC-DAMS 比 PSRO-lite / vanilla self-play 学得**更快**且**更强** | 单纯增加样本量即可达到同样效果 |
| **H2** | TC-DAMS 在 task-axis 多样性上比 uniform/Nash meta-solver 更有效 | TC-DAMS 提升仅来自随机性 |
| **H3** | kill_radius success-gated curriculum 比 fixed / time-decay curriculum 样本效率更高 | 任何 curriculum 都能到 0.2m，时间无关 |
| **H4** | CTDE TeamCritic 比 decoupled critic 在 jamming/kill credit 上更准确 | CTDE 与 IIPPO 无显著差异 |
| **H5** | 3-role exploiters 比 main-only 联赛产生更多样化、可泛化的 policy | exploiter 角色对最终 main 强度无影响 |

**5 个假设都成立才能论证 FluxLeague 是 "alpha-star-style league for laser precise-kill" 的充分设计**。

---

## 2. 实验矩阵

### 2.1 主实验（Cell A-G，应用贡献）

每个 cell 跑 **3 seeds**，30 PSRO iters（完整 league）。

| Cell | 配置 | 验证假设 | 关键改动 |
|---|---|---|---|
| **A: FluxLeague (full)** | 3-role + Nash + TC-DAMS + curriculum + CTDE | （主方法基线） | `laser_25x25_pro6000_league.yaml` 原配置 |
| **B: PSRO-lite** | train_laser.py PSRO-lite（frozen pool） | H1 | `laser_25x25_pro6000.yaml` |
| **C: Self-play** | 单 policy per team，无 league | H1 | 关闭 league，每队 1 policy |
| **D: No-TC-DAMS** | meta_solver="nash" | H2 | `league.meta_solver=nash` |
| **E: No-curriculum** | kill_radius 固定 25m | H3 | `training.kill_rate_threshold=∞` |
| **F: No-CTDE** | team_critic_enabled=False | H4 | `league.team_critic_enabled=False` |
| **G: No-exploiters** | 仅 main policy（无 main_exploiter/league_exploiter） | H5 | league_cfg.mutation=None |

### 2.2 理论分析（无 GPU 消耗）

| 分析 | 数学工具 | 论文呈现形式 |
|---|---|---|
| TC-DAMS 收敛性 | 鞅收敛定理 + Lyapunov | Theorem 1 + 证明 |
| kill_radius 退火最优性 | regret bound + success-gated SGD 类比 | Theorem 2 + 证明 |
| CTDE 信用分配方差 | bias-variance decomposition | Proposition + 推导 |

理论部分**必须有对应的实证验证**（如 TC-DAMS 收敛性 → 画 sigma-entropy vs iter 曲线；curriculum 最优性 → 画 kill_radius vs wall-clock 对比固定退火）。

---

## 3. 度量指标

### 3.1 主指标（论文表格用）

| 指标 | 单位 | 期望值（cell A） |
|---|---|---|
| **Final kill_radius** | m | ≤ 0.5m（接近 train_laser 的 0.2m） |
| **Wall-clock to kr=1m** | GPU-hours | < 50h |
| **Final NashConv** | win-rate | < 0.05 |
| **Final effK** | policies | > 3.0 |
| **Held-out exploitability** | win-rate vs unseen opponent | > 0.7 |

### 3.2 训练曲线指标（论文 figure 用）

- **kill_radius vs PSRO iter**（不同 cell 对比，证明 curriculum 优势）
- **NashConv vs PSRO iter**（不同 cell 对比，证明 league 优势）
- **Task fingerprint entropy vs iter**（证明 TC-DAMS 维持多样性）
- **Cross-team win rate matrix heatmap**（最终 policy pool 可视化）

### 3.3 消融指标

- 每个去掉的组件（TC-DAMS/CTDE/exploiters）造成的退化百分比
- 形式：`performance_loss(cell_X) = (metric_A - metric_X) / metric_A × 100%`

---

## 4. GPU 预算估算

### 4.1 单次 run 成本

- 30 iters × ~2h/iter（当前小批量配置的延伸） = **60 GPU-hours/run**
- 优化后（更精炼的 config）可能降到 30-40h

### 4.2 总成本

| 实验组 | cells × seeds × hours | 总 GPU-hours |
|---|---|---|
| 主实验 A-G | 7 × 3 × 60 | **1260 h** |
| 调参/失败重跑 | ~30% 缓冲 | 380 h |
| Held-out eval | 1 × 3 × 5 | 15 h |
| **总计** | | **~1655 GPU-hours** |

### 4.3 时间折算

| 资源配置 | 串行 wall-clock |
|---|---|
| 单 PRO 6000 | ~69 天 |
| PRO 6000 + 4090 | ~50 天 |
| 加 1 台 H100 | ~30 天 |
| 集群 4 GPU | ~17 天 |

**建议**：先用 PRO 6000 跑 Cell A+B（核心对比），3 seeds × 60h × 2 cells = 360h ≈ 15 天。这个能验证主假设 H1，再决定是否扩展。

---

## 5. 当前状态评估

### 5.1 已完成

- ✅ Phase D/E/reset fix：win_rate=0.50 bug 已解决（iter 0 有 8/9 非 0.5）
- ✅ kill_radius curriculum 触发（50m → 25m）
- ✅ TC-DAMS sigma 混合策略出现（iter 1: effK=1.89）

### 5.2 待解决（论文级必须修）

| 问题 | 严重性 | 修复方案 |
|---|---|---|
| iter 1 66% pairs 是 0.50 | 高 | 加长 eval（n_eval_games=50）+ kill_radius 退火更慢 |
| kill_radius 卡在 25m（目标 0.2m） | 高 | 跑 30 iters，看是否能继续退火；或调 reward shaping |
| 单 seed | 高 | 跑 3 seeds |
| 无 baseline 对比 | 高 | 实现 Cell B/C 配置 |
| 无 ablation | 高 | 实现 Cell D/E/F/G 配置 |

---

## 6. 实施路线（推荐顺序）

### Phase 1：核心 baseline 对比（~15 天）
1. 修 iter 1 0.5 占比过高问题（kill_radius 退火调慢、eval 加长）
2. 跑 Cell A（FluxLeague full）3 seeds × 30 iters
3. 跑 Cell B（train_laser PSRO-lite）3 seeds × 30 iters
4. **决策点**：Cell A 是否显著优于 Cell B？是→继续；否→重新评估贡献

### Phase 2：消融实验（~30 天）
5. 跑 Cell D（No-TC-DAMS）
6. 跑 Cell E（No-curriculum）
7. 跑 Cell F（No-CTDE）
8. 跑 Cell G（No-exploiters）
9. 每跑完一个 cell 做 ablation 分析

### Phase 3：理论分析（与 Phase 2 并行）
10. 写 TC-DAMS 收敛性证明
11. 写 kill_radius 退火最优性证明
12. CTDE 方差分析

### Phase 4：写作（~4 周）
13. 实验结果整理 + figure 制作
14. 论文初稿
15. 补实验（reviewer 提的）

---

## 7. 立即可做的第一件事

**修 iter 1 66% 0.5 问题**，因为这影响所有后续 cell。两个改动：

### 7.1 eval 配置加严

```yaml
# 修前（小批量验证用）
league:
  n_eval_games: 4
  max_steps_per_game: 200

# 修后（论文级）
league:
  n_eval_games: 50        # 12.5× 加长，降方差
  max_steps_per_game: 500  # 让真击杀有时间发生
```

### 7.2 kill_radius curriculum 退火变慢

```yaml
training:
  kill_radius_init: 100.0      # 修前 50（一开始就太严）
  kill_radius_decay: 0.7       # 修前 0.5（每次只退火 30%，更平稳）
  kill_rate_threshold: 0.7     # 修前 0.5（要求更稳定的胜利才退火）
```

跑一次小验证（3 iters）确认 0.5 占比下降到 < 30%，再启动 Cell A/B 正式实验。

---

## 8. 不做的事

- ❌ **不引入新算法**：当前 5 个组件（3-role/Nash/TC-DAMS/curriculum/CTDE）已足够多，再加会让 ablation 失控
- ❌ **不换任务**：laser precise-kill 是核心场景，论文卖点是"在真实物理任务上"
- ❌ **不做超大数据集实验**：当前 25×25 array + 2 teams 足够；扩到 50×50 没有论文价值
- ❌ **不做硬件部署实验**：DSP/FPGA 实现是另一篇论文
- ❌ **不开放 codebase 给他人 PR**：单人/小组保持代码一致性

---

## 9. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Cell A 不显著优于 Cell B | 中 | 致命 | 先做小尺度 pilot 验证 |
| kill_radius 卡在 >5m | 中 | 严重 | 加 reward shaping、调 sensing noise |
| TC-DAMS 理论证不出来 | 高 | 中 | 退化为实证规律（empirical claim） |
| 单机 GPU 不够 | 高 | 中 | 申请集群或缩短到 20 iters |
| reviewer 嫌任务太窄 | 中 | 中 | 加一节 "extensions to missile/comm tasks" |

---

## 10. 决策点

要我现在开始做哪一步？

| 选项 | 投入 | 产出 |
|---|---|---|
| 修 §7 的两个配置问题，跑小验证（3 iters） | 4-6h | 确认 0.5 占比下降，可以启动正式实验 |
| 直接启动 Cell A 3 seeds × 30 iters | 180h GPU | 主方法 baseline 数据 |
| 先写 §3 度量指标的代码（tracking infrastructure） | 1-2 days | 后续实验都能用 |
| 评审这个设计文档先 | 0 | 你给反馈，我再细化 |
