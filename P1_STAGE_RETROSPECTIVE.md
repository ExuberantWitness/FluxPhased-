# P1 阶段（修改建议验证）得失复盘草稿

**日期**: 2026-06-28
**状态**: 草稿（等 B iter 1 kill_rate 数据补全后定稿）

---

## 阶段目标

按修改建议.md：
1. 治本修复 alpha>0.5 崩塌（kill_rate→0）
2. 最小实验（alpha=0.7 fixed + kr=24.5m 冻结）验证 F1+F2 方向
3. 验证方向后再大规模实验

---

## 得

### 1. 代码修复落地（核心交付）
- **F1** [buffer.py:220](training/ppo/buffer.py#L220)：重写 compute_team_returns（done-mask + RMS 归一化）
- **F2** [ppo_trainer.py:91](training/ppo/ppo_trainer.py#L91)：删 team_adv 双重归一化
- **合成数据验证**：team_value loss 改善 ~92x

### 2. 🔥 发现 2 个 pre-existing 严重 bug（**最大收获**）
**Bug #1**: [train.py:116](training/train.py#L116) `return FluxLeague(...)` → config override 全是 dead code
- 影响：**v3_adaptive_alpha 30min 验证从未真正跑过 adaptive**（实际都跑 linear）
- ALPHA_COLLAPSE_REPORT.md 关于 adaptive 的结论作废

**Bug #2**: F1 GAE 路径漏调用 compute_team_returns
- 影响：前两次 P1 跑了 ~35min F1 完全没生效（team_returns=0）
- 修复后第三次 P1 才是真正的 F1+F2 测试

**全仓 AST 审计**：training/ + radar_sim/ 无其他 return-then-dead-code 模式

### 3. S0 reward 量级真相
| 来源 | 数值 |
|---|---|
| 配置 kill_bonus (raw) | 100000 |
| 实际 team_rewards max | 91-181 |
| team_returns (F1 前, v3_scaling) | 22977 |
| team_returns (F1 后, B 实测) | 7.79-13.50 ← **改善 ~1700x** |
- ALPHA_COLLAPSE_REPORT.md 的 "100000" 数据**必须修正**

### 4. alpha schedule 真正生效
- iter 0 输出 `alpha=0.700 (schedule=constant)`（修复 train.py bug 后）
- 历史所有"linear"输出都是因为 Bug #1

---

## 失

### 1. 节奏严重低估
- 估计每 iter ~25 min，**实际 85 min**（payoff 评估 36 matchups × ~120s = 70 min 是瓶颈）
- 原计划 4 备选 × 3 iter ≈ 5h，**实际要 17h**
- 应该一开始就砍 n_eval_games: 36→12

### 2. 代码改动后没做单 iter smoke test
- F1 改完直接跑 sweep，没验证 GAE 路径调用
- 浪费 35 min × 2 次（两次 P1 都因 F1 没生效而结果无效）

### 3. 配置 review 不充分
- train.py dead code 在第一次启动就暴露（alpha=0 schedule=linear vs 配置 constant），但归因为"配置未读"，没深究
- 直到 cron 监控触发才回头审计

### 4. 没主动做边界 audit
- 修改建议明确写"F1 在 n_step>0 分支"，我没问"n_step=0 怎么办"
- 这是基础代码审计应想到的边界

---

## 关键不确定性（待 B iter 1 数据回答）

- B iter 0 kill_rate=1.00（kr=24.5m 太宽松，**这数字无意义**）
- **B iter 1 kill_rate = ?**（待 watcher kill 后从 summary 读）
  - 若 > 0 → F1+F2 修复方向正确，可进 A/C/D 对照或直接长训练
  - 若 = 0 → F1+F2 不足，需 F3 (PopArt) 或 F4 (cap alpha)

---

## 下一阶段候选方向

| 方向 | 触发条件 | 工作量 | 备注 |
|---|---|---|---|
| **A/C/D 完整 sweep** | B iter 1 kill_rate > 0 | 3-4h（缩短版） | isolates F1/F2 各自贡献 |
| **F3 PopArt** | B iter 1 kill_rate = 0 | 2h 实现 + 4h 验证 | 自适应 value-target 归一化 |
| **F4 cap alpha + COMA** | F1+F2 不足 | 1h + 4h | 治标+counterfactual baseline |
| **直接长训练** | B iter 1 kill_rate > 0 | 5-7 天 | adaptive alpha + F1+F2 完整 24 iter |
| **重做 adaptive 验证** | 任何情况 | 30 min | 因 Bug #1 adaptive 从未真正测过 |

---

## 论文叙事修正（按修改建议 §7）

- ❌ 别写 "CTDE 隐藏缺陷"（overclaim）
- ✅ 改成 "诊断+修复三个已知反模式"
  - reward normalization [MAPPO Yu'22]
  - N-step cross-episode [GAE Schulman'16]
  - shared advantage [COMA Foerster'18]
- 加入"两个工程实现 bug"作为 supplementary：alpha schedule 配置 dead code、F1 GAE 路径漏调用

---

## 工件清单

- `training/ppo/buffer.py` — F1 重写 + ablation 开关 f1_disable
- `training/ppo/ppo_trainer.py` — F2 + GAE 路径 F1 调用补全
- `training/train.py` — Bug #1 修复
- `configs/ablation_f1f8/v3_p1_f1f2_alpha07.yaml` — B 配置
- `logs/diag/p1_B_f1f2_20260628_143504.log` — B 完整训练日志
- `logs/diag/p1_B_iter1_summary.txt` — watcher 自动生成的 iter 1 summary
- `scripts/run_p1_ablation_sweep.sh` — sweep 脚本（被 kill 未完成）
