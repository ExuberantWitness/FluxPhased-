# FluxLeague 实验现象报告

> **生成日期**：2026-06-28
> **覆盖时段**：2026-06-21（Tier 1 kill-fix）→ 2026-06-28（F1–F8 修复 + 4 变体消融 + anchor 诊断）
> **目的**：把过去一周观察到的所有"反常"现象集中成一份可检索的实验档案，方便后续研究与复现。

---

## 一、研究目标回顾

FluxLeague 是一个 AlphaStar 风格的多智能体 PPO 训练框架，目标是让红蓝两队的指挥官（commander）通过 kalman 滤波 + 残差瞄准（residual_aim）+ 火力门控（hybrid_fire）学会使用机载激光武器击毁敌方雷达。

任务的成功条件：
- `kill_rate ≥ 0.7` 时 `kill_radius` 从 50m 退火到 0.2m
- 在 0.2m 真实物理阈值下保持击杀率

EAAI Q1 论文框架：**League 框架本身作为 SOTA**，IPPO / MAPPO 作为 baseline。

---

## 二、四阶段实验时间线

| 阶段 | 日期 | 配置代号 | 训练时长 | 主要结果 |
|------|------|---------|---------|---------|
| Tier 1 kill-fix | 2026-06-21 | `laser_league_tier1_kill_fix` | ~6 h | `policy_loss` 正常下降，`vl` 1.9M→132K，`kr` 50→25m，3/98 真实击杀 |
| R0-R5 league | 2026-06-23 | `laser_league_R01234` | ~7 h | `policy_loss` → 0（不学习），`kr` 仍能 50→12.5m，4/293 真实击杀 |
| F1–F8 修复 + 4 变体消融 | 2026-06-25 | `v1_conservative` / `v2_aggressive` / `v3_scaling` / `v4_control` | 每个约 2 h | 4 个变体全部 0 真实击杀，`vl` 差异巨大 |
| Anchor 诊断 | 2026-06-28 | `v3_scaling` + ANCHOR-* 日志 | ~20 min | **本次诊断的核心**：报告的 `min_dist_init=139m` 是统计假象，anchor 实际工作正常 |

---

## 三、核心实验现象

### 现象 1：`kill_rate` 上升 ≠ PPO 在学习

**观察**：所有四个阶段 `kill_radius` 都能从 50m 退火下去，看起来"任务在进步"。

| 阶段 | kr 起始 → 终止 | 真实击杀率 | `policy_loss`（commander） |
|------|---------------|-----------|---------------------------|
| Tier 1 | 50m → 25m | 3/98 (3.1%) | 0.35 → 0.30（健康） |
| R01234 | 50m → 25m → **12.5m** | 4/293 (1.4%) | **-0.004 → 0.0001**（崩塌） |
| F1-F8 诊断 (本次) | 50m → 35m | **58/58 (100%)** | 0.025 → 0.020（可疑低） |

**关键反常**：
- R01234 的 `policy_loss` 已坍缩到 ~0，但 kr 仍能退火到 12.5m（比 tier1 更深）。
- 本次诊断 100% 击杀（58/58），远高于历史 1-3%。

**结论**：`kill_rate` 由 `kill_radius` 阈值主导，阈值越大当然"击中"越多；它**不能**反映 PPO 是否真的在学习策略。kr 退火只看到阈值放宽，没看到策略收敛。

---

### 现象 2：`min_dist_init=139m` 是**统计假象**，不是 anchor 失效

这是本次诊断的核心发现，需要详细说明。

#### 2.1 最初假设（错误）

5 月份以来的工作假设是："anchor（kalman-fused 敌人位置）偏离真实位置 ~139m，导致残差瞄准永远瞄不准。"

诊断脚本输出的 `dart_min_dist_avg`（min_dist 的 episode 平均）确实显示 139m 量级：
```
dart min_dist_avg: 139.2 m
dart min_dist_init: 138.7 m
```

#### 2.2 诊断设计

在关键路径加 4 处只读日志：
- **[ANCHOR-A/B]** Kalman warm-start 阶段（sensing.py）：检查 anchor 注入 commander_obs 之前的误差
- **[ANCHOR-DD]** vec_drone.py：检查 commander_action → _commander_aim 的解码
- **[ANCHOR-E]** reward.py：检查 shaper 实际计算 min_dist 时用的 aim / enemy_pos

#### 2.3 数据证据

**[ANCHOR-AB]（Kalman warm-start）**：
```
[ANCHOR-B] team=0 kalman warm-start err: max=0.068 m  (over 12 envs)
[ANCHOR-B] team=1 kalman warm-start err: max=0.088 m
```
→ Anchor 在注入 obs 前的精度 < 0.1m，Kalman 完美。

**[ANCHOR-DD]（解码）**：
```
[ANCHOR-DD] n=0 team=0 cmd_in[0,t]=[fire=0.864, aim_x=0.0312, aim_y=-0.5000]
            → _commander_aim[0,t]=(625.0, -5000.0, 0.0) m
[ANCHOR-DD] n=0 team=1 cmd_in[0,t]=[fire=0.864, aim_x=0.0312, aim_y=0.5000]
            → _commander_aim[0,t]=(625.0, 5000.0, 0.0) m
```
→ aim_x × half_x (10000) = 312m, aim_y × half_y (10000) = ±5000m。两个队伍的 anchor 都指向了正确的敌方雷达位置。

**[ANCHOR-E]（shaper 计算）—— 关键证据**：
```
[ANCHOR-E] n=0 team=0 aim[0]=(625.0,-5000.0,0.0)
           enemy0=(638.7,-5000.0)  dist0=13.7m
           enemy1=(14000,-5000.0)  dist1=13375m
           | all_envs min_dist=['2.0','2.0','2.1','2.1','2.0','2.0','2.0','2.1',
                                '2.0','2.0','2.1','2.1']
           | team_mean=2.05 m  max=2.1 m
```

12 个 env 全部 min_dist ≈ 2m，最大 2.1m。这说明 shaper 计算出的 min_dist 完全正常，**anchor 完美对准 enemy0**。

#### 2.4 那为什么 `dart_min_dist_avg` 报告 139m？

继续看后续 step 的 all_envs 分布（96 个样本统计）：

| min_dist 区间 | 样本数 | 占比 |
|---------------|--------|------|
| 0–5 m | 94 | **97.9%** |
| 5–100 m | 0 | 0% |
| 100–1000 m | 0 | 0% |
| 4000–5000 m（离群） | 2 | 2.1% |

两个离群点（425m、439m）把平均值从 ~2m 拉到 ~139m。

**离群点来源**：某 env 的某个 enemy radar 被 kill 了，但 `bf.alive` 标志没在该 step 同步更新，shaper 把它当 alive，于是 aim 距离它 ~5000m（地图尺度）。下一个 step alive 标志同步好，离群消失。

按 env 分布（哪个 env 触发离群）：
```
env0/2/4/6/7/8/9/10: 0 次离群
env1/3/5:            1 次离群
env11:               2 次离群
```

#### 2.5 结论

- **Anchor 完全正常**，Kalman + 解码 + shaper 链路一致，亚米级精度。
- `dart_min_dist_avg=139m` 是 2% 离群点拉高的结果，应该用**中位数**或**截尾平均**代替。
- 之前基于"修 anchor"的所有方案（F1 alpha 钳位、F2 anchor clamp、tight-kr 等）**不是真根因**。

---

### 现象 3：PPO `policy_loss → 0` 始于 R01234，F8 修复未验证

**Tier 1（0621，健康）**：
```
ep 1/200  cmd pl=0.35  vl=1.9M
ep 50/200 cmd pl=0.31  vl=840K
ep 100/200 cmd pl=0.30  vl=132K
```

**R01234（0623，崩塌）**：
```
ep 1/200   cmd pl=-0.004  vl=1.65M
ep 50/200  cmd pl=0.0003  vl=1.95M
ep 100/200 cmd pl=0.0001  vl=2.13M
```

policy_loss 接近 0 但 vl 上升 → critic 仍在尝试拟合，actor 完全停滞。

**F1–F8 修复（0625）**针对这个：
- F1：clip alpha 到 [0, 0.5]
- F2：clamp anchor 到合法 map 区间
- F3-F7：buffer / GAE / observation encoding 修正
- **F8：reward_normalize（按 team 归一化 reward，避免 advantage 被 outlier 主导）**

**问题**：4 变体消融（v1/v2/v3/v4）每个只跑了 2 小时，policy_loss 仍然接近 0，但**没有跑长训练验证 F8 是否真的让 PPO 学起来**。

下次实验：用 `v3_scaling` 跑 ≥6h，如果 cmd pl 从 ~0.02 升到 0.1+ 且 kr 能从 35m 继续退火到 <10m，则 F8 修复有效。

---

### 现象 4：v2_aggressive 比 v1/v4 差 ~10×

4 变体消融（F1-F8 全部应用，只差超参）的对比：

| 变体 | 关键差异 | avg_r | value_loss |
|------|---------|-------|------------|
| v4_control | reward_normalize=False（baseline） | **+18,137** | 200 M |
| v1_conservative | reward_normalize=True, lr 保守 | **+18,102** | 200 M |
| v3_scaling | reward_normalize=True, n_envs=12 | +15,400 | 280 M |
| **v2_aggressive** | reward_normalize=True, lr ×2 | **-5,776** | **2.5 B** |

**v2 的诊断**：
- value_loss 飙到 2.5B（其他 200M）→ critic 拟合的目标值在剧烈漂移
- avg_r 负值说明 reward 信号失稳
- 高 lr 让 critic 跟不上 value target 的变化，每步 value 跳跃幅度让 advantage 完全不可信

**结论**：reward_normalize 配合激进 lr 会破坏训练稳定性；保守 lr（v1）或不开 reward_normalize（v4）反而稳。

---

## 四、假设演化路径

这一周的核心假设变迁：

```
[0621 Tier 1] 一切正常，3/98 真实击杀
      ↓
[0623 R01234] policy_loss 坍塌 → 怀疑 league PFSP 导致 advantage 噪声
      ↓
[0625 F1-F8] 修了 alpha/anchor/buffer/reward_normalize 8 处
   ├── 假设 A: alpha>0.5 让 fire 概率饱和 → F1 修复
   ├── 假设 B: anchor 偏离 139m → F2 修复
   ├── 假设 C: reward 量级悬殊 → F8 修复
   └── 4 变体消融验证
      ↓
[0628 诊断] anchor 实际 <0.1m 误差，139m 是 2% 离群
   ├── F1/F2 都不是根因
   ├── 真根因可能是 F8 没充分长训练验证
   └── 或 PPO advantage/observation 有更深 bug
```

**当前认知**：
1. Anchor 是清白的，可以排除。
2. F8（reward_normalize）理论上是对的方向，但**没有长训练证明**。
3. v2_aggressive 的失败说明 lr 与 reward_normalize 交互敏感，需要更保守地调。

---

## 五、复现 / 验证清单

下面是把当前认知转化为可执行实验的清单：

### 5.1 立即可做（≤30 min）
- [ ] 把 `dart_min_dist_avg` 改成截尾平均（trim 10%）或中位数 → 避免 2% 离群主导。
- [ ] 在 shaper 加 `min_dist_median` 字段，并存到 episode summary。

### 5.2 中期（≥6 h 训练）
- [ ] 用 `v3_scaling` 跑 ≥6h 长训练，验证 F8 + reward_normalize 是否让 cmd pl 从 0.02 升到 0.1+。
- [ ] 若 cmd pl 升不起来，跑 `/training/diagnose_grad.py` 检查 actor head gradient norm。
- [ ] 若 cmd pl 升起来但 kr 不退火，说明 reward shaping 量级不够强，加 `dartboard_weight=50`。

### 5.3 长期（论文级）
- [ ] IPPO / MAPPO baseline 实现（目前 league 是 SOTA 但缺 baseline 对比）。
- [ ] 4 变体消融的完整图表（已有数据，整理成 paper figure）。
- [ ] EAAI 投稿前的 reviewer-difficulty=hard 复审。

---

## 六、附录：关键日志路径

| 内容 | 路径 |
|------|------|
| Tier 1 训练 log | `logs/laser_league_tier1_kill_fix_20260621_193136.log` |
| R01234 训练 log | `logs/laser_league_R01234_20260623_070612_run01_final.log` |
| 4 变体消融 | `logs/ablation_f1f8/v{1,2,3,4}_*.log` |
| Anchor 诊断 log | `/tmp/anchor_diag3.log`, `/tmp/anchor_diag4.log` |
| Tier 1 anchor 早期假设（已推翻） | `ANCHOR_DIAGNOSIS.md` |
| Kalman 单元测试 | `/tmp/test_anchor_unit.py` |

---

## 七、致谢与版本

- 诊断脚本由 Claude Code 协助生成，所有数据可由上述日志路径复现。
- 本文档将随实验进展持续更新；下次更新预计在 v3_scaling 长训练（≥6h）完成后。

**文档版本**：v1.0（2026-06-28）
