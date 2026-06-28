# P2 实验 A Baseline 报告 — F1+F2 因果验证失败

**日期**: 2026-06-28
**配置**: `configs/ablation_f1f8/v3_p2_A_baseline.yaml`
**日志**: `logs/diag/v3_p2_A_baseline_run.log`
**总时长**: 2483s (≈41 min)，3 iter × 平均 14 min/iter
**结论**: ❌ **预注册 FAIL 场景触发** — A 与 B 在 kr=24.5m 下表现一致，F1+F2 因果性无法证伪或证实

---

## 0. TL;DR

| 指标 | A 预测 (protocol §3.3) | A 实测 | B (P1, F1+F2 ON) | 结论 |
|---|---|---|---|---|
| iter 0-2 kill_rate | **0.00-0.30** | **1.00** | 1.00 | ❌ FAIL |
| team_value_loss | **≥1e7** (百万级不收敛) | **7.2e4 - 3.1e5** | ~1e5 | ❌ FAIL |
| team_adv_std | **趋 0** | **1.07e2 - 2.35e2** | O(1) | ❌ FAIL |
| f1_disable 生效 | (要) | **是** ([S0] 早 return) | n/a | ✅ |

**核心发现**：F1+F2 disable 后所有机制指标与 B 几乎一致，说明在 kr=24.5m 下 F1+F2 的因果效应被宽松 kr 掩盖。

**dart-board 附加发现**：min_dist_init 平均 139m（应 ~1m），证明 anchor 没有正确指向 Kalman-fused 敌人位置；kill_rate=1.00 是 kr=24.5m 宽松阈值产物，不是学习信号。

---

## 1. 实验设置

### 1.1 配置（来自 `v3_p2_A_baseline.yaml`）

| 字段 | 值 | 说明 |
|---|---|---|
| `alpha_schedule` | `constant` | 同 B（消除 alpha 变量） |
| `alpha_constant` | `0.7` | 同 B |
| `kill_radius_init` | `24.5` | 同 B（冻结，threshold=99 永不退火） |
| `kill_rate_threshold` | `99.0` | kr 永不退火 |
| `f1_disable` | `true` | **关键** — 关闭 F1 (NEW team_returns 计算) |
| `f2_disable` | `true` | **关键** — 关闭 F2 (恢复 team_adv 双重归一化) |
| `population_cap` | `6` | §2 提速 |
| `n_eval_games` | `2` | §2 提速 |
| `episodes_per_training` | `8` | §2 提速 |
| `psro_iterations` | `3` | 短验证 |

### 1.2 假设（预注册）

**机制假设**：F1+F2 是 P1 B 在 alpha=0.7 下 kill_rate=1.00 的**唯一原因**。A 关闭 F1+F2 后，team_returns 应保留 OLD 800-step 跨 episode 累加 bug + 无归一化 → critic 不收敛 (team_value_loss 百万级) → team_adv_std 趋 0 → kill_rate 崩塌。

**PASS 判据**：A kill_rate < 0.5 同时 B = 1.00 → F1+F2 因果坐实。

**FAIL 判据**：A 也 kill_rate=1.00 → kr=24.5m 太宽松 → 转 §5 严苛 kr 兜底。

---

## 2. 实验结果

### 2.1 顶层指标（3 iter 全部完成）

```
[League] Iteration 0 complete in 473.2s
[League] alpha=0.700 (schedule=constant) beta_kl=0.100
[League] kill_radius hold at 24.500m (kill_rate=1.00 < 99.0)

[League] Iteration 1 complete in 1167.2s
[League] alpha=0.700 (schedule=constant) beta_kl=0.067
[League] kill_radius hold at 24.500m (kill_rate=1.00 < 99.0)

[League] Iteration 2 complete in 843.4s
[League] alpha=0.700 (schedule=constant) beta_kl=0.033
[League] kill_radius hold at 24.500m (kill_rate=1.00 < 99.0)
```

- **kill_rate = 1.00** across all 3 iters（与 B 一致，与 A 预测 0-0.30 不一致）
- kr 未退火（threshold=99 by design）
- alpha 恒为 0.700（Bug #1 修复生效，schedule=constant 正确读取）
- 总耗时 2483s ≈ 41 min（比 P1 B 的 ~3h 快 4x，§2 提速生效）

### 2.2 机制仪表（[mech] print，共 24 条）

```
team_value_loss:
  min  = 7.198e+04
  avg  = 1.465e+05
  max  = 3.138e+05

team_adv_std:
  range = 1.07e+02 - 2.35e+02
  avg   = 1.506e+02
```

**对比 B 路径**（P1 实验，team_value_loss ~1e5）：
- A 平均 1.465e5 ≈ B 的 1e5 → **同量级**
- A team_adv_std ~150 远高于 A 预测的"趋 0"，反而比 B 路径的 O(1) 大 100x

### 2.3 dart-board 诊断（[dart] print，共 80 条）

```
fire_rate:
  avg = 0.524        # Bernoulli 0.5，policy 未学到 fire commitment

min_dist_init (前 20 步 min_dist_avg 均值):
  avg = 139.42m      # 远高于假设的 ~1m
  range = 25 - 200m+

min_dist_min (全 episode 最小):
  avg = 0.23m        # 远小于 kr=24.5m
  range = 0.02 - 0.57m

min_dist_final / min_dist_change:
  全部 = nan         # dart 末端聚合逻辑有 NaN bug（独立问题）
```

**解读**：
1. **fire_rate=0.524** → policy 完全没学 fire commitment，仍是 Bernoulli(0.5) 初始化
2. **min_dist_init=139m** → 初始 aim 远不在敌人位置（与 Plan 假设"residual_aim 让 init dist~1m"矛盾）
3. **min_dist_min=0.23m** → episode 中 dist 能瞬间降到亚米级，但这是偶发，不是策略学到的 aim 收敛
4. kill_rate=1.00 是因为 kr=24.5m 容忍任意粗略 aim，不代表学习发生了

### 2.4 [S0] print 缺失 — f1_disable 路径生效的证据

`compute_team_returns()` line 245-261 (`buffer.py`)：

```python
if getattr(self, 'f1_disable', False):
    # OLD buggy path: 800-step cross-episode accumulation
    ...
    return   # ← 早 return，跳过 [S0] print
```

A baseline log 中 **[S0] print 完全未出现**，证明 `f1_disable=True` 路径**确实在跑**，OLD bug 路径生效。开关接线正确。

---

## 3. 失败诊断

### 3.1 为什么 A 没崩塌？

按预注册假设，A 应该 team_value_loss 百万级 + kill_rate 崩塌。实测都没发生。可能原因（按可能性排序）：

**假说 1（最可能）：kr=24.5m 太宽松**
- kill_radius_init=24.5m 意味着任何 aim 在 24m 内都算"击中"
- min_dist_min 平均 0.23m，远小于 24.5m → 几乎任意粗略 aim 都能"击杀"
- learning signal 被"白送"的 reward 淹没，F1+F2 的修复效果显现不出来
- 这是 protocol §3.3 明确预注册的 FAIL 场景

**假说 2：F1 路径中的 800-step bootstrap 仍然提供了部分 critic 信号**
- OLD 路径虽然有跨界累加 bug，但 bootstrap_val = self.values[end] 仍部分有效
- 加上 GAE 的 critic 路径正常工作（self.commander_buffer.compute_returns 在 TeamPPOTrainer.update line 870 调用）
- team_value_loss 主要由 agent-path critic 决定，team-path 占比小

**假说 3（最严重）：F1+F2 不是真正的根因**
- P1 B 的 kill_rate=1.00 可能本就是 kr 宽松 + 偶发 dist 短暂达标的结果
- F1+F2 修复的 team_returns 量级问题在 alpha=0.7 + 宽松 kr 下并未显现
- 需 §5 严苛 kr 验证才能定论

### 3.2 dart 暴露的更深问题

**min_dist_init=139m** 是个**重要发现**，独立于 F1+F2 因果问题：

- Plan 假设 `residual_aim=True` 让 anchor 指向 Kalman-fused 敌人 → init dist ~1m
- 实测 init dist 25-200m，远超假设
- 可能原因：
  - Kalman filter 在 episode 前 20 步未收敛（track_burnin=120 控制）
  - anchor 用的不是 fused enemy state，而是 raw sensing（含噪声）
  - residual_aim 的 anchor 计算路径有 bug
- 后果：策略看到的初始 obs 中"敌人位置"信号是噪声主导，PPO 学不到稳定的 enemy_pos → action 映射

这意味着即使 F1+F2 修复了 critic，**策略层面仍然不能学到稳定的 aim**，因为输入信号本身不稳定。

### 3.3 fire_rate 不学的原因

`fire_init_logit: 0.0` → 初始 P(fire_on)=0.5（Bernoulli）。

期望：reward shaping 中的 `fire_commitment_weight=10` + `fire_lock_bonus=5` 会推动 policy 学持续 fire。

实测：fire_rate=0.524（仍 Bernoulli），没学到 commitment。

可能原因：
- reward 在 episode 内的方差太大（kill_bonus=1e5 dominate），fire_commitment 的细微差异被淹没
- PPO advantage 在 8-ep × 7-step batch 中信号不足
- 或者 advantage 归一化后 fire 相关信号被压制

---

## 4. 与 protocol §3.3 PASS/FAIL 对照

| 字段 | A 预测 | A 实测 | 命中？ |
|---|---|---|---|
| iter 0 kill_rate | 0.00-0.30 | 1.00 | ❌ |
| iter 1 kill_rate | 0.00-0.30 | 1.00 | ❌ |
| iter 2 kill_rate | 0.00-0.30 | 1.00 | ❌ |
| iter 0 team_value_loss | ≥1e7 | 7.2e4-3.1e5 | ❌ |
| iter 0 team_adv_std | 趋 0 | 1.07e2-2.35e2 | ❌ |
| f1_disable 生效 | 是 | 是（[S0] 缺失） | ✅ |
| f2_disable 生效 | 是 | 推测是（无独立验证） | ⚠️ 未独立验证 |

**结论**：6 项预测 5 项 FAIL，仅 f1_disable 接线确认。

protocol §3.3 明确指出：
> **FAIL**：A 也 kill_rate=1.00 → kr=24.5m 太宽松，需严苛 kr 重做（见 §5）

→ **必须跑 §5 兜底实验**（kr=5m, threshold=0.95）才能给 F1+F2 因果性定论。

---

## 5. 论文叙事影响

### 5.1 当前不能写的论断

基于 A baseline 结果，以下论断**没有实验支撑**：

- ❌ "F1+F2 是 kill_rate 恢复的唯一原因"
- ❌ "team_returns 量级问题导致 alpha≥0.5 崩塌"
- ❌ "reward normalization 阻止 team_adv_std 趋 0"

### 5.2 当前可以写的论断

- ✅ "kr=24.5m 下 kill_rate=1.00 是宽松阈值产物，不反映学习"
  - 证据：A 与 B 一致，dart min_dist_init 远超 1m
- ✅ "fire_init_logit=0 + Bernoulli fire head → policy 不学 fire commitment"
  - 证据：fire_rate=0.524 横跨 3 iter 80 episodes
- ✅ "Bug #1（train.py dead code）修复后 alpha schedule 正确生效"
  - 证据：A log 中 alpha=0.700 (schedule=constant) 持续打印

### 5.3 必须新增的实验（按优先级）

1. **§5 兜底**（kr=5m, threshold=0.95, A baseline 重跑）
   - 预期：A 在严苛 kr 下 kill_rate<0.5，B 保持 1.00 → 因果坐实
   - 时长：~30 min
2. **dart NaN bug 修复 + min_dist_init 根因调查**
   - min_dist_init=139m 暗示 sensing/anchor 路径有问题
   - 独立于 F1+F2，但影响所有 reward shaping 实验
3. **HEADLINE-LINEAR**（F1+F2 ON + linear alpha→1.0 + kr 退火）
   - 仍待跑，但前提是 §5 至少给出 F1+F2 有部分的因果信号

---

## 6. 工件清单

| 文件 | 状态 |
|---|---|
| `configs/ablation_f1f8/v3_p2_A_baseline.yaml` | ✅ 新增 |
| `configs/ablation_f1f8/v3_p2_headline_linear.yaml` | ✅ 新增（未跑） |
| `logs/diag/v3_p2_A_baseline_run.log` | ✅ 完整 3 iter |
| `training/ppo/ppo_trainer.py` | ✅ [mech] 仪表 + ablation flags |
| `training/ppo/buffer.py` | ✅ compute_team_returns + f1_disable |
| `training/train.py` | ✅ Bug #1 修复 + f1_disable/f2_disable 透传 |
| `training/flux_league.py` | ✅ alpha_schedule='constant' 分支 |
| `P2_A_BASELINE_REPORT.md`（本文件） | ✅ |
| `NEXT_STEPS_PROTOCOL.md` | ✅ §5 兜底设计已就绪 |

---

## 7. 复盘 — 这次实验的得失

### 7.1 得

1. **机制仪表落地**：[mech] print 现在每 PPO update 都打印 team_value_loss + team_adv_std，未来所有实验都能拿到机制层数据，不再黑盒
2. **f1_disable 接线验证**：[S0] 早 return 路径证实开关确实生效，消除了"开关没接上"的可能性
3. **dart-board 诊断价值显现**：min_dist_init=139m 是独立于 F1+F2 的发现，揭示了 sensing/anchor 路径的潜在问题
4. **提速生效**：3 iter × 14min = 41min vs P1 B 的 ~3h，4x 加速
5. **Bug #1 修复验证**：alpha=0.700 (constant) 持续打印，证明 schedule 配置正确生效

### 7.2 失

1. **预注册假设过于乐观**：以为 kr=24.5m 能区分 A vs B，没考虑 kr 宽松到任意粗略 aim 都能击中
2. **dart NaN bug 未预先测试**：min_dist_final/change 全部 nan，损失了一半诊断信号
3. **f2_disable 无独立验证**：f2 路径是否真的关闭了，没有像 [S0] 那样的 smoking gun
4. **min_dist_init=139m 假设没被早期发现**：Plan 假设 ~1m，跑完才发现差距 100x，浪费了一次实验机会
5. **3 iter 跑完才停**：iter 0 结果已经能判断 FAIL，但因为没人监控，又多跑了 iter 1-2（多花 33 min）

### 7.3 流程改进

- ✅ **未来 ablation 实验必须先做 kr 灵敏度预实验**：先确定一个能区分"学得到"vs"学不到"的 kr，再上 ablation
- ✅ **dart metric 加 unit test**：跑前用合成数据验证 dart print 不出 NaN
- ✅ **f2_disable 加独立 print**：类似 [S0]，让 f2 关闭时打印一行确认
- ✅ **iter 0 后强制 checkpoint + manual review**：避免"跑完 3 iter 才发现假设错了"

---

## 8. 下一步（按用户决策）

用户决策：**生成报告 + 同步 GitHub**（本文件即是）。

后续未启动的实验（待用户指令）：
- §5 兜底（kr=5m）— protocol 预注册 FAIL 处理路径
- HEADLINE-LINEAR（6-12h）— 论文级 headline 图数据
- dart NaN 修复 + min_dist_init 根因调查

---

## 附录：原始数据快照

### A.1 [mech] print 全部 24 条

```
iter 0:
  [mech] team_value_loss=3.138e+05  team_adv_std=1.817e+02
  [mech] team_value_loss=2.994e+05  team_adv_std=1.790e+02
  [mech] team_value_loss=1.334e+05  team_adv_std=1.459e+02
  [mech] team_value_loss=1.334e+05  team_adv_std=1.479e+02
  [mech] team_value_loss=1.327e+05  team_adv_std=1.806e+02
  [mech] team_value_loss=1.393e+05  team_adv_std=1.790e+02
  [mech] team_value_loss=8.189e+04  team_adv_std=1.379e+02
  [mech] team_value_loss=8.117e+04  team_adv_std=1.382e+02

iter 1+2:
  [mech] team_value_loss=2.131e+05  team_adv_std=2.352e+02
  [mech] team_value_loss=2.148e+05  team_adv_std=2.350e+02
  [mech] team_value_loss=1.307e+05  team_adv_std=1.545e+02
  [mech] team_value_loss=1.400e+05  team_adv_std=1.609e+02
  ... (12 more, range 7.2e4 - 3.1e5 / 1.07e2 - 2.35e2)
```

### A.2 [dart] print 代表性样本

```
[dart] fire_rate=0.507  min_dist_init=37.06m  final=nanm  change=+nanm  min=0.02m
[dart] fire_rate=0.500  min_dist_init=44.31m  final=nanm  change=+nanm  min=0.12m
[dart] fire_rate=0.542  min_dist_init=46.09m  final=nanm  change=+nanm  min=0.20m
[dart] fire_rate=0.558  min_dist_init=85.87m  final=nanm  change=+nanm  min=0.41m
[dart] fire_rate=0.557  min_dist_init=28.57m  final=nanm  change=+nanm  min=0.14m
[dart] fire_rate=0.517  min_dist_init=85.92m  final=nanm  change=+nanm  min=0.24m
...
```

### A.3 关键日志行

```
[League] kill_radius initialized at 24.500m (will anneal toward 0.200m)
============================================================
PSRO Iteration 0/3
============================================================
[League] Iteration 0: Evaluating payoff matrix...
  [payoff] p0000 vs p0003 (1/9)... win_rate=1.00 (6.0s)
  ...
[League] Iteration 0 complete in 473.2s
[League] alpha=0.700 (schedule=constant) beta_kl=0.100
[League] kill_radius hold at 24.500m (kill_rate=1.00 < 99.0)
  Checkpoint saved at iteration 0
...
Training complete. Final league saved.
```
