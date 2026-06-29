# Phase 1 League v2 — 失败实验报告

> **生成日期**：2026-06-29
> **训练时段**：07:42 → 13:40 (~6h) — 在 iter 27/48 用户手动终止
> **配置**：[configs/laser_25x25_pro6000.yaml](configs/laser_25x25_pro6000.yaml)
> **训练日志**：[logs/phase1_league_v2_failure.log](logs/phase1_league_v2_failure.log)
> **关联计划**：[plan §1](https://github.com/ExuberantWitness/FluxPhased-/blob/main/.claude/plans/snuggly-exploring-parrot.md) (改进版 league plan)
> **关联报告**：[EXPERIMENTAL_PHENOMENA_REPORT.md](EXPERIMENTAL_PHENOMENA_REPORT.md) (前序现象档案)

---

## TL;DR

Phase 1 改进版 league 训练在 iter 13 (kr=0.48m) 达到巅峰后持续退步，iter 14 起首次出现 BLUE 胜，iter 18 起 `cmd_pl` 坍缩到 ≈0、`adv_std` 从 ~10 爆炸到 **235**，iter 27 (kr=0.20m) **RED 0 kills / 0 eval_kills**。**Phase 1.4 严格 gating 全面失败**（cum red 0.58 < 0.8，cum blue 0.11 > 0.1，draw 0.31）。这是 PPO 学习失败，**不是技术 bug**（无 NaN/Traceback）。

---

## 一、实验配置摘要

| 段 | 参数 | 值 | 来源 |
|-----|------|----|------|
| training | psro_iterations | 48 | plan §1.6 |
| training | league | true | plan §1.1 |
| training | **reward_normalize** | **true** | **Phase 1.0 改动 1 (F8)** |
| training | league_snapshot_every | 3 | 原 plan |
| training | episodes_per_training | 10 | config |
| training | log_std_init / floor / decay | -1.0 / -6.0 / 0.40 | 原 plan |
| training | bc_pretrain_iters / cmd_bc_weight_init→final | 0 / 5.0 → 2.0 | 原 plan |
| reward_shaping | **dartboard_weight / scale_m** | **50.0 / 5.0** | **Phase 1.0 改动 2b (全场稠密)** |
| reward_shaping | kill_bonus | 100.0 | 原 plan |
| env | num_envs | 12 | config |
| env | kill_radius_m | 0.2 (final) | 课程目标 |
| env | min_radar_baseline_m | 5000.0 | 方案1（verified p14） |
| sensing_noise | mode | tracked (Kalman) | 方案2 |

**Phase 1.0 五项工程改动**（详见 plan §1.0）：
1. ✅ F8 reward_normalize 开启（[train_laser.py:216,220](training/train_laser.py#L216)）
2. ✅ dartboard 改为全场稠密（[reward.py:298-310](training/laser/reward.py#L298)）
3. ✅ adv_std / adv_mean 监控加入 metrics dict
4. ✅ kill_radius 退火门控用 eval_kill_rate（发现已存在，未改）
5. ✅ aim_residual_norm 监控（验证 BC + dartboard 兼容性）

---

## 二、27 iter 完整轨迹表

| iter | kr (m) | kills | eval_kills | eval_kill_rate | cmd_pl | adv_std | aim_res | this (R/B/D) | cum red | cum blue | cum draw |
|------|--------|-------|------------|----------------|--------|---------|---------|--------------|---------|----------|----------|
| 1 | 50→35 | 175 | 16 | 0.667 | -0.003 | 10.0 | 0.411 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| 2 | 35→24.5 | 166 | 13 | 0.542 | -0.002 | 9.4 | 0.429 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| 3 | 24.5→17.15 | 169 | 15 | 0.625 | -0.000 | 6.4 | 0.305 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| 4 | 17.15→12 | 199 | 24 | 1.000 | -0.003 | 11.7 | 0.204 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| 5 | 12→8.4 | 188 | 24 | 1.000 | -0.009 | 10.4 | 0.140 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| 6 | 8.4→5.88 | 196 | 24 | 1.000 | -0.004 | 9.9 | 0.101 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| 7 | 5.88→4.12 | 173 | 24 | 1.000 | +0.001 | 14.5 | 0.069 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| 8 | 4.12→2.88 | 187 | 24 | 1.000 | -0.001 | 10.7 | 0.048 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| 9 | 2.88→2.02 | 120 | 14 | 0.583 | +0.028 | 7.8 | 0.040 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| 10 | 2.02→1.41 | 120 | 13 | 0.542 | +0.009 | 7.7 | 0.032 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| 11 | 1.41→0.99 | 188 | 24 | 1.000 | +0.020 | 10.6 | 0.025 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| 12 | 0.99→0.69 | 118 | 12 | 0.500 | +0.120 | 7.0 | 0.023 | 12/0/0 | 1.00 | 0.00 | 0.00 |
| **13** | **0.69→0.48** | **185** | **24** | **1.000** | **+0.196** | **10.7** | **0.023** | **12/0/0** | **1.00** | **0.00** | **0.00** |
| 14 | 0.48→0.34 | 97 | 12 | 0.500 | +0.184 | 5.5 | 0.055 | 8/2/2 | 0.98 | 0.01 | 0.01 |
| 15 | 0.34(卡) | 44 | 8 | 0.333 | +0.223 | 37.3 | 0.064 | 4/2/6 | 0.93 | 0.02 | 0.04 |
| 16 | 0.34(卡) | 33 | 9 | 0.375 | +0.167 | 10.8 | 0.062 | 7/1/4 | 0.91 | 0.03 | 0.06 |
| 17 | 0.34→0.24 | 27 | 15 | 0.625 | +0.063 | 26.8 | 0.067 | 3/9/0 | 0.87 | 0.07 | 0.06 |
| 18 | 0.24(卡) | 6 | 2 | 0.083 | **-0.006** | 39.1 | 0.069 | 2/0/10 | 0.83 | 0.06 | 0.10 |
| 19 | 0.24→0.28 | 5 | **0** | **0.000** | -0.003 | 49.7 | 0.072 | 0/0/12 | 0.79 | 0.06 | 0.15 |
| 20 | 0.28(卡) | 10 | 6 | 0.250 | -0.003 | **190.2** | 0.075 | 3/2/7 | 0.76 | 0.07 | 0.17 |
| 21 | 0.28(卡) | 10 | 1 | 0.042 | +0.422 | 80.8 | 0.075 | 1/0/11 | 0.73 | 0.06 | 0.21 |
| 22 | 0.28(卡) | 14 | 3 | 0.125 | +0.348 | 80.4 | 0.067 | 3/0/9 | 0.71 | 0.06 | 0.23 |
| 23 | 0.28(卡) | 45 | 3 | 0.125 | -0.003 | 24.9 | 0.081 | 0/3/9 | 0.68 | 0.07 | 0.25 |
| 24 | 0.28→0.20 | 11 | 12 | 0.500 | -0.005 | 35.2 | 0.087 | 0/12/0 | 0.65 | 0.11 | 0.24 |
| 25 | 0.20(卡) | 2 | 3 | 0.125 | -0.004 | 72.1 | 0.080 | 0/3/9 | 0.62 | 0.11 | 0.26 |
| 26 | 0.20(卡) | 3 | 3 | 0.125 | -0.008 | 106.0 | 0.089 | 0/3/9 | 0.60 | 0.12 | 0.28 |
| **27** | **0.20→0.24** | **0** | **0** | **0.000** | **-0.004** | **179.97** | 0.084 | **0/0/12** | **0.58** | **0.11** | **0.31** |

**最后一条 Update（iter 27 之后）**：`adv_std=235.11`（持续爆炸）。

---

## 三、Phase 1.4 Gating 验证

| 指标 | 阈值 | 实际 | 状态 |
|------|------|------|------|
| kr 退火 | ≤ 0.5m 单调 | 50→0.20m 但**非单调**（iter 19/27 反向放宽） | ⚠️ 部分 |
| 紧 kr 真实击杀 | kr≤0.5m 时 `eval_kill_rate` > 0 | iter 14-27 大部分 iter 在 0.04-0.5 | ✅ 通过（弱） |
| 瞄准精度 | `eval_min_aim_dist` → 0m | 全程 0m（min，非 avg/median） | ⚠️ 指标失真 |
| 对手池压制 | cum red ≥ 0.8, cum blue ≤ 0.1 | **cum red=0.58, cum blue=0.11** | ❌ **失败** |
| **policy_loss 不坍缩** | 训练全程 > 1e-4 且非单调下降 | iter 18 起 ≈0，最低 -0.008 | ❌ **失败** |
| **advantage std** | 训练全程 > 1e-3 | 7-235 区间，iter 18 起剧烈震荡 | ⚠️ 通过但失稳 |
| 健康 | 无 NaN/crash | 全程无 NaN/Traceback | ✅ 通过 |

**结论**：7 项 gating 中 3 项失败、2 项部分通过。**Phase 1.4 整体未通过**。

---

## 四、关键失败模式

### 模式 1：PPO 策略损失坍缩（重现历史 R01234 现象）

`cmd_pl` 轨迹（取每 iter 最后一条 Update）：

```
iter 1-12:  cmd_pl ∈ [-0.01, +0.12]   健康学习区间
iter 13:    cmd_pl = +0.196             ← 巅峰
iter 14-17: cmd_pl ∈ [+0.06, +0.22]     健康但震荡
iter 18-20: cmd_pl ∈ [-0.006, -0.003]   ← 坍缩！
iter 21-22: cmd_pl = [+0.42, +0.35]     ← 短暂反弹
iter 23-27: cmd_pl ∈ [-0.008, -0.003]   ← 再次坍缩
```

这是 [EXPERIMENTAL_PHENOMENA_REPORT.md](EXPERIMENTAL_PHENOMENA_REPORT.md) §3 描述的 R01234（2026-06-23）现象：`policy_loss ≈ 0` 持续多次 iter，actor 完全停滞。本实验证实 **F8 reward_normalize + dartboard 全场稠密 + BC weight 退火** 三件套**未能根治**该问题。

### 模式 2：adv_std 指数爆炸

`adv_std` 轨迹：

```
iter 1-17:  adv_std ∈ [5, 27]    健康区间（plan §1.4 阈值 > 1e-3，远超）
iter 18:    adv_std = 39          ← 开始爆
iter 19:    adv_std = 50
iter 20:    adv_std = 190         ← 4× 跳跃
iter 21:    adv_std = 81
iter 23:    adv_std = 25
iter 26:    adv_std = 106
iter 27:    adv_std = 180
iter 27 Update: adv_std = 235     ← 训练终止前最高
```

**根因假设**：dartboard_weight=50 + 全场稠密（per-step 50 reward）在精细 kr=0.2-0.3m 区间产生极端 reward outlier。F8 reward_normalize 按 running std 归一化，但当 std 本身被 outlier 拉大时，归一化反而放大了 outlier 对 advantage 的污染。

### 模式 3：策略无法学击败多样化对手池

```
iter 1-13:  pool=1→5,  cum red=1.00 (RED 全胜 vs 单一/少量对手)
iter 14:    pool=5,    cum red=0.98, blue=0.01 (首次 BLUE 胜)
iter 17:    pool=6,    cum red=0.87, blue=0.07 (BLUE 9 胜)
iter 20:    pool=7,    cum red=0.76, blue=0.07
iter 24:    pool=8,    cum red=0.65, blue=0.11 (BLUE 12 胜！)
iter 27:    pool=9,    cum red=0.58, blue=0.11, draw=0.31
```

**对手池越多，RED 越弱**——典型的 PSRO 多样性反噬。RED 策略被推到 Kr=0.20m 物理极限后，无法在该精度下击败 8-9 个不同历史快照。

### 模式 4：aim_residual 始终低位

```
iter 1:  aim_res = 0.411
iter 5:  aim_res = 0.140
iter 11: aim_res = 0.025   ← BC 主导，残差 ≈0
iter 14: aim_res = 0.055   ← 精细 kr 下被迫开始学残差
iter 27: aim_res = 0.084
```

**结论**：BC weight=5.0 主导 + dartboard_weight=50 dense reward **没给 PPO 留出足够的策略学习空间**。aim_res 一直 ≤ 0.1，PPO 在 commander action 上的策略贡献极弱（plan §1.0 改动 5 验证：BC 完全主导模式）。

---

## 五、假设根因（按可能性排序）

### H1（最可能）：dartboard_weight=50 + 全场稠密让 reward 量级失控

- **证据**：iter 20 adv_std=190、iter 27=235；iter 14 起 `avg_cmd_r` 从 ~130 升到 ~200（reward 量级膨胀 50%）
- **机制**：dartboard per-step 给 `50 × exp(-dist/5m)`，近距离时 ≈50。原 reward 量级（guidance + fire_lock + illum）≈50-100，dartboard 加入后总 reward 跳到 100-200。F8 按 std 归一化但 std 本身被推大
- **可验证**：降 `dartboard_weight=20` 重跑，看 adv_std 是否回到 10 量级

### H2（高）：log_std floor=-6.0 让策略探索过早收敛

- **证据**：iter 14 起 `log_std=-6.0`（达到 floor），iter 1 时 `log_std=-1.0`。decay=0.40/iter × 12 iters = -4.8 + init=-1.0 → -5.8（iter 12 已接近 floor）
- **机制**：log_std floor -6.0 对应 std ≈ 0.0025（fire logit, aim_x, aim_y）。在精细 kr=0.2m 下，该探索熵不足以跳出局部最优
- **可验证**：用 `log_std_floor=-4.0`（std ≈ 0.018）重跑

### H3（中）：BC weight 衰减节奏不匹配 kr 课程

- **证据**：iter 12 `bc_w=2.25` 时 cmd_pl=+0.12（PPO 仍在学），iter 13 `bc_w=2.0` 时 cmd_pl=+0.20（巅峰），iter 14 起 BC weight 不再降（已到 final=2.0），但 cmd_pl 开始坍缩
- **机制**：BC weight 在 kr 退火到 0.5m 前已到 final，PPO 必须独立学精细瞄准但探索熵不足
- **可验证**：拉长 `cmd_bc_decay_iters: 12 → 24`，让 BC 在精细 kr 阶段继续退火

### H4（中）：pool snapshot 节奏（每 3 iter）+ kr 退火门控（≥0.5）形成不稳定性

- **证据**：iter 13 (pool=5) → iter 14 (pool=5, opp=0) → iter 15 (pool=5, opp=0)，连续 2 iter 不 snapshot 但 eval_kill_rate 大幅波动（24→12→8）
- **机制**：snapshot_every=3 + pool_cap=30 让对手多样性增长慢于 RED 能力下降速度，形成"打不过老对手 → 又来新对手"恶性循环
- **可验证**：用 `league_snapshot_every: 6` 或 `pool_cap: 5` 重跑

### H5（低）：reward_normalize 实现可能有 bug

- **证据**：adv_std=235 是极端异常，远超任何合理 PPO 训练
- **机制**：[ppo/buffer.py:163-166](training/ppo/buffer.py#L163) 实现按 running std 归一化，但若 running std 因 cold-start 估计不准，反而放大 noise
- **可验证**：用 [diagnose_grad.py](training/diagnose_grad.py) 在 iter 20 状态下抓 advantage 直方图

---

## 六、与历史 R01234 比较

| 维度 | R01234 (0623) | Phase 1 League v2 (0629) |
|------|---------------|--------------------------|
| 训练时长 | ~7h | ~6h（手动终止） |
| kr 起始 → 终止 | 50 → 12.5m | **50 → 0.20m**（更深） |
| 真实击杀率 | 4/293 (1.4%) | 24/24 at iter 13, **0/12 at iter 27** |
| cmd_pl 终态 | -0.0001 | -0.004（相同量级） |
| adv_std | 未监控 | 7-235 |
| F8 reward_normalize | ❌ 未开 | ✅ 开启 |
| dartboard 全场稠密 | ❌ 未开 | ✅ 开启 |
| aim_res 监控 | ❌ 未加 | ✅ 加入 |

**结论**：Phase 1.0 五项工程改动让 kr 退火深度从 12.5m 提升到 0.20m（4× 改善），但 **未解决 PPO 策略坍缩的根本问题**。kr 能退到 0.20m 主要靠 BC anchor + 早期优势，iter 14 起 PPO 主导阶段崩溃。

---

## 七、下一步建议（消融实验清单）

按优先级排序，每个 ≤6h 验证：

### 优先级 P0（必做）

1. **dartboard_weight=20 重跑**（验证 H1）
   - 改动：`reward_shaping.dartboard_weight: 50 → 20`
   - 预期：adv_std 回到 10-30 区间，cmd_pl 不再坍缩
   - 失败回退：dartboard_weight=10

2. **log_std_floor=-4.0 重跑**（验证 H2）
   - 改动：`training.log_std_floor: -6.0 → -4.0`
   - 预期：探索熵足够，精细 kr 下 RED 能击败多样对手

### 优先级 P1（建议）

3. **BC weight 衰减拉长**（验证 H3）
   - 改动：`training.cmd_bc_decay_iters: 12 → 24`
   - 预期：精细 kr 阶段 BC 继续退火，PPO 不必独立承担精细瞄准

4. **diagnose_grad 在 iter 20 状态运行**（验证 H5）
   - 需要从 checkpoint 加载 iter 20 的模型状态
   - 看 PPO-ONLY action_head gradient 是否为 0

### 优先级 P2（论文级）

5. **snapshot_every=6 + pool_cap=5**（验证 H4）
   - 对手多样性增长放慢
   - 适合做 ablation figure

---

## 八、可复现性

### 复现命令

```bash
cd /home/ubuntu/CODE/FluxPhased-
PYTHONUNBUFFERED=1 /home/ubuntu/miniconda3/envs/fluxphased/bin/python -u \
  -m training.train_laser --config configs/laser_25x25_pro6000.yaml \
  2>&1 | tee logs/phase1_league_v2_failure.log
```

### Checkpoints

```
checkpoints/laser_pro6000/
├── red_iter_03.pt
├── red_iter_06.pt
├── red_iter_09.pt
├── red_iter_12.pt   ← 巅峰状态 (kr=0.99m, cum red=1.0)
├── red_iter_15.pt
├── red_iter_18.pt
├── red_iter_21.pt
├── red_iter_24.pt
├── red_iter_27.pt   ← 失败终止状态
└── (opponent pool snapshots)
```

可加载 iter 12/13 checkpoint 做精细诊断。

---

## 九、附录：日志关键时间点

| 时间 | iter | 事件 |
|------|------|------|
| 07:42 | — | 训练启动 |
| ~08:00 | 1 | 第一次 PSRO 完成 |
| ~10:30 | 13 | **巅峰**：cum red=1.0, kr→0.48m |
| ~10:50 | 14 | **首次 BLUE 胜**：R8/B2/D2 |
| ~11:00 | 15-16 | kr 卡 0.34m |
| ~11:25 | 17 | kr 突破到 0.24m，但 R3/B9/D0 |
| ~11:45 | 18 | **cmd_pl 坍缩开始** |
| ~12:00 | 19 | **eval_kills=0 首次出现**，kr 反向放宽到 0.28m |
| ~12:20 | 20 | adv_std 飙到 190 |
| ~13:20 | 24 | kr 突破到 0.20m final，但 R0/B12/D0 |
| 13:40 | 27 | **手动终止**：kills=0, adv_std=235 |

---

## 十、致谢与版本

- 训练由 Claude Code 协助监控与终止
- 所有数据可由上述命令与配置文件复现
- 本文档与 [EXPERIMENTAL_PHENOMENA_REPORT.md](EXPERIMENTAL_PHENOMENA_REPORT.md) 形成完整的"现象档案 → 失败实验"链条

**文档版本**：v1.0（2026-06-29 13:50）
