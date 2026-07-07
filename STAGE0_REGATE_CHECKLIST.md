# Stage 0 重判门 — 精确执行清单(给 PRO6000 agent)

**目的**:把 Concerto pilot 从 "0/4 plumbing 失败" 推进到**对核心命题(工作正常的 Concerto 在高难度赢经典)的真裁决**。0/4 是坏 MAPPO checkpoint + 编排器高难度过度触发造成的,**thesis 未测**。修三处 → 重跑同矩阵 → 判 v2 是否赢经典。
**关联**:`EAAI_RESEARCH_PLAN.md §9` · `concerto_pilot_results.md`(根因) · `CONCERTO_RRM_PLAN.md`

---

## 前置事实(已核实,别改错)
- pilot 入口:`algo/_shared/pilot/run_pilot.py`;配置:`configs/pilot/concerto_pilot.yaml`(4 方法 × [L0,L1,L3] × seed[42-46]);
- **坏 checkpoint**:`mppo_checkpoint: checkpoints/laser_mappo/iter_019.pt`(旧激光任务,`task_head` argmax=0=recon → 四功能 QoS 全零);
- composer:`algo/_shared/concerto/composer.py`,`ComposerV2` 参数 `theta1_jsr_db=10.0 / theta2_trace=0.6 / epsilon_margin=0.2 / K`;触发后 commit K 个 RL 时隙、countdown 递减(**无冷却→重复触发→L1/L3 占 98-100%**);
- 经典:`algo/_shared/baselines/classical_qos_rrm.py`(water-fill,`qos_floor_per_fn=3`),L0=0.890(略低于 0.9);
- QoS-RRM env:`env/gpu/qos_rrm/qos_rrm_env.py` + reward `algo/_shared/ppo/reward_shaping.py:DenseRewardShaper`。

---

## 修复 1(主,决定成败)— 训一个 QoS-RRM 任务上的新 MAPPO
**问题**:现 checkpoint 从没在四功能分配任务上训过 → 交给它的 RL 时隙全零。
**做法**:
1. 写/改一个训练入口(改 `algo/_shared/train_laser.py` 或新建 `algo/_shared/train_qos_mappo.py`),**在 `qos_rrm_env` 上训**,reward 用 **`DenseRewardShaper`**(四功能 QoS:detect_snr/detect_coverage/jam_effectiveness/comm_reliability/recon_intel),`use_mappo=true`(CTDE);
2. 关键:确保 **commander `task_head` 在四功能间真分配**(不是恒 argmax)——训练目标含全部四功能 QoS,别只给单一奖励(否则又退化);
3. 训 ~20 iter(~1.5 GPU-h),存 `checkpoints/qos_mappo/iter_019.pt`(radar_ac + commander_ac state_dict,与 `SimpleMAPPOTrainer` 兼容);
4. **验证(必过再往下)**:单独用这个 checkpoint 跑 1 局,打印 commander `task_id` 直方图 → **四功能都有非零分配**(不是恒 recon);per-function QoS 非零。

## 修复 2(设计关键)— composer 加冷却,让 RL 在 L3 也只占少数时隙
**问题**:L1/L3 触发 98-100% → 高难度退化成纯 RL(经典底座被绕过 + 无加速)。
**做法**(改 `composer.py:ComposerV2`):
1. 加 `cooldown_M` 参数:**K 个 RL 时隙结束后,强制 M 个经典时隙**内不许再触发(加一个 cooldown countdown);
2. 调 `K / M` 使 **L3 的 RL 占比 ∈ [0.3, 0.5]**(经典保留 50-70% 保 QoS 平衡 + 出加速);起点 K=4, M=6(RL≤40%);
3. (可选)略升阈值 `theta2_trace 0.6→0.7`、`epsilon 0.2→0.15` 降低触发频率;
4. **验证**:重跑后看日志 `n_rl_steps/n_classical_steps` → L0≈0%、**L3 RL 占比落在 [0.3,0.5]**(不再 98-100%)。

## 修复 3(强 baseline 前提)— 调经典到 L0 近最优
**问题**:强 classical 是全故事前提;现 L0=0.890 略低。
**做法**(改 `classical_qos_rrm.py`):
1. 打印 L0 的 **per-function QoS**,定位拖后腿的功能(大概率 track 或 comm 分配不足);
2. 调 water-fill / `qos_floor_per_fn`(3→按需)使**每功能 L0 都达标**,目标 **L0 总 QoS ≥ 0.95**;
3. **验证**:纯经典 @ L0 ≥ 0.95、@ L3 明显跌(相变仍在)。

---

## 重跑(三修复都验证过后)
```bash
# 1. 训新 MAPPO
python -m algo._shared.train_qos_mappo --config configs/pilot/qos_mappo_train.yaml   # ~1.5h
# 2. 重跑 pilot,指向新 checkpoint
python -m algo._shared.pilot.run_pilot \
    --config configs/pilot/concerto_pilot.yaml \
    --mppo-checkpoint checkpoints/qos_mappo/iter_019.pt \
    --methods classical mappo concerto_v1 concerto_v2 \
    --difficulties L0 L1 L3 --seeds 42 43 44 45 46 \
    2>&1 | tee experiments/concerto_pilot_regate.log
```
> **L3 收敛风险(R4)**:L3 学习型干扰机可能在 pilot 预算内不收敛。若 L3 jammer QoS/触发异常 → 退回 `ReactiveJammer(tau=1)` 当高难度代理(config 已注明 fallback),别让干扰机没训好污染判据。

---

## 判据(Stage 0 门)—— 逐 cell + CI,不看单数
| # | 判据 | 阈值 | 决定 |
|---|---|---|---|
| **G1(命门)** | **concerto_v2 > 强经典 @ L3** | gap > 0.05,95%CI 不含 0 | 这条是 EAAI 成不成立的核心 |
| G2 | v2 > 纯MAPPO @ L3 | gap > 0.05 | 证交错 > 纯 RL |
| G3 | MAPPO QoS 非零(checkpoint 修好) | 四功能都有分配 | 修复 1 验证 |
| G4 | L3 RL 占比 ∈ [0.3,0.5] | 不再 98-100% | 修复 2 验证 |
| G5 | 强经典 @ L0 ≥ 0.95、@ L3 明显跌 | 相变在 + baseline 强 | 修复 3 验证 |
| G6 | 无功能退化(min dwell ≥ 0.05 ∀ cell) | ≥0.05 | 不退化 |
| G7 | Concerto 墙钟 < 纯MAPPO(RL 少数时隙) | ratio < 1 | 加速显现 |

## 决策树
- **G1 PASS(v2 赢经典 @ L3)** → **命题成立** → 进 **Stage 1 全难度扫描**(主轴 τ∈{∞,16,8,4,2,1,0}+L3,4 方法 × 7-8 点 × 5 seed,出 headline 包线图);
- **G3-G6 修好但 G1 FAIL(v2 仍不赢经典)** → **thesis 真失败**(即便 RL 正常也榨不出争用时隙增量)→ **退 Path A**(传感 C1 + 基准 C0,投 IEEE TAES);
- **G3/G4 没修好** → 别判 G1,先把 checkpoint/composer 修对(否则又是 plumbing 假失败)。

## 纪律(这一路踩坑换来的)
1. **逐 cell + CI**,不看聚合单数;头对头比 v2 vs 经典 @ L3,不各打各的;
2. **修复必逐个验证**(G3/G4/G5)再判 G1——别让 plumbing 未修就误判 thesis;
3. **机制核查**:打印 task_id 直方图、RL 占比、per-function QoS、trace(P)——查 WHY 不只看结果;
4. **诚实**:低难度 v2≈经典是正常(诚实汇报),不 p-hack;强 classical 是前提。

跑完把 **v2/经典/MAPPO 在 L0/L1/L3 的 QoS(逐 seed + 均值 + CI)+ L3 的 RL 占比 + MAPPO task_id 直方图** 贴回,据此判 G1 → 决定进 Stage 1 还是退 Path A。
