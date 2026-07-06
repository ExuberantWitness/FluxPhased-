# EAAI C2 方案 — EW 任务下的"信念感知 CTDE"(给 PRO6000 agent)

**背景**:phase1.5 证明(cross-play)无 EW 任务里 PFSP≈CTDE、**所有 RL 输给 ClassicalMPC**(mappo 0.472)——根因是**好 anchor 把任务解掉了,classical"瞄均值+一直开火"近最优**。要冲 EAAI(AI 须赢 classical),必须换到**经典法必然失效的 EW 任务**,并做一个从问题机理推导的新算法。**本方案两个门,先廉价证伪再投入。**

**关键前提(已核实,基础设施都在 phase1.5)**:
- EW:`reward.py` `_jam_level`(commander_action[4]=jam)/`jam_cost`;`sensing.py` `jam_mul` 削弱 range+crossrange σ(`jam_gain`)+ `exposure_gain` home-on-jam;`configs/laser_25x25_ew_exposure.yaml` 有全套值。
- CTDE 特权:`commander_privileged_dim:10`(critic 已能看 EW 状态)。
- **Kalman 协方差 P00/P01/P11 已在 `sensing.py:_kalman_step` 算出并存 `_trk_P`**,但**只把均值 x0,x1 写进 obs[68:70]**——belief conditioning 就是把 P 也喂给 actor。

---

## Gate 0 — 先证 EW 让 classical 失效(纯配置,~8h,决定整条线成不成立)
**不写任何新算法,先回答:开 EW 后 classical 是否掉下 0.5、RL 是否反超?** 这一步决定 C2 有没有戏。

1. **建 EW 版配置**:把 `configs/laser_25x25_ew_exposure.yaml` 的 EW 段合并进 algo 配置:
   ```
   cp algo/mappo/code/config.yaml algo/ew_mappo/code/config.yaml
   # 加入(值取自 ew_exposure):
   #   training.jam_log_std: -1.0, training.jam_kr_threshold_m: 0.5, training.commander_privileged_dim: 10
   #   reward_shaping: jam_gain: 8.0, jam_cost: 0.01, exposure_gain: 50.0, race_time_cost: 0.01, race_death_penalty: 30.0
   #   checkpoint_dir 改到 algo/ew_mappo/data
   同理建 algo/ew_ippo（use_mappo:false）
   ```
2. **跑 ew_mappo + ew_ippo(seed 42, psro=20)**,再用 phase1.5 的 cross-play 脚本让它们 vs **ClassicalMPC**(classical 用同一 EW env)。
3. **判据(命门)**:
   | 结果 | 结论 |
   |---|---|
   | ClassicalMPC 掉到 **<0.5**(被 ew_mappo 赢) | ✅ **EW 任务立论成立** → 进 Gate 1(做新算法赢 MAPPO) |
   | classical 仍 ≥0.5 | EW 不够狠 → 调 `jam_gain 8→16`、`exposure_gain 50→100`,或加维度(机动目标/多目标);再测 |
   | ew_mappo 也学不动(NaN/adv_std 爆/kills=0) | EW 下 RL 不稳 → 先诊断(逐轮 R/B/D + adv_std),别急着上新算法 |

> **为什么 classical 会失效(机理)**:`jam_gain=8` 使敌方半干扰把我的 σ ×(1+8·0.5)=×5 → anchor 均值被污染 → classical 瞄污染均值+一直开火 → 打空;且它一直照射=辐射拉满 → home-on-jam(exposure_gain 50)把它暴露 → 被杀。**这正是 RL 该赢的结构。**

---

## Gate 1 — C2 算法:信念感知 CTDE(仅 Gate 0 通过后做,~1-2 天代码 + 跑)
现有 MAPPO+EW 的 **actor 只看 anchor 均值**——干扰下均值不可信它却照信。**新算法 = 让去中心 actor 感知自己的定位不确定度(Kalman 协方差),学会"低置信时收火/机动/反干扰"。** 这是 EAAI 的 C2 贡献(belief-conditioned decentralized policy for cognitive EW radar,文献未见)。

### 组件 1(核心新颖点)— belief conditioning:把协方差喂进 commander obs
- **`sensing.py`**:`_kalman_step` 已返回 `nP00,nP01,nP11`,`_trk_P` 已存。在写 obs[68:70]=均值 之后,**把该敌的协方差(3 个唯一元,或降成 1 个标量 `trace=P00+P11`=定位不确定度)写进新 obs 维**。两个敌×(1 或 3)= +2 或 +6 维。
  - 归一化:P 是 m² 量级,除以 `half_map²` 或取 `log(trace)` 稳定。
- **`actor_critic.py`**:`CommanderActorCritic(obs_dim=76)` → 改 **76+2**(用 trace,最省)或 **76+6**;更新 commander obs 组装处同步补维。
- **`train_laser.py`/obs 组装**:填这几维时,**tracked 模式取 `_trk_P`,fused/single 模式取该步的 R(测量协方差)**,保证各 sensing 模式都有值。

### 组件 2 — exposure/EW 特权 critic(大部分已有,确认接上)
- `commander_privileged_dim:10` 已让 CTDE critic 看 EW 状态。**确认它含"敌方对我的定位置信/我的 jam 暴露"分量**(exposure 对偶);若无,把"我的 jam_level² × exposure_gain"这个暴露量加进 privileged_info。actor 保持只看 76+belief 的局部 obs(去中心执行)。

### 对比 + 消融(EAAI headline)
在 **EW 任务**上 cross-play(双向平均 + 共同 held-out),方法集:
| 方法 | 说明 |
|---|---|
| ClassicalMPC | 传统法(Gate 0 已证失效) |
| IPPO / MAPPO(EW) | 标准 baseline |
| **BeliefCTDE(本方法)** | MAPPO + belief conditioning |
| **消融**:BeliefCTDE −belief / −exposure | 证每个新组件的因果贡献 |
**目标**:BeliefCTDE **显著赢 classical 且赢 MAPPO**;消融证 belief/exposure 各自有效。

---

## 论文三层贡献(故事重构)
- **C0 环境/基准**:IQ 级多功能相控阵 EW 对抗 MARL 基准(MATLAB 83/83 校验)——"认知雷达 EW 的 SMAC";
- **C1 传感/工程**:多基地 Kalman 融合→亚米定位 + **CRLB 验证**(`WP3_CRLB_ANCHOR.md`);
- **C2 算法(新)**:belief-conditioned CTDE,在 EW 认知资源管理上赢经典法与标准 MARL。

## 执行顺序
1. **Gate 0**(纯配置,~8h)→ 判 classical 是否失效。**这是最便宜的决定性一跳,先做。**
2. 通过 → **Gate 1** 组件1(belief,~1天)→ BeliefCTDE vs MAPPO vs classical + 消融;
3. 阳性 → 补 C1 CRLB + Phase2 消融/seed/统计;
4. Gate 0 就没戏(classical 调狠了还赢 / RL 学不动)→ 转维度(机动目标+延迟)或退守 C0+C1(传感贡献 + "AI≈classical 因传感已解"的诚实叙事)。

## 诚实缺口 / 纪律
- **Gate 0 未验证前一切是假设**——classical 在 EW 下是否真败、RL 是否真赢,必须实测;
- belief conditioning 是从 **POMDP + 辐射对偶机理**推的(符合"文献锚定/机理归因,不开关式试错"的纪律),但**新颖性需 research-lit 锚定**(belief-state MARL / cognitive-radar RL / exposure-aware)后再定稿;
- 全仿真+军事色彩对 EAAI 偏险 → C1 的 CRLB/损伤真实性必做,否则退 IEEE TAES / Neurocomputing;
- 全程逐轮 R/B/D + policy_loss/adv_std 判健康,cross-play 双向平均 + held-out 不相交。
