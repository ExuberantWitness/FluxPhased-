# FLUXPH — 部分可观测两队多智能体对抗:RL vs 盲态胜任经典(详细实现 spec)

**版本**: 2026-07-16 · **分支**: `twoteam/bc-ppo`(现有 WP-A~D5 代码基线)
**前序链**: `6f91f8c`(infra)→ `176c0bd`(D1 AdaptiveSpectrum)→ `4105e3e`(D3 killer)→ `491f0f5`(D5 channel) → `fc3d0dc`(WP-B)
**取代**: 之前所有含"全信息 oracle / 近最优中心化调度"的 spec(那是 toy,见 §0.2)

---

## 0. 背景与纠正(为什么必须改)

### 0.1 不变 thesis
系统真身 = **2 队 × 每队 2 部 IQ 级 侦/干/探/通 一体化相控阵 = MARL**(队内 CTDE 协同 + 队间自博弈对抗)。**经典在 IQ 级高干扰下失效是实践 ground truth(前提,不是待验证)**;RL 立命点 = **高干扰 + 部分可观测下的多智能体协同/决策**,不是"打赢估计器"。

### 0.2 致命 toy 的诊断(已核实代码)
当前 `env/gpu/twoteam/twoteam_env.py::get_obs`(远端 `fc3d0dc`)的 obs 结构:
```
per enemy-radar r (r∈{0,1}):
  obs[base+0..3] = tracker_x[x,vx,y,vy]     # 对敌雷达 r 的跟踪估计
  obs[base+4]    = trace_P                    # 协方差
  obs[base+5]    = radar_E / e_kill           # kill 进度
  obs[base+6]    = jam_matrix                 # 受干扰指示
  obs[base+7]    = track_ok flag
own: exposure, radar_alive×2, comm_link, step, task_alloc×8, freq_hop(own+enemy), channel(own+enemy)
action: task_alloc[4], beam_target∈{0,1}, laser_target∈{0,1}, emission_on, freq_hop, channel_select
```
**问题(= 用户点死的 god-view)**:
1. **tracker 永远有一个含噪测量**(true 位置 + jam_mul 放大的噪声)→ **敌方永远被跟着,从不消失**;
2. **`beam_target`/`laser_target` 直接按索引选"敌雷达 0/1"** → **敌方存在与身份已知**,无需搜索、无数据关联;
3. **敌方不能关机、不能隐藏、无虚警**。
→ **"知道敌方在哪连经典都不需要"** —— 真正的难点(**隐藏、关机、搜索、数据关联、虚警**)完全没建模。**这就是把问题的本质(不确定性)抹掉的 toy。**

### 0.3 反 toy 铁律(memory `feedback_no_reactive_decisions` 钉死,违反=白做)
1. **任何一方都不得有 god-view**:obs 只能是**自己的、含噪、被干扰削弱的探测/belief**,读不到敌方真状态;
2. **敌方可隐藏 + 可关机**:不保证有测量;要**搜索**才可能探到;
3. **干扰是真 IQ(敌扰+他扰)**,不是 scalar jam_mul(WP-A 已接入,保留);
4. **经典 baseline = 胜任盲态自适应栈**,不是全信息 oracle、不是固定规则稻草人;
5. **RL production 规模训练**(几小时 / ~5e7 steps / league),不是 60-iter sanity;
6. **demonstrate 不 gate**:经典高干扰失效是前提;若没崩=干扰/不确定性没配到现实量级,加大,不退。

---

## 1. 目标与命题

**核心命题**:在**两方都盲**(敌方隐藏、可关机,只有含噪被干扰的探测)+ **高干扰**(敌扰+他扰)+ **两队多智能体**下,**学习式 MARL 指挥官在交战结局上显著优于胜任盲态经典**,因为经典的模块化 搜/探/跟/调度 在盲态高干扰下维护不住 belief、协调不好 2 雷达;而 RL 学到非近视主动感知 + 协同。

**env 天生 ≥4 分 RL-favorable**(checklist:#1 部分可观测 / #5 自适应对手 / #8 多智能体 / #9 竞争频谱)——**这正是文献证明 RL 赢的 regime**(Kreucher-Hero 主动感知 POMDP、Track-MDP、congested-spectrum RL)。我们之前的错都是把 #1/#9 抽象掉。

---

## 2. WP-1 — env 引入部分可观测(核心改动,最大工作量)

**文件**: `env/gpu/twoteam/twoteam_env.py` + `env/gpu/twoteam/iq_interference.py`(保留)+ 新 `env/gpu/twoteam/detection.py`

### 2.1 敌方隐藏 + 可关机(去"永远被跟着")
- 加敌方 **emit 状态** `enemy_emitting[E, T, R]`(bool):敌雷达可**关机**(不辐射)——由敌方策略的 action 控制(新增 `emit_on` 已有 `emission_on`,复用);
- **关机的敌雷达无被动辐射** → 你的**被动探测(侦)拿不到它**;
- 只有**主动探测(探)**照到它、且回波在干扰下过检测门,才拿到一次含噪测量;
- **敌方位置对你隐藏**:你不再直接拿 `tracker_x`;`tracker_x` 只在**你成功探测到并关联后**才更新。

### 2.2 搜-探测-跟踪链(替换"永远有测量")
新 `detection.py::detect(...)`,每 control step:
```
for each own aperture 分给"侦/探"的:
  照到的方位/距离单元 = f(beam_target OR search_direction, beam_width)
  for each enemy radar r (env 知 true 位置,仅用于算物理,不入 obs):
    if 该单元覆盖 r AND enemy 在检测范围:
      SNR_r = radar_eq(range, RCS, P_tx) / (thermal + 敌扰 + 他扰)   # iq_interference 算干扰
      P_detect = sigmoid((SNR_r - detect_threshold)/width)          # 干扰↑ → SNR↓ → P_detect↓
      if bernoulli(P_detect): 生成一条含噪测量 z = true_pos + N(0, σ(SNR))
  # 虚警:每单元以 P_fa 生成假测量(CFAR 门下的杂波/干扰尖峰)
detections = 真测量(部分) + 虚警  → 一个 [E, T, K_max, 3] 的检测列表(range,az,SNR),K 变长
```
**关键**:
- **搜索**:敌方未跟踪时,必须分孔径给"侦/探"去**扫描**未覆盖区,才可能探到;不扫 = 探不到 = 丢目标;
- **数据关联**:检测列表含虚警 → 跟踪器要做关联(哪条是敌方);
- **关机敌方**:被动侦拿不到,主动探能拿到但**你主动辐射→暴露**(exposure,已有);
- **干扰**:敌扰+他扰抬高噪声 → P_detect 降、σ 升(iq_interference 已算 JNR)。

### 2.3 obs 重构(去 god-view,换成"探测 + 自己的 belief")
**删除**:直接的 `tracker_x` god-view 语义不变(它是你的估计),**但它只在成功探测+关联后更新**;新增"是否当前有航迹/检测"。
每 enemy-radar-slot r 改为(仍 obs[base+…],但语义变):
```
obs[base+0..3] = own_tracker_x_hat[r]        # 你的估计(未探到则为 stale/predicted,不是 truth)
obs[base+4]    = own_trace_P[r]              # 你的不确定度(未探到会随时间膨胀)
obs[base+5]    = radar_E[r]/e_kill           # kill 进度(仅当持续跟住才涨)
obs[base+6]    = measured_JNR[r]             # 你测到的干扰(不是敌方 true jam)
obs[base+7]    = track_active[r]             # 当前有没有活跃航迹(探到=1,丢了=0)
+ NEW obs[base+8]  = frames_since_last_detection[r]/H   # 多久没探到(belief 老化)
```
新增全局:
```
+ search_coverage[E,T]        # 已扫区域占比(未探到时用来引导搜索)
+ n_detections_this_step      # 本步检测数(含虚警)
+ detection_list [E,T,K_max,3]# 变长检测(range,az,SNR),pad+mask;供 attention/set encoder
```
**铁律**:obs **绝不含敌方 true 位置/true emit 状态/true jam**;只含你自己传感器+跟踪器产物。

### 2.4 action 去 god-view(`beam_target` 不再按敌方索引)
- **删** `beam_target∈{敌雷达0,1}` 这种"按敌方索引指向"(那是 god-view);
- **改为**:`beam_direction`(连续方位/距离)或 `beam_cell`(离散搜索单元)——**指向一个空间方向,不是一个已知敌方**;
- `laser_target` 改为**指向一条自己的活跃航迹**(track id),而非敌方索引;没航迹 = 不能开激光;
- `commander_actor_critic.py`:`beam_head` 从 Categorical{0,1} 改成方向输出(Beta/Gaussian 连续 或 搜索单元 Categorical over N_cells);`laser_head` over 活跃航迹。

### 2.5 no-god-view assert(反 toy 的机器验证)
新 `env.assert_no_godview()`:
- **置换测试**:把敌方 true 位置随机置换(不改你已探测到的测量)→ **obs 必须逐元素不变**(证 obs 只是你观测的函数,不含 truth);
- 敌方 `enemy_emitting=False` 且未主动探到 → 对应 slot `track_active=0`、trace_P 随时间膨胀(证隐藏/关机真生效);
- 每次 `get_obs` 后跑一次轻量 assert(debug 模式)。

### 2.6 WP-1 验证(必过再往下)
① **no-godview assert 通过**(置换 truth,obs 不变);② 关机敌方 → 被动侦探不到(track_active=0);③ 主动探能探到但抬 exposure;④ 虚警存在、跟踪器要关联;⑤ IQ 干扰生效(JNR,承 WP-A);⑥ 镜像自博弈无偏(reward asymmetry=0);⑦ NaN-free。
**TBD(你核实定)**:detect_threshold / P_fa / σ(SNR) 的现实量级;搜索单元离散化;beam_width;敌方关机建模。

---

## 3. WP-2 — 胜任盲态经典 baseline(反稻草人,无 god-view)

**文件**: 扩 `algo/_shared/baselines/twoteam_strong_rule_commander.py` → `twoteam_blind_classical.py`

盲态经典栈(全在**探测/belief**上,不给敌方 truth):
1. **搜索**:未跟踪目标时,把"侦/探"孔径按 `search_coverage` 扫未覆盖区(raster 或 max-uncertainty 引导);
2. **CFAR 探测**:对检测列表做恒虚警门(干扰下,`detection.py` 已产回波+虚警);
3. **IMM-PDAF 跟踪**:对检测做概率数据关联 + IMM(CV+CT),维护航迹 + 协方差(**Stone Soup 或自实现**);
4. **资源管理**:按 belief(track 质量、未覆盖区、kill 优先级)在 2 雷达间分配 侦/干/探/通(现有 rule 逻辑 + 搜索项);
5. **ECCM**:被干扰(measured_JNR>阈)→ 频率跳变 + 错频(承现有 StrongRule);
6. **火控**:仅对**活跃且置信足够**的航迹开激光。
**验证(必过)**:① 低干扰 + 敌方常辐射 → 能搜到+跟住+kill(证胜任);② **高干扰 / 敌方关机穿插 → 搜/跟失效、track_active 频繁掉、kill 链断**(前提);③ 若高干扰它还行 = **干扰/隐藏没配到现实量级,加大**。

---

## 4. WP-3 — RL MARL(盲态,production 规模)

**文件**: `commander_actor_critic.py`(obs/action 改)+ `br_trainer.py`(训练)+ `run_wp2_league.py`(league)

### 4.1 obs / action(同 §2.3/§2.4,与经典完全同样的盲态)
- obs = 探测列表(set encoder / attention)+ 自己的航迹 belief + measured_JNR + 资源/exposure/channel 状态;**无 truth**;
- action = 孔径→功能分配 + **beam_direction/search_cell** + laser→track_id + emission_on + freq_hop + channel_select。

### 4.2 架构
- **检测列表用 permutation-invariant 编码**(mean-pool / self-attention over K 检测)——变长目标/检测;
- CTDE 中央 critic(α_eff,现有;**用前打印 priv[:,4] 核归一化,α_eff bug 咬过一次**);
- (可选)recurrent(GRU)处理"多久没探到"的 belief 老化 + 非近视搜索。

### 4.3 训练配置(production,不是 60-iter toy)
| 项 | 值 |
|---|---|
| 规模 | **~5e7 steps / 1000+ iter / 几小时**(不是 60 iter) |
| 算法 | PPO/MAPPO(CTDE),lr 3e-4/1e-3,clip 0.2,GAE λ0.95 γ0.99,entropy 0.01 anneal |
| **league/自博弈** | 队间 PFSP(现有 `opponent_pool`,EMA 已修)——对手是**同构盲态指挥官**,共适应 |
| log_std_floor | **-6**(勿 -4) |
| BC warmup | 可选:BC 从盲态经典 bootstrap(现有 `bc_pretrain`)|
| checkpoint | `checkpoints/blind/`(**严禁 /tmp**)|

---

## 5. WP-4 — 比较协议(扫干扰/隐藏强度)

**文件**: 新 `algo/_shared/pilot/twoteam/run_blind_compare.py`(复用 `run_wp2_crossplay.py` eval 框架)

- **方法**:{RL(production), 盲态胜任经典, (对照)IPPO} —— **两者完全同样的盲态 + IQ 干扰**;
- **难度轴**:干扰强度(敌扰功率/覆盖 + 他扰密度)× 敌方隐藏程度(关机占空比)× 目标数;从低(经典 fine)扫到高(经典失效);
- **指标**:① 交战结局 **kill-rate / survival-rate**;② 对**隐藏敌方**的 **track 质量(trace_P)/ 探测率 / 重捕时间**;③ **搜索效率**(找到隐藏敌方的时间);
- **统计**:≥5 seed,mean±95%CI(bootstrap 1e4),cross-play 双向平均 + held-out 对手,Welch-t/Mann-Whitney。

### Make-or-break / 判据(demonstrate 姿态)
| 结果 | 决定 |
|---|---|
| **RL 在高干扰/隐藏下显著优于胜任盲态经典**(kill/survival/track,CI 分离,同样的盲+干扰)| **真赢** → 铺全套 → 论文(RL 在部分可观测+竞争频谱+多智能体 = 文献证明它赢的 regime pk 过经典)|
| 胜任盲态经典高干扰下也行 | **干扰/隐藏没配到现实量级 → 加大,不退** |
| 加到现实极限经典仍行 | 诚实记录(该域盲态也 classical-favorable)|

---

## 6. 反 toy 验证 checklist(每步交回必带)
- [ ] `assert_no_godview` 通过(置换 truth,obs 不变)——**任何一方无全信息**;
- [ ] 敌方能隐藏/关机,被动探不到 → track_active=0、trace_P 膨胀;
- [ ] 检测含虚警,跟踪器做数据关联;
- [ ] 干扰是 IQ(JNR),非 scalar;
- [ ] 经典是盲态胜任栈(低干扰胜任 / 高干扰失效),非 oracle 非稻草人;
- [ ] RL production 规模(~5e7 steps),非 60-iter;
- [ ] cross-play 双向 + CI;checkpoint 非 /tmp;log_std_floor=-6;priv[:,4] 归一化核过。

---

## 7. 顺序(硬性)
**WP-1(引入部分可观测 + no-godview assert + IQ 干扰)→ 验证 §2.6 → WP-2(盲态经典 + 验证低干扰胜任/高干扰失效)→ WP-3(production RL)→ WP-4(扫干扰/隐藏比较)。**
**先交 WP-1 的 no-godview assert + 经典高干扰失效验证,再动 RL。** 别在还有 god-view / 还是 toy 规模的 env 上训 RL。

## 8. 交回给我
① WP-1:no-godview assert 结果 + 关机/隐藏/虚警/搜索生效证据 + 镜像无偏;② WP-2:盲态经典 低干扰 kill vs 高干扰失效曲线(前提验证);③ WP-4:RL vs 盲态经典 kill/survival/track vs 干扰强度(CI)+ 搜索效率。

## 9. TBD(你核实定,我不臆断)
detect_threshold / P_fa / σ(SNR) 现实量级;搜索单元离散化 + beam_width;敌方关机/隐藏建模;IMM-PDAF 用 Stone Soup 还是自实现;env 检测列表 K_max;现实高干扰/隐藏量级(锚实践反馈)。

---

## 10. 诚实的 scope
这是**大改**(env 从"永远跟着"改成"部分可观测搜-探-跟 + 数据关联 + 虚警 + 隐藏/关机敌方",obs/action 重构,盲态经典栈,production 训练)。但这是**去掉 god-view toy、进入文献证明 RL 赢的部分可观测+高干扰 regime** 的必经改动——不再有 oracle、不再 god-view、不再抹掉不确定性。
