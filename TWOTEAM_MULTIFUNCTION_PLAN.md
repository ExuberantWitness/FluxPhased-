# 两队多功能相控阵对抗 — 定版执行计划(给 PRO6000)

**系统(不得再阉割)**:**2 队 × [2 部相控阵雷达 + 1 指挥官 + 1 激光],对称零和对抗。** 每部雷达 25 子阵(5×5)单孔径时分复用 **侦查/探测跟踪/干扰/通信**;每队 2 部雷达 = 多基地对(几何分集→融合出亚米级定位,CRLB);指挥官协调 2 部 + 指挥激光。**队内 CTDE 协同、队间自博弈对抗。**
**核心动作**:每队在 **2 孔径(2×25 子阵)上动态分配四功能** + 波束指向 + 激光目标 + 辐射控制,against 同构敌队。
**胜负(对称对决)**:我用激光杀敌(需持续 track)vs 敌杀我;管理暴露避免被 home-on-jam 反杀。
**目标期刊**:TAES 主线(两队×多功能×CTDE×CRLB×自博弈 = Li'22/Xiong'23 TAES、Dolinger'25 IET 地盘);IET 兜底;AppInt slice 备选。

## 立命点(为什么必须这套,缺一不可)
① 相控阵多功能:4 功能 × 2 孔径动态重分配,常规雷达做不到;② 多智能体协同:每队 2 部雷达配合 → CTDE;③ 双方对抗:对称博弈,固定多功能策略可被反演 → 自博弈/league;④ 多基地:2 雷达几何融合(CRLB)是感知优势,但被对抗争夺。

---

## ⚠️ 规避的历史 bug(逐条堵死,这是这次不重蹈覆辙的关键)
| 历史根因 | 之前怎么坏 | 这次怎么规避 |
|---|---|---|
| **根因 B:对手被阉割成 1-bit 标量**(S1/Task B jammer 收敛成常数)| 对手无信息可自适应→无博弈→league 训到虚无 | **干扰是每队雷达的一个功能,红方 = 完整对称指挥官(同 `CommanderActorCritic`),不是标量 jammer** |
| **根因 A:平静海面**(Phase1.5 无 EW→好传感解掉任务→classical 近最优→league 零增益)| 传感把难点解了,RL 无发挥 | **传感被争夺**:敌队实时干扰你的多基地融合→固定策略可被利用(WP1 命门先证) |
| PFSP `pool_winrate` 死代码 | 从不更新→均匀采样 | 用已修的 `update_pool_winrate` EMA + `pfsp_p` 开关 |
| **α_eff bug**(priv[:,4] raw≈200→MAPPO 暗塌成 IPPO)| CTDE 没生效 | 已修;**用前打印 priv 核对归一化**;CTDE 真生效再信 MAPPO |
| checkpoint→/tmp 崩 | 磁盘写满丢工作 | `checkpoints/twoteam/`,**严禁 /tmp** |
| log_std_floor=-4 自毁 | 稳定性崩 | **-6**(p14 proven-stable) |
| 无效对比(各打各的)| cross-play 才对 | **cross-play 双向平均 + 共同 held-out** |
| NaN/adv_std 爆 | league 训崩 | 逐轮监控 R/B/D + adv_std,爆则先诊断不硬训 |

---

## WP0 — 两队对称多功能 testbed(~1-1.5 周)
把 env 从"单方 + 标量 jammer"改回**两队对称**。
**0.1 结构**:2 队,各 2 雷达(25 子阵)+ 指挥官 + 激光;**两队同构**(同动作/观测空间)。
**0.2 四功能(单孔径时分复用)**:每雷达每 step 把 25 子阵分给 {侦查, 探测跟踪, 干扰, 通信};
- **干扰 = 一个功能**(抬敌方对我方的传感 σ),**不是外挂标量 agent**;
- **通信 = 多基地融合命脉**(队内 2 雷达融合需 comm 链;被干扰/让位→退回单站精度)。
**0.3 多基地融合 + CRLB**(`sensing.py` 现有):2 雷达几何分集→融合定位;实现定位 CRLB + 跟踪 PCRLB(`crlb.py`),报 trace_P vs 下界。
**0.4 激光驻留-kill-链**:激光对敌目标只在 track 达标时积能,track 丢(被干扰)清零→时序耦合。
**0.5 暴露/home-on-jam**:任何主动辐射累积 exposure→被敌反辐射反杀(`exposure_gain`/`race_death_penalty`)。
**0.6 验证(必过)**:① 镜像自博弈(一套策略打两队)对称、无偏;② 四功能权衡真实(纯跟踪 vs 纯干扰的策略在对抗下各有胜负,不存在恒优单策略);③ CRLB 锚接上;④ NaN-free、adv_std ∈ [3,14]。

## WP1 — 强规则式多功能指挥官 + 命门 exploitability 门(~1.5 周,决定有没有戏)
**这是回答"多功能博弈非不非平凡"的命门——不过就没有 RL 论文,先测再投自博弈。**
**1.1 强规则式多功能指挥官**(anti-strawman,非"瞄+开火"):按规则在 2 孔径上分配四功能——track 优先级 × kill 进度、敌方跟踪我时转干扰、comm 维持融合、track 够好时降辐射控暴露。**要强**(WP0 验证:低对抗下近最优)。
**1.2 命门 exploitability 门(G0)**:
```
固定强规则式队 π_rule;训 best-response 学习队 BR(π_rule) 打它;
exploitability(π_rule) = U(π_rule vs 镜像 π_rule) − U(π_rule vs BR(π_rule))
```
| G0 结果 | 决定 |
|---|---|
| **π_rule 显著可被 exploit**(BR 比镜像多赢,gap>阈,CI 不含 0)| ✅ **多功能博弈非平凡** → 进 WP2 自博弈 |
| π_rule 近不可 exploit(BR≈镜像)| ❌ **多功能博弈也是平静海面** → 诚实退(该域对 RL 不友好,收 C1 传感/基准→IET) |
**这一步直接判死或点亮整条路——它测的正是杀死我们两次的"根因 A"。**

## WP2 — 自博弈/league + cross-play(仅 G0 PASS,~1.5 周)
**2.1 训练**:队内 **CTDE**(MAPPO,α_eff 已修)+ 队间 **league/PSRO**(PFSP 已修);对手池 = 规则式 + 自博弈快照;混合初始几何/目标。
**2.2 方法集**:{自博弈-league 指挥官, MAPPO(CTDE 无 league), IPPO, 强规则式多功能}。
**2.3 cross-play 锦标赛**:全方法互打,**双向平均 + 共同 held-out 对手**;指标 = 胜率/kill-survival、exploitability、trace_P vs CRLB。
**2.4 门 G1**:
| # | 判据 | 阈值 |
|---|---|---|
| **G1(命门)** | 自博弈-league 在 cross-play 显著赢强规则式 | 胜率>0.5+CI 不含 0.5 |
| G2 | 自博弈-league 赢 MAPPO(无 league)| 证 league 加值(不是又一次 PFSP≈CTDE)|
| G3 | exploitability(自博弈)< exploitability(规则式)| 证向均衡收敛 |

## WP3 — 全刻画 + 消融 + 统计(仅 G1 PASS,~3-4 周)
- 操作包线(cross-play 表现 vs 对抗强度/目标数)+ Elo;
- 消融:−league(退 CTDE)、−CTDE(退 IPPO)、−干扰功能、−通信功能、−多基地(单雷达)、−暴露;每个证组件因果;
- 物理锚:多基地 CRLB/PCRLB;
- 统计:≥5 seed,mean±95%CI(bootstrap 1e4)、Welch-t/Mann-Whitney、Cohen's d、Holm-Bonferroni、Elo±CI。

## 决策树
- **G0 FAIL** → 多功能博弈良态 → 退 IET(C1 多基地 CRLB + C0 IQ 基准),别硬上自博弈;
- **G0 PASS, G1 FAIL(自博弈不赢规则式)** → 对抗有 richness 但 RL 榨不出 → 退 IET;
- **G0+G1+G2 PASS** → **命题成立** → WP3 铺全套 → TAES 主投,IET 兜底。

## GPU 预算 + 时间线
| WP | 内容 | GPU-周 |
|---|---|---|
| WP0 | 两队 testbed + CRLB + 验证 | 1-1.5 |
| WP1 | 强规则式 + G0 exploitability 门 | 1.5 |
| WP2 | 自博弈/league + cross-play + G1 | 1.5 |
| WP3 | 全刻画 + 消融 + 统计(仅过门)| 3-4 |
| 合计 | | **~2-2.5 月** |

## 回报格式 + 交接纪律
- **先交 WP0 验证(镜像对称 + 四功能权衡真实 + CRLB)+ WP1 的 G0 exploitability 数**;
- **G0 是最先的命门**:π_rule 可不可被 exploit?可 → 我放行 WP2;不可 → 别烧自博弈,一起退 IET;
- 每 WP 逐 cell + CI + cross-play 双向 + adv_std/NaN 健康记录。

## 诚实缺口
- **G0 未过前,一切是假设**——"多功能博弈非平凡"必须实测,不是拍脑袋相信联赛;
- Phase1.5 已证"传感解掉任务时 RL 全输",这次靠"被争夺的传感"翻案,但**是否真翻得了,G0 说了算**;
- 两队对称 + 四功能是新代码,WP0 验证必须扎实(镜像无偏、权衡真实、CRLB 对齐);
- 全仿真+军事色彩:CRLB/损伤真实性做满;框架"多传感器-效应器对抗资源分配";venue TAES/IET 对纯仿真+EW 友好。
