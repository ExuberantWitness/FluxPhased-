# WP2 — BC→League 自博弈:赌"RL 打赢近 Nash 经典规则"(给 PRO6000)

**这一注的定位(贪心第一注)**:成本≈恒定 → 按价值贪心。**地板(IET:两队 IQ testbed + BC→PPO pipeline + 近 Nash 表征,~85%)已建好在手**,所以这一注全砸天花板:**BC→league 能否产出一个稳健打赢强规则的策略 → TAES 冠**。用最强方法(种群自博弈,比 G0 单 best-response 探索更广)赌最高价值。
**诚实赔率**:~20-25%(G0 近 Nash 平局是压着它的真证据)。但**地板已保、失败只是坐实近 Nash(还给 IET 加证据),成功是 TAES**——期望价值上该先打。
**目标**:TAES 主;失败 → 退 IET 地板 + 转 Bet B(鲁棒/泛化)抬 AppInt。

## 为什么 league 比 G0 的 BC→PPO 更有戏(不是重复)
G0 测的是"BC→PPO **单 best-response** 从 rule 盆地做 local 搜索"→ 打平。**league 不同**:
- **种群 + PFSP + 共演化**做**无向、非局部**搜索,能找到 local best-response 探不到的**非传递/混合策略**(这正是 AlphaStar 用 league 而非单 best-response 的原因);
- 把**人想到的 candidate exploit 塞进对手池当种子**,league 既无向搜索、又带上人的假设。

---

## 复用(已建,别重写)
| 组件 | 文件 | 状态 |
|---|---|---|
| BC 预训练(从 rule) | `bc_pretrain.py` | ✅ 已建,G0 验证过(完美学 rule 70% vs 71%) |
| PPO+GAE+CTDE/IPPO+freq_hop | `br_trainer.py` | ✅ |
| PFSP 对手池(EMA,无死代码) | `self_play/opponent_pool.py` | ✅ |
| Commander AC(freq_hop_head + **α_eff CTDE critic**) | `commander_actor_critic.py` | ✅ |
| 两队 env(抗干扰技能 + exposure 生效) | `env/gpu/twoteam/twoteam_env.py` | ✅ WP0-decisive 过 |
| 强规则多功能 | `twoteam_strong_rule_commander.py` | ✅ |
| 极端策略 + track_agile | `extreme_commanders.py` | ✅ |
| 评测 harness | `run_g0_gate.py::eval_method` | ✅ 复用 |

## 新写(少量)
1. **Candidate-exploit 脚本策略**(扩 `extreme_commanders.py`,~150 LOC):
   - `exposure_pump`:专攻 rule 的 duck 阈值(60)——持续高辐射逼 rule duck,duck 时击杀;
   - `jam_spread`:专攻 rule 抗干扰触发阈值(0.30)——分散 jam 到多目标,每个 < 0.30 不触发 rule 的 hop 反应,累积压制;
   - `false_target`:专攻 rule 聚焦射杀——beam/laser 频繁切换制造关联歧义。
   - 用途:**塞进 league 对手池当种子 + 独立 eval(顺带回答 G0 遗留的"这三条 exploit 到底成不成立")**。
2. **`run_wp2_league.py`**(~350 LOC):BC 预训练 → 对称自博弈/PSRO 主循环 → 快照入池。
3. **cross-play 锦标赛 eval + Elo + 非传递性检测**(~200 LOC)。

---

## Step 1 — BC 预训练(bootstrap,复用 `bc_pretrain.py`)
从强规则收集 50K 样本(7 极端对手 + 对称增强)→ BC 15 epoch → 得到 rule-equivalent 起点策略。**这是 league 的起跑线(免费把 reward 从 -1.82 推到 -0.68)。**

## Step 2 — 对称自博弈/League 主循环(`run_wp2_league.py`)
```
pool = {强规则, 7 极端策略, 3 candidate-exploit 脚本, RL 快照(初始=BC)}
for iter in range(N_iters):          # N ~ 800-1500
    opp = PFSP_sample(pool)          # 优先采难打对手(update_pool_winrate EMA)
    # 对称自博弈:commander 训 vs opp,两队都训
    rollout(commander vs opp, both teams)   # CTDE 队内(α_eff),league 队间
    PPO_update(commander)            # log_std_floor=-6
    if iter % snapshot_every == 0:
        pool.add(commander.snapshot())
    monitor: adv_std ∈ [0.1,100], no NaN, entropy 不崩
```
- **CTDE(α_eff 已修)**队内协同 2 雷达;**league**队间对抗;
- checkpoint → `checkpoints/twoteam/wp2_league/`(**严禁 /tmp**);
- **⚠️ 开训前 assert `priv[:,4]` 是归一化 trace_P(非 raw≈200)**——α_eff bug 咬过一次,两队 obs 重排易复发。

## Step 3 — cross-play 锦标赛评测
方法集:{**league-commander**, 强规则, MAPPO(CTDE 无 league), IPPO, 7 极端, 3 candidate-exploit}。
- **all-vs-all cross-play**,双向平均 + 共同 held-out(不训练用的对手);
- 指标:cross-play 胜率、**Elo**、**head-to-head vs 强规则(冠军指标)**、exploitability、trace_P vs CRLB;
- ≥5 seed,bootstrap 1e4 CI。

---

## 门(逐 cell + CI)
| # | 判据 | 阈值 | 决定 |
|---|---|---|---|
| **G1(冠)** | league-commander **head-to-head 打赢强规则** | 胜率>0.5,95%CI 排除 0.5 | 可能 TAES |
| **G1-clean(关键)** | 赢是**稳健支配**不是石头剪刀布:league Elo 显著最高 **且** league 不输给"规则能赢的东西" | Elo 一致性 + 无环 | 干净 TAES |
| G2 | league > MAPPO(无 league) | 证 league 加值,非仅 CTDE | 排除"又一次 PFSP≈CTDE" |
| (顺带) | 3 条 candidate-exploit 单独 vs 规则 | 任一胜率>0.55 | 回答 G0 遗留 + 定位 exploit 机制 |

## 决策树(硬停,不开第 4 轮)
- **G1 PASS + G1-clean(稳健支配规则)** → **TAES 冠成立** → WP3 全刻画(操作包线 + Elo + 消融 + CRLB + 统计);
- **G1 PASS 但非传递(石头剪刀布,league 赢规则但输别的)** → 不是干净"RL>经典" → **诚实写成"非传递对抗动态"→ IET**(仍是贡献,不吹碾压);
- **league 收敛回规则(打平,近 Nash 坐实)** → **退 IET 地板**(testbed + 近 Nash,证据更硬)+ **转 Bet B(鲁棒/泛化 eval)抬 AppInt**。

---

## 规避的历史 bug(逐条,和 TWOTEAM plan 一致)
- α_eff:priv[:,4] 归一化 assert(防 MAPPO 暗塌 IPPO);
- PFSP:`update_pool_winrate` EMA(非死代码);
- checkpoint:`checkpoints/twoteam/wp2_league/`(非 /tmp);
- log_std_floor=-6(非 -4);
- 评测:cross-play 双向平均 + held-out(非各打各的);
- NaN/adv_std/entropy 逐轮监控,爆则先诊断不硬训;
- **近 Nash 陷阱**:league 从 BC-of-rule 起跑可能收敛回 rule——**snapshot 多样性 + PFSP 采难打对手 + candidate-exploit 种子**逼它探非局部;若仍收敛回 rule,那就是近 Nash 的真答案,诚实退。

## GPU 预算 + 时间线
| Step | 内容 | GPU |
|---|---|---|
| candidate-exploit 脚本 + 独立 eval | 便宜 | ~0.5 天 |
| BC 预训练 | 分钟级 | — |
| league 主循环(N~800-1500 iter,种群)| 主成本 | ~1-2 周 |
| cross-play 锦标赛 + Elo + 统计 | eval | ~2-3 天 |
| **合计** | | **~2-3 周** |

## 回报格式
贴回:**cross-play 矩阵 + Elo(±CI)+ head-to-head league vs 强规则(per-seed+mean+CI)+ 非传递性检测(有无环)+ G2(league vs MAPPO)+ 3 条 candidate-exploit vs 规则**。据此判 G1/G1-clean → TAES 冠 / 非传递→IET / 打平→IET 地板+Bet B。

## 诚实钉死
- **~20-25% 赔率**(近 Nash 是真证据),但地板已保、这是价值最大的一注,该先打;
- **非传递"赢"不是干净 TAES**——G1-clean 必须查环,别把石头剪刀布吹成支配;
- **硬停**:赢→WP3;非传递→IET;打平→IET 地板+Bet B。**不开第 4 轮 league 调参**——若种群自博弈 + 种子 exploit 都赢不了规则,那"规则近 Nash"就是这个域的真答案。
