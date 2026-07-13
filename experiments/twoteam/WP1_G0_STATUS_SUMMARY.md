# WP1 G0 状态总结(2026-07-13)

> 给用户判读的完整事实+诊断+决策树。**不要直接执行下一步,等用户定夺。**

---

## TL;DR

- **G0 FAIL**:exploit_gap = **−0.096**(BR 没打过 rule,BR 胜率 3%)
- **Anti-strawman 也 FAIL**:StrongRule 只赢 3/4 极端(差 pure_jam)
- **Mirror 致命信号**:rule-vs-rule = **0.000 kills 双方,240/240 episodes 全零**
- **结构性 confound**:anti-strawman 失败 + mirror=0 同时出现,G0 结果不干净
- **不推荐立刻退 IET**:先修 StrongRule 的 duck mutex(可能是 mirror=0 的元凶),再跑一次干净 G0
- **若修后仍 mirror=0 或 BR 仍打不过** → root A 确认 → 退 IET

---

## 1. WP1 构建完成度(全 PASS)

| 组件 | 路径 | 行数 | smoke |
|---|---|---|---|
| StrongRuleCommander | `algo/_shared/baselines/twoteam_strong_rule_commander.py` | 124 | ✅ |
| CommanderActorCritic(Dirichlet) | `algo/_shared/pilot/twoteam/commander_actor_critic.py` | 236 | ✅ |
| BRTrainer(PPO+α_eff blend) | `algo/_shared/pilot/twoteam/br_trainer.py` | 358 | ✅ |
| G0 gate driver | `algo/_shared/pilot/twoteam/run_g0_gate.py` | 472 | ✅ |
| Smoke 测试 | `tests/twoteam/test_wp1_smoke.py` | 99 | 4/4 PASS |

spec D2=A 守护:`task_alloc` 用 Dirichlet(α) per aperture,**不是** Categorical(4);α_eff bug 守护:trainer 内 `assert priv_4_max < 100`。

---

## 2. G0 主跑结果(200 BR iters, 6.8 min)

### 2.1 Anti-strawman 检查 → **TOO_WEAK**

| 对手 | rule 胜率 | 杀伤 rule vs opp | 备注 |
|---|---|---|---|
| pure_track | 1.00 ✅ | 1.98 vs 0.97 | OK |
| **pure_jam** | **0.15** ❌ | 0.15 vs 0.00 | **打不过 pure_jam** |
| pure_comm | 1.00 ✅ | 2.00 vs 0.00 | OK |
| pure_detect | 1.00 ✅ | 2.00 vs 0.00 | OK |
| balanced | 0.20 | 0.20 vs 0.00 | 大量 draw |
| balanced_jam_heavy | 0.25 | 0.25 vs 0.00 | 大量 draw |

阈值:≥80% 赢至少 4/4。实际 3/4 → **TOO_WEAK**(plan §R1 触发)。

### 2.2 Cell 1: rule vs rule(mirror)→ **0.000 kills**

| 指标 | 值 | 解读 |
|---|---|---|
| kills_t0 mean | 0.000 | t0 队 240 episodes 全零 |
| kills_t1 mean | 0.000 | t1 队 240 episodes 全零 |
| exposure_t0 | 4.49(constant) | **duck mutex 嫌疑**(duck=off 时辐射恰好常量) |
| trace_P_t0 mean | 0.389 | 高(>>tau_track=0.04,没人能 track) |
| trace_P_t1 mean | 0.389 | 完全对称 |
| winner dist | 100% draw | — |

**240/240 episodes,双方都打 0 杀伤。** 这是 G0 结果不干净的核心 confound。

### 2.3 BR 训练曲线(200 iters)

| iter | reward | entropy | approx_kl | v_loss | early_stop |
|---|---|---|---|---|---|
| 0 | −1.818 | −0.029 | 0.007 | 43.8 | 0 |
| 50 | −1.660 | −0.170 | 0.006 | 19.6 | 0 |
| 100 | +0.138 | −0.474 | 0.055 | 4.8 | 1 |
| 150 | +0.869 | −0.729 | 0.025 | 4.4 | 0 |
| 199 | **+0.739** | −0.761 | 0.067 | 6.2 | 1 |

- reward −1.82 → +0.74,**真实学习**(phase transition ~iter 70)
- entropy 单调下降,策略在锐化
- 14/200 iters early-stop(kl>target 0.03)—— PPO 在更新,但 kl 偏高
- adv_std=1.0 全程(归一化后,正常)
- **健康,无 NaN,无塌缩**

### 2.4 Cell 2: rule vs BR → BR **0/240 wins**

| 指标 | rule(t0) | BR(t1) | 比值 |
|---|---|---|---|
| kills mean | 1.096 | 1.000 | rule 略胜 9.6% |
| kills max | 2 | 1 | BR 单 episode 上限只到 1 |
| exposure mean | 4.49 | 2.55 | **BR 学会了 duck** |
| trace_P mean | **0.029** | **1.308** | **rule 跟踪效率 45× BR** |
| trace_P max | 0.044 | **6.522** | BR 跟踪有时彻底崩 |

| 杀伤 margin(rule-BR)分布 | n | % |
|---|---|---|
| −2(BR 赢 2) | 0 | 0.0% |
| −1(BR 赢 1) | 0 | **0.0%** |
| 0(平) | 217 | 90.4% |
| +1(rule 赢 1) | 23 | 9.6% |
| +2(rule 赢 2) | 0 | 0.0% |

**BR 没赢过一局。** 217 平局里大概率双方各杀 1 个(偶发的"互相致盲")。

### 2.5 G0 verdict

| 检查 | 阈值 | 实际 | 通过? |
|---|---|---|---|
| exploit_gap ≥ 0.5 | 0.500 | **−0.096** | ❌ |
| CI excludes 0 | > 0 | −0.133 | ❌ |
| BR 胜率 ≥ 0.55 | 0.55 | **0.03** | ❌ |
| BR 健康 | adv_std∈[0.1,100] | 1.0 | ✅ |

**G0 FAIL**(3/4 项不达标,只有训练健康)。

---

## 3. 关键诊断:为什么 G0 FAIL 不干净

### 3.1 Confound #1: StrongRule 本身太弱(anti-strawman FAIL)

`twoteam_strong_rule_commander.py` 的策略:
- 基线 alloc = [0.10, 0.45, 0.30, 0.15](detect/track/jam/comm)
- 当 enemy tracking me(trace_P < jamreact_tau=0.04)→ +0.20 jam
- 当 exposure > 30 AND no near-kill(E_max < 0.5)→ **duck(emission_off)**

**为什么打不过 pure_jam**:pure_jam 持续 jam,strong rule 的 `enemy_tracking_me` 判定要看 enemy 的 tracker trace_P,但 pure_jam 不 track(只 jam),所以 strong rule 不会 boost jam 反制;反而 strong rule 的 0.30 jam 分配被 pure_jam 完全压制。

**为什么 vs balanced/balanced_jam_heavy 都是大量 draw(0.80/0.75)**:双方都开 duck → 双方 emission_off → 双方都没杀伤 → draw。

### 3.2 Confound #2: Mirror=0 是结构性问题(duck mutex)

镜像下两队同时触发 `duck_mask = (exposure > 30) & (E_max < 0.5)`:
- 双方 exposure 同步上升(WP0 验证过物理严格对称)
- 双方 E_max 同步低(没人能 track 到对方)
- → 双方同时关辐射 → exposure 同步下降 → duck_mask 同时解除 → 双方同时恢复辐射 → 循环

结果:240/240 episodes 全 0 杀伤。`exposure_t0 = 4.49` 几乎是常量,印证了 duck 循环。

**这不是 BR 训练不够,是 StrongRule 自己和自己的镜像就是死锁。**

### 3.3 Cell 2 BR 学到了什么(why trace_P_BR=1.308)

BR 的 PPO 找到了一个**坏局部最优**:
- exposure=2.55(< rule 的 4.49)→ 学会了 duck/hide
- trace_P_BR=1.308(>> tau_track=0.04)→ **完全没学会 track**
- 但还能拿 1.00 kill/episode → 通过盲射激光偶发杀伤

即 BR 学到的是"当 rule 不注意时偷一刀",不是"用 track 反制 rule 的强防"。这是 reward +0.74 但 eval 0 wins 的原因。

### 3.4 这是 root A 吗?(用户最关心)

**目前证据不足以确认 root A**,但有强嫌疑:
- 嫌疑证据:rule-vs-rule = 0.000 kills 是 calm-sea 的典型症状
- 反证:rule-vs-BR 双方都能杀伤(1.10 vs 1.00)→ env 并非"完全无杀伤"
- 反证:rule-vs-extreme 都能拿到 1.98-2.00 kills → env 物理允许杀伤

更可能的解释:**StrongRule 的 duck 设计在镜像下创造了对称死锁,而 BR 训练时间不足以发现 break-the-symmetry 的策略**(比如先开辐射的一方暴露,另一方反制——这种 chicken-game 的解需要更长探索)。

---

## 4. 决策树

```
当前: G0 FAIL (3 confounds: 强rule太弱 + mirror=0 + BR local opt)
   │
   ├─ A. 修 StrongRule 重跑(1-2h)
   │     修 duck mutex: duck_mask 改成 (exposure>30)&(E_max<0.5)&(enemy_duck==False)
   │     修 pure_jam 反制: boost jam 用 enemy_jam_power 而非 enemy_trace_P
   │     → 期望 mirror 不再死锁,anti-strawman 通过
   │     → 重跑 G0:若 PASS → WP2;若仍 FAIL → root A 确认 → IET
   │
   ├─ B. 直接退 IET(严格按原规则)
   │     rule-vs-rule=0 已是 root A 强信号
   │     把 env + StrongRule 作为 IET 的 baseline 复用
   │     节省 WP2 self-play 算力(可能数百 GPU-h)
   │
   ├─ C. 续训 BR 到 500-1000 iters(2-4h)
   │     training reward 已稳态在 +0.7,继续训收益边际
   │     不解决 mirror=0 的结构问题
   │     不推荐
   │
   └─ D. 先深诊断(30 min)
         单跑 10 局 rule-vs-rule,trace exposure/duck/trace_P 时序
         确认 duck mutex 是否真锁死,还是另有原因
         → 确认后再决定 A/B
```

---

## 5. 我的建议

**推荐 D → A**(深诊断 30min → 若确认 duck mutex 则修 StrongRule → 重跑 G0 6.8min → 二次 G0 判)。

理由:
1. 当前 G0 FAIL 有 2 个 confound,直接退 IET 会浪费已建好的非平凡 env
2. duck mutex 是 1-2 行代码可修的具体 bug(不是设计哲学问题)
3. 修完只需 6.8 min 重跑 G0(数据来自本次 elapsed)
4. 若修后 G0 仍 FAIL,**那时的 FAIL 才是干净的 root A 信号**,退 IET 才有底气

**不推荐 B(直接退 IET)**:虽然严格执行你的"G0 FAIL → IET"规则,但当前数据未排除"StrongRule 设计 bug"这个解释。退 IET 应该在排除所有非 root-A 因素之后。

**不推荐 C(续训 BR)**:不解决 mirror=0 结构问题。

---

## 6. 产物索引(供后续复用)

| 文件 | 用途 |
|---|---|
| `experiments/twoteam/g0_gate_report.md` | G0 自动生成报告(本次) |
| `experiments/twoteam/g0_mirror_metrics.csv` | Cell 1 raw(240 rows) |
| `experiments/twoteam/g0_br_metrics.csv` | Cell 2 raw(240 rows) |
| `experiments/twoteam/g0_br_training_log.csv` | BR 200 iters 曲线 |
| `checkpoints/twoteam/br_vs_strong_rule_final.pt` | BR checkpoint(可加载续训) |
| `experiments/twoteam/wp0_check_report.md` | WP0 PASS 报告(env 健康基线) |

---

## 7. 等用户定夺

**我不会自动执行下一步。** 请选 A/B/C/D,或给其他指示。
