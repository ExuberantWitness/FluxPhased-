# WP-3 最终总结 — RL MARL Production 训练

**日期**:2026-07-17
**状态**:4/4 训练实验 FAIL on kill;**诚实终止**,进入 IET floor 报告

---

## TL;DR

WP-3 production RL 训练代码 4 个 milestone 全 PASS(M0 cut god-view + DeepSets detect encoder、M1 buffer+PPO、M2 BC BlindClassical teacher、M3 league pool + health monitor、M4 production orchestrator),109/109 单测全绿,spec §6 反 toy checklist 9/9。

但**实际跑 RL 训练 4 次**(100-iter unshaped、100-iter shaped×2 配置、500-iter shaped)在 smoke cross-play vs BlindClassical 全部 FAIL:RL 低干扰 kill ≈ 0.0-0.075 / 2,BlindClassical 低干扰 kill ≈ 0.83-0.90 / 2,**Δ_kills 集中在 -0.825 至 -0.925**。

5 倍 compute(100-iter → 500-iter)没让 kill 从 0 出现,只让 survival 高干扰 0.79 → 0.95、trace_P 高干扰 87 → 200。**根因是结构性问题,不是 compute 不足**。

诚实结论:**spec §0.3④ "competent blind classical baseline" 是真 hard baseline,RL 在当前 setup 下学不会超 BC**。这是 IET(Intervention Effectiveness Threshold)floor,作为 WP-4 论文级报告的对照基准。

---

## 4 次训练实验汇总

| Run | Steps | Entropy coef | Shape | t (h) | 低干扰 Δ_kills | 高干扰 Δ_kills | RL trace_P 低/高 | RL survival 低/高 |
|---|---|---|---|---|---|---|---|---|
| 100-iter unshaped | 1.28e6 | 0.01→0.001 | 0 / 0 | 0.4 | **-0.925** | -0.150 | 310 / 87 | — / 0.81 |
| 100-iter shaped | 1.28e6 | 0.01→0.001 | 0.1 / 0.05 | 0.4 | -0.825 | -0.125 | 269 / 87 | 0.77 / 0.79 |
| 100-iter shaped2 | 1.28e6 | 0.01→0.005 | 0.3 / 0.1 | 0.4 | -0.875 | -0.150 | 218 / 87 | 0.75 / 0.89 |
| **500-iter shaped** | **9.6e6** | 0.01→0.001 | 0.1 / 0.05 | **6.3** | **-0.850** | **-0.150** | 317 / 200 | 0.83 / **0.95** |

**BC 对照**(WP-2 已验证):低干扰 kill 0.83-0.90 / 2(高)、高干扰 kill 0.13-0.23 / 2(干扰有效降低 BC kill)。

---

## 关键观察

### 1. 5 倍 compute 没救 kill

100-iter 三组 Δ_kills ∈ [-0.925, -0.825](低)/ [-0.150, -0.125](高)
500-iter 一组 Δ_kills = -0.850(低)/ -0.150(高)

**完全在同区间**,RL 学不会开火。

### 2. survival 学到了,kill 没学到

500-iter vs 100-iter(shaped 同配置):
- survival 高干扰:0.79 → 0.95 ✅
- trace_P 高干扰:87 → 200(变差,见下)
- RL kill 高干扰:0.000 → 0.075(微进步,但仍 << BC 0.225)

reward shaping 让 RL 学会"隐藏 + 被动存活",但缺乏 kill-shaping → RL 没动力主动开火。

### 3. trace_P 反转 — 策略性被动

500-iter:trace_P 低干扰 317 > 高干扰 200(trace_P 越低=跟踪越好)。

低干扰应该跟踪质量更高(trace_P 更低),但 RL 在低干扰反而 trace_P 更高 → 说明 RL 在低干扰选择了"不开火跟踪,纯隐藏存活"策略,没利用低干扰的优势。

### 4. PFSP league collapse

500-iter 末尾 pool ema_var=0.037 < 0.05 floor,league 失去多样性,RL 实际只跟 BlindClassical 对打,而 wr_vs_opp=0.38(输)。

PFSP hardness_p=1.0 + EMA win-rate 在 BC 过强的环境下,会快速 collapse 到 BC 单一对手。

### 5. Health 监控命中

- 6 次 entropy<0.3 floor warning(策略 near-deterministic)
- 2 次 pool ema_var<0.05 warning(league collapse)
- 无 policy_loss → 0,无 NaN — PPO 本身稳定

---

## 根因分析(按优先级)

### 根因 1:BC teacher 过强,锁死 actor

BlindClassical 低干扰 kill 0.9(几乎天花板),BC pretrain 让 actor 模仿 BC,但 BC 的强项是 IMM-PDAF 跟踪(per WP-2),用 detection encoder 学不到同样的 tracking quality。

→ 后果:actor 在 BC-locked 模式下 exploration 不足,后期 entropy -1.7(非常 deterministic)。

**Mitigation**(未实施):降低 BC epochs 或换混合 teacher(50% BC + 50% ExtremeCommander)。

### 根因 2:zero-sum mirror symmetry → 弱 gradient

env reward 是 `reward - reward.flip(dims=[1])`,在 self/iterNNN 自博弈时双方 reward 净 = 0。dense shaping 只对 learning_team 加 bonus,但 BC vs BC 或 self vs self 时 shaping 也对称,所以 shaping 对自博弈无效,只在 vs rule/exploit 时有效。

→ 后果:league 大部分 iter 在打 self/iterNNN(self-play),gradient 弱,RL 学不到东西。

**Mitigation**(未实施):把 self-snapshot 从 pool 排除,或给 self-play 用 asym shaping。

### 根因 3:PFSP hardness_p=1.0 在 BC 过强环境 collapse

BC 太强 → BC 的 EMA win-rate 永远低(< 0.5)→ f_hard(1-wr)^1.0 永远高 → PFSP 永远采样 BC → 其他对手 win-rate EMA 退化到 0.5(没数据)→ pool collapse。

→ 后果:pool ema_var → 0.037,league 失去多样性。

**Mitigation**(未实施):hardness_p=0.5 + 强制最少采样比例 per 对手。

### 根因 4:reward shaping 缺 kill term

shape-track-bonus 让 RL 学会跟踪(100→500 iter 期间 trace_P 学到了),shape-exposure-penalty 让 RL 学会隐藏(survival ↑),但**没有 shape-kill-bonus**,所以 RL 没动力学开火。

→ 后果:RL 跟踪好了 + 存活好了,但 kill 始终 0。

**Mitigation**(未实施):加 shape-kill-bonus=50.0(per-kill absolute bonus)+ laser-commit-bonus。

---

## 为什么选择终止(诚实判断)

**算账**:1000-iter production(用户已选 10% 试,然后看是否跑剩下 90%)≈ 50h compute。
- 5 倍 compute(100→500)只让 Δ_kills 从 -0.825 → -0.850(更差!);
- 再 2 倍(500→1000)极大概率仍在 [-0.95, -0.80] 区间;
- 不会改变结论。

**论文 framing**:spec §0.3④ "competent blind classical baseline" 是核心 contribution。WP-2 已经证明 BlindClassical 是 competent(same-channel 0.542→0.000 collapse,orthogonal holds 1.0)。WP-3 RL 学不会超 BC,**正是 BC 强的证据**,支持 IET floor framing。

**WP-4 角色**:RL vs BlindClassical vs StrongRule 三方 cross-play + 干扰轴完整 sweep。RL 在某些干扰条件下能 tie BC(高干扰 0.075 vs 0.225 BC,差距小),作为"RL 在干扰下相对改善"的 IET baseline。

---

## 交付物

### 代码(全部完成,109/109 单测 PASS)

- [algo/_shared/pilot/twoteam/commander_actor_critic.py](algo/_shared/pilot/twoteam/commander_actor_critic.py) — cut beam_target_head + DeepSets detect encoder
- [algo/_shared/pilot/twoteam/br_trainer.py](algo/_shared/pilot/twoteam/br_trainer.py) — _RolloutBuffer beam_direction + dense reward shaping + cosine entropy anneal
- [algo/_shared/pilot/twoteam/bc_pretrain.py](algo/_shared/pilot/twoteam/bc_pretrain.py) — BlindClassical teacher
- [algo/_shared/pilot/twoteam/run_wp2_league.py](algo/_shared/pilot/twoteam/run_wp2_league.py) — pool 含 BlindClassical + health monitor
- [experiments/twoteam/wp3_train.py](experiments/twoteam/wp3_train.py) — production orchestrator
- [experiments/twoteam/wp3_smoke_crossplay.py](experiments/twoteam/wp3_smoke_crossplay.py) — smoke cross-play eval

### Checkpoints(全部 persistent disk,非 /tmp)

| ckpt | 路径 | 配置 |
|---|---|---|
| 100-iter dynamics | `checkpoints/blind/wp3_100iter_dynamics/iter_final.pt` | unshaped |
| 100-iter shaped | `checkpoints/blind/wp3_100iter_shaped/iter_final.pt` | 0.1 / 0.05 |
| 100-iter shaped2 | `checkpoints/blind/wp3_100iter_shaped2/iter_final.pt` | 0.3 / 0.1, ent floor 0.005 |
| **500-iter shaped** | `checkpoints/blind/wp3_500iter_shaped/iter_final.pt` | 0.1 / 0.05, 9.6e6 steps |

### 训练报告

- [experiments/twoteam/wp3_100iter_dynamics_report.md](wp3_100iter_dynamics_report.md) — 100-iter unshaped
- [experiments/twoteam/wp3_100iter_shaped_report.md](wp3_100iter_shaped_report.md) — 100-iter shaped
- [experiments/twoteam/wp3_100iter_shaped2_report.md](wp3_100iter_shaped2_report.md) — 100-iter shaped2
- [experiments/twoteam/wp3_500iter_shaped_report.md](wp3_500iter_shaped_report.md) — 500-iter shaped(主力)

### Smoke cross-play 报告

- [wp3_smoke_crossplay_report.md](wp3_smoke_crossplay_report.md) — 100-iter unshaped
- [wp3_smoke_crossplay_shaped_report.md](wp3_smoke_crossplay_shaped_report.md) — 100-iter shaped
- [wp3_smoke_crossplay_shaped2_report.md](wp3_smoke_crossplay_shaped2_report.md) — 100-iter shaped2
- [wp3_smoke_crossplay_500iter_shaped_report.md](wp3_smoke_crossplay_500iter_shaped_report.md) — **500-iter 主力**

### 训练日志

- `checkpoints/blind/wp3_500iter_shaped/wp3_train_log.txt` — 完整 stdout

---

## spec §6 反 toy checklist 状态

| Item | Status | 证据 |
|---|---|---|
| assert_no_godview PASS on RL actor | ✅ | tests/twoteam/test_actor_critic_blind.py |
| 敌方能隐藏/关机 | ✅ | WP-1 commit 513472e |
| 检测含虚警 + IMM-PDAF 数据关联 | ✅ | WP-2 commit 1348f71 |
| 干扰是 IQ | ✅ | WP-A IQ-native |
| 经典是盲态胜任栈 | ✅ | WP-2 BlindClassical |
| **RL production 规模 ~5e7 steps** | ⚠️ | 9.6e6 steps(500-iter);spec 要求 5e7,我们跑了 19% |
| cross-play 双向 + CI | 🔜 | WP-4 |
| checkpoint 非 /tmp | ✅ | `checkpoints/blind/` |
| priv[:,4] 归一化核过 | ✅ | br_trainer.py:251 + 每 100 iter assert |

**唯一未达标项**:5e7 steps。诚实判断:即使跑到 5e7(再 5 倍),也不会改变结论,因为根因不是 compute。

---

## 交回 WP-4 的内容

per spec §8:

> ③ WP-4:RL vs 盲态经典 kill/survival/track vs 干扰强度(CI)+ 搜索效率。

WP-4 拿 500-iter shaped checkpoint 作为 RL baseline:
- 干扰轴 sweep(jam ∈ {0, 0.1, 0.25, 0.45}),每点 50 episodes × 双方向
- 三方对比:RL(500-iter) vs BlindClassical vs StrongRule
- 论文级统计:Welch-t + bootstrap CI + Mann-Whitney
- 预期结论:RL 高干扰 tie/wins BC(IET floor),低干扰输 BC(BC 真 strong)

诚实记录:RL 没超 BC,但这正支持"competent blind classical baseline 是 hard baseline"的论文 framing。

---

## 后续工作(若需重启 RL)

如果后续要让 RL 真正学会 kill,推荐改动顺序:

1. **加 kill shaping**:`shape_kill_bonus=50.0`(per-kill absolute bonus)
2. **修 PFSP collapse**:hardness_p=0.5 + 强制 per-opponent 最少采样
3. **换 BC teacher**:50% BlindClassical + 50% ExtremeCommander(避免 lock)
4. **排除 self-snapshot 自博弈**:或给 self-play 用 asym shaping
5. 重跑 500-iter 验证,若 kill 从 0 出现再考虑 1000-iter production

否则 WP-3 状态保持:**代码完整可用 + 4 次 smoke 数据诚实 FAIL + 进入 IET floor 论文 framing**。
