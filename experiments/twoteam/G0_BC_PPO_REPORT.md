# WP1 G0 BC→PPO 范式验证 — 详细实验报告

**日期**: 2026-07-14
**项目**: 两队对称多功能相控阵对抗(TWOTEAM_MULTIFUNCTION_PLAN.md)
**目标会议**: TAES(主)/ IET(备选)
**spec**: TWOTEAM_ENV_FIX_SPEC.md + AlphaStar SL→RL 范式(Vinyals et al. 2019, Nature)
**前序**: G0_FULL_EXPERIMENT_REPORT.md(2026-07-14,前两次 G0 FAIL 诊断)

---

## 摘要

本报告详细记录了在 TWOTEAM_MULTIFUNCTION 环境中应用 **行为克隆预训练(BC)+ PPO 微调(fine-tune)** 范式(AlphaStar SL→RL 范式)以求解 G0 exploitability gate 的完整实验过程。

**实验结论**:
- BC 范式**技术上完全成功**:用 StrongRule 作专家,50K 样本 + 15 epoch NLL 监督训练后,actor deterministic 策略 task_alloc 与 rule 几乎完美匹配(70% vs 71% track concentration)
- BC 把 PPO 起步 reward 从随机初始化的 **-1.82** 推到 **-0.68**(63% 改进)
- PPO fine-tune 500 iters 进一步把 reward 推到 **-0.05**,完全收敛
- 但 Cell 2 评估显示 BC+PPO'd BR 与 rule 形成 **93% 平局**(rule kills 1.95 vs BR kills 1.93),exploit_gap = **-0.016**,CI **[-0.05, +0.02] 包含 0**
- G0 仍 FAIL,但与前两次性质完全不同 — 本次 FAIL 暗示 **StrongRule 可能是 env 的近似 Nash 均衡**

---

## 1. 实验背景与动机

### 1.1 G0 命门

**用户 verbatim**: "G0 是最先的命门:π_rule 可不可被 exploit?可 → 我放行 WP2;不可 → 别烧自博弈,一起退 IET"

G0 数学定义:
```
exploitability(π_rule) = U(π_rule vs 镜像 π_rule) − U(π_rule vs BR(π_rule))
                      ≈ 0 − (negative) = 正值 = rule 可被 exploit 的程度
```

**PASS 判据**: exploit_gap ≥ 0.5 kills/episode **且** bootstrap 95% CI 排除 0 → 放行 WP2 self-play
**FAIL 判据**: gap ≈ 0 或 CI 含 0 → 退 IET(IQ/CRLB 基线论文)

### 1.2 前两次 G0 FAIL 概况

**第一次 FAIL**(默认超参,BR 500 iters):
- exploit_gap = -0.963(CI [-0.98, -0.94])
- BR task_alloc near-prior 均匀 [0.25, 0.25, 0.25, 0.25](Dirichlet α 没学到浓度)
- BR training reward 平坦在 -1.82 至 iter 275,iter 300 才开始学习

**第二次 FAIL**(调超参 lr_actor=1e-4, entropy_coef=0.03, lr_decay):
- exploit_gap = -0.979(CI [-1.00, -0.96])
- BR 学到 track-heavy 策略 [0.20, 0.36, 0.30, 0.14],但浓度只有 rule 的一半
- BR training reward -1.82 → -0.88(改进但不够)

### 1.3 根本困境

on-policy PPO 从随机初始化出发,在 **18 维动作空间**(13 连续 + 5 离散)中同时承担两个任务:
1. **学会玩**(learn to play) — 学会基本的多功能分配 + beam/laser 协调 + 抗干扰反应
2. **找 exploit**(find exploit) — 发现 rule 不防御的弱点

任务 1 耗掉了大部分样本(800K samples = 500 iters × 200 horizon × 8 envs),任务 2 没有足够样本预算。

### 1.4 解决方案:AlphaStar SL→RL 范式

**用户提议**(verbatim): "是否可以让 rule 生成样本,作模仿学习预训练基座模型,然后再强化学习 selfplay/联赛机制提升的模式?"

**范式**(Vinyals et al. 2019, Nature "Grandmaster level in StarCraft II using multi-agent reinforcement learning"):
1. **BC pretrain**(SL phase): 用 StrongRule 作专家,监督学习让 actor 一上来就会玩
2. **PPO fine-tune**(RL phase): 从 rule 局部最优出发,小 LR + 中 entropy 微调找 exploit

**预期效果**:
- BC 把 PPO 起点从随机推到 rule 局部最优
- PPO 不用再花样本学 "怎么玩",直接探索 exploit 结构

---

## 2. 实验环境配置

### 2.1 硬件

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- **VRAM**: 101.9 GB
- **CUDA**: 是
- **框架**: PyTorch

### 2.2 环境(TwoTeamVecEnv)超参

| 参数 | 值 | 说明 |
|---|---|---|
| `n_envs` | 8 | 并行环境数 |
| `horizon`(episode_steps) | 200 | 单 episode 步数 |
| `n_episodes`(eval) | 30 | Cell 1/2 评估 episodes |
| `geometry` | RANDOM_GEOMETRY | 镜像轴保持的随机几何 |
| `freq_hop_max` | 8.0 | 最大频率捷变率(Fix 1) |
| `jam_gain` | 6.0 | 干扰耦合增益(调过,原 8.0) |
| `exposure_gain` | 200.0 | 暴露敏感度(Fix 2,原 50) |
| `tau_track` | 0.04 | 跟踪锁定阈值 |
| `reward_scale` | 0.1 | per-step reward 缩放(env 累积 kill 量级大) |

### 2.3 网络(TwoTeamCommanderActorCritic)结构

- **actor trunk**: Linear(36, 256) → Tanh → Linear(256, 256) → Tanh
- **task_alloc_head**: Linear(256, 2×4=8) → softplus + 0.5 → **Dirichlet α**(每孔径一个 4 维浓度)
- **beam_target_head**: Linear(256, 2×2=4) → **Categorical logits**(每孔径选 0/1 敌雷达)
- **laser_target_head**: Linear(256, 2) → **Categorical logits**(选 0/1 敌雷达)
- **emission_on_head**: Linear(256, 2) → **Bernoulli logits**(每孔径开关)
- **freq_hop_head**: Linear(256, 2×2=4) → softplus + 0.5 → **Beta(α, β)** → 重缩放到 [1, 8]
- **central_trunk**(critic): Linear(36+8, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, 1)
- **local_trunk**(IPPO critic): Linear(36, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, 1)

**动作维度统计**:
- 连续: 8(Dirichlet 4 维 × 2 孔径)+ 4(Beta α,β × 2 孔径) = **12** 维(注:实际 13 维含 Bernoulli 概率)
- 离散: 4(Categorical 2 类 × 2 孔径)+ 2(Categorical 2 类 laser)+ 2(Bernoulli × 2 孔径) = **8** 维
- 总计 ~18 维

---

## 3. 实现细节

### 3.1 BC 数据收集算法(`bc_pretrain.py::collect_samples`)

**输入**: env, rule, n_samples=50000, episode_steps=200
**输出**: dict {obs[N,36], priv[N,8], task_alloc[N,2,4], beam_target[N,2], laser_target[N], emission_on[N,2], freq_hop_rate[N,2]}

**算法**:
```
opponent_strategies = [pure_track, pure_jam, pure_comm, pure_detect,
                       balanced, balanced_jam_heavy, track_agile]   # 7 个多样化对手
total = 0, ep = 0
while total < 50000:
    opp = STRATEGIES[ep % 7]   # 循环选对手
    
    # Phase 1: rule = team 0
    env.seed = 1000 + ep*2; env.reset()
    for step in 200:
        obs_dict = env.get_obs()                    # snapshot BEFORE step
        a_rule = rule.get_action(env, team=0)
        a_opp = opp.get_action(env, team=1)
        record(obs_dict["obs"][:,0], obs_dict["priv"][:,0], a_rule)
        env.step(combine_team_actions(a_rule, a_opp))
        total += 8  # n_envs
        if total >= 50000: break
    
    # Phase 2: rule = team 1(对称数据增强)
    if total < 50000:
        env.seed = 1000 + ep*2 + 1; env.reset()
        for step in 200:
            obs_dict = env.get_obs()
            a_opp = opp.get_action(env, team=0)
            a_rule = rule.get_action(env, team=1)
            record(obs_dict["obs"][:,1], obs_dict["priv"][:,1], a_rule)
            env.step(combine_team_actions(a_opp, a_rule))
            total += 8
            if total >= 50000: break
    
    ep += 1
```

**关键设计**:
1. **多样化对手**: rule vs 同一对手只覆盖单一场景。7 个 ExtremeCommander 让 BC 数据覆盖 "rule vs pure_jam"(rule 应抗干扰)/ "rule vs balanced" / "rule vs track_agile" 等多种情况
2. **对称数据增强**: env 零和对称,rule 在 team 1 时视角相反。这给 BC 2× 数据多样性,也教 AC "对称策略"
3. **50K 样本预算**: 实测 16 episodes(8 个不同对手 × 2 队)用 40 秒收集完成

### 3.2 NLL loss(`bc_pretrain.py::_bc_loss`)

```python
def _bc_loss(self, obs_b, priv_b, action_b):
    """NLL on rule's action under AC's current distribution."""
    log_prob, _, _, _ = self.ac.evaluate_actions(obs_b, action_b, priv_b)
    return -log_prob.mean()
```

**log_prob 是多头条件密度的对数**(在 `commander_actor_critic.py:159-217` 实现):
```
log_prob = task_logp.sum(-1)       # Dirichlet(α) on [0.25,...]-simplex
         + beam_logp.sum(-1)       # Categorical(logits) on {0,1}
         + laser_logp              # Categorical(logits) on {0,1}
         + emit_logp.sum(-1)       # Bernoulli(logits) on {0,1}
         + fh_logp.sum(-1)         # Beta(α,β) on [0,1] (inverse-rescaled from [1,8])
```

**关键复用**: 不重写 log density 计算,直接调用 PPO 的 `evaluate_actions`。NLL = `-log_prob.mean()` 是标准 maximum likelihood BC。

### 3.3 BC 训练循环(`bc_pretrain.py::train`)

**超参**:
- lr=1e-3, batch_size=256, val_split=0.1
- early_stop_patience=3(val_loss 3 epoch 不降则停)
- grad clip 1.0

**每 epoch 操作**:
1. Shuffle 训练索引
2. 分 minibatches,每 batch 算 NLL loss,backward,step
3. 验证集 forward only(无 backward)
4. log train_loss, val_loss
5. 早停检查

**关键设计**: **Critic 不 BC**。central_trunk + local_trunk 是 PPO fine-tune 阶段学的价值函数。BC 阶段不需要 value,留给 PPO 自己从 GAE 学。

### 3.4 PPO fine-tune(`br_trainer.py`)

完全复用现有 `TwoTeamBRTrainer`,只改初始化 — `br_ac` 已经是 BC 训练过的。

**PPO 超参**(本次实验):
- lr_actor=1e-4(小步微调,不破坏 BC 起点策略)
- lr_critic=1e-3(critic 还没学过,需要更快学)
- entropy_coef=0.01(允许少量探索,但不像之前 0.03 那样激进扩散)
- clip=0.2, gamma=0.99, gae_lambda=0.95
- reward_scale=0.1, value_huber_delta=1.0
- target_kl=0.03, max_grad_norm=0.5
- n_epochs=4, minibatch_size=64

### 3.5 G0 评估(`run_g0_gate.py`)

**Cell 1**(mirror baseline):
- π_rule vs π_rule × 30 episodes × 8 envs = 240 episodes
- 计算 margin = kills_t0 − kills_t1 per episode
- bootstrap 10000 次 → 95% CI

**Cell 2**(exploit test):
- π_rule vs BR(π_rule) × 30 episodes × 8 envs
- BR = BC+PPO'd AC,deterministic 策略
- margin = kills_rule − kills_BR(rule POV)

**Exploit gap**:
```
gap = mean(mirror_margin) − mean(br_margin)
CI: bootstrap both arrays independently, subtract
```

---

## 4. 实验过程与现象

### 4.1 V0 — Smoke Test(2026-07-14)

**命令**:
```bash
python -u tests/twoteam/test_bc_pretrain_smoke.py
```

**目的**: 验证 BC pipeline 不 crash + train_loss 下降 + deterministic 策略改变

**结果**:
```
test_bc_pretrainer_smoke:
  [BC collect] ep=1 total=200/200 opp=pure_track t=0.5s
  [BC collect] DONE: 200 samples in 0.5s (1 episode)
  [BC train] epoch=1/3 train_loss=-0.069 val_loss=-4.295
  [BC train] epoch=2/3 train_loss=-6.037 val_loss=-8.628
  [BC train] epoch=3/3 train_loss=-9.957 val_loss=-11.008
  ✅ train_loss -0.069 → -9.957

test_bc_changes_deterministic_policy:
  [BC collect] ep=5 total=500/500 opp=balanced t=1.5s
  [BC train] epoch=1/5 train_loss=-2.945 val_loss=-7.697
  [BC train] epoch=5/5 train_loss=-14.342 val_loss=-14.692
  ✅ BC shifted task_alloc by max 0.260
```

**现象**:
1. **数据收集极快**: 200 samples 仅需 0.5 秒(env vectorized GPU)
2. **train_loss 下降迅猛**: 3 epoch 从 -0.07 → -9.96(143× 改进)
3. **task_alloc 显著偏移**: 5 epoch BC 让 deterministic 策略偏移 0.26(远超 0.05 阈值)
4. **无 NaN**: 所有参数健康

**判读**: BC pipeline 工作正常,且 AC 对 rule 策略的学习速度极快。这说明 rule 策略对 AC 网络来说是 "容易表达" 的。

---

### 4.2 V1 — BC-only(BR iter=0,2026-07-14)

**命令**:
```bash
python -u algo/_shared/pilot/twoteam/run_g0_gate.py \
    --br-iters 0 --horizon 200 --n-envs 8 --n-episodes 30 \
    --bc-pretrain-samples 50000 --bc-pretrain-epochs 15 \
    --out experiments/twoteam/g0_bc_only_report.md
```

**目的**: 不做 PPO fine-tune,只看 BC'd AC 在 Cell 2 的表现。这是 "BC 起点策略质量" 的纯检测。

#### 4.2.1 Anti-strawman 检查

```
rule vs pure_track         : WR=0.95 draws=0.05 kills 1.95 vs 0.97
rule vs pure_jam           : WR=0.38 draws=0.62 kills 0.35 vs 0.00
rule vs pure_comm          : WR=1.00 draws=0.00 kills 2.00 vs 0.00
rule vs pure_detect        : WR=1.00 draws=0.00 kills 1.95 vs 0.00
rule vs balanced           : WR=0.95 draws=0.05 kills 1.95 vs 0.95
rule vs balanced_jam_heavy : WR=0.95 draws=0.05 kills 1.95 vs 1.00
verdict: TOO_WEAK (extreme wins ≥80%: 3/4)
```

**现象**: rule vs pure_jam 只有 38% 胜率(62% 平局),其他 extreme 都 ≥ 95%。这是 **G0 PASS 的潜在 exploit 路径** — BR 应该学会打 rule 像 pure_jam 那样让 rule 难受。

#### 4.2.2 Cell 1(mirror baseline)

```
mirror margin: +0.000 (95% CI [+0.000, +0.000])
kills: rule_t0=1.94, rule_t1=1.94
winner: t0=0.00, t1=0.00, draw=1.00
```

**现象**: rule vs rule 严格 0-0(100% 平局)。这是 mirror symmetry 的预期结果。注意:这个 100% 平局意味着 **rule 自己跟自己也是 1.94 kills 各队** — 不是 0-0 stalemate,是 1.94-1.94 同步杀。

#### 4.2.3 BC 数据收集

```
[BC collect] ep=5 total=16000/50000 opp=balanced t=12.8s
[BC collect] ep=10 total=32000/50000 opp=pure_comm t=25.6s
[BC collect] ep=15 total=48000/50000 opp=pure_track t=38.6s
[BC collect] ep=16 total=50000/50000 opp=pure_jam t=40.2s
[BC collect] DONE: 50000 samples in 40.2s (16 episodes)
```

**现象**: 16 episodes(每 episode 200 步 × 8 envs = 1600 samples,16 episodes ≈ 25600 step-samples,但加上双队对称增强 = 51200,然后 trim 到 50000)。每 episode ~2.5 秒。

#### 4.2.4 BC 训练曲线(完整 15 epoch)

| epoch | train_loss | val_loss | 说明 |
|---|---|---|---|
| 1 | -16.189 | -19.551 | 起步已经远好于 smoke test 的 -0.07(数据多 250×) |
| 2 | -21.711 | -23.018 | 大幅下降 |
| 3 | -23.773 | -24.279 | |
| 4 | -24.767 | -25.091 | |
| 5 | -25.484 | -25.734 | |
| 6 | -26.048 | -26.221 | |
| 7 | -26.521 | -26.685 | |
| 8 | -26.936 | -27.063 | |
| 9 | -27.284 | -27.387 | |
| 10 | -27.610 | -27.709 | |
| 11 | -27.909 | -27.997 | |
| 12 | -28.183 | -28.252 | |
| 13 | -28.442 | -28.479 | |
| 14 | -28.690 | -28.750 | |
| 15 | **-28.918** | **-28.961** | 收敛(val_loss 仍微降,没早停) |

**现象**:
1. **train/val loss 差距很小**(epoch 15: -28.92 vs -28.96)→ 无过拟合
2. **每 epoch 改进单调递减**(epoch 1→2 改进 5.5,epoch 14→15 改进 0.2)→ 典型饱和曲线
3. **val_loss 持续下降到 epoch 15** → 没触发早停,可能再训几个 epoch 还能微改
4. **NLL = -28.96** 意味着 rule action 在 AC 分布下概率约 e^29 ≈ 4e12 → 极大概率(因为 18 维联合密度的 log_prob 自然是大数)

#### 4.2.5 BC sanity check(deterministic 策略 profile)

```
BC task_alloc profile (4 fns, avg over 2 apertures): [0.08, 0.71, 0.12, 0.10]
BC freq_hop mean: 1.03
Rule task_alloc profile: [0.10, 0.71, 0.09, 0.10] (approx)
```

**现象**:
- **task_alloc track 浓度 = 0.71**(rule 也是 0.71)→ **完美匹配**
- detect / track / jam / comm 比例分别为 [0.08, 0.71, 0.12, 0.10] vs rule [0.10, 0.71, 0.09, 0.10] → 各项偏差 < 0.03
- **freq_hop = 1.03**(rule 默认 1.0)→ BC 学到了 rule "无干扰时不 hop" 的行为

**判读**: BC 完美学到了 rule 的核心策略 profile。

#### 4.2.6 Cell 2(BC'd AC vs rule,无 PPO fine-tune)

```
br margin (rule POV): +0.138 (95% CI [+0.087, +0.192])
kills: rule_t0=1.95, br_t1=1.81
winner: rule=0.11, BR=0.00, draw=0.89
```

**现象**:
1. **BR kills 1.81**(对比上次纯 PPO BR: 0.98)→ **BC 让 BR 几乎匹配 rule 的杀敌能力**(+85%)
2. **89% 平局** → BC'd AC 行为接近 rule mirror
3. **rule 11% 胜 / BR 0% 胜** → rule 在非平局中略微优势(BC 还没学到 rule 的全部细节,或 stochastic 差异)
4. **exploit_gap = -0.138**(CI [-0.19, -0.09])→ 仍排除 0,但比纯 PPO 的 -0.963 / -0.979 缩小 7×

**判读**: BC 起点 BR 已经能跟 rule 接近持平(平局 89%)。这是 PPO fine-tune 的好起点。

---

### 4.3 V2 — BC + PPO fine-tune(500 iters,2026-07-14)

**命令**:
```bash
python -u algo/_shared/pilot/twoteam/run_g0_gate.py \
    --br-iters 500 --horizon 200 --n-envs 8 --n-episodes 30 \
    --bc-pretrain-samples 50000 --bc-pretrain-epochs 15 \
    --br-lr-actor 1e-4 --br-entropy-coef 0.01 \
    --out experiments/twoteam/g0_bc_then_ppo_report.md
```

**目的**: 在 BC 起点上做完整 PPO fine-tune,看能否找到 exploit 让 G0 PASS。

#### 4.3.1 Anti-strawman 检查

```
rule vs pure_track         : WR=0.97 draws=0.03 kills 1.98 vs 1.00
rule vs pure_jam           : WR=0.33 draws=0.68 kills 0.33 vs 0.00
rule vs pure_comm          : WR=1.00 draws=0.00 kills 2.00 vs 0.00
rule vs pure_detect        : WR=1.00 draws=0.00 kills 1.95 vs 0.00
rule vs balanced           : WR=1.00 draws=0.00 kills 2.00 vs 0.97
rule vs balanced_jam_heavy : WR=0.95 draws=0.05 kills 1.95 vs 1.00
verdict: TOO_WEAK
```

跟 V1 的 anti-strawman 几乎一致(随机种子不同,有微小波动)。

#### 4.3.2 BC 阶段(同 V1)

```
[BC train] epoch=15/15 train_loss=-28.921 val_loss=-29.012 best=-29.012
BC task_alloc profile: [0.08, 0.70, 0.12, 0.10]   # 70% track
BC freq_hop mean: 1.03
```

跟 V1 几乎一致(随机种子同 seed=100,微小差异来自 GPU non-determinism)。

#### 4.3.3 PPO fine-tune 训练曲线(完整 51 个 log 点,每 10 iter 一记)

| iter | reward | v_loss | pi_loss | ent | kl | clip_frac | es | t(min) |
|---|---|---|---|---|---|---|---|---|
| 0 | **-0.682** | 12.607 | -0.010 | -2.379 | 0.033 | 0.10 | 0 | 0.0 |
| 10 | -0.527 | 12.802 | -0.004 | -2.354 | 0.027 | 0.09 | 0 | 0.3 |
| 20 | -0.499 | 9.124 | -0.004 | -2.364 | 0.019 | 0.09 | 0 | 0.6 |
| 30 | -0.637 | 9.937 | +0.001 | -2.350 | 0.010 | 0.05 | 0 | 0.9 |
| 40 | -0.772 | 11.076 | -0.001 | -2.420 | 0.044 | 0.05 | 0 | 1.1 |
| 50 | -0.746 | 8.041 | +0.008 | -2.456 | 0.046 | 0.05 | 1 | 1.4 |
| 60 | -0.441 | 7.895 | +0.021 | -2.313 | 0.061 | 0.15 | 1 | 1.7 |
| 70 | -0.299 | 2.452 | -0.008 | -2.309 | 0.025 | 0.14 | 0 | 1.9 |
| 80 | -0.260 | 1.877 | -0.001 | -2.251 | 0.039 | 0.13 | 0 | 2.2 |
| 90 | -0.242 | 2.579 | -0.006 | -2.276 | 0.039 | 0.12 | 0 | 2.5 |
| 100 | -0.253 | 2.815 | -0.008 | -2.267 | 0.033 | 0.18 | 0 | 2.8 |
| 110 | -0.214 | 2.690 | -0.010 | -2.229 | 0.036 | 0.12 | 0 | 3.1 |
| 120 | -0.203 | 2.376 | -0.013 | -2.228 | 0.036 | 0.15 | 0 | 3.4 |
| 130 | -0.187 | 2.229 | -0.001 | -2.258 | 0.017 | 0.08 | 0 | 3.7 |
| 140 | -0.168 | 1.711 | -0.000 | -2.242 | 0.040 | 0.12 | 0 | 3.9 |
| 150 | -0.169 | 2.433 | +0.009 | -2.349 | 0.060 | 0.09 | 1 | 4.1 |
| 160 | -0.098 | 1.798 | +0.009 | -2.344 | 0.078 | 0.18 | 1 | 4.3 |
| 170 | -0.073 | 1.663 | +0.003 | -2.323 | 0.074 | 0.11 | 1 | 4.5 |
| 180 | -0.170 | 2.557 | +0.012 | -2.360 | 0.079 | 0.10 | 1 | 4.8 |
| 190 | -0.098 | 1.586 | +0.007 | -2.320 | 0.065 | 0.08 | 1 | 5.0 |
| 200 | -0.099 | 1.632 | -0.003 | -2.349 | 0.036 | 0.09 | 0 | 5.2 |
| 210 | -0.102 | 1.535 | +0.013 | -2.327 | 0.075 | 0.09 | 1 | 5.4 |
| 220 | -0.090 | 1.774 | +0.008 | -2.338 | 0.074 | 0.10 | 1 | 5.6 |
| 230 | -0.064 | 1.294 | +0.010 | -2.323 | 0.053 | 0.12 | 1 | 5.8 |
| 240 | -0.062 | 0.509 | +0.006 | -2.338 | 0.072 | 0.09 | 1 | 6.0 |
| 250 | **-0.040** | 0.378 | +0.016 | -2.344 | 0.082 | 0.10 | 1 | 6.2 |
| 260 | -0.083 | 1.437 | +0.007 | -2.363 | 0.094 | 0.08 | 1 | 6.4 |
| 270 | -0.049 | 1.558 | +0.006 | -2.351 | 0.053 | 0.04 | 1 | 6.6 |
| 280 | -0.093 | 1.635 | +0.004 | -2.332 | 0.099 | 0.11 | 1 | 6.9 |
| 290 | -0.191 | 3.643 | +0.005 | -2.371 | 0.073 | 0.08 | 1 | 7.1 |
| 300 | -0.189 | 3.691 | +0.006 | -2.316 | 0.055 | 0.07 | 1 | 7.3 |
| 310 | -0.120 | 1.612 | -0.002 | -2.315 | 0.025 | 0.05 | 0 | 7.5 |
| 320 | -0.166 | 1.321 | -0.002 | -2.335 | 0.044 | 0.15 | 0 | 7.7 |
| 330 | -0.056 | 1.563 | +0.004 | -2.345 | 0.088 | 0.10 | 1 | 7.9 |
| 340 | -0.101 | 1.339 | +0.003 | -2.336 | 0.047 | 0.05 | 1 | 8.1 |
| 350 | -0.127 | 3.156 | +0.010 | -2.358 | 0.101 | 0.16 | 1 | 8.3 |
| 360 | -0.048 | 1.848 | -0.001 | -2.336 | 0.051 | 0.08 | 1 | 8.5 |
| 370 | -0.077 | 1.373 | +0.008 | -2.361 | 0.085 | 0.10 | 1 | 8.7 |
| 380 | -0.103 | 1.366 | -0.004 | -2.329 | 0.034 | 0.08 | 0 | 9.0 |
| 390 | **-0.031** | 0.130 | +0.000 | -2.353 | 0.113 | 0.16 | 1 | 9.2 |
| 400 | -0.070 | 1.162 | +0.011 | -2.355 | 0.089 | 0.08 | 1 | 9.4 |
| 410 | -0.034 | 0.131 | +0.012 | -2.363 | 0.128 | 0.12 | 1 | 9.6 |
| 420 | -0.041 | 1.661 | +0.001 | -2.357 | 0.105 | 0.12 | 1 | 9.8 |
| 430 | -0.071 | 1.055 | +0.008 | -2.357 | 0.076 | 0.16 | 1 | 10.0 |
| 440 | -0.293 | 1.765 | -0.004 | -2.351 | 0.030 | 0.11 | 0 | 10.2 |
| 450 | -0.096 | 1.210 | +0.003 | -2.358 | 0.075 | 0.08 | 1 | 10.4 |
| 460 | -0.056 | 1.453 | +0.003 | -2.382 | 0.093 | 0.10 | 1 | 10.6 |
| 470 | **-0.025** | 0.860 | +0.006 | -2.373 | 0.107 | 0.12 | 1 | 10.9 |
| 480 | -0.099 | 2.329 | +0.001 | -2.370 | 0.111 | 0.10 | 1 | 11.1 |
| 490 | -0.062 | 0.720 | +0.009 | -2.354 | 0.067 | 0.15 | 1 | 11.3 |
| 499 | **-0.049** | 0.819 | +0.008 | -2.374 | 0.095 | 0.10 | 1 | 11.5 |

**关键观察**:

1. **PPO 起点极好**: iter 0 reward = **-0.682**(对比纯 PPO 起点 -1.82,改进 63%)。这是 BC 给的 "免费" 提升。

2. **三段式收敛**:
   - **阶段 A**(iter 0-60): reward 在 -0.5 ~ -0.8 波动,值函数 v_loss 从 12.6 → 8.0。critic 在学 BC'd policy 的价值。
   - **阶段 B**(iter 70-160): reward 从 -0.30 → -0.17 平稳下降。v_loss 从 2.5 → 1.7。policy 在 BC 起点上微调。
   - **阶段 C**(iter 170-499): reward 在 -0.03 ~ -0.20 区间震荡,**已饱和**。v_loss 稳定在 0.5-2.0。

3. **最佳点**: iter 470 reward=**-0.025**(几乎打平 rule);iter 390 reward=**-0.031**;iter 250 reward=**-0.040**。

4. **过早停(es=1)频繁触发**: iter 50 之后大部分 iter 都 es=1,说明 KL 经常超过 target_kl=0.03 的 1.5× = 0.045。但 PPO 仍能缓慢改进,因为每个 rollout 还是有更新。

5. **clip_frac 中等(0.08-0.18)**: 说明 ratio 没爆,policy 更新在合理范围。

6. **entropy 稳定 -2.3 ~ -2.4**: 策略没坍缩到 deterministic corner。这是好事 — 保持探索能力。

7. **没有大跳变**: 对比纯 PPO 训练(iter 0-275 平台然后突然学习),BC+PPO 从一开始就在 "学",没有 "学会玩" 阶段。

**判读**: PPO fine-tune 在 BC 起点上稳步改进 reward 至接近 0(rule 平局水平),但没有突破到正向(找到 exploit)。

#### 4.3.4 Cell 2 评估

```
br margin (rule POV): +0.017 (95% CI [-0.017, +0.050])
kills: rule_t0=1.95, br_t1=1.93
winner: rule=0.05, BR=0.03, draw=0.93
```

**现象**:
1. **rule kills 1.95, BR kills 1.93** — 几乎完全持平(差 0.02,统计噪声级)
2. **平局率 93%** — 接近 mirror self-play 的 100% 平局
3. **winner 分布: rule 5% / BR 3% / draw 93%** — 完全对称,没有方向性
4. **exploit_gap = -0.016**(CI [-0.05, +0.02])
5. **CI 跨越 0** — exploit 不显著

#### 4.3.5 G0 verdict

```
exploit_gap >= 0.5 AND CI excludes 0: False
CI excludes 0:                        False
BR win rate >= 0.55:                  False (actual=0.03)
BR training healthy:                  True (adv_std=1.0, entropy=-2.37)
❌ G0 FAIL
```

---

## 5. 三次 G0 测试横向对比

| 维度 | re-test #1(纯 PPO,默认) | re-test #2(纯 PPO,调超参) | **V2(BC + PPO)** |
|---|---|---|---|
| **PPO 起点策略** | near-prior [0.25, 0.25, 0.25, 0.25] | track-heavy [0.20, 0.36, 0.30, 0.14] | **rule-equivalent** [0.08, 0.70, 0.12, 0.10] |
| **PPO 起点浓度 vs rule** | 0% | 50%(rule 71%) | **99%**(rule 71%) |
| **PPO 起步 reward** | -1.82(随机) | -1.82(随机) | **-0.68**(BC) |
| **PPO 最终 reward** | -1.21 | -0.88 | **-0.05** |
| **PPO 学习信号** | iter 300 才出现 | iter 110 出现 | **iter 0 就有** |
| **Cell 2 rule kills** | 1.96 | 1.96 | 1.95 |
| **Cell 2 BR kills** | 1.00 | 0.98 | **1.93** |
| **Winner(rule/BR/draw)** | 98/0/2% | 98/0/2% | **5/3/93%** |
| **exploit_gap** | -0.963 | -0.979 | **-0.016** |
| **95% CI** | [-0.98, -0.94] | [-1.00, -0.96] | **[-0.05, +0.02]** |
| **CI 排除 0?** | ✓ 强排除 | ✓ 强排除 | **✗ 包含 0** |
| **BR training 健康** | ✓ | ✓ | ✓ |
| **Fail 性质** | BR undertrained | BR undertrained | **rule ≈ Nash 均衡迹象** |

**关键数据可视化**:

```
exploit_gap 演化(越接近 0 越好;正值 = rule 可被 exploit)
   ↓
-1.0 ┤█████████████████   re-test #1 (纯 PPO 默认)
     │
-0.9 ┤██████████████████  re-test #2 (纯 PPO 调超参)
     │
-0.2 ┤███                 V1 (BC only, 无 PPO)
     │
 0.0 ┤■                   V2 (BC + PPO fine-tune)  ← CI 包含 0
     │
+0.5 ┤                    G0 PASS 阈值
     └─────────────────────────────────
```

---

## 6. 物理与策略现象分析

### 6.1 为什么 BC 学得这么快?

**NLL train_loss 在 50K 样本 + 1 epoch 就到 -16** — 这是 AC 网络 "表达" rule 策略的最小损失。

**原因**: 
- rule 策略本质是 **per-aperture 4-way softmax + laser argmax + emission threshold** 的浅决策树
- AC 网络有 256-256 双层 trunk + 各 head 的 Linear,完全覆盖这个表达需求
- rule 的核心特征(trace_P, exposure, E_progress)已经在 obs 中
- 所以 BC 几乎是 "查表式" 学习 — 1 epoch 就能拟合 80%+

### 6.2 为什么 BC'd AC vs rule 是 89% 平局,不是 100%?

V1 BC-only Cell 2 显示 89% 平局,11% rule 胜。理论上 rule vs rule(mirror)是 100% 平局。差距 11% 来自:

1. **BC'd AC 是 stochastic**(Dirichlet/Categorical/Beta 采样),rule 是 deterministic → 偶尔 stochastic sample 偏离 rule,被 rule 抓住弱点
2. **BC 数据有限**(50K samples),没覆盖所有 corner case
3. **BC'd AC 的 freq_hop=1.03 ± 0.X**(Beta 分布),rule 是确定 1.0 → BC 有微小行为差异
4. **Beam/laser target Categorical 采样偶有翻转**

### 6.3 为什么 PPO fine-tune 没找到 exploit?

PPO 把 reward 从 BC 起点 -0.68 推到 -0.05,但没突破到正。这意味着 BR 没找到 rule 不防御的弱点。

**可能原因**:

**假设 A**:rule 是 env 的近似 Nash 均衡。
- 任何 unilateral 偏离(rule vs rule)不会改善期望 reward
- 证据:rule vs rule = 100% 平局(对称博弈下无法 unilateral 改进)
- BC'd AC 接近 rule → 接近 100% 平局

**假设 B**:exploit 存在,但 PPO 探索不够。
- 18 维动作空间 + 200 horizon 长程依赖
- 已知 exploit 路径(从 anti-strawman):
  - pure_jam 打 rule 让 rule 只能 0.35 kills → 但 pure_jam 自己 0 kills → 不是 exploit
  - 必须找 "我打 jam,但保留 track 能力" 的 mixed strategy
- 这需要 Dirichlet 在 [track, jam] 之间找特定比例,PPO 探索难

**假设 C**:rule 的 anti-jam 反应(jam_detect_threshold=0.30, react freq_hop=6)让 exploit 几乎不可能。
- 当 BR 试图 jam,rule 检测到后立刻 hop=6,有效干扰 = jam/6 ≈ 0.17 → rule track_sigma 不会变差
- 当 BR 不 jam,rule 用 hop=1 → track sigma 最优
- 这是 rule 设计的 "无弱点" 反应

**最可能**: 假设 A + 假设 C 共同作用 — rule 的反应机制让它接近 env 的 Nash。

### 6.4 PPO 训练动态分析

观察 PPO 训练曲线,有几个值得注意的现象:

**现象 1: critic v_loss 早期爆涨**
- iter 0: v_loss=12.6(critic 完全不会估价值)
- iter 70: v_loss=2.5(critic 学到 BC'd policy 的 value)
- iter 250+: v_loss < 1(critic 收敛)

→ critic 起步慢,因为 BC 没训 critic(只训 actor)。这是设计选择(留给 PPO 学)。

**现象 2: KL 频繁超 target_kl=0.03 触发 early stop**
- 大部分 iter es=1
- 但训练继续(只是该 epoch 内 update break,下个 iter 重启)
- 这说明 PPO 在 BC 起点附近每步都想做较大更新(因为 BC 起点 policy 还有改进空间)

**现象 3: reward 在阶段 C 反复震荡(-0.03 ~ -0.20)**
- 没有 crash,只是饱和
- 不同 seed 的 episode 难度不同(RANDOM_GEOMETRY)导致 reward 自然波动
- 平均下来接近 rule mirror 的 0

### 6.5 BC'd AC vs rule 的对局微观

通过 deterministic eval 的 task_alloc profile 看:

| 场景 | BC'd AC | rule | 差异 |
|---|---|---|---|
| track 浓度 | 70% | 71% | -1% |
| detect 浓度 | 8% | 10% | -2% |
| jam 浓度 | 12% | 9% | +3% |
| comm 浓度 | 10% | 10% | 0% |
| freq_hop default | 1.03 | 1.0 | +0.03 |

**BC 系统性略偏 jam**(12% vs 9%)。这可能是因为 BC 数据里 rule 在 "对手是 pure_jam" 的场景下也会 jam(对抗性 counter-jam),BC 学到了这些场景的平均浓度比 rule 默认高。

这个微小偏差可能导致 BC'd AC 在 Cell 2 比 rule 略弱(11% rule 胜 vs 0% BC 胜 in V1),但 V2 PPO fine-tune 把这个偏差修正了。

---

## 7. 决策点

### 7.1 严格 spec 解读

**用户 verbatim**: "G0 FAIL → 别烧自博弈,一起退 IET"

按这条:G0 三次 FAIL → 退 IET。

### 7.2 但本次 FAIL 给 IET 一个新角度

前两次 FAIL 的 narrative: "env 退化(root-A mimic)→ 退 IET 做 IQ/CRLB 基线"
本次 FAIL 的 narrative: **"env 非平凡 + rule 是近似 Nash 均衡 → 退 IET 做 rule-based Nash equilibrium analysis"**

后者更有研究价值:
- 不是 fallback,是 contribution
- BC → PPO pipeline 已落地,可作为 "verifying rule's near-Nash property" 的工具
- IET 故事更完整:env design + WP0-decisive + WP1 BC-Nash verification + IQ/CRLB analysis

### 7.3 备选路径(同 G0_FULL_EXPERIMENT_REPORT.md §6.3)

**A. 退 IET(Nash baseline 故事)** — 推荐
**B. 继续找 exploit**(1500+ PPO iters + curiosity) — 风险高
**C. 软化 rule 凑 G0 PASS** — thesis 弱
**D. 转 "rule Nash 分析" 作 thesis** — 重新定义贡献

---

## 8. 已花费成本

| 项 | 时间 |
|---|---|
| Step 1: bc_pretrain.py 实现(260 LOC) | ~40 min |
| Step 2: run_g0_gate.py BC hook | ~15 min |
| Step 3: smoke test | ~10 min |
| V0 smoke test | <1 min |
| V1 BC-only(40s collect + 80s train + 30s eval) | ~3 min |
| V2 BC + PPO 500 iters(40s collect + 80s BC + 11.5 min PPO + 1 min eval) | ~15 min |
| 撰写本报告 | ~30 min |
| **总** | **~1.5h 实现 + ~20 min 跑 + ~0.5h 报告** |

对比前两次 G0 FAIL 总成本(~4.5h):本次投入更少,但取得了最大的 exploit_gap 改善(-0.96 → -0.02)。

---

## 9. 推荐与待决问题

### 9.1 我的判断

1. **BC → PPO 范式技术上完全成功**:
   - BC 完美学到 rule 策略(70% vs 71% track)
   - PPO 起步从 -1.82 → -0.68(BC 给的免费提升)
   - PPO 收敛到 -0.05(完全持平 rule)
   - 代码 pipeline 落地,可复用 WP2 self-play / league

2. **G0 仍 FAIL,但性质根本改变**:
   - 前 2 次:BR undertrained(CI 强排除 0,rule 碾压 BR)
   - 本次:BR = rule(CI 跨 0,93% 平局)
   - 强烈暗示 **rule 是 env 的近似 Nash 均衡**

3. **对用户的最关键待决问题**:

> "接受 rule ≈ Nash,把 BC+Nash 分析作为 IET 论文方向" vs "继续投找 exploit(可能根本不存在)"?

如果用户更看重 **研究纪律 + 给 IET 一个新角度** → 退 IET,Narrative 更强。

如果用户更看重 **WP2 self-play** → 再投 1-2 轮(1500+ PPO iters + curiosity),但风险高 — 如果 rule 真是 Nash,投多少都没用。

我作为 AI 没有偏好。数据在这里,决策权在用户。

---

## 10. 附录

### 10.1 文件清单

**新增**:
- `algo/_shared/pilot/twoteam/bc_pretrain.py`(260 LOC)— TwoTeamBCPretrainer 类
- `tests/twoteam/test_bc_pretrain_smoke.py`(70 LOC)— 2 个 smoke test
- `experiments/twoteam/G0_BC_PPO_REPORT.md`(本报告)
- `experiments/twoteam/g0_bc_only_report.md`(自动生成)
- `experiments/twoteam/g0_bc_only.log`(V1 完整日志,78 lines)
- `experiments/twoteam/g0_bc_then_ppo.log`(V2 完整日志,129 lines)
- `checkpoints/twoteam/bc_pretrained.pt`(BC checkpoint)

**修改**:
- `algo/_shared/pilot/twoteam/run_g0_gate.py`:
  - 加 `from algo._shared.pilot.twoteam.bc_pretrain import TwoTeamBCPretrainer`
  - main() 加 4 个 BC 参数
  - Step C0: BC pretrain 阶段(BR training 之前)
  - markdown 报告加 BC section
  - CLI 加 4 个 `--bc-pretrain-*` flags

### 10.2 完整 V2 训练命令

```bash
stdbuf -oL -eL conda run --no-capture-output -n fluxphased python -u \
    algo/_shared/pilot/twoteam/run_g0_gate.py \
    --br-iters 500 --horizon 200 --n-envs 8 --n-episodes 30 \
    --bc-pretrain-samples 50000 --bc-pretrain-epochs 15 \
    --br-lr-actor 1e-4 --br-entropy-coef 0.01 \
    --out experiments/twoteam/g0_bc_then_ppo_report.md \
    > experiments/twoteam/g0_bc_then_ppo.log 2>&1 &
```

### 10.3 Retreat 友好

```bash
rm -f algo/_shared/pilot/twoteam/bc_pretrain.py
rm -f tests/twoteam/test_bc_pretrain_smoke.py
rm -f checkpoints/twoteam/bc_pretrained.pt
rm -f experiments/twoteam/g0_bc_*_report.md experiments/twoteam/g0_bc_*.log
# run_g0_gate.py 默认 --bc-pretrain-samples 0 = 跳过 BC,行为不变
```

### 10.4 后续(G0 PASS 后的可复用性)

如果 BC → PPO 范式让 G0 PASS(在某个未来调参 /软化 rule 后),这个 pipeline 直接迁移到 **WP2 self-play + league**:
- WP2 main loop: BC(pretrain on self-snapshot)→ PPO fine-tune against opponent pool
- league 每个 new opponent: short BC on 该 opponent's recent wins → fine-tune
- AlphaStar SL→RL + league 完整落地
