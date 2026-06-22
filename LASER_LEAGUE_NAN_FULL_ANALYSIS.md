# FluxLeague iter-1 NaN — 完整代码分析报告

**日期**: 2026-06-22
**分析基线**: origin/main @ 99527dc + evo/laser-fix
**崩溃**: `actor_critic.py` `Normal(aim_mean, aim_std)` — `aim_mean` 全 NaN
**关联**: 取代 `LASER_LEAGUE_NAN_ROOT_CAUSE.md`(Adam-eps 误诊)与 `LASER_LEAGUE_NAN_REAL_ROOT_CAUSE.md`(简版)

---

## 0. 执行摘要

NaN 的真正根因是 **`_apply_residual_aim` 把"要存进 PPO buffer 的 commander 动作"in-place 覆盖成了
"绝对 aim 并 clamp 到正好 ±1"**。这制造了两个独立缺陷:

- **缺陷 A(统计失配)**:存进 buffer 的动作是"绝对 aim",但与它配对的 `old_log_prob` 是策略对
  "残差"算的 → PPO 重要性比 `exp(logp_new − logp_old)` 比较的是两个不同随机量 → ratio 爆破(日志里被
  误读成 Adam 病征的 policy_loss spike)。
- **缺陷 B(数值奇点)**:覆盖后的 aim 可能正好 = −1.0;PPO update 反算 `atanh(−1.0) = log(0)/… = −∞`
  → `log_prob = −∞` → 梯度 NaN → `action_head` 权重 NaN → 下一次 forward `aim_mean = NaN` → 崩溃。

**触发条件**:只要**融合/跟踪 anchor 饱和到 ±1**(`sensing.py` 把估计 clamp 到 ±half 再 /half),`(anchor+residual)`
就 clamp 到 ±1 → 喂进 atanh → NaN。**这与 win_rate=0.50 bug 同源**(同一个 sensing clamp-to-±1)。
`log_std_floor=−6`(σ≈2.5e-3)是放大器,不是根因。

**先前 Adam-eps 诊断是错的**,且那个修复**从未进代码**(`ppo_trainer.py:49` 仍默认 `eps=1e-8`),所以"效果不好"。

**最小可行修复 = F1**:让 env-动作与 buffer-动作解耦(镜像 train_laser),NaN 立除。

---

## 1. 崩溃定位

| 项 | 值 |
|---|---|
| 异常 | `ValueError: Expected parameter loc … to satisfy Real(), found invalid values: tensor([[nan,…` |
| 抛出点 | `actor_critic.py` `aim_dist = torch.distributions.Normal(aim_mean, aim_std)` |
| 直接原因 | `aim_mean = self.action_head(features)` 输出 NaN ⇐ `action_head` 权重已是 NaN |
| 权重变 NaN 的步 | 前一次 PPO update 的 `loss.backward()` 产生 NaN 梯度 |
| NaN 梯度来源 | `evaluate_actions` 里 `log_prob = −∞`(atanh 奇点)→ `policy_loss = NaN/Inf` |

---

## 2. 完整数据流追踪(逐跳 + 行号)

```
[采样] actor_critic.py forward (hybrid_fire 分支)
   aim_raw = aim_dist.rsample()            # ~L213  残差,Normal(aim_mean, σ) 采样
   aim     = tanh(aim_raw)                 # ~L213  ∈ (-1,1) 严格开区间
   action  = [fire, aim]; log_prob(对 aim_raw/残差)  → 返回
        │
        ▼
[覆盖] ppo_trainer.py _apply_residual_aim   # L351-370   ★缺陷源★
   dx_norm = cmd_action[1] * residual_scale_m / half_x
   aim_x   = (anchor_x + dx_norm).clamp(-1.0, 1.0)     # L366  ← clamp 到正好 ±1
   cmd_action[1] = aim_x                                # L368  ← in-place 覆盖 buffer 动作
        │   (anchor_x = cmd_obs[68] = 融合 anchor,sensing 已 clamp 到 ±1 —— §4)
        ▼
[存储] get_own_actions transition           # L532-543
   "cmd_action": cmd_action(=覆盖后的绝对 aim)   ← 与
   "cmd_logp"  : cmd_logp   (=覆盖前对残差算的)  ← 失配(缺陷 A)
        │
        ▼
[更新] ppo_trainer.py PPO loop              # L108-115
   log_prob,… = self.ac.evaluate_actions(obs, old_actions=cmd_action, …)   # L108
        │
        ▼
[反算] actor_critic.py evaluate_actions     # L264   ★NaN 奇点★
   aim_raw = 0.5*log((aim+1.0)/(1.0-aim+1e-6).clamp(min=1e-6))
        aim=-1.0 → (aim+1)=0 → log(0)=-∞ → aim_raw=-∞       (缺陷 B)
   log_prob += aim_dist.log_prob(-∞) = -∞                    # L267
        │
        ▼
   ratio = exp(log_prob - old_log_probs)    # L112  无 clamp → 0 或 +∞
   policy_loss = -min(surr1,surr2).mean()   # L115  → NaN/Inf
   loss.backward() → action_head.grad = NaN → 权重 NaN → 崩溃(§1)
```

---

## 3. 两个独立缺陷

### 3.1 缺陷 A — 动作/log_prob 失配(统计破坏)
- `forward` 对**残差**算并返回 `cmd_logp`;
- `_apply_residual_aim` 之后把 `cmd_action` 覆盖成 **anchor+残差(绝对 aim)**;
- transition 把"绝对 aim"与"残差的 logp"配对存入。
- PPO update `evaluate_actions(obs, old_actions=绝对aim)` 重算 log_prob 时,是用**残差的高斯**
  在 `atanh(绝对aim)` 处取值——与 `old_logp`(残差在采样点的 logp)**毫无对应关系**。
- 后果:`exp(logp_new − logp_old)` 量级失控,即日志里 `policy_loss` 从 ~0.3 单点跳到 2.05(ratio≈20)。
  **这不是探索到了强对手 p0014,是数学上比错了量。**

### 3.2 缺陷 B — atanh 奇点(数值崩溃)
`actor_critic.py:264`(及非 hybrid 的 :275)的 inverse-tanh **clamp 不对称**:
```python
aim_raw = 0.5 * torch.log( (aim + 1.0) / (1.0 - aim + 1e-6).clamp(min=1e-6) )
#                            └ 分子未 clamp ┘   └────── 分母已 clamp ──────┘
```
| 输入 | 分子 | 分母 | aim_raw | 结果 |
|---|---|---|---|---|
| `aim = +1.0` | 2.0 | clamp(1e-6)=1e-6 | 0.5·ln(2e6)=**7.25** | 有界(被分母 clamp 救了) |
| `aim = −1.0` | **0.0** | 2.0 | 0.5·ln(0)=**−∞** | **NaN 源**(分子无保护) |
- 所以 **+1 这侧侥幸安全、−1 这侧必爆**。`Normal.log_prob(-∞)` = `-0.5·((-∞-μ)/σ)² - …` = `-∞`。

---

## 4. 触发条件 — anchor 何时正好 ±1(与 0.50 bug 同源)

`anchor_x = cmd_obs[68]`,由 `fused_sensing` 写入。该值在 **`sensing.py`** 中被**先 clamp 到 ±half、再 /half**:
```python
# fused 路径
zx = zx.clamp(-half_x, half_x);  obs[..., off] = zx / half_x        # L280, L282
# tracked 路径
x0 = x0.clamp(-half_x, half_x);  obs[..., off] = x0 / half_x        # L315, L319
```
→ **任何时刻估计饱和到地图边界,anchor 就 = ±1.0 精确值**。饱和发生于:
1. 退化几何(GDOP,det(FIM)→0,融合估计爆掉被 clamp)—— 即 0.50 bug 的同一机制;
2. **目标本就靠近地图边界**(合法的正常对局也会发生);
3. NaN 守卫 `nan_to_num(…, posinf=1.0, neginf=-1.0)`(ppo_trainer:345)把 inf 映成 ±1。

随后 `aim = (±1 + residual).clamp(-1,1)`,只要 residual 同号或量级小,就**仍 = ±1** → §3.2 奇点。

> **重要**:reset-timing 修复(fa98871)减少了退化几何,但**没消除边界目标与残余饱和**,所以 NaN
> 仍会在某局触发(日志正是 iter1 某 episode)。**这是潜伏 bug,靠近边界即引爆,与 sensing 修复无关。**

---

## 5. 放大器 — log_std_floor=−6

- `train_laser.py:1531` 默认 `log_std_floor=−6.0`,退火 `target=max(floor, init − iter·decay)`(:1533),
  联赛跑得久 → σ 收到 `exp(−6)≈2.5e-3`。
- 极窄 σ 使残差高斯的 `log_prob` 对均值**超敏感**:`logp = −0.5·((x−μ)/σ)² − log(σ√2π)`,σ=2.5e-3 时
  `(x−μ)/σ` 极大 → 缺陷 A 的失配被放大成 ratio≈20 的爆破,缺陷 B 的 `-∞` 也来得更快。
- 这正是 `LASER_RESULTS_SUMMARY` 记过的 p12 "log_std 退太低 → 均值漂"现象,p12b 用"抬高 floor"缓解;
  FluxLeague 路径未继承该缓解。

---

## 6. 误诊纠正 — 为什么不是 Adam

先前文档称根因是 "Adam exp_avg_sq=0 → 步长 = lr·grad/eps = 3e4·grad 爆破"。**数学上不成立**:
- Adam 更新 = `lr · m̂ / (sqrt(v̂) + eps)`,其中 `v̂` 在**同一步**累入 `grad²`;
- 某维首次出现非零 grad 时:`m̂ ≈ (1−β1)g`,`v̂ ≈ (1−β2)g²` → 更新 ≈ `lr·(1−β1)/sqrt(1−β2)·sign(g)`
  = `lr · 0.1/0.0316 ≈ 3.16·lr` —— **有界**,不是"乘 3e4"。
- eps 仅在 `m̂≠0 且 v̂→0` 时才主导,而 `m̂`(β1=0.9)比 `v̂`(β2=0.999)衰减快得多,自然几乎不可达。
- 即便如此,该 eps 修复**根本没进代码**(`ppo_trainer.py:49` 仍 `Adam(…, lr=lr)`,默认 eps=1e-8)。

**所以"效果不好"= 改错了文件想象出来的病因,真凶(覆盖 + atanh)分毫未动。**

---

## 7. 参照对比 — train_laser 为什么不崩(正确范式)

`train_laser.py:1136-1139` 把绝对 aim 算成**单独的 `env_a` 张量**:
```python
env_a[..., 1] = anchor[..., 0] + raw[..., 1] * (residual_scale_m / half_x)   # 不 clamp 到 ±1
env_a[..., 2] = anchor[..., 1] + raw[..., 2] * (residual_scale_m / half_y)
```
- **buffer 存的是 `raw`(策略残差)** → PPO 对残差算 log_prob → 一致(无缺陷 A);
- **绝对 aim 只进 env、从不被 atanh** → 无缺陷 B。

FluxLeague 的 `_apply_residual_aim` 把"env-动作"与"buffer-动作"**合二为一并 in-place 覆盖**,
是移植时引入的回归。**修复方向 = 让两者重新分离。**

---

## 8. 次生问题(顺带修,Q1 鲁棒性)

| # | 问题 | 位置 |
|---|---|---|
| S1 | PPO update 无 NaN/Inf 守卫,坏一个 minibatch 即污染权重 | `ppo_trainer.py:108-129` |
| S2 | `ratio = exp(logp_new − logp_old)` 无 log-ratio clamp | `ppo_trainer.py:112` |
| S3 | 两处 atanh 的 clamp 都不对称(分子未 clamp) | `actor_critic.py:264, 275` |
| S4 | NaN 守卫 `nan_to_num(posinf=1.0)` 把 inf→±1,**反而**喂给 atanh 制造新奇点 | `ppo_trainer.py:345` |

---

## 9. 修复方案(按重要性)

| # | 改动 | 位置 | 修什么 |
|---|---|---|---|
| **F1(根因)** | `_apply_residual_aim` 改为**返回单独的 env-动作张量**(anchor+residual,可不 clamp 或 clamp 到 ±(1−1e-4));`get_own_actions` 用它做 env step,但 transition 仍存**未覆盖的 `cmd_action`(残差)** | `ppo_trainer.py:351-370` + 调用处 | 缺陷 A + B 一并根除(残差恒在 (-1,1)) |
| **F2(防御)** | 两处 atanh 前 `aim = aim.clamp(-1+1e-6, 1-1e-6)`,分子分母都 clamp | `actor_critic.py:264, 275` | 数值兜底(含 S3) |
| **F3(去敏)** | `log_std_floor: -6 → -3~-4` | league 配置 `training.log_std_floor` | 降窄-σ 放大(§5) |
| **F4(护栏)** | `ratio = exp((logp_new−logp_old).clamp(-20,20))` + update 前 `if not torch.isfinite(loss): continue` | `ppo_trainer.py:112, 108-129` | S1 + S2 |
| — | ~~Adam eps 1e-8→1e-4~~ **撤掉** | — | 治错病 |

**最小可行 = F1 单独即可根除 NaN。** F2/F4 兜底,F3 提稳健,建议一起上以过 Q1 鲁棒性审查。

---

## 10. 验证协议
1. **定向单测**:构造 `aim=-1.0` 的 batch 喂 `evaluate_actions`,F2 前应得 `-inf`,F2/F1 后有限;
2. **复现**:用 `diagnose_nan.py` 的崩溃 checkpoint+对手,F1 后 iter1 ep8 update 不再 NaN;
3. **量级一致性**:同策略在 train_laser 与 FluxLeague 两路径下 `policy_loss/ratio` 量级一致(无 10× spike);
4. **回归**:跑 iter 0→3 不崩;`policy_loss` 无单点跳、entropy 无伪崩;
5. **不掩盖**:验证时**临时关掉** S4 的 `nan_to_num` 兜底,确认 F1 后无 inf 进入 atanh(否则只是被兜底掩盖)。

---

## 11. 严重性与影响范围
- **严重性**:致命(训练崩溃),且**潜伏**——不只在退化几何下,**任何目标接近地图边界的正常对局都会引爆**;
- **影响**:所有走 `_apply_residual_aim` + `mode=tracked/fused` 的 FluxLeague 训练(即 Q1 主实验全 cell);
- **不影响** `train_laser.py` 路径(env/buffer 动作本就分离);
- **与 0.50 bug 同根**:都源自 sensing 的 clamp-to-±1。F1 修 NaN;若要彻底,sensing 的边界饱和也应改为
  软 clamp 或在 anchor 进 residual 前做 `±(1−ε)` 收缩。
