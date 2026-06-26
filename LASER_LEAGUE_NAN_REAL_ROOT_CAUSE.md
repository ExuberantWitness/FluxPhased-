# FluxLeague iter-1 NaN — 真正根因(纠正 LASER_LEAGUE_NAN_ROOT_CAUSE.md)

**结论先行**:NaN **不是 Adam 数值退化**(那份诊断是误诊,且它提的 eps 修复**根本没进代码**——
`ppo_trainer.py:49` 仍是默认 `eps=1e-8`)。真正根因是 **`_apply_residual_aim` 把存进 buffer 的
commander 动作覆盖成"绝对 aim 并 clamp 到 ±1"**,与训练时的 log_prob 失配、且让 PPO 反算 `atanh(±1)=±∞`。

---

## 1. 因果链(逐行)

1. **采样**(`actor_critic.py` forward):`aim_raw = aim_dist.rsample()` → `aim = tanh(aim_raw)` ∈ (-1,1),
   连同 `log_prob`(对 `aim_raw`/残差算的)一起返回。**此时 action 是策略的残差,严格在 (-1,1) 内。**
2. **覆盖**(`ppo_trainer.py:366-368` `_apply_residual_aim`):
   ```python
   aim_x = (anchor_x + dx_norm).clamp(-1.0, 1.0)   # ← 绝对 aim,clamp 到正好 ±1
   cmd_action[..., 1] = aim_x                       # ← 覆盖了要存进 buffer 的动作
   ```
   transition 存的是**覆盖后的绝对 aim**,但 `cmd_logp` 是**覆盖前对残差**算的 → **二者失配**。
3. **PPO update**(`actor_critic.py:264` evaluate_actions):对存的(被覆盖的)动作做 inverse-tanh
   ```python
   aim_raw = 0.5*torch.log((aim+1.0)/(1.0-aim+1e-6).clamp(min=1e-6))
   ```
   - `aim = -1.0`(clamp 命中):分子 `(aim+1)=0` → `log(0) = -∞` → `aim_raw = -∞`
     (注意分母 clamp 了、分子**没**clamp,所以 -1 这侧无保护);
   - `aim_dist.log_prob(-∞)` → `-∞` → `policy_loss` = NaN/Inf → 梯度 NaN → `action_head` 权重 NaN
     → 下一次 forward `aim_mean = NaN` → `Normal(aim_mean, ...)` 抛 ValueError。**这就是崩溃点。**
4. **放大器**:`log_std_floor=-6`(σ≈2.5e-3)使残差高斯极窄,log_prob 对均值超敏感 → 失配的
   ratio `exp(logp_new − logp_old)` 爆到 20+(日志里的 policy_loss spike,被误读成 Adam 病征)。

## 2. 对照证据:train_laser 为什么不崩(它是对的)

`train_laser.py:1136-1139` 把绝对 aim 算成**单独的 `env_a` 张量**,**只发给 env**:
```python
env_a[..., 1] = anchor[..., 0] + raw[..., 1] * (residual_scale_m / half_x)   # 不 clamp 到 ±1
```
- **buffer 里存的是 `raw`(策略残差)**,PPO 对残差算 log_prob → 一致;
- 绝对 aim 只进 env,**从不被 atanh** → 无奇点。

**FluxLeague 的 `_apply_residual_aim` 把这条改坏了**:它 in-place 覆盖了 `cmd_action`(= 要存进 buffer 的动作),
把 env-动作和 buffer-动作**混为一谈**。这是移植引入的 bug,不是 Adam,不是 checkpoint。

## 3. 修复(按重要性,镜像 train_laser)

| # | 改动 | 位置 | 作用 |
|---|---|---|---|
| **F1(根因)** | **不要覆盖存进 buffer 的动作**。`_apply_residual_aim` 改为返回一个**单独的 env-动作张量**(anchor+residual),buffer/transition 仍存策略原始残差 `cmd_action`。env step 用 env-动作,PPO update 用残差。 | `ppo_trainer.py:351-370` + `get_own_actions` 里调用处 | 同时修 (A) log_prob 失配 和 (B) atanh 奇点(残差恒在 (-1,1)) |
| **F2(防御)** | atanh 前把 action clamp 到 `(-1+1e-6, 1-1e-6)`,分子分母都 clamp | `actor_critic.py:264` 和 `:275` | 数值兜底,杜绝 `log(0)` |
| **F3(去敏)** | `log_std_floor: -6 → -3~-4`(σ 2.5e-3→0.02~0.05) | league 配置 `training.log_std_floor` | 降低窄-σ 的 ratio 爆破 |
| **F4(标准护栏)** | ratio 前 clamp log-ratio:`ratio = exp((logp_new−logp_old).clamp(-20,20))` | `ppo_trainer.py:112` | 标准 PPO 数值安全 |
| ~~Adam eps 1e-8→1e-4~~ | **撤掉**——治错了病,且本就没进代码 | — | Adam 用 sqrt(exp_avg_sq) 归一化、同步累加 grad²,grad/sqrt(grad²)≈1 有界,不会"乘 3e4" |

**最小可行**:只做 **F1** 即可根除 NaN(env/buffer 动作解耦);F2/F4 作兜底,F3 提稳健性。

## 4. 验证
- 复跑 `diagnose_nan.py` 的崩溃配置:F1 后 evaluate_actions 不再出现 `aim_raw=±inf`/`log_prob=-inf`;
- 跑 iter 0→2 不崩;monitor `policy_loss` 不再单点跳 10×、entropy 不再伪崩;
- 与 train_laser 对齐:同一策略在两条路径下 log_prob/ratio 量级一致。
