# Anchor 诊断 + 修复指令 — min_dist_init=139m 根因调查

**优先级**: P0（最高，所有 alpha/F1+F2 实验在此修好前都 inconclusive）
**前置**: P2_A_BASELINE_REPORT.md（A baseline FAIL 暴露的真问题）
**目标**: 让 league 路径的 min_dist_init 从 139m 回到 ≤1m（train_laser 路径能做到 0.2m）

---

## 0. 问题陈述

**症状**: A baseline 全 80 个 [dart] 打印显示 `min_dist_init` 平均 139m（25-200m 范围）。

**应有行为**: `residual_aim=True` 配合 Kalman 跟踪，初始 aim 应锚定 Kalman 估计的敌人位置；warm-start 120 步后 Kalman 精度应 ≤0.2m。min_dist_init 应 ~1m。

**实际行为**: aim 偏离敌人 139m → kill_rate=1.00 全靠 kr=24.5m 宽松阈值白送 reward，**不是学习信号**。

**对照基准**: train_laser.py 路径相同 σ=5cm / cf=7.4e-5 / track_burnin 配置下能稳定到 0.2m（V5_DARTBOARD_REPORT.md 历史 iter_min_dist 数据）。

---

## 1. 已确认的代码路径（已审计，无 bug）

整个 anchor → aim → min_dist 链路：

```
[LaserEpisodeRunner.reset()]
  ↓ kalman.reset()  → tracker._initialized = False
  ↓ enforce_radar_baseline(env, 5000.0)  → 推开 own 2 部雷达到 5km baseline

[step_control → env.step × 5 pulses (first_step uses _cached_cmd=zeros)]
  ↓ process_commander_actions(zeros) → _commander_aim = 0  ← 第一次步骤 aim 是 0
  ↓ env.step 返回 result + CPI

[step_control → get_own_actions(team=0)]
  ↓ _get_observations(env, spectrum, events)
    ↓ env.battlefield.get_commander_observation(radar_pos, comm_input)
      → drone.get_commander_obs() returns obs[E, T, 76]
         obs[..., 68:70] = enemy_idx[0] pos / half_map  ← 这是**真值** enemy pos
         obs[..., 70:72] = enemy_idx[1] pos / half_map
  ↓ _apply_laser_sensing(commander_obs, env)
    ↓ mode=tracked + σ=0.05, cf=7.4e-5 → 走 fused_sensing(tracked) 分支
    ↓ tracker not initialized → warm-start 120 次 _fuse_one(true_enemy) + _kalman_step
    ↓ obs[68:70] 被覆写为 Kalman 估计（normalized [-1,1]）
    ↓ tracker._initialized = True
  ↓ cmd_obs = commander_obs[:, team, :]  ← 已 Kalman 处理的 per-team slice
  ↓ commander_trainer.ac.get_action(cmd_obs) → cmd_action（raw policy 输出）
  ↓ _apply_residual_aim(cmd_action, cmd_obs, env)
    ↓ anchor_x = cmd_obs[..., 68]  ← **应是** Kalman 估计的 enemy x
    ↓ dx_norm = cmd_action[..., 1] × residual_scale_m(=6) / half_x(=10000)
    ↓ aim_x = (anchor_x + dx_norm).clamp(...)
    ↓ env_action[1:3] = aim_x, aim_y
    ↓ env_action[3] = 0  (aim_z=0)
  ↓ return commander_action=env_action（已 anchor + 残差）

[step_control Phase 4-5: _cached_cmd = new_cmd]

[下一个 step_control → env.step(_cached_cmd) × 5 pulses]
  ↓ process_commander_actions(_cached_cmd)
    ↓ _commander_aim[0] = cmd[1] × half_x  ← Kalman enemy + tiny residual
    ↓ _commander_aim[1] = cmd[2] × half_y
    ↓ _commander_aim[2] = cmd[3] × 1000 = 0  (forced)

[reward shaper (reward.py:237)]
  ↓ aim = drone._commander_aim[:, t, :].unsqueeze(1)
  ↓ enemy_pos = radar_pos[:, enemy_idx, :]  ← 真值
  ↓ min_dist = ||aim - enemy_pos||.min()
```

**结论**: 代码接线**理论正确**。139m 偏离必来自以下五个候选之一。

---

## 2. 五个候选根因（按可能性排序）

### 候选 A（最可能）：Kalman warm-start 没真正运行

**症状**: `tracker.is_initialized` 在第一次 `_apply_laser_sensing` 调用时**不是 False**，跳过 warm-start，直接走 line 309-314 的 else 分支（只做一次更新）。

**触发条件**: `KalmanTracker.reset()` 把 `_initialized = False`，但**没有重置 `_trk_x / _trk_P`**（看 sensing.py:199-203）。第二次 reset 调用前后状态可能不一致。

**或**: `runner.reset()` 在某个分支没把 trainer 传进去，导致 `kalman.reset()` 没被调用。

**验证步骤**:
```python
# 在 ppo_trainer.py _apply_laser_sensing 入口加 print
def _apply_laser_sensing(self, cmd_obs, env):
    if self.task_type == "laser":
        before_init = self.kalman_tracker.is_initialized
        ...
    # 出口加
    print(f"[ANCHOR-A] tracker init before={before_init} after={self.kalman_tracker.is_initialized}", flush=True)
```

第一次调用应 `before=False after=True`。后续调用应 `before=True after=True`。
若**第一次就是 True** → reset 没生效，找根因。

---

### 候选 B：warm-start 跑了但 Kalman 估计仍偏离

**症状**: warm-start 120 次后，Kalman 估计的 (x0, x1) 与 true (ex, ey) 偏离 >10m。

**触发条件**:
- `_fuse_one` 的 info matrix 接近奇异（near-collinear geometry）
- `clamp(-half_x, half_x)` 把估计截断到地图边
- jam_mul 异常放大噪声

**验证步骤**:
```python
# 在 sensing.py fused_sensing warm-start 循环末尾加 print（仅第一次）
if not tracker.is_initialized and not getattr(self, '_dbg_warmup_printed', False):
    self._dbg_warmup_printed = True
    err = (x0 - ex).abs().max().item()
    print(f"[ANCHOR-B] warm-start final: max_err={err:.2f}m "
          f"(target_x={ex.flatten()[:2].tolist()}, est_x={x0.flatten()[:2].tolist()})", flush=True)
```

`err` 应 <0.5m。若 err >10m → 几何退化或噪声过大。

---

### 候选 C：cmd_obs[68:70] 进 _apply_residual_aim 时**不**是 Kalman 输出

**症状**: Kalman 输出写到 `commander_obs`，但 `cmd_obs = commander_obs[:, team, :]` 后又被某个步骤覆写回真值。

**触发条件**:
- `_apply_laser_sensing` 用 `return torch.nan_to_num(cmd_obs, ...)` 在 line 410 — nan_to_num 是 in-place 操作的别名，可能没生效
- 或 `_get_observations` 在 `_apply_laser_sensing` 之后又被调用（重新读 env 真值）

**验证步骤**:
```python
# 在 _apply_residual_aim 入口加 print（仅第一次调用）
if not getattr(self, '_dbg_anchor_printed', False):
    self._dbg_anchor_printed = True
    # 读取 env 真值 enemy pos
    enemy_idx = env.battlefield.team_radar_indices[1]  # 假设 team=0 时 enemy=1
    true_ex = env.radar_pos[0, enemy_idx[0], 0].item()
    true_ey = env.radar_pos[0, enemy_idx[0], 1].item()
    half_x = float(env.map_size[0]) / 2.0
    anchor_x_m = cmd_obs[0, 68].item() * half_x
    print(f"[ANCHOR-C] team={team} anchor=({anchor_x_m:.1f}, ...) "
          f"true_enemy=({true_ex:.1f}, {true_ey:.1f})", flush=True)
```

`anchor` vs `true_enemy` 应偏差 <1m。若偏差 >10m → Kalman 输出没传到 anchor。

---

### 候选 D：env.process_commander_actions 接收的 action 被覆写

**症状**: `_apply_residual_aim` 返回正确的 env_action，但 `env.step(env_action)` 内部某处把 aim 改了。

**触发条件**:
- `vec_drone.process_commander_actions` 解码顺序错误
- `vec_mfar_env.step` 中的 `_apply_vehicle_actions` 误改 commander_actions

**验证步骤**: 在 `vec_drone.py:193` 之前加：
```python
# 在 process_commander_actions line 193 之前加
if not hasattr(self, '_dbg_cmd_printed'):
    self._dbg_cmd_printed = True
    print(f"[ANCHOR-D] cmd_action[0,0,:]={commander_actions[0,0,:].tolist()}", flush=True)
self._commander_aim[..., 0] = commander_actions[..., 1] * half_x
...
```

若 `cmd_action[1]` 在归一化空间是 ~0.3 而 enemy_x_norm 也是 ~0.3 → 一致。若不一致 → action 在中间被改了。

---

### 候选 E：min_dist 计算用的 enemy_pos 与 Kalman 跟踪的不是同一个 enemy

**症状**: Kalman 跟踪 enemy_idx[0]，但 reward.py 的 min_dist 取 min over enemy_idx（包含 enemy_idx[1]）。两个 enemy 雷达位置差很大，min 选错。

**触发条件**: 无 — 这是设计行为（aim 应指向最近的敌人）。但若 Kalman 跟踪错了 enemy_idx[1] 而 reward 取了 enemy_idx[0] 的 dist，会显示偏大。

**验证步骤**:
```python
# reward.py line 234 之后加（仅 team=0 第一步）
if not hasattr(self, '_dbg_enemy_printed'):
    self._dbg_enemy_printed = True
    print(f"[ANCHOR-E] t={t} enemy_pos[0]={enemy_pos[0, 0].tolist()} "
          f"enemy_pos[1]={enemy_pos[0, 1].tolist()}", flush=True)
```

观察 enemy_pos[0] vs enemy_pos[1] 距离。若 >200m，Kalman 跟踪 enemy[0] 但 min_dist 选了 enemy[1] → 解释 139m。

---

## 3. 执行计划（按优先级，每步 ≤10 min）

### 步骤 1：加 5 个诊断 print（一次性，全部加上）
按上面 5 个候选的 print 代码片段，全部插入对应位置。打印只在第一次触发，不会刷屏。

### 步骤 2：跑 1 个 episode 即停
```bash
timeout 120 python -m training.train --config configs/ablation_f1f8/v3_p2_A_baseline.yaml 2>&1 | tee /tmp/anchor_diag.log
# 等 [ANCHOR-X] 5 条都出现后 ctrl-C
grep "\[ANCHOR-" /tmp/anchor_diag.log
```

### 步骤 3：根据 print 结果定位
- 候选 A 触发（`before=True` 第一次）→ 修 `LaserEpisodeRunner.reset()` 确保 kalman.reset() 被调用
- 候选 B 触发（err >10m）→ 修 `_fuse_one` 几何 / jam_mul / clamp 边界
- 候选 C 触发（anchor vs true 偏 >10m）→ 修 obs 传递路径，确保 Kalman 后的 obs 不被覆盖
- 候选 D 触发（cmd_action[1] 异常）→ 修 vec_drone / vec_mfar_env 的 action 解码
- 候选 E 触发（enemy_pos[0] vs [1] 差大）→ 修 Kalman 跟踪哪个 enemy 或 reward 取 min 的逻辑

### 步骤 4：修复后回归验证
重跑 A baseline 1 个 episode，确认 `[dart] min_dist_init` ≤1m。然后才能跑 §5 tight-kr。

---

## 4. 最容易被忽视的"沉默杀手"

**沉默杀手 1：torch.nan_to_num 的 in-place 陷阱**

ppo_trainer.py:410:
```python
return torch.nan_to_num(cmd_obs, nan=0.0, posinf=1.0, neginf=-1.0)
```

`torch.nan_to_num(input, ...)` 默认**out-of-place**——返回新 tensor，不修改 input。所以这里返回的 tensor 是清洗过的，但**原始 cmd_obs 没被修改**。如果后续代码引用原始 cmd_obs，就会用到未清洗的版本。

**检查**: 调用方 `commander_obs = self._apply_laser_sensing(commander_obs, env)`（line 605）用了返回值，所以**league 路径 OK**。但若有其他调用方没用返回值，就有问题。

**沉默杀手 2：team_radar_indices 在 reset 后变了**

`enforce_radar_baseline` 修改 `env.radar_pos`，**不**修改 `team_radar_indices`。但若 env.reset 重新分配 indices 而 Kalman tracker 缓存了旧 indices → 跟踪错 enemy。

**检查**: sensing.py 的 `fused_sensing` 用 `obs[..., 68:70]` 读取 enemy 位置（来自最新 obs），**不**缓存 indices。OK。

**沉默杀手 3：warm-start 用的是真值，runtime 用的是 noisy 测量**

warm-start（sensing.py:285-308）调用 `_fuse_one(ex, ey, ...)` 时，`ex, ey` 是 `obs[..., off] * half_x`——**真值** enemy pos（因为 obs 是从 env 真值构建的）。

但**Kalman update 本身**（line 276-279）也用 `ex, ey`（真值）→ Kalman 估计必收敛到真值。

所以**warm-start 必收敛**。问题不在 warm-start 本身。

---

## 5. 最关键的怀疑：观测被重复构建

看 ppo_trainer.py 的 `get_own_actions`:
```python
state, commander_obs = self._get_observations(env, spectrum, events)  # line 596
# ...
if self.task_type == "laser":
    commander_obs = self._apply_laser_sensing(commander_obs, env)  # line 605
```

`commander_obs` 被 `_apply_laser_sensing` 的返回值**覆写**——OK。

但 `_apply_residual_aim(cmd_action, cmd_obs, env)` 用的是 `cmd_obs = commander_obs[:, team, :]`（line 607）。如果 line 605 的赋值**没生效**（例如 `_apply_laser_sensing` 内部 bug 返回了未修改的旧 obs），anchor 就是真值（这反而对，min_dist_init 应 ~0）。

**反向思考**: 如果 anchor 是真值，min_dist_init 应**很小**（policy 残差 6m 最大）。实测 139m，**远超**这个上限。所以 anchor 不是真值。

那 anchor 是什么？可能是：
- **未经 Kalman 的 raw sensing**（但本配置没用 raw sensing）
- **被某个 stale cache 覆盖**（前面提到的 `nan_to_num` 陷阱）
- **完全错误的索引**（如 obs[0:2] 是 own radar 0，不是 enemy 0）

**强烈怀疑**: `_apply_residual_aim` 的 `cmd_obs[..., 68]` 在 league 路径下不是 enemy 位置而是别的。需要按候选 C 的 print 验证。

---

## 6. 备用方案：跳过 Kalman，用真值 anchor

**如果诊断 4-6 小时后仍找不到根因**，临时绕过：

```python
# ppo_trainer.py:436-441
if not self.residual_aim:
    return cmd_action
# BYPASS: use true enemy pos directly (skip Kalman) for debugging
half_x = float(env.map_size[0]) / 2.0
half_y = float(env.map_size[1]) / 2.0
# cmd_obs[68:70] should already be Kalman-fused, but for debugging read from env
enemy_idx = env.battlefield.team_radar_indices[1]  # enemy team when team=0
anchor_x = env.radar_pos[:, enemy_idx[0], 0] / half_x  # TRUE enemy x
anchor_y = env.radar_pos[:, enemy_idx[0], 1] / half_y
```

如果这个 bypass 让 min_dist_init 回到 ≤1m → 确认 anchor 链路是 bug，Kalman 路径有问题。
如果仍 139m → bug 在 process_commander_actions 或 reward 计算侧。

---

## 7. 论文叙事（修好后的样子）

修好 anchor 后，预期：
- min_dist_init 从 139m → ≤1m
- kill_rate 在 kr=24.5m 下仍是 1.00（kr 太宽松），**但** learning 信号变得有意义
- 跑 §5 tight-kr (5m)：A baseline 此时 kill_rate → 0（学不到 5m 精度），B（F1+F2 ON）能否保持 1.00 才是真正的因果验证

**论文图素材**: min_dist_init vs iter 曲线，对比 v3_scaling（坏 anchor，139m 不收敛）vs fixed（≤1m 收敛）。这本身就是一个独立于 alpha 的贡献。

---

## 8. 工件清单

- `ANCHOR_DIAGNOSIS.md`（本文件）
- 修复后新增/修改的代码文件（待诊断完成）
- 修复后回归验证日志：`/tmp/anchor_diag.log` + `[dart]` 打印的 min_dist_init 量级

---

## 9. 给 agent 的执行指令（一句话）

> 加 5 个 [ANCHOR-A/B/C/D/E] print，跑 1 个 episode 即停，根据哪个 print 显示异常定位根因，修复后重跑确认 min_dist_init ≤1m。**绝不能**在 anchor 没修好前跑 §5 tight-kr 或 HEADLINE-LINEAR——那些实验在 broken anchor 下都 inconclusive。
