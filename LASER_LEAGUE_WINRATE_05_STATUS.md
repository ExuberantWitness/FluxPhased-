# Status: FluxLeague Laser Task — win_rate=0.50 持续问题

**Date:** 2026-06-18
**Branch:** `evo/laser-fix`
**Related docs:** [DIAG_WIN_RATE_ZERO.md](DIAG_WIN_RATE_ZERO.md), [LASER_ROOT_CAUSE_ANALYSIS.md](LASER_ROOT_CAUSE_ANALYSIS.md)

---

## 1. 问题陈述

FluxLeague 激光精确击杀任务在 PSRO 评估中，**全部 36 个 cross-team 对局胜率 = 0.50**（red 主政策 vs blue 主政策、exploiter 对主政策、所有方向都一样）。

这导致 league 学习信号完全失效：
- Nash meta-solver 收到均匀矩阵 → sigma 退化到 `[1, 0, 0]`
- PFSP 失效（没有胜负区分）
- Elo 不更新
- TC-DAMS 拿不到指纹差异

---

## 2. 已识别的 3 个独立 bug

经 3 个并行 Explore agent 调查，确认 root cause 是 3 个 bug 叠加：

| # | Bug | 位置 | 严重性 |
|---|---|---|---|
| PRIMARY | `create_team_policy` 没透传 `hybrid_fire`/`decouple_value` 给 `CommanderActorCritic` | [actor_critic.py:983-1008](training/ppo/actor_critic.py#L983-L1008) | 致命 |
| SECONDARY | `PayoffMatrix.evaluate_pair` 在 timeout 时硬编码 `red_wins += 0.5` | [payoff_matrix.py:144-149](training/self_play/payoff_matrix.py#L144-L149) | 信号丢失 |
| TERTIARY | `laser_cfg` 没读 `training.hybrid_fire`/`training.decouple_value` | [train.py:143-151](training/train.py#L143-L151) | 参数缺失 |

**对照基线**：[train_laser.py:1075-1083](training/train_laser.py#L1075-L1083) 直接构造 `CommanderActorCritic(hybrid_fire=True, decouple_value=True)`，所以 baseline `red=0.85` 工作正常；FluxLeague 走 `create_team_policy` 把 flag 丢了。

---

## 3. 已实施的 3 阶段修复

### Phase A — `create_team_policy` 加新参数（5 LOC）

[training/ppo/actor_critic.py](training/ppo/actor_critic.py) `create_team_policy` 签名扩展：

```python
def create_team_policy(
    ...
    hybrid_fire: bool = False,        # NEW
    decouple_value: bool = False,     # NEW
) -> dict:
```

透传到 `CommanderActorCritic` 构造。默认 `False` 保持向后兼容（通用导弹任务、smoke_tcdams_25 等不破坏）。

### Phase B — `laser_cfg` 透传 + 3 处调用点更新（~15 LOC）

[train.py:143-153](training/train.py#L143-L153) `laser_cfg` 加 2 字段：

```python
laser_cfg={
    ...
    "hybrid_fire": config.get("training", {}).get("hybrid_fire", False),
    "decouple_value": config.get("training", {}).get("decouple_value", False),
},
```

[flux_league.py](training/flux_league.py) `__init__` 末尾存储：

```python
self.hybrid_fire = (laser_cfg or {}).get("hybrid_fire", False)
self.decouple_value = (laser_cfg or {}).get("decouple_value", False)
```

3 处 `create_team_policy` 调用全部加 kwargs：主初始化、mutant 生成、mutant_trainer 重建。

### Phase C — PayoffMatrix timeout tiebreaker（~20 LOC）

[training/self_play/payoff_matrix.py](training/self_play/payoff_matrix.py)：

1. `__init__` 加 `self._last_step_progress = None`
2. step loop 缓存 `illumination_progress [E, n_teams]`
3. 替换硬编码 `+= 0.5`：
   ```python
   if last_progress is not None and self.task_type == "laser":
       p0, p1 = float(last_progress[e, 0]), float(last_progress[e, 1])
       if p0 - p1 > 0.01:      red_wins += 1.0
       elif p1 - p0 > 0.01:    red_wins += 0.0
       else:                   red_wins += 0.5
   else:
       red_wins += 0.5
   ```

**总改动量**：~45 LOC，全为参数透传 + 1 个 tiebreaker 替换。

---

## 4. 验证结果

### 4.1 静态测试 ✅ PASS

```bash
python -c "
from training.ppo.actor_critic import create_team_policy
p1 = create_team_policy(team=0, hybrid_fire=True, decouple_value=True)
p2 = create_team_policy(team=0, hybrid_fire=False, decouple_value=False)
w1 = p1['commander'].action_head.weight[1:].abs().mean().item()
w2 = p2['commander'].action_head.weight[1:].abs().mean().item()
print(f'hybrid_fire=True:  {w1:.6f}')
print(f'hybrid_fire=False: {w2:.6f}')
print(f'ratio: {w2/w1:.1f}x')
"
```

输出：`hybrid_fire=True: 0.0007` vs `hybrid_fire=False: 0.07`，**100× 比率确认零初始化生效**。

### 4.2 有效小批量训练 ✅ PASS

```bash
python -m training.train \
  --config configs/laser_25x25_pro6000_league.yaml \
  --override training.psro_iterations=3 \
  --override env.num_envs=8 \
  --override league.episodes_per_training=3 \
  --override league.max_steps_per_episode=50 \
  --override league.n_eval_games=4
```

- **PPO 真在更新**：54+ 次 `[PPO]` 日志行，cmd/rad 的 pl/vl/ent 都有数值
- **Reward 在涨**：p0000 episode reward 18.54 → 25.02 → 27.03（4× 高于修前 4-7）
- **hybrid_fire 真生效**：随机 commander 的 aim-head weight ≈ 0.0007

### 4.3 症状持续 ❌ FAIL

iter 1 评估：全部 36 个 cross-team win_rate **仍然 = 0.50**。

诊断（直接读 env result）：
```
team0 illumination_progress = 0.0000
team1 illumination_progress = 0.0000
laser_aim = (10000, 10000, 0)  # 地图角落，完全饱和
```

---

## 5. 为什么修复没有真正解决 0.5

Phase C 的 tiebreaker **只在至少一队有 progress > 0 时才能区分胜负**。当前两队 progress 都 = 0，所以 tiebreaker 走 `else: red_wins += 0.5` 分支——**和修前完全一样**。

**根因链**（独立于上述 3 bug）：

```
fused_sensing 把 anchor (cmd_obs[68:70]) 饱和到 ±1.0
  ↓
hybrid_fire 公式: aim = anchor + residual × (scale_m / half_map_m)
  当 anchor=±1 时, residual 的影响 << anchor 的影响
  ↓
aim 落到 ±10000m（地图角落）
  ↓
kill_radius_m=50m 永远不满足 → illumination_progress=0
  ↓
两队 progress 都 = 0 → tiebreaker 退化到 0.5
```

**关键认识**：Phase C 修复了"如何记录 0.5"的机制，但没有修复"为什么会产生 0.5"的根因。当前 tiebreaker 形式上正确，实践上无效。

---

## 6. 为什么 train_laser.py 工作但 FluxLeague 不工作

| 维度 | train_laser.py (baseline) | FluxLeague (evo/laser-fix) |
|---|---|---|
| Agent 数 | 单 agent（1 commander） | 多 agent CTDE（每队 1 cmd + N radars） |
| Sensing 路径 | 直接读 env state | `fused_sensing` 多雷达融合 + KalmanTracker |
| CommanderActorCritic 构造 | 直接传 `hybrid_fire=True` | 走 `create_team_policy`（修前丢 flag） |
| red 胜率 | **0.85** | **0.50**（卡死） |

train_laser.py 的 sensing 路径不经过 `fused_sensing`，所以没有 anchor 饱和问题。FluxLeague 的多 agent Kalman 融合产生饱和的 anchor，使 `hybrid_fire` 的"零初始化 aim-head"机制失效。

---

## 7. 未完成的工作

要真正让 win_rate ≠ 0.50，必须解决 sensing 饱和。三个候选方向：

| 方向 | 预期成本 | 风险 | 是否根治 |
|---|---|---|---|
| A. 诊断 `fused_sensing` anchor 饱和 | 1-2 小时 | 低 | ✅ 根治 |
| B. 临时放宽 `kill_radius_init` 50m → 1000m | 5 分钟 | 中（curriculum 退火后又复发） | ❌ 治标 |
| C. 扩展训练到 24 iter（baseline 配置） | ~2 小时 | 高（policy 可能学不到规避饱和） | ❌ 赌博 |

**推荐方向 A**：诊断 [training/laser/sensing.py](training/laser/sensing.py) 的 `KalmanTracker` 和 anchor 计算路径。

---

## 8. 修改的文件清单

| 文件 | 改动量 | 内容 |
|---|---|---|
| [training/ppo/actor_critic.py](training/ppo/actor_critic.py) | +10 | Phase A: `create_team_policy` 加 2 个 flag 参数 + 透传 |
| [training/flux_league.py](training/flux_league.py) | ~250 行（含迁移） | Phase B: `__init__` 存 2 字段 + 3 处调用透传（其余为先前 laser 迁移） |
| [training/train.py](training/train.py) | +26 | Phase B: `laser_cfg` 加 2 字段 |
| [training/self_play/payoff_matrix.py](training/self_play/payoff_matrix.py) | ~128 行 | Phase C: tiebreaker + progress 缓存（含先前 laser 集成） |
| [training/ppo/ppo_trainer.py](training/ppo/ppo_trainer.py) | ~236 行 | 先前 laser 迁移：LaserRewardShaper/KalmanTracker/residual_aim 钩子 |

---

## 9. 回退方案

- Phase A 默认 `hybrid_fire=False`，对通用任务零影响
- Phase B `laser_cfg.get(..., False)` 兜底，老 config 不破坏
- Phase C 阈值 0.01 太紧可放宽到 0.05；如果 sensing 修后 progress 仍全 0，退回 `+= 0.5` 也无害
- 每个 Phase 独立 commit，可单独 revert

---

## 10. 结论

**当前状态**：3 个 bug 的修复在代码层面正确完成，静态测试和小批量训练证明修复生效（PPO 在更新、reward 在涨、aim-head 零初始化生效）。

**未解决的问题**：0.50 症状持续，因为存在**第 4 个独立 bug**（fused_sensing anchor 饱和），不在本次修复范围内。

**下一步**：需要单独诊断 sensing 路径，或者放宽 kill_radius_init 作为临时绕过。
