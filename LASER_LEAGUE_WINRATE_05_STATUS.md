# Status: FluxLeague Laser Task — win_rate=0.50 持续问题

**Date:** 2026-06-18 (corrected)
**Branch:** `evo/laser-fix`
**Related docs:** [DIAG_WIN_RATE_ZERO.md](DIAG_WIN_RATE_ZERO.md), [LASER_ROOT_CAUSE_ANALYSIS.md](LASER_ROOT_CAUSE_ANALYSIS.md)

---

## ⚠️ Correction Notice (2026-06-18 late)

Earlier version of this doc claimed 0.5 win rate was caused by "3 fixed bugs + 1 unknown fused_sensing saturation bug". **That was wrong**. The truth is simpler and worse:

- The `training/laser/` package (sensing/reward/episode) was referenced by 6 import sites in committed code but **never committed** — anyone pulling `17fcb77` hits ImportError at first laser call.
- The "saturation bug" was **not pre-existing** — it was a regression introduced by incomplete config threading in `train.py`'s `laser_cfg` dict.

See §5 below for the corrected root cause.

---

## 1. 问题陈述

FluxLeague 激光精确击杀任务在 PSRO 评估中，**全部 36 个 cross-team 对局胜率 = 0.50**。

这导致 league 学习信号完全失效：
- Nash meta-solver 收到均匀矩阵 → sigma 退化到 `[1, 0, 0]`
- PFSP 失效（没有胜负区分）
- Elo 不更新

---

## 2. 真实 root cause：laser_cfg 配置透传缺失

[training/train.py](training/train.py) `laser_cfg` 字典漏传 4 个关键字段（Phase B 阶段只补了 2 个）：

| 字段 | 配置位置 | 默认值 | 缺失后果 |
|---|---|---|---|
| `hybrid_fire` | `training:` | `False` | aim-head 随机初始化（Phase B 已修） |
| `decouple_value` | `training:` | `False` | value head 不解耦（Phase B 已修） |
| **`residual_aim`** | `training:` (line 70 = `true`) | `False` | aim 不锚定敌位 → hybrid_fire 零初始化无效 |
| **`min_radar_baseline_m`** | `env:` (line 101 = `5000.0`) | `0.0` | `enforce_radar_baseline` 是 no-op → 几何退化 |

### 因果链（修正版）

```
laser_cfg.min_radar_baseline_m = 0  (未透传)
  ↓
enforce_radar_baseline(env, 0) → 立即 return（no-op）
  ↓
雷达随机部署，可能近共线
  ↓
_fuse_one 的信息矩阵 L 接近奇异: det(L) → 0
  ↓
zx = (L11·e0 − L01·e1) / det → 爆炸到 ±Inf
  ↓
zx.clamp(±half_x) → 饱和到 ±1.0（地图边界）
  ↓
cmd_obs[68:70] = ±1.0 → residual_aim 把 anchor ±1 当真值
  ↓
aim = (±1) × half_map + residual × scale = ±10000m（地图角）
  ↓
kill_radius_m=50m 永远不满足 → illumination_progress=0
  ↓
tiebreaker 退化到 0.5
```

### 为什么 train_laser.py 不出问题

[training/train_laser.py:323](training/train_laser.py#L323) 直接读 `cfg["env"]["min_radar_baseline_m"]`，不走 `laser_cfg` 字典中转；[train_laser.py:287](training/train_laser.py#L287) 直接读 `tcfg["residual_aim"]`。两个字段都正确赋值，所以 baseline 雷达几何被强制 5km，融合矩阵良态，估计收敛到 0.2m。

---

## 3. 已实施的修复（4 阶段）

### Phase A — `create_team_policy` 加 `hybrid_fire`/`decouple_value` 参数

[training/ppo/actor_critic.py](training/ppo/actor_critic.py) ~10 LOC。

### Phase B — `laser_cfg` 透传 `hybrid_fire`/`decouple_value`

[train.py](training/train.py) + [flux_league.py](training/flux_league.py) ~15 LOC。

### Phase C — PayoffMatrix timeout tiebreaker（illumination_progress）

[training/self_play/payoff_matrix.py](training/self_play/payoff_matrix.py) ~20 LOC。

### Phase D — `laser_cfg` 透传 `residual_aim`/`min_radar_baseline_m`（**真正解决 0.5**）

[train.py](training/train.py) laser_cfg 加：

```python
"residual_aim": config.get("training", {}).get("residual_aim", False),
"min_radar_baseline_m": config.get("env", {}).get("min_radar_baseline_m", 0.0),
```

### Phase E — 提交 `training/laser/` 包（修复破损 build）

之前 commit `17fcb77` 引用了 `training.laser.{episode,reward,sensing}` 但没把包提交。本 commit 补上：

- [training/laser/__init__.py](training/laser/__init__.py)
- [training/laser/sensing.py](training/laser/sensing.py) — `fused_sensing` / `KalmanTracker` / `enforce_radar_baseline` / `add_sensing_noise`
- [training/laser/reward.py](training/laser/reward.py) — `LaserRewardShaper`
- [training/laser/episode.py](training/laser/episode.py) — `LaserEpisodeRunner`

---

## 4. 验证（pending — Phase D/E commit 后重跑）

### 4.1 静态（已通过）
- training.laser 4 个 import 全部 OK
- `laser_cfg.get("min_radar_baseline_m", 0.0)` 现在返回 5000.0（来自 env.min_radar_baseline_m）

### 4.2 待重跑：有效小批量训练

```bash
python -m training.train \
  --config configs/laser_25x25_pro6000_league.yaml \
  --override training.psro_iterations=3 \
  --override env.num_envs=8 \
  --override league.episodes_per_training=3 \
  --override league.max_steps_per_episode=50 \
  --override league.n_eval_games=4
```

**PASS 标准**：
1. iter 0 eval 后任一 commander 的 `laser_aim` **不在地图角**（|x|, |y| < half_map × 0.9）
2. illumination_progress > 0（至少一队）
3. payoff matrix 出现非 0.5 值

---

## 5. 之前错误诊断的反思

### 错误 1：把 0.5 归因于"未知 fused_sensing 饱和 bug"

实际上 fused_sensing 代码本身没问题（line 556-559 的 clamp 是退化几何的兜底）。问题是上游 `enforce_radar_baseline` 因 `min_radar_baseline_m=0` 没被触发，导致几何本身退化。

### 错误 2：Phase C 的 tiebreaker 不能解决 0.5

tiebreaker 只在两队 progress 至少一个 > 0 时才能区分胜负。两队都 0 时退化到 0.5。Phase D 修好 sensing 后，progress 才会 > 0，tiebreaker 才有意义。Phase C 不是没用，但它依赖 Phase D 先修好 sensing。

### 错误 3：未提交 training/laser/ 包

最严重的错误。让 commit `17fcb77` 处于不可构建状态。Phase E 修复。

---

## 6. 战略建议（来自用户）

| 路径 | 状态 | 建议 |
|---|---|---|
| `python -m training.train_laser` (train_laser.py) | ✅ 工作（4090 上验证 red=0.88, kr→0.2m） | **PRO 6000 立即用这条**，配置 [configs/laser_25x25_pro6000.yaml](configs/laser_25x25_pro6000.yaml) |
| `python -m training.train` (FluxLeague) | ❌ 之前坏，Phase D/E 后待验证 | 需要全套元博弈（Nash/exploiter/TC-DAMS）时才用 |

如果只需要多智能体效果，train_laser.py 内置的 PSRO-lite 联赛已达成目标。FluxLeague 那条路径的价值在于 alpha-star 风格的 3-role exploiter，目前还没验证修好后能否跑出比 train_laser 更好的样本效率。

---

## 7. 修改文件清单

| 文件 | Phase | 改动 |
|---|---|---|
| [training/ppo/actor_critic.py](training/ppo/actor_critic.py) | A | `create_team_policy` 加 `hybrid_fire`/`decouple_value` |
| [training/flux_league.py](training/flux_league.py) | B | `__init__` 存 2 字段 + 3 处调用透传 |
| [training/train.py](training/train.py) | B + D | `laser_cfg` 加 4 字段（hybrid_fire, decouple_value, residual_aim, min_radar_baseline_m） |
| [training/self_play/payoff_matrix.py](training/self_play/payoff_matrix.py) | C | timeout tiebreaker + progress 缓存 |
| [training/laser/__init__.py](training/laser/__init__.py) | E | NEW: 包初始化 |
| [training/laser/sensing.py](training/laser/sensing.py) | E | NEW: 270 LOC KF/融合/基线 |
| [training/laser/reward.py](training/laser/reward.py) | E | NEW: LaserRewardShaper |
| [training/laser/episode.py](training/laser/episode.py) | E | NEW: LaserEpisodeRunner |
| [training/ppo/ppo_trainer.py](training/ppo/ppo_trainer.py) | (prior migration) | laser 钩子集成 |

---

## 8. 结论

**真正的 root cause**：`train.py` 的 `laser_cfg` 字典漏传 `residual_aim` 和 `min_radar_baseline_m`，让 train_laser.py 工作良好的 sensing 逻辑在 FluxLeague 路径下被静默禁用。

**之前 3 阶段修复（A/B/C）**：是真实小问题但不是 0.5 的根因。

**真正解决 0.5**：Phase D（透传 residual_aim + min_radar_baseline_m）+ Phase E（提交 training/laser/ 包让 build 不坏）。

**待验证**：Phase D/E push 后重跑小批量训练，确认 aim 不再饱和到地图角、progress > 0、win_rate 出现非 0.5 值。
