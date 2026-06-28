# 下一步实验指令 — 把"初步有效"推成"因果坐实 + 论文级证据"

**面向**: PRO 6000 上的 agent
**前置**: P1_EXPERIMENT_REPORT.md（B 备选已验证 kill_rate=1.00 @ alpha=0.7，但 kr=24.5m 宽松信号弱）
**目标**: 用最少 GPU 时间拿到 2 个关键证据
1. **因果坐实**：A 对照证明"F1+F2 是 kill_rate 的唯一原因"
2. **论文级 headline**：F1+F2 + 原版 linear α→1.0 斜坡冲过 iter 6（v3_scaling 崩塌点）
**总预算**: ~6-8 GPU-h（半天）

---

## 0. 纪律提醒（必读）

- ✅ Bug #1（train.py dead code）**只让 adaptive-alpha 结论作废**，**不影响 linear-斜坡崩塌**这个核心发现
  - v3_scaling 本就跑 linear schedule（Bug #1 让它 fallback 到 linear，与配置声称一致）
  - 论文里把 adaptive 相关结论删掉即可，**核心故事还在**
- ❌ 不要再单独跑 adaptive schedule 验证（30 min 那次已确认是 linear，作废）
- ❌ 不要修改 sensing / aim_z（已修，与本阶段无关）
- ❌ 不要在没有 A 对照的情况下宣称 F1+F2 是"治本"

---

## 1. 必做的代码补丁（机制仪表，10 min）

### 1.1 ppo_trainer.py — 补 team_adv_std 累计 + 输出

**位置**: `training/ppo/ppo_trainer.py` PPOTrainer.update()

**改动 1**（line 65 附近，metrics 累计初始化）：
```python
total_team_value_loss = 0.0
total_team_adv_std = 0.0      # ← 新增：归一化前 team_adv 的 std
```

**改动 2**（line 90 之后，team_adv 计算后）：
```python
team_adv = team_returns.unsqueeze(-1) - team_value.detach()
# 仪表：累计归一化前 team_adv.std（F1 修复前应趋 0，修复后应离 0）
total_team_adv_std += float(team_adv.std().item())
```

**改动 3**（line 180 附近，metrics 输出）：
```python
if total_team_value_loss > 0:
    metrics["team_value_loss"] = total_team_value_loss / max(n_updates, 1)
    metrics["team_adv_std"] = total_team_adv_std / max(n_updates, 1)   # ← 新增
```

### 1.2 flux_league.py — 每 iter 打印机制仪表

**位置**: `training/flux_league.py` `_train_against` 或 PPO update 调用后

**改动**：找到 `cmd_metrics = trainer.update(...)` 调用处，紧跟着加：
```python
# 机制仪表：per-iter 打印（不依赖 metrics dict 是否被汇总）
if isinstance(cmd_metrics, dict):
    tvl = cmd_metrics.get("team_value_loss", None)
    tas = cmd_metrics.get("team_adv_std", None)
    if tvl is not None:
        print(f"  [mech] team_value_loss={tvl:.3e}  team_adv_std={tas:.3e}", flush=True)
```

**验证**：跑 1 个 episode 后日志应出现 `[mech] team_value_loss=... team_adv_std=...`

---

## 2. 提速配置（必做，节省 70% 时间）

### 2.1 全局 override（所有下一步实验默认）

| 字段 | 原值 | **新值** | 原因 |
|---|---|---|---|
| `n_eval_games` | 8 | **2** | payoff 矩阵每 matchup 跑 2 ep 够判 wr |
| `population_cap` | 12 | **6** | pool 小，payoff matchup 数从 36 → 9 |
| `episodes_per_training` | 10 | **8** | 训练样本略减 |
| `psro_iterations` | 3 | **见各实验** | 各实验指定 |

**预期效果**: 每 iter 从 130min → **~30-40min**

### 2.2 Isolation 实验额外（可选更激进）

如果只关心训练臂的 kill_rate + team_value_loss（不看 league win_rate），可在 `FluxLeague.run()` 主循环里加 `bypass_payoff=True` 开关：
```python
# flux_league.py run() 主循环
if not getattr(self, 'bypass_payoff', False):
    payoff = self.evaluate_payoff_matrix(...)   # 跳过这步
# 仍跑训练
self._train_against(...)
```
**预期**: 每 iter ~15min（仅训练，无 payoff）。

⚠️ **PASS 判据仍可用**：kill_rate 是训练臂 episode 内的事件，不依赖 payoff。

---

## 3. 实验 A — Baseline 对照（因果坐实）

### 3.1 配置

```yaml
# configs/ablation_f1f8/v3_p2_A_baseline.yaml（基于 v3_p1_f1f2_alpha07.yaml 派生）
league:
  alpha_schedule: constant
  alpha_constant: 0.7
  kill_radius_init: 24.5
  kill_rate_threshold: 99.0      # 同 B：kr 冻结
  population_cap: 6              # 提速
  n_eval_games: 2                # 提速
  episodes_per_training: 8       # 提速
  psro_iterations: 3             # 短验证
  checkpoint_dir: "checkpoints/laser_pro6000_league_v3_p2_A_baseline"

ppo:
  f1_disable: true               # ← 关 F1（恢复 OLD 800-step cross-ep 累加）
  f2_disable: true               # ← 关 F2（恢复 team_adv 双重归一化）
```

### 3.2 打印字段（每 iter 末汇总）

```
[League] alpha=...  schedule=...  kill_rate=...
[mech] team_value_loss=...  team_adv_std=...
```

### 3.3 PASS 判据（预注册）

| 字段 | A 预测 | B 实测 | 因果结论 |
|---|---|---|---|
| iter 1 kill_rate | **0.00-0.30**（崩塌或濒崩） | 1.00 | 若 A << B → F1+F2 是唯一原因 ✅ |
| iter 1 team_value_loss | **≥1e7**（百万级不收敛） | ~1e5 | 若 A >> B → F1 让 critic 收敛 ✅ |
| iter 1 team_adv_std | **趋 0**（噪声被放大） | O(1) | 若 A 趋 0 → F1+F2 阻止噪声主导 ✅ |

**核心 PASS**：A 在 alpha=0.7 + kr=24.5m 下 kill_rate < 0.5（vs B=1.00）→ 干净因果证明

**FAIL**：A 也 kill_rate=1.00 → kr=24.5m 太宽松，需严苛 kr 重做（见 §5）

### 3.4 预算

3 iter × ~35min = **~105 min**

---

## 4. 实验 HEADLINE-LINEAR — 论文级 headline 图

### 4.1 配置

```yaml
# configs/ablation_f1f8/v3_p2_headline_linear.yaml（基于 v3_scaling.yaml 派生）
league:
  alpha_schedule: linear         # ← 原版斜坡 0→1 by iter 12（v3_scaling 同款）
  alpha_constant: 0.0            # 不用
  kill_radius_init: 50.0         # ← 正常起点（v3_scaling 同款）
  kill_rate_threshold: 0.7       # ← 正常退火阈值（v3_scaling 同款）
  kill_radius_decay: 0.7
  kill_radius_m: 0.2
  population_cap: 8              # 略宽于 A（headline 需要更可信的 wr）
  n_eval_games: 4                # 略宽于 A
  episodes_per_training: 10
  psro_iterations: 12            # ← 关键：必须跑到 iter 12 让 alpha=1.0
  checkpoint_dir: "checkpoints/laser_pro6000_league_v3_p2_headline_linear"

# F1+F2 默认 ON（不加 f1_disable / f2_disable）
```

### 4.2 PASS 判据（预注册 — 论文核心图）

**v3_scaling 历史**（baseline，F1+F2 OFF）：
| iter | alpha | kill_rate |
|---|---|---|
| 5 | 0.417 | 0.62 |
| 6 | 0.500 | **0.00** ← 崩塌 |
| 7 | 0.583 | **0.00** |

**HEADLINE-LINEAR 目标**（F1+F2 ON）：
| iter | alpha | kill_rate PASS | kill_rate FAIL |
|---|---|---|---|
| 6 | 0.500 | **≥0.3** | <0.3 |
| 9 | 0.750 | **≥0.3** | <0.3 |
| 12 | 1.000 | **≥0.3** | <0.3 |

**核心 PASS**: **iter 6-12 全部 kill_rate ≥ 0.3**（v3_scaling iter 6 崩到 0）
- 论文图：双曲线 kill_rate vs iter，v3_scaling 在 iter 6 悬崖，F1+F2 一路平
- 配 team_value_loss 曲线：v3_scaling 百万级，F1+F2 ~10万级

**FAIL**: 任一 iter 6-12 kill_rate < 0.3 → F1+F2 不足以撑过 linear 斜坡，需 F3 (PopArt)

### 4.3 预算

12 iter × ~60min（含正常 payoff）= **~12h**（**建议 bypass_payoff=True 跑 ~6h**）

---

## 5. 兜底 — 严苛 kr 验证（仅当 A 也 kill_rate=1.00 时）

如果 A baseline 在 kr=24.5m 下也 kill_rate=1.00，说明 kr 太宽松，必须收紧：

```yaml
# configs/ablation_f1f8/v3_p2_A_tight_kr.yaml
league:
  kill_radius_init: 5.0          # ← 严苛
  kill_rate_threshold: 0.95      # 几乎必须 hit 才不退火
  # 其他同 A
```

此时 A 应该 kill_rate→0（baseline 学不会 5m 精度），B 看是否能保持。

---

## 6. 执行顺序（推荐）

```
Day 1 上午（~3h）：
  1. 应用代码补丁 §1（机制仪表）+ 提速配置 §2 [10min]
  2. 跑实验 A [~105min]
     → 立刻判断：A kill_rate < 0.5？
       - YES → F1+F2 因果坐实，进 §3
       - NO  → kr 太宽松，跑 §5 兜底

Day 1 下午（~6h，或过夜 12h）：
  3. 跑 HEADLINE-LINEAR（bypass_payoff=True）[~6h]
     → 判断：iter 6-12 全部 kill_rate ≥ 0.3？
       - YES → 论文 headline 图素材到手，整理 P1_FINAL_REPORT
       - NO  → F1+F2 不足，进 F3 (PopArt) 或 F4 (cap alpha + COMA)
```

---

## 7. 数据汇总表（实验完成后填）

### 7.1 因果对照（A vs B）

| 指标 | A baseline (F1+F2 OFF) | B (F1+F2 ON) | 改善倍数 |
|---|---|---|---|
| iter 1 kill_rate | TBD | 1.00 | — |
| iter 1 team_value_loss | TBD | ~1e5 | TBD |
| iter 1 team_adv_std | TBD | O(1) | TBD |

### 7.2 Headline（v3_scaling vs HEADLINE-LINEAR）

| iter | alpha | v3_scaling kill_rate | HEADLINE-LINEAR kill_rate |
|---|---|---|---|
| 0 | 0.000 | 1.00 | TBD |
| 5 | 0.417 | 0.62 | TBD |
| **6** | **0.500** | **0.00** ← 崩塌 | **TBD** |
| 9 | 0.750 | — | TBD |
| **12** | **1.000** | — | **TBD** |

---

## 8. 论文叙事更新（实验完成后）

### 8.1 主线（不变）

> naive α 斜坡在 α≥0.5 崩塌，归因于三个已知 CTDE 反模式（reward 未归一化 / N-step 跨界 / shared advantage）。F1+F2 套用 MAPPO/GAE/COMA 最佳实践修复后，linear α→1.0 一路稳定。

### 8.2 新增支撑（来自本指令的实验）

- **因果坐实图**：A vs B 在 alpha=0.7 + kr=frozen 下的 kill_rate/team_value_loss 对照
- **Headline 图**：linear α 斜坡下 v3_scaling vs F1+F2 的 kill_rate vs iter 双曲线
- **机制图**：team_value_loss + team_adv_std 随 iter 变化（A 不收敛，B 收敛）

### 8.3 删除（因 Bug #1）

- 所有 adaptive schedule 相关结论（"已验证 adaptive 可避开崩塌"等）
- ALPHA_COLLAPSE_REPORT.md §4.1-4.3 的 adaptive 数据（标"作废，因 dead code bug"）

---

## 9. 工件清单（实验完成后 commit）

- `configs/ablation_f1f8/v3_p2_A_baseline.yaml`
- `configs/ablation_f1f8/v3_p2_headline_linear.yaml`
- `configs/ablation_f1f8/v3_p2_A_tight_kr.yaml`（如跑兜底）
- `logs/diag/v3_p2_A_*.log`
- `logs/diag/v3_p2_headline_*.log`
- `P2_EXPERIMENT_REPORT.md`（新报告，含两张论文级图的数据）
- 代码改动（机制仪表补丁）：
  - `training/ppo/ppo_trainer.py`（team_adv_std 累计 + 输出）
  - `training/flux_league.py`（per-iter 机制仪表 print）
  - 可选 `bypass_payoff` 开关
