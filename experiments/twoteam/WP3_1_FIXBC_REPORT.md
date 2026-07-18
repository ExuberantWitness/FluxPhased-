# WP-3.1 Fix B+C 实施报告

**日期**:2026-07-18 · **branch**:`twoteam/bc-ppo`
**前置**:Fix A 100-iter 命门通过(r=+1.4 at StrongRule),但 smoke vs BlindClassical 仍 kill=0.025
**本报告**:Fix B+C 装好 + 100-iter 复跑 + smoke + 行为 probe 全套数据

---

## 0. 背景回顾

### Fix A 命门结果(2026-07-18 早上)
- 训练 r=+1.4 at iter 70/90 vs StrongRule(kill bonus 触发)
- Smoke cross-play RL vs BlindClassical:**RL kill=0.025**(低干扰)/ 0.025(高干扰)
- 训练 vs BC r=-0.05/-0.07(iter 40/50),wr=0.34-0.39

### 详查训练数据(2026-07-18 上午)
behavior probe (`wp3_fix_a_behavior_probe.py`) 揭示:

| Opp | n_det | tracker_init | beam_d 误差 | radar_E_et | RL kill |
|---|---|---|---|---|---|
| BlindClassical | 16 | **0/2 全程** | 1919m (vs 50m hit radius) | **0 全程** | 0.03 |
| StrongRule | 16 | **4/2 ~3 step 后** | 部分命中 | max 0.50×e_kill | 0.37 |
| extreme/* | - | 0/2 | - | 0 | 0 |

**核心诊断**:RL 一直开火(emission_on=2 全程),**激光完全打偏**(radar_E_et=0)。两边都 emit=2 都有 16 detection,只有 vs StrongRule 时 RL 的 tracker 能 init;vs BC tracker 全程不 init → beam_direction_head 输出 garbage。

---

## 1. Fix B 实施(PFSP f_var + ema_var 健康门)

### 改动文件
- [algo/_shared/pilot/twoteam/opponent_pool.py](algo/_shared/pilot/twoteam/opponent_pool.py):
  - `__init__` 加 `pfsp_var_mix: float = 0.0`、`ema_var_uniform_floor: float = 0.0`
  - `sample_pfsp` 改 AlphaStar f_hard ⊕ f_var 混合:`weights = (1-mix)·f_hard + mix·f_var`
  - health gate:pool 已知 wr 的方差 < floor 时强制均匀采样
- [algo/_shared/pilot/twoteam/run_wp2_league.py](algo/_shared/pilot/twoteam/run_wp2_league.py):CLI 透传
- [experiments/twoteam/wp3_train.py](experiments/twoteam/wp3_train.py):subprocess cmd 透传

### 微验证(`/tmp/fix_b_microverify.py`)
- 假池 wr=[0.1, 0.5, 0.9]:var_mix=0.0 时 P(mid)=0.326 → var_mix=0.5 时 P(mid)=0.378 ✓
- 假池 wr=[0.9, 0.9, 0.9] + floor=0.05:强制均匀,P=0.336 ≈ 1/3 ✓
- 旧池 API 不破(test_pool_mixed_kinds 仍 PASS)✓

---

## 2. Fix C 实施(entropy gate + self-play 80/20)

### 改动文件
- [algo/_shared/pilot/twoteam/br_trainer.py](algo/_shared/pilot/twoteam/br_trainer.py):
  - `__init__` 加 `entropy_gate_on_kill: bool = False` + `self._kill_appeared = False`
  - `collect_rollout`:首次 `new_kills.sum() > 0.5` 时设 `self._kill_appeared = True`
  - `_entropy_coef`:gate 开启且 kill 未现时保持 max;kill 出现后才按 cosine 衰减
- [algo/_shared/pilot/twoteam/run_wp2_league.py](algo/_shared/pilot/twoteam/run_wp2_league.py):
  - CLI 加 `--entropy-gate-on-kill`、`--self-play-frac`
  - 主循环:20% 概率用 `copy.deepcopy(br_ac)` 包 `ACCommander` 当对手(OpenAI Five 80/20)

### 微验证(`/tmp/fix_c_microverify.py`)
- gate on + kill_appeared=False:coef 在 it=0/50/99 全 = 0.01(max)✓
- kill_appeared=True 后:coef at it=0/50/99 = 0.01/0.0055/0.001(cosine 启动)✓
- gate off(legacy):coef at it=50 = 0.0055(旧行为)✓

### Fix C 第 3 项未装(BC teacher 混合)
spec §3 第 3 项:`bc_pretrain.py` teacher 换 50% BlindClassical + 50% ExtremeCommander。
**未实施**:teacher 混合对 tracker init 问题无直接帮助;留作后续 ablation。

---

## 3. pytest 回归

```
tests/twoteam/  →  109 passed in 500.24s
```

无回归。

---

## 4. Fix B+C 100-iter 训练数据

### 训练参数
```
--iters 100 --n-envs 64 --horizon 300 (1.92M steps)
--shape-kill-bonus 50 --shape-dwell-bonus 1 --shape-exposure-penalty 0
--pfsp-var-mix 0.5 --ema-var-uniform-floor 0.05
--entropy-gate-on-kill --self-play-frac 0.2
--bc-samples 20000 --bc-epochs 8 --blind-teacher
ckpt_dir = checkpoints/blind/wp3_20260718_090802/  (非 /tmp ✓)
耗时 = 140.3 min (vs Fix A 单独 137.5 min,self-play 几乎无 overhead)
```

### 训练曲线
| Iter | Opp | r | entropy | kl | wr_vs_opp | ema_var |
|---|---|---|---|---|---|---|
| 0 | exploit/hard_jam_focus | -0.000 | -1.629 | 0.1253 | 0.50 | 0.000 |
| 10 | extreme/track_agile | +0.003 | -1.591 | 0.0499 | 0.51 | 0.188 |
| 20 | blind_classical | **-0.053** | -1.519 | 0.4409 | 0.35 | 0.130 |
| 30 | blind_classical | **-0.098** | -1.478 | 0.2202 | 0.36 | 0.130 |
| 40 | exploit/jam_spread | -0.188 | -1.441 | 0.0604 | 0.61 | 0.094 |
| 50 | self/iter050 | +0.029 | -1.413 | 0.0643 | 0.51 | 0.083 |
| 60 | extreme/track_agile | -0.165 | -1.341 | 0.0162 | 0.59 | 0.083 |
| **70** | **blind_classical** | **+0.105** | -1.389 | 0.0283 | 0.38 | 0.083 |
| **80** | **blind_classical** | **+0.063** | -1.383 | 0.0797 | 0.35 | 0.082 |
| 90 | self/iter050 | +0.135 | -1.387 | 0.0252 | 0.53 | 0.076 |
| **99** | **blind_classical** | **+0.067** | -1.387 | 0.0326 | 0.24 | 0.073 |

### vs Fix A 单独对比
| Iter | Opp | Fix A r | Fix B+C r | Δ |
|---|---|---|---|---|
| 40 | blind_classical | -0.051 | -0.188 (vs jam_spread) | — |
| 50 | blind_classical | -0.075 | +0.029 (vs self) | — |
| 70 | strong_rule / blind_classical | +1.401 (SR) | **+0.105 (BC)** | BC 上 r 从负翻正 |
| 99 | exploit/jam_spread | -0.020 | **+0.067 (BC)** | BC 上 r 持续正 |

**关键变化**:Fix B+C 训练后期(70-99)vs BlindClassical r 从 Fix A 时的 -0.05/-0.07 升到 +0.06~+0.10,**翻正**。

### 健康指标
- pool ema_var:0.000 → 0.073(没塌到 0,Fix B 工作)
- entropy:-1.629 → -1.387(略升,Fix C gate 让 entropy 不退得太快)
- 但 it=99 仍触发 HEALTH WARN(entropy=-1.387 < 0.3 floor)—— gate 在 kill 出现后释放,但 kill 出现得早(早期就有 vs SR)

---

## 5. Smoke cross-play vs BlindClassical(50 episodes)

| Condition | RL kill | BC kill | Δ | RL survival | RL trace_P |
|---|---|---|---|---|---|
| low_interference (orthogonal) | **0.031 ± 0.174** | 0.938 ± 0.242 | **-0.906** | 0.815 | 285 |
| high_interference (same_channel) | **0.039 ± 0.194** | 0.211 ± 0.408 | **-0.172** | 0.939 | 327 |

### 与历次 smoke 对比
| 跑次 | low Δ | high Δ | 备注 |
|---|---|---|---|
| WP-3 M4 100-iter(no Fix) | -0.92 | -0.13 | baseline |
| WP-3 M4 500-iter(no Fix) | -0.85 | -0.15 | 5× compute 无效 |
| Fix A 100-iter | -0.975 | -0.175 | reward 装对 |
| **Fix B+C 100-iter** | **-0.906** | **-0.172** | **PFSP+entropy+self-play** |

**Smoke kill 几乎不变**(0.025 → 0.031)。Fix B+C 让训练 r 翻正,但 smoke 实际 kill 没改善。

---

## 6. Behavior probe 关键发现

### `wp3_fix_a_behavior_probe.py`(新 ckpt `wp3_20260718_090802/iter_final.pt`)

| Opp | rl_kills | opp_kills | survival | dwell@max | fire | dwell steps | kill |
|---|---|---|---|---|---|---|---|
| BlindClassical | **0.04** | 0.92 | 0.58 | **0.00** | 200/200 | **0/200** | False |
| **StrongRule** | **0.95** | 1.00 | 0.65 | **1.02** | 200/200 | **180/200** | **True** |
| extreme/balanced | 0.00 | 0.01 | 1.29 | 0.00 | 200/200 | 0/200 | False |
| extreme/pure_track | 0.00 | 0.01 | 1.19 | 0.00 | 200/200 | 0/200 | False |
| extreme/pure_jam | 0.00 | 0.00 | 1.23 | 0.00 | 200/200 | 0/200 | False |

### vs Fix A 单独对比
| Opp | Fix A kill | Fix A dwell@max | **Fix B+C kill** | **Fix B+C dwell@max** | Δ kill |
|---|---|---|---|---|---|
| BlindClassical | 0.03 | 0.00 | **0.04** | **0.00** | ≈ 0 |
| **StrongRule** | **0.37** | **0.50** | **0.95** | **1.02** | **+0.58** |

**核心结论**:
- ✅ **Fix B+C 在 StrongRule 上显著有效**:kill 0.37 → 0.95(+157%),dwell@max 0.50 → 1.02(2×)
- ❌ **Fix B+C 在 BlindClassical 上无效**:dwell@max 仍 0.00,kill 仍 0.04
- ❌ **vs extreme/* 也无效**:所有 passive 对手 dwell=0

---

## 7. 真根因 = tracker 在 BC 上 init 失败(物理瓶颈)

### 排除 reward 接线错(spec §5 验证)
- ✅ dwell_frac 真进 rew_lt(microverify 已验证:16 envs × 300 step,16/16 envs dwell_frac>0 时 rew_dwell>0)
- ✅ kill_bonus 真触发(microverify:31 kills × 50 = 1550 bonus exact)
- ✅ reward 接线对

### 物理瓶颈定位
1. **vs BC**:两边都 emit=2 都有 16 detection,但 RL 的 IMM-PDAF tracker 全程 0/2 init
   - BC 的 detection 位置 step-to-step 变化大(BC 机动 + 虚警)→ PDAF M-of-N 关联断裂
2. **vs SR**:StrongRule god-view 给稳定 RF → tracker 4/2 init(3 step 后)→ beam 部分指向 enemy
3. tracker 不 init → `tracker_x` 保持 (0,0) → belief 距 enemy 1919m vs laser_hit_radius=50m(差 38×)
4. → beam_direction_head 没有有效输入信号 → 学到 garbage(atan2 输出从 ±π 随机漂移)
5. → 激光完全 miss → radar_E=0 → 没 dwell → 没 kill

### 关键 invariant
- env reward 接线正确(Fix A 验证)
- Fix B+C 在能 init tracker 的对手上有效(SR 数据证明)
- **vs BC tracker init 物理性失败 = 真正瓶颈**

---

## 8. 修正:不是 "vs BC 真 floor",是 "BC 杀不动规避目标"(对称 floor)

### 8.1 上游用户 2026-07-18 反驳(原 §8 判断过早)

原 §8 判 "vs BC 真 floor" 的结论被上游用户推翻,理由:
> vs BC 的 kill=0.04 不是物理 floor,是稀疏墙上移了一层。

代码硬理由(逐行核实):
- [twoteam_env.py:699-700](env/gpu/twoteam/twoteam_env.py#L699-L700): `lsr_track_ok = (trace_P < tau_track) & tracker_initialized`
- [twoteam_env.py:718](env/gpu/twoteam/twoteam_env.py#L718): `accum_mask = lsr_track_ok & hit_mask & emitting` → `radar_E` 累积被门控
- [twoteam_env.py:723, 782](env/gpu/twoteam/twoteam_env.py#L723): `track_bonus` 也门控 `tracker_initialized`

→ tracker 不 init 时,`radar_E` 恒 0 → Fix A 的 `dwell_bonus`、`kill_bonus`、原 `track_bonus` 三个 shaping **同时归零**。**RL 对"获取航迹"这一步完全没有奖励梯度**,跟最初 kill=0 的病同构,只是稀疏墙从 "dwell" 上移到 "track acquisition"。

### 8.2 决定性对照:BC vs BC(`/tmp/bc_vs_bc.py`,2026-07-18)

| 配置 | t0 kill | t1 kill | tracker_init | radar_E |
|---|---|---|---|---|
| BC vs BC (orthogonal) | **0.000** | **0.000** | 0.45/2 | 0.063 |
| BC vs BC (same_channel) | **0.000** | **0.000** | 0.00/2 | 0.000 |
| (对比) RL vs BC | 0.031 | **0.938** | — | — |

**关键判决**:BC 杀 RL=0.83-0.90 是因为 RL 不规避(盲态 forward),BC 锁定容易;**BC 杀不动 BC(规避目标)**。

→ 这就是用户 stop criterion (a):**BC vs BC 也杀不动规避目标 → 对称 floor**。
→ 不是 "RL 输给 BC",是 "经典 tracker 杀不动规避对手"。
→ Smoke cross-play "BC kill RL=0.938" 不能证明 BC 能跟踪规避目标,因为 RL 不规避。

### 8.3 架构核实:RL 不能绕过经典 tracker

[twoteam_env.py:417-426](env/gpu/twoteam/twoteam_env.py#L417-L426) obs 包含 `tracker_x`(IMM-PDAF 输出)、`trace_P`、`tracker_initialized`。同时 RL 也喂原始 detection(DeepSets)。

[twoteam_env.py:699-718](env/gpu/twoteam/twoteam_env.py#L699-L718) **激光命中门控** `tracker_x`(`lsr_belief = tracker_x[:, t, laser_slot]` → 比 laser_hit_radius_m)。

→ 即使 RL 拿原始 detection,**物理命中门控 tracker_x**。tracker 不 init 时 RL 物理性 miss,不能绕过。

→ Fix D 必须**让 tracker init**,不是绕过它。

### 8.4 Thesis reframe(用户 2026-07-18)

现在的瓶颈 = 规避对手 + 杂波/干扰下的**主动感知 / 航迹获取**——这正是 RL 立命的 regime(认知雷达 / active-perception POMDP,Kreucher-Hero;经典波束调度做不好的地方)。

若 RL 学会在经典跟不住规避目标的地方主动搜索、获取航迹、再击杀 → 这就是论文最强的点。**这不是 floor,这是被正确隔离出来的真正研究问题。**

### 8.5 修正后的停止判据(用户 2026-07-18)

floor 只有在以下两者之一成立时才是真 floor:
- (a) BC vs BC 也杀不动规避目标 → **已成立**,但这是对称 floor,不是 RL 劣势
- (b) RL 装了正确主动感知 reward + curriculum 后仍无法获取航迹 → 未测试

**当前状态**:(a) 成立但叙事需修正;(b) 未测试 = Fix D 待做。

---

## 9. 下一步选项

### 当前判断(修正后)
"vs BC 真 floor" 判断**过早**,被上游用户推翻。当前状态:
- (a) BC vs BC 也 0 kill(对称 floor)— 已确认
- (b) RL 装主动感知 reward + curriculum 后能否获取航迹 — 未测

→ Fix D 是下一步,**不是 "救不活",是被正确隔离的研究问题**。

### Fix D 设计草案(active perception shaping + reverse curriculum)

**Fix D1 — track-init shaping**(让 RL 因主动获取航迹而得奖):
- `shape_init_bonus`:每次新 `tracker_initialized[laser_slot]` 从 False→True 时给 bonus
- `shape_belief_bonus` = -Δ‖tracker_x - 最近 detection‖(belief 收敛奖励)
- `shape_detect_in_beam_bonus`:检测落在 beam_direction 半功率波束内时给 bonus

**Fix D2 — reverse curriculum**(Florensa CoRL 2017):
- 部分 episode 从"航迹已 init"起步,逐步退火到"全 cold start"
- 或从"弱规避 BC"→"标准 BC"逐步加强对手
- 直接打破 "track 不 init ⟺ beam 学不会" 的鸡生蛋

**Fix D3 — 架构侧(若 D1/D2 不够)**:
- RL actor 加 "tracker_init mask" 输入(让策略知道哪些 slot 没 init)
- 或:beam_direction_head 加 explicit sweeping fallback(no init → 周期扫描)

### 决定性实验顺序
1. Fix D1(track-init shaping)单独装 + 100-iter → smoke + probe
2. 若仍 0 kill → 加 Fix D2(curriculum)
3. 若仍 0 → Fix D3(架构)
4. 三层做完仍 0 kill → **带证据的真 floor**(stop criterion (b))

### 跳过项
- ~~A. 接受真 floor,写 WP-4 报告~~(过早,不诚实)
- ~~B. Fix D beam_direction fallback(架构)~~(在 D1/D2 shaping 之前跳过)
- ~~C. Fix E obs tracker_init mask~~(并入 Fix D3)
- ~~D. 调 IMM-PDAF init 阈值(env 侧)~~(改 env 物理不是本研究路线)

---

## 10. 交付物

### Checkpoints(非 /tmp ✓)
- `checkpoints/blind/wp3_20260718_090802/iter_final.pt`
- `checkpoints/blind/wp3_20260718_090802/iter100.pt`
- `checkpoints/blind/wp3_20260718_090802/pool_metadata.json`
- `checkpoints/blind/wp3_20260718_090802/wp3_train_log.txt`

### 训练 log
- `/tmp/wp3_fixbc_100iter.log`(140 min 完整 stdout)

### 报告
- 本文件:`experiments/twoteam/WP3_1_FIXBC_REPORT.md`
- Smoke:`experiments/twoteam/wp3_smoke_crossplay_fixbc_report.md`

### 代码改动(branch `twoteam/bc-ppo`)
- `algo/_shared/pilot/twoteam/opponent_pool.py`(Fix B)
- `algo/_shared/pilot/twoteam/br_trainer.py`(Fix C entropy gate)
- `algo/_shared/pilot/twoteam/run_wp2_league.py`(Fix B+C CLI + self-play 80/20)
- `experiments/twoteam/wp3_train.py`(subprocess 透传)
- `tests/twoteam/test_wp2_smoke.py`(回归兼容)

### 微验证脚本
- `/tmp/fix_b_microverify.py`(PFSP f_var + ema_var floor)
- `/tmp/fix_c_microverify.py`(entropy gate kill-appeared)
- `experiments/twoteam/wp3_fix_a_microverify.py`(Fix A 3 asserts)
- `experiments/twoteam/wp3_fix_a_behavior_probe.py`(behavior 详查)

---

## 附:WP-3 跑次历史

| # | 配置 | 训练时长 | smoke Δ_low | smoke Δ_high | 备注 |
|---|---|---|---|---|---|
| 1 | M4 100-iter no-fix | 25 min | -0.92 | -0.13 | baseline |
| 2 | M4 500-iter no-fix | 137 min | -0.85 | -0.15 | 5× compute 无效 |
| 3 | M4 1000-iter no-fix | 280 min | -0.83 | -0.15 | 10× 无效 |
| 4 | Fix A 100-iter | 137 min | -0.975 | -0.175 | reward 装对;r=+1.4 at SR |
| **5** | **Fix B+C 100-iter** | **140 min** | **-0.906** | **-0.172** | **SR kill 0.37→0.95;BC 仍物理瓶颈** |
