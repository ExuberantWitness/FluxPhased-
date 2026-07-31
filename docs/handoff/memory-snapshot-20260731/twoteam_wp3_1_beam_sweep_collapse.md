---
name: twoteam-wp3-1-beam-sweep-collapse
description: "WP-3.1 真根因 (2026-07-19, 4 跑后确认): PPO 把 BC 教的 deterministic sweep 压缩成 fixed bd mode; RL cold-start 时 beam_direction 不扫, 永远到不了 enemy bearing HPBW, tracker 永不 init. 不是 tracker_init floor, 是 beam sweep 被 entropy anneal 杀死."
metadata:
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

2026-07-19: 4 个 100-iter 跑(Fix D / Fading ε=0.05 / Run A ε=0.005 / Run B det-reset)全 FAIL smoke(kill ≤ 0.047 vs BC 0.91)。深度 probe 找到真 root cause,推翻 [[twoteam-wp3-1-tracker-init-floor]] 的"稀疏墙上移到 tracker init"框架。

**真 root cause(深度诊断已确认)**:
- BC 的 beam_direction 策略 [twoteam_blind_classical.py:114-146](algo/_shared/baselines/twoteam_blind_classical.py#L114):
  - tracker 未 init: deterministic 全方位 sweep,步长 0.0375 rad (HPBW/2,2× oversample)
  - 168 步扫一圈必然命中 enemy bearing → tracker init → 切到 track_az
- RL 的 beam_direction 策略(probe env-0, seed=42 实测):
  - step 0-49: bd_r0 ≈ -60° 固定,**根本不扫**
  - enemy bearing = -93°,HPBW=4.3° → 偏 30°+ 远超 HPBW → 永远无检测 → tracker 永不 init
- **PPO entropy anneal (0.01→0.001) 把 BC 教的高熵 sweep 压缩成 deterministic mode** = 经典 PPO+BC 陷阱

**Why curriculum 改动救不了**:
- warm_start / fading ε / det reset 都在改 tracker 初始条件
- 它们假设 "tracker init 后 RL 能保持"
- 但 RL cold-start 时永远到不了 init 状态(beam 不扫)
- warm_start 可能让情况更坏:RL 学到 "tracker init 是免费的",更不学 search

**关键 probe 数据(Run A,唯一一次半破墙)**:
| step | n_init | trace_P | beam_dir | enemy_bearing | radar_E_et |
|---|---|---|---|---|---|
| 0-49 | 0 | 9→49 | ~-60° | -93° | 0.000 |
| 50 | **2** | **2.04** | -97° | -93° | **0.200** |
| 60 | 2 | 4.94 | -105° | -93° | 0.903 |
| 90 | 2 | 10.79 | -44° | -93° | 1.167 |
Run A env-0 在 step 50 偶然扫到 enemy → tracker init → 但 90 步后 beam 漂走 → 失锁。

**4 跑对比**:
| 配置 | 低干扰 kill | 高干扰 kill |
|---|---|---|
| Fix D (anneal p only) | 0.016 | 0.008 |
| Fading ε=0.05 | 0.000 (regression!) | 0.000 |
| Run A ε=0.005 | **0.047** (best) | 0.000 |
| Run B det reset N=3 | 0.031 | 0.016 |

**Why**:之前 [[twoteam-wp3-1-tracker-init-floor]] 框架"tracker init 是稀疏墙"被实证推翻——稀疏墙其实是 **beam_direction 的 search 行为被 PPO 杀死**。"tracker init 失败"是症状,不是因。连续 3 次:①WP3_FINAL_SUMMARY "IET floor" ②WP3_1_FIXBC "vs BC floor" ③WP3_1_tracker_init_floor "tracker-init floor" → 都被更深一层的诊断推翻。**下次再下 "RL 学不会 X" 判断前,先问"是不是 X 上游还有一层稀疏墙"**。

**Fix E 候选(用户在讨论中)**:
1. **Freeze BC sweep head**: PPO 不动 beam_direction_head;保住 search。最简单,最直接。
2. **Search-when-cold reward shaping**: tracker 未 init 时 reward ∝ beam_dir delta。符合 Kreucher-Hero active-perception framing。
3. **BC-as-search override**: 训练时未 init 状态下 beam_direction 用 BC output。ECPO-style。
4. **接受墙**: 写 WP-4 negative result case study + diagnosis。

**How to apply**:
1. 用户问"RL 为何杀不动 BC"时:答"PPO 把 BC sweep 杀了,RL cold-start beam 不扫 → tracker 永不 init"。不是"tracker init 太难"。
2. 若选 Fix E.A (freeze): 加 `freeze_beam_direction: bool` flag,opt param group 排除 beam_direction_head。
3. 若选 Fix E.B (search shaping): trainer 加 prev_bd 缓冲,新 shaping = `|bd_now - bd_prev|` × not_init mask。
4. Curriculum 改动方向已穷尽,**不要再调 curriculum 参数**(eps / N / anneal_iters)。
5. 论文叙事:RL 的瓶颈是 PPO+BC 在 active-perception 上的结构性限制,不是"RL 学不会"——是"标准 PPO+BC pipeline 不够,需要保住 BC 的 search 行为"。

**完整数据**:`experiments/twoteam/WP3_1_FIXE_ROOTCAUSE_REPORT.md`(2026-07-19)。`checkpoints/blind/wp3_20260718_{131822,202825,234749}/` + `wp3_20260719_074048/`。相关:[[twoteam-wp3-1-tracker-init-floor]](被本文推翻)、[[twoteam-wp3-production-smoke-fail]]、[[twoteam-wp3-m0123-pass]]。
