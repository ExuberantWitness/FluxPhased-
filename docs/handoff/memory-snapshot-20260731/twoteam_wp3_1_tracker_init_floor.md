---
name: twoteam-wp3-1-tracker-init-floor
description: "WP-3.1 Fix A+B+C 后 vs BC kill=0 真因 = 稀疏墙上移到 tracker init (dwell+kill+track 三个 shaping 同时归零); 不是物理 floor, BC vs BC 也 0 kill = 对称 floor; 真研究问题 = active perception"
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

2026-07-18: WP-3.1 Fix A+B+C 装好 + 100-iter + smoke + probe + 报告 + commit/push (0202183)。**原"vs BC 真 floor"判断被上游用户推翻**(过早),修正如下:

**真根因(逐行代码核实)**:
- [twoteam_env.py:699-700](env/gpu/twoteam/twoteam_env.py#L699-L700): `lsr_track_ok = (trace_P<tau_track) & tracker_initialized`
- [twoteam_env.py:718](env/gpu/twoteam/twoteam_env.py#L718): `accum_mask = lsr_track_ok & hit_mask & emitting` → radar_E 累积门控
- [twoteam_env.py:723, 782](env/gpu/twoteam/twoteam_env.py#L723): track_bonus 也门控 tracker_initialized
- tracker 不 init 时 → radar_E 恒 0 → Fix A 的 dwell_bonus + kill_bonus + 原 track_bonus **三个 shaping 同时归零**
- = RL 对"获取航迹"零梯度,跟最初 kill=0 的病同构,只是稀疏墙从 "dwell" 上移到 "track acquisition"

**架构核实**:[twoteam_env.py:417-426](env/gpu/twoteam/twoteam_env.py#L417-L426) obs 含 tracker_x(IMM-PDAF 输出); L699-718 激光命中门控 tracker_x。即使 RL 拿原始 detection(DeepSets),物理命中门控 tracker_x → RL 不能绕过经典 tracker → Fix D 必须让 tracker init,不是绕过。

**决定性对照(BC vs BC, 2026-07-18 `/tmp/bc_vs_bc.py`)**:
- orthogonal: 双方 kill=0.000, tracker_init 0.45/2, radar_E 0.063
- same_channel: 双方 kill=0.000, tracker_init 0.00/2, radar_E 0.000
- 对比 RL vs BC: BC kill 0.938(RL 不规避)
- → **stop criterion (a) 成立**:BC 杀不动 BC(规避目标)
- → 不是 "RL 输给 BC",是 "经典 tracker 杀不动规避对手"
- → BC kill RL 0.83-0.90 不能证明 BC 能跟规避目标(RL 凑着往前送)

**Thesis reframe(上游用户 2026-07-18)**:瓶颈 = 规避对手 + 杂波/干扰下的主动感知 / 航迹获取 = active-perception POMDP (Kreucher-Hero; 认知雷达)。**这不是 floor,这是被正确隔离出来的真正研究问题**——RL 立命的 regime。

**修正停止判据**:
- (a) BC vs BC 也杀不动规避目标 → **已成立**(对称 floor,叙事需修正,非 RL 劣势)
- (b) RL 装主动感知 reward + curriculum 后仍无法获取航迹 → 未测试 = Fix D 待做
- 只有 (b) 成立才是 "RL 学不会" 的真 floor

**Why**:之前 [WP3_FINAL_SUMMARY](WP3_FINAL_SUMMARY.md) "IET floor" 推翻 → Fix A 通过(r=+1.4 at SR)→ 我又过早下 "vs BC 真 floor" → 上游用户又推翻 → 真问题是被稀疏墙掩盖的 tracker-init 奖励缺口。**连续两次过早下 floor 教训**:每次"RL 学不会"判断前,必须先问"是不是另一层稀疏墙"。

**How to apply**:
1. 若用户问 "WP-3 还能再救吗":Fix D(active perception shaping + reverse curriculum)是下一步;Fix A/B/C 已穷尽 reward/PFSP/entropy 三层。不要再调 shape_kill_bonus magnitude(vs BC new_kills=0)。
2. 若用户提到 "RL 学不会 / floor":提醒"先看稀疏墙在哪层",不要直接接受 floor 判断。当前 known walls: kill(已修) → dwell(已修) → track-init(待修) → ?
3. BC vs BC 对照是 "BC 杀 RL 高 kill 证明力" 的硬反驳,任何时候报 "BC 强" 都要问 "vs 规避目标?"
4. 论文叙事:RL 的卖点不是 kill count(BC 也能在 easy 目标上 kill),是主动感知 + 在经典跟不住的地方获取航迹。

**完整数据**:`experiments/twoteam/WP3_1_FIXBC_REPORT.md` §8 修正版 + §9 Fix D 草案。`checkpoints/blind/wp3_20260718_090802/iter_final.pt`。相关:[[twoteam-wp3-production-smoke-fail]]、[[twoteam-wp3-m0123-pass]]。
