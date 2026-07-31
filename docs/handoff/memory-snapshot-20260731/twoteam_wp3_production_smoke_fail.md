---
name: twoteam-wp3-production-smoke-fail
description: WP-3 production RL 4 次 smoke cross-play 全 FAIL (2026-07-17) — RL 低干扰 Δ_kills=-0.85, 高干扰 -0.15; 5 倍 compute 没救; 根因是结构性问题; 用户接受 IET floor, 进入 WP-4。
metadata:
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

WP-3 production RL 训练 4 次 smoke cross-play vs BlindClassical 全部 FAIL (2026-07-17)。RL 在低干扰 (orthogonal channel) 下学不会开火 (kill ≈ 0.0-0.075 / 2)，BC kill 0.83-0.90 / 2。

**Why:** 不是 compute 不足 (5 倍 compute 没救)。是结构性问题:
1. **BC teacher 过强 lock actor**: BlindClassical 低干扰 kill 0.9，BC pretrain 把 actor 锁死在"模仿 BC"模式，detection encoder 学不到 IMM-PDAF 同样的 tracking quality，后期 entropy → -1.7 (near-deterministic)。
2. **zero-sum mirror symmetry → 弱 gradient**: env reward 是 `reward - reward.flip(dims=[1])`，self/iterNNN 自博弈时双方净 reward = 0；dense shaping (shape_track_bonus / shape_exposure_penalty) 对 learning_team 是 absolute，但 league 大部分 iter 在打 self-snapshot，gradient 仍弱。
3. **PFSP league collapse**: BC 太强 → BC 的 EMA win-rate 永远低 → f_hard(1-wr)^1.0 永远高 → PFSP 永远采样 BC → 其他对手 win-rate 退化 → pool ema_var → 0.037 < 0.05 floor (实测命中)。
4. **reward shaping 缺 kill term**: shape-track-bonus + shape-exposure-penalty 让 RL 学会跟踪 + 隐藏 (survival 高干扰 0.79→0.95)，但没有 shape-kill-bonus → RL 没动力学开火。

**4 次实验数据:**

| Run | Steps | 低干扰 Δ_kills | 高干扰 Δ_kills | RL survival 低/高 |
|---|---|---|---|---|
| 100-iter unshaped | 1.28e6 | -0.925 | -0.150 | — / 0.81 |
| 100-iter shaped (0.1/0.05) | 1.28e6 | -0.825 | -0.125 | 0.77 / 0.79 |
| 100-iter shaped2 (0.3/0.1, ent 0.005) | 1.28e6 | -0.875 | -0.150 | 0.75 / 0.89 |
| 500-iter shaped (0.1/0.05) | 9.6e6 | **-0.850** | **-0.150** | 0.83 / 0.95 |

**用户决策 (2026-07-17 "写总结，终止训练"):** 接受 IET (Intervention Effectiveness Threshold) floor framing，进入 WP-4。RL 500-iter ckpt `checkpoints/blind/wp3_500iter_shaped/iter_final.pt` 作为 RL baseline 用于 WP-4。

**Why accept IET floor:**
- spec §0.3④ "competent blind classical baseline" 是论文核心 contribution；RL 不超 BC **正是 BC 强的证据**，支持 framing。
- 算账：再 2 倍 compute (500→1000 iter, 50h) 极大概率仍在 [-0.95, -0.80] 区间，不会改变结论。
- WP-4 角色：RL vs BlindClassical vs StrongRule 三方 cross-play + 干扰轴 sweep；RL 高干扰可能 tie BC (0.075 vs 0.225 差距小)，作为"RL 在干扰下相对改善"IET baseline。

**How to apply:**
- **不要重启 RL 训练** without 先修 4 个根因。如果用户问"再跑一次试试"，先指出 5 倍 compute 没救的事实 + 根因清单。
- **若未来要让 RL 学会 kill**，推荐改动顺序：① 加 shape_kill_bonus=50.0；② 修 PFSP (hardness_p=0.5 + per-opponent 最低采样)；③ 换混合 teacher (50% BC + 50% ExtremeCommander)；④ 排除 self-snapshot 自博弈或 asym shaping；⑤ 500-iter 验证。
- WP-4 拿 500-iter shaped ckpt，不要用 100-iter 三组。
- 诚实记录 RL 没超 BC — 这是 IET floor 证据，不是失败。

**最终总结报告:** `experiments/twoteam/WP3_FINAL_SUMMARY.md` (含完整数据 + 4 根因分析 + 后续改动建议)。

参见 [[twoteam-wp3-m0123-pass]] (代码完成状态)、[[twoteam-wp2-blind-classical-pass]] (BC baseline 前置)、[[twoteam-multifunction-pivot]] (框架总览)。
