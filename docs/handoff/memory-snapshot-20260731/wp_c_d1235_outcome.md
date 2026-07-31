---
name: wp-c-d1235-outcome
description: "WP-C D1-D5 结果 — D1 PASS (动态敌方真比 fixed 难), D3 IET FLOOR (RL 0/3 crown @ 60-iter), D5 发现 PPO 确实打破 BC lock 但未形成可用策略"
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

WP-C D1-D5 完成 (2026-07-16):

**D1 PASS 3/3** — AdaptiveSpectrumJammer (channel-split follower) 在所有 jam_frac 点都比 TrueFixedJammer 难:1e-5 时 trace_P 3.63×、1e-4 时 kills 2→0、1e-3 时 kills 2→0。R2 0/3 fail 的根因是 build_fixed_enemy_action 用 env.radar_freq_hz 推 channel_select,在 mirror orth 下 enemy 自动落在 victim 的 [ch0,ch1],apple-to-apple 与 reactive 没差异。修法:TrueFixed 常数信道 + AdaptiveSpectrum split-follow。

**D3 IET FLOOR 0/3 crown** — RL (60-iter PPO vs AdaptiveSpectrum) 在所有 jam_frac 下都败给 StrongRule+orth:
  - 1e-5: RL trace_P=7.4 vs base 0.7 (10× worse), kills 1 vs 2
  - 1e-4: RL 23.2 vs base 20.4
  - 1e-3: RL 103.5 vs base 94.5
按用户 "no tuning games" 规则,诚实收 IET,不调参硬凑。D4 生产训练 NOT triggered。

**D5 发现 D3 chan_Δ/s=0 是 measurement bug** — `deterministic=True` 用 argmax 掩盖了探索。真实情况:
  - BC prior: entropy 0.003 (完全锁死 ch0/ch1), 3/2000 步换信道
  - PPO 后: entropy 0.65 (探索中), 1166/2000 步换信道, 与敌方重叠率从 0.999 降到 0.696
PPO 确实打破了 BC lock,但探索未形成 argmax 可利用的稳定策略。

**Why:** WP-C thesis "学习式动态协同 vs 胜任固定规则" 的诚实结论 = 在完美协调的 split-following 敌方下,StrongRule+orth 静态分配已经够好;RL 探索信号存在但 60-iter PPO 不足以形成 crown。Production 5e7 可能突破,但用户严格 gating D4 on D3 crown。

**How to apply:** 未来类似 MARL 协同任务,① 先建 truly dynamic enemy (split-follow),别让 "fixed" 偷偷 channel-follow;② RL eval 用 deterministic=True 是标准,但分析时一定补 stochastic 采样看 head 真实行为;③ 60-iter sanity 不足以判 crown,但用户严格 gating 时尊重规则。

相关:[[taes_wp_b_classical_collapse]] 是 WP-C 的前提;[[twoteam_multifunction_pivot]] 是 framework pivot。

关键文件:
- `experiments/twoteam/wp_c_d1_dynamic_enemy_verify.py` — D1 验证
- `experiments/twoteam/wp_c_d3_killer_contrast.py` — D3 训练+eval
- `experiments/twoteam/wp_c_d5_channel_analysis.py` — D5 head 分析
- `algo/_shared/baselines/adaptive_spectrum_jammer.py` — TrueFixed + AdaptiveSpectrum
- 提交链: 6f91f8c (infra) → 176c0bd (D1) → 4105e3e (D3) → 491f0f5 (D5)
