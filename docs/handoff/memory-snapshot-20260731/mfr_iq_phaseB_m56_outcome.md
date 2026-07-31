---
name: mfr-iq-phaseb-m56-outcome
description: "2026-07-27 Phase B M5+M6 outcome — env extension done, G1' PASSES (3 jammers break near-Nash 0.090→0.21-0.26); antijam trigger 阈值校准关键教训 (30dB 反伤, 65dB 才合理); M7-M8 待做."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c76cde4-1c23-4df6-94f5-ee29ee81afbc
---

MFR-IQ Phase B 第一段交付完成 (M5 env 扩展 + M6 rule baseline G1' gate). 用户在 2026-07-25 终裁路线 "复杂化→新算法" 后, 选了 MAXIMAL 设计 (3 policies × 2 carriers × 2 counter-levers + Phase 1+2 全交付).

**Why:** Phase A G2 负结果证明 pure-scheduling fully-observable queue 上 MAPPO 打不过 rule near-Nash. Phase B 引入 IQ 物理 (JNR/σ/tracker) + 时间动态 (reactive/blink) + EW cue, 试图打破 near-Nash floor 并给 PPO 新杠杆 (active perception).

**How to apply:** 后续 M7 (learning jammer) / M8 (league) 工作直接 read this as checkpoint. 若 G2'a (learning beats scripted) 或 G2'b (MAPPO radar beats rule) 也 negative, 整个复杂化路线复刻 Phase A 结局 — 那时回头评估是否换问题.

## 关键结果

- **M5 env 扩展**: 19/19 测试绿 (T14-T19 + Phase A 13 回归). 默认 ctor bit-identical 与 Phase A.
- **M6 G1' PASS**: 3 jammer × rule/random, 30W:
  - rule drop ∈ [0.21, 0.26] across {noise, reactive, blink}, 全部 ≥ 0.19 gate (Phase A baseline 0.090, 抬升 2-3×).
  - rule 击败 random ≥ 22 pp 每种 jammer (gate ≥ 10 pp).
  - Per-type succeeded: track 类 (0,1,2) 全归零 under noise (σ_inflation 致 trace_P 不收敛); search/jam 类 60-77%.

## 工程教训 (写下来防止重蹈)

1. **cue admissions 必须算入 n_arrived**: 否则 drop_ratio = dropped/poisson_arrived 可超 1, 且 cue 多时 metric 完全失真. 修在 task_layer.cue_jammers().
2. **antijam trigger 阈值不是越低越好**: 30 dB 默认 (我一开始用的) 让 rule drop 从 0.20 跳到 0.49 — 25% 孔径损失压垮 σ 改善. 65 dB 才合理. 不要把 "active defense 总是有用" 当默认.
3. **last_jnr_at_sub 是 linear 不是 dB**: sum of jnr_full (linear). 之前 rule_scheduler 误把 linear 当 dB, 再 10^(x/10), 触发条件永远 true. 任何新代码读 env.last_jnr_at_sub 都要注意单位.
4. **off-board virtual slot indexing**: target_slot = N_max + j. 任何 gather/scatter on pool.alive[N], matched[N], trace_p[N_trk] 都要 clamp + mask tgt_in_pool. rule_scheduler 第一版没做, 直接 CUDA assert.
5. **T19 "Phase A bit-identical" 不该 assert 50-step rollout equality**: CUDA reduction 非确定性, 50 步后 obs 漂 ~1e-4 (atomicAdd 顺序). 用 allclose(atol=1e-3) + reward equal, 或只查 reset() 复现性.

## 入口与产物

- 入口: `algo/_shared/pilot/mfr/run_stage_b.py --algo {rule,random} --jammer {noise,reactive,blink,none} --enable-antijam --jammer-pow-w 30 --seed N --out ...`
- 报告: `experiments/mfr_phaseB/REPORT.md` (含 G1' table + antijam 阈值对照 + per-type 完成 signature).
- 测试: `tests/mfr/test_mfr_jammer.py` (T14-T19).
- 相关 memory: [[mfr_iq_m03_outcome]] (Phase A G1/G2 结局), [[twoteam_mfr_pivot_20260725]] (路线终裁).
