---
name: twoteam-wp3-2-phaseA-fail
description: "WP-3.2 Phase A 失败结果 (2026-07-19~20): 4 跑 (A1+A2+A3+Run4 Beta anchor) 全 FAIL gate. Run 4 probe 揭示 root cause 转为 'beam sweeps but too fast' (-111→+13° 过 enemy 用 2 步, dwell 不够). 转 Run 5 freeze beam_direction_head 锁 BC sweep pattern."
metadata:
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

2026-07-19~20: WP-3.2 Phase A 4 跑全 FAIL gate (kill ≥ 0.5 vs BC)。Probe Run 4 ckpt 否定"beam 不扫"诊断,真因是"beam 扫太快"。

**4 跑 crossplay 结果**:

| Run | 配置 | low-int kill | high-int kill | 训练 ent |
|-----|------|--------------|---------------|---------|
| WP-3.1 Fix D baseline | shape+curriculum | 0.016 / BC 0.91 | 0.008 / BC 0.21 | -1.6→-1.7 |
| **Run 1 (A1 only)** | + BC KL 0.3 (joint lp) | **0.000** / BC 0.891 | **0.000** / BC 0.250 | -1.7 |
| **Run 2 (A1+A2)** | + coord 0.5 | **0.000** / BC 0.875 | **0.016** / BC 0.188 | -1.7 |
| **Run 3 (A1+A2+A3)** | + enemy sched (warm_start) | **0.000** / BC 0.906 | **0.000** / BC 0.156 | -1.7 |
| **Run 4 (A1+A2 + Beta anchor)** | bc_kl_coef_beam=1.0 | **0.000** / BC 0.922 | **0.000** / BC 0.250 | -1.55 (好一点) |

**Probe Run 4 (env-0, vs BC orthogonal, 60 步) — 颠覆性发现**:
```
step 0:    bd_r0 = +61.8° (开始位置)
step 1-12: bd_r0 -111 → -98 → -75° (扫得很快, ~5°/步)
step 13:   bd_r0 = -94.9° (最近 enemy -93.2°, beam_err = -1.7°)
step 14:   bd_r0 = -90.8°, beam_err = +2.5°
step 15:   bd_r0 = -86.1°, beam_err = +7.2°  ← 已超出 HPBW 4.3°
step 20-55: 继续扫到 +13° 后停下
全程: n_init=0, trace_P=2.04 固定, radar_E_et=0, 0 kills
```

**Root cause(更新)**:
1. **不是"beam 不扫"**: Run 4 Beta anchor 成功保住 sweep 能力, beam 在 60 步内从 -111° 扫到 +13°。
2. **真因: sweep 太快 + pattern 不对**: beam 在 enemy bearing (-93°) 只停留 2 步 (err 1.7°→2.5°),不够 dwell time → 0 detection → tracker 永不 init → 0 kill。
3. **BC sweep pattern 工作**: 同环境 BC 拿 0.91 kill, 证明 BC 的 sweep pattern (slower + dwell + 多 bearing 停留) 能 detect。
4. **Joint BC KL (Run 1-3) 太弱**: buffer 已坍缩, 信号传不回。Per-head Beta anchor (Run 4) 稍强但仍不够 — 保住 sweep 能力但学不到 pattern。

**Phase A 实施 (5 个 code 改动, ~85 LOC, pytest 109/109 不回归, microverify 12/12 PASS)**:
- A1: BC KL regularizer `[br_trainer.py:579-583]` bc_kl_coef × (lp_bc - lp_new).mean()
- A2: Coordination shaping `[br_trainer.py:423-429]` shape_coordination_bonus × (1-cos_overlap)
- A3: Enemy schedule `[br_trainer.py:238-262]` + run_wp2_league.py stationary/bc swap
- A4: Pool tier filter `[opponent_pool.py:108-141]` sample_pfsp(tier_min_iter=N)
- Run 4: Per-head Beta L2 anchor `[br_trainer.py:594-613]`, CLI `--bc-kl-coef-beam`
- **Run 5: Freeze beam_direction_head `[br_trainer.py:151,230-241]`, CLI `--freeze-beam-head`** (NEW 2026-07-20)

**Run 5 (freeze beam head, 2026-07-20 07:03 启动)**:
- 直接 freeze AC.beam_direction_head.parameters() (requires_grad=False) + 从 optimizer 排除
- AC iter 0 == BC (after pretrain), 所以冻结即锁 BC sweep pattern forever
- 配置: `--freeze-beam-head --curriculum-p-start 0.0` (去掉 curriculum 依赖, 纯 cold-start)
- 不用 bc_kl_coef / bc_kl_coef_beam (冗余, head 不动)
- Microverify: frozen head grad=None, task_alloc 仍学 (grad=0.19), 12/12 PASS
- Run 5 PID 1236626, ckpt `checkpoints/blind/wp3_phaseA_run5_freezebeam_20260720_070345/`

**How to apply**:
1. 用户问"BC KL / Beta anchor 没破 collapse": 答"joint KL 弱在 buffer 已坍缩; Beta anchor 保住 sweep 但 pattern 错; freeze beam_direction_head (Run 5) 锁 BC pattern 是最强干预"。
2. 论文 Phase A ablation: 4 跑全 FAIL 是关键 negative result。Probe Run 4 的"beam sweeps too fast"是关键洞察 — RL 学到 sweep 能力但学不到 dwell pattern。
3. 若 Run 5 PASS (kill ≥ 0.5 vs BC): 证明 BC sweep 是 essential, RL 的任务是学 *其他* 决策 (laser aim, channel hop, freq select, task alloc), 不是 beam sweep。
4. 若 Run 5 仍 FAIL: 候选 (b) BC override (training 时未 init 用 BC beam_direction 强制), 或回 spec §6 反 toy checklist 排查。
5. **不要再调 curriculum 参数** — A3 + Run 4 都证 warm_start 创依赖。
6. **不要再调 bc_kl_coef / bc_kl_coef_beam 数值** — joint KL 太弱本质问题, 不是数值问题。

**完整数据**:
- 训练 logs: `/tmp/wp3_phaseA_{a1only,a1a2,a1a2a3,run4_beamanchor,run5_freezebeam}.log`
- Crossplay reports: `experiments/twoteam/wp3_smoke_phaseA_{a1only,a1a2,a1a2a3,run4_beamanchor}_report.md`
- Probe: `experiments/twoteam/wp3_root_cause_probe.py`
- Memory 关联: [[twoteam-wp3-1-beam-sweep-collapse]] (旧诊断, 被 Run 4 probe 部分推翻), [[twoteam-wp3-production-smoke-fail]] (前序 baseline)
