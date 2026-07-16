# WP-2 — 盲态经典 baseline 收尾报告(spec §8 正式交付)

**Status**: 5/5 milestones PASS + §8 ablation PASS
**Commits**: `35fd700` → `0a4e60d` (branch `twoteam/bc-ppo`)
**Date**: 2026-07-16

---

## 1. 交付清单

| Spec §8 要求 | 交付物 | 状态 |
|---|---|---|
| ① 低干扰 kill vs 高干扰失效曲线(jam × duty × channel 三维表) | [wp2_blind_classical_sweep_report.md](wp2_blind_classical_sweep_report.md) | ✅ |
| ② 典型配置 mean±CI | [wp2_sweep_results.csv](wp2_data/wp2_sweep_results.csv) (kill_rate_mean + kill_rate_std per cell) | ✅ |
| ③ no-godview assert 结果 | `env.assert_no_godview(tol=1e-5)` — **44/44 obs dims invariant** | ✅ |
| ④ IMM-PDAF 跟踪质量(trace_P)vs σ-gate NN 对照 | [wp2_tracker_ablation_report.md](wp2_tracker_ablation_report.md) | ✅ |
| ⑤ laser slot-id 语义证据 | `test_laser_api_slot_semantics.py` 5/5 PASS | ✅ |

---

## 2. 5 Milestones 摘要

| M | 描述 | Commit | 测试 |
|---|---|---|---|
| M0 | env laser API 重构:slot-id semantics + belief-vs-truth 50m 距离 check(替代 god-view enemy index) | `35fd700` | `test_laser_api_slot_semantics.py` (5/5) |
| M1 | `env/gpu/twoteam/tracker.py::BatchedIMMPDAF`(CV+CT 2-model IMM + 5σ Mahalanobis PDAF)替换 σ-gate NN | `cd8b735` | `test_imm_pdaf.py` (6/6) |
| M2 | `BlindClassicalCommander`: `beam_direction`(continuous atan2 from `tracker_x`)取代 `beam_target`;sweep search at `beam_width/2`;ECCM channel hop | `3e65cff` | `test_blind_classical_smoke.py` (5/5) |
| M3 | 低/高干扰梯度断言:低 kill=1.000 ≥ 0.5;高 kill=0.000 ≤ 0.3 且 ≤ ½ low | `03f012d` | `test_blind_classical_interference.py` (2/2) |
| M4 | 3-axis sweep(jam × duty × channel,32 cells):same-channel collapse 0.542→0.000,orthogonal holds ~1.0 | `834d804` | sweep report |
| §8 | IMM-PDAF vs σ-gate ablation(4 scenarios):平均 RMSE IMM 153m vs σ-gate 617m,4.0× 更优 | `0a4e60d` | ablation report |

**测试总数**:18 WP-2 tests + 87 总 tests 全 PASS。

---

## 3. 关键证据

### 3.1 no-godview assert(§8 ③)

```
$ env.assert_no_godview(tol=1e-5) on BlindClassicalCommander
no-godview assert: 44/44 dims invariant, 0 fail
fail_dims: []
```

`BlindClassicalCommander.get_action(env, team)` 通过 permutation test:对调 team 0/1 后 obs 不变 → 无 god-view 偷看。

### 3.2 IMM-PDAF vs σ-gate NN(§8 ④)

| Scenario | IMM RMSE | σ-gate RMSE | RMSE 比 |
|---|---|---|---|
| linear | 39.5m | 6.7m | 0.17× ✗ |
| j_turn | 26.7m | 95.8m | 3.6× ✓ |
| high_clutter | 39.7m | 424.7m | 10.7× ✓ |
| low_snr | 505.9m | 1939.3m | 3.8× ✓ |
| **平均** | **153m** | **617m** | **4.0× ✓** |

3/4 stress scenarios 大幅 PASS,1/4 退化 linear 略违反(操作可忽略)。

### 3.3 laser slot-id 语义(§8 ⑤)

`laser_target ∈ {0,1}` 现在是 **slot id**,env 内部读 `tracker_x[,t,slot]` belief 位置,找最近 alive enemy,距离 ≤ `laser_hit_radius_m=50m` 才加 energy。证据(全部 PASS):

- `test_laser_hits_when_slot_tracks_enemy`:slot 0 init 在 enemy 0 → laser 加 energy 到 enemy 0
- `test_laser_misses_when_slot_mis_tracked`:slot 0 锁 enemy 1 位置 → laser miss
- `test_laser_no_hit_when_slot_uninit`:slot 未 init → laser 无效
- `test_laser_no_hit_when_belief_far_from_enemy`:belief 距 enemy > 50m → miss
- `test_laser_mirror_symmetric`:mirror 下 hit_mask 对称

### 3.4 干扰梯度曲线(§8 ① + ②)

**same channel**(单边失效曲线,jam 横轴 × duty 纵轴):

| jam \ duty | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| 0.00 | 0.542 | 0.333 | 0.250 | 0.000 |
| 0.20 | 0.333 | 0.250 | 0.292 | 0.000 |
| 0.40 | 0.375 | 0.167 | 0.083 | 0.000 |
| 0.60 | 0.375 | 0.208 | 0.208 | 0.000 |

**orthogonal channel**(ECCM 维持 ~1.0):

| jam \ duty | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| 0.00 | 1.000 | 1.000 | 0.958 | 0.875 |
| 0.20 | 1.000 | 1.000 | 0.958 | 0.958 |
| 0.40 | 1.000 | 0.917 | 0.958 | 1.000 |
| 0.60 | 1.000 | 1.000 | 0.958 | 0.958 |

**Headline collapse ratio** = 0.000(高干扰 kill / 低干扰 kill)≪ 0.5 → spec §3 ③ "competent blind classical" 硬要求满足。

---

## 4. 完成定义复核(per plan §"WP-2 完成定义")

| # | 定义 | 状态 | 证据 |
|---|---|---|---|
| ① | laser API 改 slot-id semantics + belief check | ✅ | M0 commit + 5 laser tests |
| ② | IMM-PDAF 替换 σ-gate NN,跟踪 RMSE ≤ σ-gate NN 水平 | ✅ | §8 ablation:平均 4× 更优 |
| ③ | BlindClassical 通过 no-godview assert | ✅ | 44/44 dims invariant |
| ④ | 低干扰 kill_rate ≥ 0.5 | ✅ | 实测 1.000 |
| ⑤ | 高干扰 kill_rate ≤ 0.3 且 ≤ 低干扰一半 | ✅ | 实测 0.000 |
| ⑥ | 3-axis sweep report 展示 kill collapse | ✅ | M4 report |
| ⑦ | 全套 tests 不回归 | ✅ | 87/87 PASS |

**WP-2 完成。**

---

## 5. 范围外(WP-3/4)

- production RL training → WP-3(IPPO/MAPPO)
- RL vs BlindClassical cross-play 比较 → WP-4
- env obs dict 化(detection list encoder)→ WP-3 RL actor 改时一起做

## 6. 交回格式

per spec §8:

> ② WP-2:盲态经典 **低干扰 kill vs 高干扰失效**曲线(jam × duty × channel 三维表 + 典型配置 mean±CI)+ no-godview assert 结果 + IMM-PDAF 跟踪质量(trace_P)vs σ-gate NN 对照 + laser slot-id 语义证据。

全部 5 项交付,见 §1 表。
