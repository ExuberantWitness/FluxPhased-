---
name: twoteam-wp3-2-phaseA2-ppo-variants
description: "WP-3.2 Phase A2 (2026-07-20) FAIL: HD-PPO/IPPO/MAPPO 3 变体全部 ≤0.016 kill vs BC 0.84-0.97。证明 PPO loss 结构 (joint/HD/IPPO/MAPPO) 不是 root cause。Phase A 累计 7 跑全 FAIL。下一步必须改 architecture (LSTM/belief) 或 Tier 2 env (adaptive jammer) 或 entropy_coef_min 0.001→0.05。"
metadata:
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

2026-07-20: Phase A2 跳出"加 BC regularization"路径,改为改 PPO loss 结构。Plan agent + 用户共识别 root cause = [commander_actor_critic.py:269](commander_actor_critic.py#L269) joint log_prob 把 6 个 head 求和成单一 ratio,beam_direction 高熵 sweep 被高维 head (task_alloc, channel_select) 淹没。3 个变体分别从不同维度拆 ratio。

**3 个 PPO 变体**:

| Mode | 核心机制 | Critic | LOC |
|------|----------|--------|-----|
| `joint` (baseline) | 6 head → joint log_prob → 单 ratio/clip | central + local | - |
| `hd` (HD-PPO) | per-head 独立 ratio/clip, mean over heads | both | +60 |
| `ippo` (Independent PPO) | per-aperture 独立 ratio/clip | local only | +30 |
| `mappo` (Multi-Agent PPO) | per-aperture 独立 ratio/clip | central only (CTDE) | +10 |

**实现** ([br_trainer.py](br_trainer.py) + [commander_actor_critic.py](commander_actor_critic.py) + run_wp2_league.py + wp3_train.py + tests/twoteam/test_wp2_smoke.py,~120 LOC):
- `evaluate_actions(return_decomposed=True)` 返回 6-tuple: `(log_prob, value, value_local, entropy, per_head_logp_dict, per_ap_logp)`
- `per_head_logp` = 6 head 各自 sum over aperture → [B],验证 `sum(per_head) == joint log_prob` (diff < 1e-6)
- `per_ap_logp` = sum over head 但 keep aperture dim → [B, n_ap];**laser_target 是 team-level,按 /n_ap 分配到每 aperture 保持 sum 一致**
- `_RolloutBuffer` 加 `log_prob_per_head[name][H,E]` + `log_prob_per_ap[H,E,n_ap]`
- `collect_rollout` 在 ppo_loss_mode != "joint" 时多调一次 evaluate_actions(return_decomposed=True) 填 buffer
- `update()` 按 mode 分支:joint(原),hd(per-head ratio/clip 平均),ippo(per-ap ratio + local critic only),mappo(per-ap ratio + central critic only)
- CLI: `--ppo-loss-mode {joint,hd,ippo,mappo}` (默认 joint,backward compat)

**Microverify** (`/tmp/phaseA2_microverify.py`, 4/4 PASS):
- shape 验证: joint lp [8], per-head dict 6×[8], per-ap [8, 2]
- 数学一致性: sum(per_head) - joint = 0.00e+00; sum(per_ap) - joint = 9.54e-07
- 4 mode 各跑 30-step rollout + PPO update 无 NaN
- 4 mode value_loss 差异符合预期:joint/hd ≈ 4 (central+local),ippo ≈ 0.02 (local only),mappo ≈ 1.6 (central only)
- pytest 109/109 不回归

**3 个训练配置** (2026-07-20 07:48 启动):
- 共同: 50 iter × n_envs=32 × horizon=300 = 4.8e5 steps,BC 15k samples 6 epochs,shape dwell/kill/init/detect/belief on,curriculum off,entropy anneal 0.01→0.001,fixation off,BC KL off
- 差异: `--ppo-loss-mode {hd,ippo,mappo}`
- 并行 4 训练 (Run 5 + 3 变体) CPU 竞争,ETA 09:30 全部训完
- ckpt: `checkpoints/blind/wp3_phaseA2_{hd,ippo,mappo}_20260720_074819/`

**How to apply**:
1. 用户问"3 个变体效果": 看本文 + 等训练结果 (crossplay 在 `experiments/twoteam/wp3_smoke_crossplay.py`)
2. 论文 ablation: 4 个 PPO 变体 vs BC kill rate,展示哪种 loss 结构最适合 multifunction phased array RL
3. 若 hd PASS: 证明 "joint log_prob 淹没" 是 root cause,Phase A 全 FAIL 是因 joint loss
4. 若 ippo/mappo PASS: 证明 per-aperture learning 是关键,decentralized actor 训得更好
5. 若全 FAIL: "保持 BC + 改 PPO loss" 都不行 → 必须改 architecture (LSTM + belief input) 或改 env (Tier 2 adaptive jammer)
6. **不要再调 KL coef / curriculum 参数** — Phase A 已证这些不是关键。

**Memory 关联**: [[twoteam-wp3-2-phaseA-fail]] (前序 Phase A 全 FAIL), [[twoteam-wp3-1-beam-sweep-collapse]] (前序诊断), [[twoteam-wp3-production-smoke-fail]] (基线)

---

## 结果 (2026-07-20 09:30 训练完成 + crossplay)

**3 变体 crossplay vs BlindClassical** (n=30 eps × bidirectional, total 64 samples per side):

| Mode | iter | low-int RL kill | low-int BC kill | low-int Δ | high-int RL kill | high-int BC kill | high-int Δ |
|------|------|-----------------|-----------------|-----------|------------------|------------------|------------|
| HD-PPO | 50/50 | **0.000** | 0.969 | **-0.969** | **0.000** | 0.250 | -0.250 |
| IPPO | 50/50 | **0.016** | 0.844 | **-0.828** | **0.000** | 0.141 | -0.141 |
| MAPPO | 50/50 | **0.000** | 0.906 | **-0.906** | **0.000** | 0.172 | -0.172 |

**Gate (RL kill ≥ 0.5) = 0/3 PASS**。

**训练过程观察** (50 iter, ~90 min each, 3-way 并行):
- HD-PPO iter 20: r=+3.408 wr=0.95 vs pure_comm (极端对手),iter 40 vs blind_classical wr=**0.36** (低于 0.5)
- IPPO: 全程 r≈0,v_loss 稳定在 0.01-0.3,**几乎没学到任何 kill reward** (per-ap ratio 接近 1 → 无梯度信号)
- MAPPO iter 20: r=+3.443 wr=0.95 vs pure_comm,iter 30/40 r=-0.2 vs jam_spread (退化)

**RL survival 数据有意思** (RL 学到的是"hide"而不是"kill"):
- HD-PPO survival: low 0.79 / high 0.83
- IPPO survival: low 0.73 / **high 0.89 (最高) + trace_P 35 (最低,几乎不发射)**
- MAPPO survival: low 0.73 / high 0.89
- 解释: IPPO local critic + per-ap ratio 让 RL 找到了 "shutdown emission → 不被 kill" 的 local optimum,但完全没学 detect+kill 链

**结论 (用户 + Plan agent 假设被推翻)**:
1. ❌ "joint log_prob 6 head 求和淹没 beam_direction sweep entropy" **不是** root cause — HD-PPO per-head 独立 clip 依然 0 kill
2. ❌ "decentralized actor 学得更好" **不是** root cause — IPPO/MAPPO per-aperture 独立 ratio 依然 0 kill
3. ✅ **PPO loss 结构 (joint/HD/IPPO/MAPPO) 不是问题** — 4 种都 FAIL,问题在更深层

**累计 Phase A 7 跑全 FAIL** (A1 only / A1+A2 / A1+A2+A3 / Run 4 Beta anchor / Run 5 freeze beam / HD-PPO / IPPO / MAPPO):
- 4 种 PPO loss × 多种 shaping/curriculum/BC-regularizer 组合 → RL kill 全部 ≤ 0.05 vs BC 0.84-0.97
- 用户"先做实验"路径耗 ~30 GPU-h,Phase A 全 FAIL

**Root cause 候选 (按可能性排序,需下一轮 research pipeline 验证)**:
1. **Architecture**: 缺 LSTM / frame-stack → RL 无法建模时序(BC teacher 隐式有 tracker history,RL obs 只有当前帧)
2. **Entropy coef min=0.001 是文献禁忌值** (Zhao 2025: 必须 0.05-0.1);anneal 把 BC 教的高熵 search 压死了
3. **BC teacher 太强**: BlindClassical 用 IMM-PDAF + HPBW 4.3° sweep_step 2.15° 2x oversample,RL 用 single-frame obs 永远学不到这个
4. **Symmetric 1v1 是 Nash 平局**: Bet B 已证 rule 的 kill capacity invariant,BC 同样近乎最优,RL 没有"可赢"的方向

**下一步候选 (待用户决策)**:
- **Tier 0 (修 RL)**: entropy_coef_min 0.001→0.05,加 LSTM/frame-stack,加 belief input(把 tracker state 喂进 obs)
- **Tier 2 (改 env)**: B1 机动目标 + B3 comm burst + B4 adaptive jammer(GIRL 2025 setup),让 RL 有 BC 无法编码的结构性决策点
- **接受 IET floor**: 承认 symmetric 1v1 RL 打不过 BC,直接进论文 §4 "BC strong floor, RL在其他场景未必"

**How to apply**:
1. 用户问"3 个变体效果": 看本文 + crossplay 报告 `experiments/twoteam/wp3_smoke_phaseA2_{hd,ippo,mappo}_report.md`
2. **不要再调 PPO loss 结构** — Phase A2 已证 4 种都 FAIL
3. 论文 framing: 这一轮 negative result 可作为 "RL 在 symmetric 1v1 multifunction radar 无优势" 的 honest finding
4. **下一步必须改 architecture 或 env** — 不是参数调整能救的
