# Phase 1.5 Cross-Play Tournament (Exp B) — Paper Headline Table

> **方法论**: 每个 unordered pair 双向比赛 (A_red vs B_blue + B_red vs A_blue) 平均,消除红/蓝起始位置不对称。NN-vs-NN 用 `LaserTrainer.eval_episode()` (league opp 模式);NN-vs-MPC 用 `LaserEpisodeRunner.step_control` + `NNPolicyAdapter`。每个 cell 共 72 局 (36/dir × 2 directions),binomial stderr ≈ √(p(1-p)/72) ≈ 0.059。

## Table 1: Round-robin (NN finals vs each other)

对称胜率矩阵;对角线 = 0.5 (self-play,未实际比赛);`mean` = 该 arm 对其它两 arm 的平均胜率。

| | mappo | ippo | pspfix | **mean** |
|---|---|---|---|---|
| **mappo** | — | 0.750 | 0.764 | **0.757** |
| **pspfix** | 0.236 | 0.722 | — | **0.479** |
| **ippo** | 0.250 | 0.278 | — | **0.264** |

**排名**: mappo (0.757) ≫ pspfix (0.479) > ippo (0.264)。
**统计意义**: mappo vs ippo 差 0.486 (>>2σ),mappo vs pspfix 差 0.278 (>>2σ),pspfix vs ippo 差 0.215 (>2σ)。排名无歧义。

## Table 2: Held-out (NN finals vs cross-method iter_010 snapshots)

held-out 集 = 每个 arm 自己的 iter_010 快照 (跨方法早期检查点,避免训练 seed-43 开销)。

| final | mappo_iter10 | ippo_iter10 | pspfix_iter10 | **mean** |
|---|---|---|---|---|
| **mappo** | 0.500 | 0.861 | 0.833 | **0.731** |
| **pspfix** | 0.319 | 0.653 | 0.583 | **0.519** |
| **ippo** | 0.222 | 0.208 | 0.319 | **0.250** |

**与 Table 1 一致**: 排名 mappo > pspfix > ippo 在 held-out 集上保持。mappo 即使对 iter_010 版本的自己也是 0.500 (无退化),对其它两 arm 的 iter_10 强势胜出 (0.83-0.86)。
**ippo 反常**: ippo 对自己 iter_010 只有 0.208 — 表明 ippo 训练后期出现了策略退化 (后期 iter_019 比中期 iter_010 更差)。

## Table 3: NN finals vs ClassicalMPC (EAAI engineering baseline)

ClassicalMPC = 规则化波束指向融合敌方锚点 + 始终开火,无学习/波形敏捷/干扰。共享同一 env + 同一 Kalman 融合感知前端。每行 72 局 (36/dir × 2)。

| NN final | NN wins | MPC wins | draws | **NN win rate** |
|---|---|---|---|---|
| **mappo** | 34 | 38 | 0 | **0.472** |
| **pspfix** | 31 | 41 | 0 | **0.431** |
| **ippo** | 12 | 60 | 0 | **0.167** |

**关键发现**: 所有 3 个 RL 决赛圈都未能在 5σ 显著性上击败 ClassicalMPC。
- **mappo** 0.472: 与 MPC 在统计噪声内 (差 0.028,stderr 0.059) — 实质上**平局**
- **pspfix** 0.431: 比 MPC 略差 (差 0.069,约 1.2σ) — **轻微劣势**
- **ippo** 0.167: 显著输给 MPC (差 0.333,>>5σ) — **明显劣势**

## Verdict

### V1: NN-vs-NN 排名清晰
mappo (CTDE on,PFSP off) > pspfix (CTDE off,PFSP on) > ippo (both off)。这与三方自博弈比较的排名一致,但双向 + held-out 提供了**有效的跨方法对比** (旧比较每个 arm 各打各的自博弈池,数字不可比)。

### V2: V-V critically — RL 未在头对头中击败 classical baseline
这是**论文核心警告**: EAAI 要求 "AI beats classical",但当前 3 个 arm 都没做到。原因可能:
1. **kill_radius_m=0.5 太严苛**: ClassicalMPC 始终开火 + 指向融合锚点,在 500 步内累积光照已足够杀;RL 的优势 (波形敏捷、干扰、动态资源分配) 在这种"指向即杀"的简化环境下无法变现。
2. **reward shaping 主导**: 训练时 RL 优化 shaping reward (illumination_progress),但 ClassicalMPC 直接优化 ground-truth kill,二者目标偏离。
3. **500 step 上限太短**: 短局有利于简单策略;长局可能让 RL 的多步规划能力显现。

### V3: 对 Exp A (full league) 的决策影响
**不建议立即跑 Exp A**。理由:
- Exp A 假设 "PFSP+CTDE 组合 > 单独任一",但如果组合后仍 ≤ 0.50 vs MPC,4h 训练成本无法兑现论文价值主张。
- 应**先排查 V2 的根因** (kill_radius / reward shaping / episode length),再决定是否调整环境参数后重训。

### V4: 论文叙事选项
- **C1 工程洞察叙事**: "RL 在严苛环境下未击败 classical — 揭示 sensing frontend 才是主要贡献" (诚实但弱化贡献)
- **C2 2×2 ablation 框架叙事**: 保留 PFSP×CTDE 矩阵作为方法论贡献,把 RL-vs-MPC 平局当作"RL 达到 classical 等价性能 + 提供 deployment-time 学习能力" (需要重新定位论文卖点)
- **C3 调参重训**: 调整 kill_radius_m / episode_length / reward_shaping 让 RL 优势可变现,再跑全量 (4-6h 延迟)

## Reproducibility

- **代码**: `scripts/crossplay.py` (NN-vs-NN) + `scripts/crossplay_mpc.py` (NN-vs-MPC)
- **配置**: `algo/mappo/code/config.yaml` (actor 架构三 arm byte-identical)
- **Checkpoints**: `algorithms/{mappo,ippo,pspfix}/data/checkpoints/iter_019.pt` (20 个/arm)
- **Seed**: 42 (set_global_seed 锁定 torch/np/random/cudnn)
- **Log**: `experiments/crossplay.log` + `experiments/crossplay_mpc.log`
- **环境**: NVIDIA RTX PRO 6000 (101.9 GB),CUDA 12.8,PyTorch 2.12.0
- **耗时**: NN-vs-NN 22 min (round-robin 4 min + held-out 18 min),NN-vs-MPC 22 min (3 arm × ~7 min)
