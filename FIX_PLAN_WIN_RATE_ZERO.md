# FluxPhased win_rate=0% 完整修改报告

**日期:** 2026-06-12
**分析基线:** origin/main @ b2880bf(诊断当天代码,只读分析,未运行)
**结论先行:** DIAG_WIN_RATE_ZERO.md 方向正确(episode 内不可能出现击杀),但关键数字错了约 1000 倍,且跑错了训练入口。按其推荐方案(仅 max_steps 50→1000)修复后 win_rate 仍将是 0%。

---

## 1. 症状回顾

- Phase B/C/D 三个消融 cell(R0/R1/R3)win_rate 全部为 0%,所有对局超时截断;
- 支付矩阵全平 → NashConv=0,sigma 恒为 [1,0,0,…],任务熵 0,effK=1;
- Phase A 单任务 policy_loss 正常收敛(纯感知学习,与战斗无关)。

## 2. 根因链(按因果顺序)

### 2.1 直接根因:导弹运动学时间尺度差 3 个数量级

DIAG 那次跑的实际物理参数(均已核对到代码):

| 参数 | 实际值 | 出处 |
|---|---|---|
| 每步时长 dt | **0.1 ms** | `dt = pri × n_pulses`(vec_mfar_env.py:686);单体 ENV_DEFAULTS `pulses_per_cpi: 1`(train_league.py:75),PRF=10kHz |
| 导弹速度 | **244.4 m/s** | vec_missile.py:42 默认值;入口未传 `speed_ms`,不读任何 YAML |
| 每步飞行距离 | **2.44 cm** | 244.4 × 0.0001 |
| 50 步 episode 总飞行 | **1.2 m** | 击杀半径 500 m,典型交战距离 5–15 km |
| 命中 15 km 所需步数 | **≈ 61 万步** | 与 5/22 commit 实测吻合:"First verified end-to-end kill: 494k-step missile flight"(README:1953,速度+20%、距离 14.5 km) |

→ win_rate=0 在数学上必然,与训练算法无关。

### 2.2 配置根因:双训练体系分裂,跑在了陈旧单体上

仓库存在两套并行训练体系:

| | 单体 `train_league.py`(1935 行) | 模块化 `training/` 包 + YAML |
|---|---|---|
| 创建 | 05-19,"原有 16 个训练文件整合为单一文件,消除 YAML 配置"(README:2235) | 05-14 |
| 配置 | 硬编码 ENV/REWARD/PPO/LEAGUE/CURRICULUM_DEFAULTS + ABLATION_CELLS | `training/train.py --config configs/league_*.yaml --override k=v` |
| 5/22 后的开发 | 几乎为零(仅 b2880bf 被动改 4 行) | 全部:GPU 并行 HPEDF 数据采集、data_collector、CTDE TeamCritic、BPSK 末制导、评估强制发射、OOM 分块修复 |
| 导弹提速配置 | ✗(无 speed_ms,用默认 244.4) | ✓ `league_25x25_configA/12env/quick_test.yaml` 已写 `speed_ms: 62500` + `pulses_per_cpi: 4`(25 m/步,15 km ≈ 600 步) |

**DIAG 运行用的是单体**,证据:
- checkpoint 命名 `checkpoints/league_R0_seed42` = train_league.py:1853 的格式串(模块化入口为 `checkpoints/league_tcdams/...`);
- DIAG 文档头部"2 radars, fft_size=32768, streaming"与单体 ENV_DEFAULTS 逐项一致;
- 诊断 commit b2880bf 的 obs 68→76 修复打在 train_league.py 上。

### 2.3 DIAG 文档的两处事实错误

1. "missile speed = 62,500 m/s in env" — 该值只存在于 YAML(league_25x25_configA.yaml:91 等),单体入口不读 YAML,实际运行速度是 244.4 m/s;
2. "~600 steps needed" — 仅在 speed_ms=62500 **且** pulses_per_cpi=4 同时生效时成立。其推荐修复(max_steps 50→1000)下导弹只飞 24 m,win_rate 仍为 0,但会白烧约 20 倍算力(其自估 ~60 GPU 小时)。

### 2.4 下游机制(无需单独修复,击杀出现后自愈)

全部超时 → payoff_matrix.py:161 将截断记 0.5 平局 → 矩阵全 0.5 → Nash LP 任意单纯形顶点皆为解,返回 [1,0,…] → NashConv=0、effK=1;TC-DAMS 从该 Nash 解出发,在零方差矩阵上同样塌缩。

---

## 3. 修改清单

### P0-1:接通导弹时间尺度(二选一)

**方案 A(推荐,零代码改动):放弃单体入口,用模块化体系 + 现成 YAML**

```bash
python -m training.train --config configs/league_25x25_configA.yaml --seed 42 \
    --override league.meta_solver=nash ...
# 或直接用消融封装(默认 config 为 league_tcdams.yaml,注意换成含 speed_ms 的配置):
python run_tcdams_ablation.py --config configs/league_25x25_configA.yaml
```

configA 已包含全套修复:`speed_ms: 62500`、`pulses_per_cpi: 4`(dt=0.4ms → 25 m/步)、`max_steps_per_episode: 2000`、`urgency_penalty: -0.05`、`num_envs: 4`。

若需复现 DIAG 的小规模设置(2 radars、fft 32768、streaming、num_envs=1),新建 `configs/league_25x25_small.yaml`:抄单体 ENV_DEFAULTS + LEAGUE_DEFAULTS,并加入 `speed_ms: 62500`、`pulses_per_cpi: 4`、`max_steps_per_episode: 1000~2000`。

**方案 B(若坚持物理真实速度):运动学 dt 解耦**

在 vec_mfar_env.py:686 引入 `kinematic_dt_scale` 参数:`dt = self.pri * self.n_pulses * kinematic_dt_scale`,保持雷达信号仿真不变、只放大导弹/车辆运动积分。物理上更干净(不扭曲多普勒),但需改环境代码并重验证。

### P0-2:删除单体 train_league.py(消除事故复发路径)

迁移步骤(前 4 步本机即可完成):

1. **抢救语义改动进 YAML:**
   - `urgency_penalty`:单体 -1.0("was -0.01 — strong incentive to launch")vs 模块化 YAML -0.05 vs physics.yaml -0.01——选定一个值写入正式配置(建议先 -0.05,过强的负奖励会主导 50/2000 步内的回报);
   - 单体 ENV_DEFAULTS 小规模配置 → `league_25x25_small.yaml`(见 P0-1);
   - obs 68→76:模块化 actor_critic.py:82 默认已是 76,无需移植。
2. **迁移两个测试**:tests/minimal_detect_test.py、tests/selfplay_detect_test.py 共导入 6 个符号(DenseRewardShaper、RadarActorCritic、RolloutBuffer、TeamPPOTrainer、PPOTrainer、create_team_policy),training/ppo/ 包内全部有同名实现(reward_shaping.py / actor_critic.py / buffer.py / ppo_trainer.py)。注意模块化版本签名有漂移(TeamCritic 等),逐个适配。
3. `git rm train_league.py`。
4. **更新 README**(7 处引用,含 Quick Start 的 `python train_league.py --cells ...` 改为 `python -m training.train --config ...`),并在 DIAG_WIN_RATE_ZERO.md 顶部加更正说明(或以本报告替代)。
5. 旧 checkpoint 兼容性:单体保存的是 state_dict 字典(train_league.py:939、1575),非类 pickle,删除模块不影响加载;但单体网络与模块化网络结构已分叉,league_R0/R1/R3_seed42 的权重价值有限,建议归档不复用。

### P1:次级问题(不阻塞 win_rate>0,建议随后处理)

| 问题 | 位置 | 建议 |
|---|---|---|
| 训练/评估不一致:payoff 评估第 0 步强制发射,训练不强制 | training/self_play/payoff_matrix.py:123-139 | 时间尺度修复后先观察;若 Commander 仍学不会发射,再考虑训练期 ε-强制发射课程 |
| missile_guidance_reward 只发给雷达 [E,R],Commander 看不到导弹进度 | training/ppo/reward_shaping.py:325-370 | 增加 per-team 分量并入 commander_rewards |
| TeamCritic N-step=800 远超 episode 长度 | training/ppo/buffer.py(n_step_team=800) | max_steps≥2000 后该设置才有意义;保持并监控 bootstrap 值 |
| Phase A 完全不训练发射 | training/scripted_policy.py(预训练不强制发射) | 按 logical-purring-naur.md 的 v4 计划:HPEDF 示教 + Commander critic 预训练 |

### P2:卫生项

- `.git/config` 的 remote URL 内嵌 GitHub PAT(明文)——改用 credential helper 并**轮换该 token**;
- 若暂不删单体,至少在 train_league.py 与各 YAML 顶部互相注明"对方不读本文件"。

---

## 4. 验证计划

**本机(24G 内存,无需 GPU 大显存):**
- 测试迁移后 `python -m pytest tests/ --collect-only` 级别的 import 冒烟;
- 新 YAML 经 training/train.py 的 load_config + compute_env_params 解析(纯 CPU)。

**训练机(98G 显存):**
1. 单集冒烟:configA 配置 + 评估强制发射,期望 **≤800 步内出现 kill 事件**,kill_bonus(+10)进入 commander_rewards;
2. 短程 Phase C(1 个 PSRO 迭代):支付矩阵出现非 0.5 条目,NashConv>0,sigma 不再 [1,0,…];
3. 全量 3-cell 消融(预算参考 DIAG 自估:~15–60 GPU 小时,num_envs=4 取下限)。

**成功判据:** Phase B 双方 win_rate 不再恒 0;Phase C effK>1、任务熵>0;Phase D 至少一方 win_rate>0。

## 5. 风险

- `speed_ms=62500` 是"游戏速度"(约 183 马赫),若导弹回波参与雷达多普勒仿真,提速会使多普勒非物理——configA 既有设计已接受此近似,但建议在冒烟时确认 detect 奖励未被异常多普勒破坏;介意则改用方案 B;
- `pulses_per_cpi` 1→4 使每步信号处理量约 ×4,需复核 streaming 模式显存(DIAG 实测 14G/98G,余量充足);
- 删除单体后若训练机上有未提交的本地脚本引用它,会断链——执行前在训练机 `grep -rn "train_league"` 确认。
