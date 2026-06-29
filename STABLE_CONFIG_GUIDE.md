# 稳定联赛配置说明 — `laser_25x25_pro6000_stable.yaml`

**面向**: PRO 6000 上的 agent。**目的**: 修掉 v2(`PHASE1_LEAGUE_V2_FAILURE_REPORT.md`)的**过训练退化**,把验证过的稳定配方固化。
**关联**: [PHASE1_LEAGUE_V2_FAILURE_REPORT.md](PHASE1_LEAGUE_V2_FAILURE_REPORT.md) · [LEAGUE_TRAINING_PLAN.md](LEAGUE_TRAINING_PLAN.md)

---

## 0. v2 不是"没学会",是"学会了又被练崩"

v2 用 `pro6000.yaml`(clean `use_mappo=false`)**iter 13 真的到了 kr=0.48m、eval_kill_rate=1.0、cum red=1.0、cmd_pl=0.196**——**底座是 work 的、闸门那一刻 PASS**。问题是 iter 14-27 退化:adv_std 从 ~10 爆到 **235**,cmd_pl 坍缩到 ≈0,cum red 跌到 0.58。

**退化的三个原因(本配置逐一对治)**:
1. **dartboard_weight=50 全场稠密奖励**(v2 自加,**不在原验证配方里**)→ per-step `50·exp(-d/5)` 在精细 kr 把 reward 量级推到 100-200 → F8 归一化的 std 被离群拉大 → advantage 被噪声污染(adv_std 爆)。
2. **psro=48 过训练**:kr 在 iter ~16 就到物理底,后面 30 轮在临界点空转 → 退化。
3. **追到 kr=0.2m 抖动临界**(RESULTS_SUMMARY:0.20m 是临界,0.24m 才有裕度)+ **log_std_floor=-6 探索塌缩** → 精细 kr 下打不过多样对手池。

**决定性对照**: 早先 4090 联赛(**无 dartboard、跑 20 轮、kr→0.2m**)cum red=0.88、blue=0.00、**全程稳定不退化**。所以退化是 v2 加料+过训练造成的,不是底座问题。

---

## 1. 5 处改动(vs `pro6000.yaml`,before → after + 理由)

| # | 参数 | v2(失败) | stable | 对治 |
|---|---|---|---|---|
| 1 | `reward_shaping.dartboard_weight` | **50.0** | **0.0** | 去掉 adv_std 爆炸的头号原因(回验证过的 reward shaping) |
| 2 | `env.kill_radius_m`(=课程地板 kr_final,见 train_laser:1468) | 0.2 | **0.24** | 钉在鲁棒裕度,不追抖动临界 |
| 3 | `training.log_std_floor` | -6.0 | **-4.0** | 精细 kr 下保留探索熵(σ 0.0025→0.018) |
| 4 | `training.psro_iterations` | 48 | **20** | 对齐验证过的 4090 run,防过训练空转 |
| 5 | `training.cmd_bc_decay_iters` | 12 | **24** | BC 在精细 kr 阶段继续支撑,PPO 不必独扛精细瞄准 |

其余**全部沿用验证过的 p14 三件套**(5km 基线 + Kalman tracked + 6m 残差)+ `use_mappo` 不设(=False,**不经过 alpha-blend,无 alpha-collapse**)+ `reward_normalize:true`(F8,agent 路径已验证有效)。

---

## 2. 本地 smoke 已验证(4090, num_envs=2, 2 轮)
```
League ON → [PSRO 1/2] kills=11 kr=50→35m → [Eval 1] eval_kill_rate=0.5 cum red=1.00
          → [PSRO 2/2] kills=8  kr=35→24.5m → [Eval 2] eval_kill_rate=0.5
```
- ✅ 配置有效、能跑、无 NaN、kr 退火、出击杀、dartboard 已关、league 正常。
- ⚠️ **诚实**: 2 轮 smoke **只证明"能正常起训",证不了"抗退化"**(退化在 iter 14+);`cmd_pl/adv_std=0.000` 是 4-episode 小样本退化,非健康证据。

---

## 3. 怎么跑(PRO 6000)+ 看什么
```bash
python -m training.train_laser --config configs/laser_25x25_pro6000_stable.yaml 2>&1 | tee logs/phase1_stable.log
```
**全程盯这三条曲线判抗退化是否成立**:
| 指标 | 健康(修复成功) | 退化(仍失败) |
|---|---|---|
| `adv_std` | 全程稳定 ~5-30,**不爆** | 像 v2 那样窜到 100-235 |
| `cmd_pl` | 有量级、不持续坍缩 | iter 14+ 持续 ≈0 |
| `cum red` | 退火到 0.24m 后**保持 ≥0.8** | 像 v2 跌到 0.58 |

**预期**: 复现 4090 的稳定结果——kr→0.24m、cum red≈0.85+、adv_std 不爆、20 轮不退化。

## 4. 若仍退化(回退树)
- adv_std 仍爆 → 查 reward 是否还有别的大幅值项;最坏 `reward_normalize:false`(回 v4_control 那档,4090 验证过稳);
- cum red 仍跌 → kr 地板再放宽到 0.3m;或 `league_snapshot_every:6`(放慢对手多样性增长);
- cmd_pl 仍坍缩 → 这才回到深层 PPO 问题,跑 `diagnose_grad.py` 抓 actor grad norm。

> 纪律: 这是**验证过的配方 + 抗退化改动**,改动小、有 4090 稳定结果背书。**一次只改一处,改完对照本表判 PASS。**
