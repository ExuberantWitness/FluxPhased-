# WP1 — 联赛收敛门(思路,非代码)

**一句话**:在投入任何正式实验前,先证明 **FluxLeague 路径自己能跑到收敛**(亚米 kr + 非退化 Nash + 残余 0.5 可忽略)。
门过不了,后面 baseline/消融/真实性全是空中楼阁——这是**止损闸门**。

> 注意区分:`train_laser.py` 内置 PSRO-lite **已验证**收到 0.2m。本门考的是**更完整的 FluxLeague**
> (Nash + 3-role exploiter + TC-DAMS + CTDE)能否**匹配或超过**它。若 FluxLeague 只到 ~5m 而 train_laser 到 0.2m,
> 这本身就是一个必须**如实上报**的负结果(联赛贡献存疑),不要粉饰。

## 1. 配置思路(改什么、为什么)
- 基底:`configs/laser_25x25_pro6000_league.yaml`。
- 套用 `EXPERIMENT_DESIGN_Q1.md §7` 的稳态化:`n_eval_games=50`(降评估方差)、`kr_init=100m`、
  `kr_decay=0.7`(每次只退 30%,平稳)、`kill_rate_threshold=0.7`(更稳的胜利才退火)、`max_steps=500`(给真击杀留时间)。
- `num_envs` 按 98GB 上调(12→16,看 `nvidia-smi` 余量);落盘持久盘。
- 先跑 **1 个 seed × 30 iters**;门过后这一跑直接作为 WP2 cell-A 的 seed-0。

## 2. 监控协议(每轮记什么)
逐轮抓:`kr`(课程)、`eval_kill_rate`、`cum red/blue/draw`、`NashConv`、`effK`、**0.50 占比**、`jam`(若 EW)。
另存一条**诊断量**:每局 reset 后己方两雷达的**两两间距**(必须 ≥ `min_radar_baseline_m`),用来确认修复真生效。

## 3. 三种失败模式 → 对应诊断(出问题照这查)
| 现象 | 根因方向 | 查法 |
|---|---|---|
| kr 卡 > 5m | 几何/课程/残差 | 打印 reset 后雷达间距(应 ≥5km);确认 `residual_aim` 在 eval 路径生效;看 anchor 是否仍退化到地图中心 |
| 0.50 占比 > 20% | 评估太短 / 课程太激进 | 加长 `n_eval_games`、`max_steps`;放慢 `kr_decay` |
| Nash 退化(sigma=[1,0,0]、effK=1) | 元博弈塌缩 | 看 payoff 矩阵是否有方差;若全平 → 回到"击杀信号"问题,不是 Nash 问题 |

## 4. 通过判据(全部满足)
| 指标 | 阈值 |
|---|---|
| Final kill_radius(联赛路径) | ≤ 0.5m(理想);≤ 1m 为最低可接受 |
| 残余 0.50 占比 | < 10% |
| Final NashConv / effK | < 0.05 / > 3(或至少单调改善) |
| 课程单调 | kr 全程不回升 |
| 复现 | 1 个完整 seed,log + checkpoint 留档 |

## 5. 决策逻辑
- **PASS** → 进 WP2:这一跑作 cell-A seed-0,再补 4 个 seed + 4 条外部 baseline。
- **FAIL** → **立即停**,按 §3 定位,**不要再烧 GPU 铺其他 cell**。
- **若联赛收敛但明显劣于 train_laser**(如卡 ~5m) → 论文重心移到 **C1 工程洞察 + train_laser 的 PSRO-lite**,
  联赛降级为"训练手段之一",并把这一对比作为**诚实负结果**写进论文。

## 6. 产出物
- `logs/wp1_gate_seedX.log` + checkpoint;
- 一张 `kr / 0.5占比 / effK vs iter` 三联曲线(后续直接进论文 figure);
- 一句**门结论**:PASS/FAIL + 收敛到的 kr + 与 train_laser 0.2m 的差距。
