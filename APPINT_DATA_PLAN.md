# Applied Intelligence 数据生产计划(给 PRO6000 agent)— 一次算齐全部论文数据

> **目标期刊**:Applied Intelligence(Springer,Q2 应用 AI,IF~5;对方法新意/军事框架宽容,收纯仿真)。
> **论文命题**:自适应 EW 下,双相控阵雷达 + 指挥官 + 激光 DEW 联合交战的 **MAPPO/CTDE 学习式指挥官**,打赢强模块化经典基线并在 **kill-survival Pareto** 上占优,across 难度包线。
> **明确砍掉(那是 TAES,AppInt 不需要)**:league/PSRO、exploitability/Nash、fictitious-play 博弈论经典。**保留** WP1 已建的强模块化经典(IMM-PDAF + Q-RAM + 火控 + ECCM)当 baseline。
> **执行伴档**:env/kill-链/指标/CRLB 公式见 `TAES_IMPLEMENTATION_SPEC.md` §1/§4,本文件只定**要跑哪些 run、产哪张图表、什么 scope**。
> **地雷**:checkpoint_dir→`checkpoints/appint/`(严禁 /tmp);log_std_floor=-6。

---

## 0. 论文要的产物(每个 = 一个数据 run 的输出)
| # | 论文 artifact | 由哪个 run 产 |
|---|---|---|
| T1 | **主结果表**:{MAPPO, 强模块化经典, static 经典, IPPO} × 代表 cell × {kill, ttk, survival, track-loss} ± 95%CI | R2 eval 网格 |
| T2 | **消融表**:−belief / −noise-robust-critic / MAPPO-vs-IPPO / −survival项 / −exposure项 | R3 消融 |
| F2 | **操作包线图(headline)**:kill & survival vs 难度,全方法,经典 D_c 断崖 + RL 扩 ΔD | R2 网格 |
| F3 | **学习曲线**:reward/kill vs steps,MAPPO vs IPPO 收敛 | R1 训练日志 |
| F4 | **Pareto 前沿**:kill vs survival(或 exposure),RL 占优 | R2 网格 |
| F5 | **CRLB 锚**:track 质量(trace_P)vs PCRLB across 干扰档 | R2 + crlb.py |
| F6 | **定性 episode 轨迹**:子阵分配 + kill-链 E_i + exposure over time(展示 RL 学到啥)| R4 rollout dump |
| T3 | **统计检验汇总**:per-cell Welch-t/Mann-Whitney + Cohen's d + Holm-Bonferroni | R2 后处理 |
| F1 | 系统/testbed 示意图 | 无需算(画) |

---

## 1. 前置修复(必须先做 + 验证,再跑贵的 eval;省算力关键)
**三个 bug 不修,数据无效——先修先验证。**
1. **正经训练 LearnedJammer(修随机初始化)**:PPO 训到收敛(每 curriculum 档 L0→L1→L3 都真训),监控 entropy 不崩、kill≠0;产出 **有效的 L3 checkpoint**。(不用红蓝 PSRO——单边训练即可,AppInt 不要求均衡。)
2. **混合 n_targets 训练**:MAPPO/IPPO 训练时 `N_targets` 随机 ∈{1,2,4,8}(解决 n8 OOD)。
3. **MAPPO 微调**:lr/entropy/clip 小扫(如 lr∈{1e-4,3e-4}、entropy∈{0.005,0.01}),选验证集最优;log_std_floor=-6。
- **验证门(必过再跑 R2)**:① L3 jammer entropy 稳、对固定雷达真造成 kill 下降(证它自适应有效);② MAPPO 在 n∈{1,2,4,8} 都不崩;③ 单目标无 EW 时 MAPPO 与经典都 kill≈1(sanity)。**没过别跑 R2。**

## 2. R1 — 训练 run(产 checkpoint + 学习曲线 F3)
训练这些策略(各 ≥5 seed,混合 n,课程含**有效 L3**):
- **MAPPO(主方法,belief-conditioned + noise-robust critic)**;
- **IPPO**(CTDE 对照,`use_mappo:false`);
- **消融变体**(供 R3):−belief、−noise-robust-critic(fixed α)、−survival 奖励项、−exposure 奖励项;
- (经典 baseline 无需训练,WP1 已建)。
- 记录每 run 的 reward/kill/survival vs steps → **F3 学习曲线**;total ~2e7-5e7 steps/run;checkpoint 存 `checkpoints/appint/<name>_seed<k>.pt`。

## 3. R2 — Eval 网格(产 T1 主表 + F2 包线 + F4 Pareto + F5 CRLB)
**方法** × **难度 cell** × **≥5 seed × 8 env = 40 episode/cell**:
- 方法:{MAPPO, 强模块化经典, static 经典, IPPO};
- 难度:`N_targets{1,2,4,8} × jammer{L0, L1-τ16, τ8, τ4, τ2, τ1, L3-trained} × exposure{low,high}`;
- 每 cell 记:kill-rate、time-to-kill、survival-rate、track-loss、mean(trace_P/PCRLB);
- **报完整曲线含低难度经典够用区(诚实,防 p-hack)**;
- 输出 `experiments/appint/eval_grid.csv`(方法×cell×seed×指标 长表)。

## 4. R3 — 消融(产 T2)
在**代表性硬 cell(如 n4_L1、n4_L3-trained)**上,eval R1 训好的消融变体(同 40 ep/cell、5 seed):
- −belief-conditioning、−noise-robust-critic、MAPPO vs IPPO、−survival 项、−exposure 项;
- 每个报 kill/survival Δ vs full-MAPPO → **证每个组件的因果贡献**。

## 5. R4 — 定性轨迹(产 F6)
取 1-2 个代表 episode(n4_L1,MAPPO),dump 每 step:子阵→功能分配、每目标 E_i(kill-链累积/清零)、exposure、jam_level、kill/death 事件 → 画时间轴图,展示"RL 学到:低置信收火 / 换被动传感降暴露 / 干扰下重分配保 kill 链"。

## 6. 统计协议(产 T3)
- 每 cell:mean ± 95%CI(bootstrap 1e4);
- 头对头 MAPPO vs 强经典:Welch-t + Mann-Whitney(非参兜底);
- 效应量 Cohen's d;多重比较 Holm-Bonferroni(cell 数多);
- 操作包线 D_c(kill 跌破可接受阈的难度)bootstrap CI;曲线交叉点 CI;
- 输出 `experiments/appint/stats_summary.csv`。

---

## 7. 数据完整性检查表(交付即论文可写)
- [ ] R1:MAPPO/IPPO/4 消融变体 × ≥5 seed checkpoint + 学习曲线;
- [ ] R2:eval_grid.csv(4 方法 × 4×7×2 cell × 5 seed × 40 ep);
- [ ] R3:消融表数据;
- [ ] R4:1-2 条定性轨迹 dump;
- [ ] R5:CRLB/PCRLB 每档 trace_P vs 下界;
- [ ] T3:stats_summary.csv;
- [ ] 前置验证门通过记录(L3 jammer 有效、混合 n 不崩、sanity)。

## 8. 诚实底线(AppInt 也会查)
1. **L3 必须是正经训练的自适应干扰机**——不得拿随机初始化当"自适应 EW"报;
2. **低难度 RL≈经典照实报**(相变叙事需要它,不 p-hack);
3. **强经典非稻草人**(IMM-PDAF/Q-RAM,WP1 已验证近满 kill@低难度);
4. **scope 诚实**:框架"多智能体资源管理/决策 under 竞争环境",剥 kill-chain 军事措辞;局限(纯仿真/单一 sim)明写。

## 9. 预算 + 时间线(轻活,非那 2 月 league)
| 步 | 内容 | GPU |
|---|---|---|
| 前置修复 + 验证门 | 修 jammer/混训/MAPPO 微调 | ~3-5 天 |
| R1 训练 | MAPPO+IPPO+4 消融 × 5 seed(混训) | ~1 周 |
| R2 eval 网格 | 4 方法 × 56 cell × 5 seed × 40ep(纯 eval,快) | ~2-3 天 |
| R3+R4+R5+T3 | 消融 eval + 轨迹 + CRLB + 统计 | ~2-3 天 |
| **合计** | | **~2.5-3 周** |

## 10. 回报格式
按 §7 检查表逐项交付 + 前置验证门记录。**先交前置验证门 + R2 的 n1/n4 sanity 行,我确认 L3 jammer 真有效、经典非稻草人后,再放行全网格 R2**(避免又在坏 pipeline 上烧算力)。全齐后我据 eval_grid + stats 定论文主张(kill 赢 / Pareto 占优 / 相变)并起草。
