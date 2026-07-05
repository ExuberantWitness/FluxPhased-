# 实验 A + B 精确执行方案(给 PRO6000 agent)

**背景**:phase1.5 三方对比(PfspFix/MAPPO/IPPO)得出"PfspFix 打不过 MAPPO",但这个结论**站不住**,有两个致命缺陷:
1. **对比无效**:三个 arm 全是 `league:true`,`cum_red`=红方打**自己训练池里的冻结 blue**(within-method 自博弈),不是三方最终策略互打。MAPPO cum_red=0.97 = CTDE 让 red 看到 blue 全队状态→碾压自己的瞎眼冻结影子(信息不对称产物),**不是"MAPPO 比 League 强"**。`eval_kr` 也各测各的对手分布(PFSP 打更难对手→均匀 eval 分数看着低)。
2. **真·League 没跑**:三个 arm 的 2×2 是
   ```
                  use_mappo:false │ use_mappo:true (CTDE)
     pfsp_p:2  │  PfspFix          │  ★没跑★ = 真·完整 League
     pfsp_p:0  │  IPPO             │  MAPPO
   ```
   右上角(PFSP + CTDE 都开)从没跑过。你的 League 从没拿到 CTDE 的 +0.16。

**目标**:① 补跑真·League ② 用有效协议(cross-play + 共同 held-out)裁决"谁更强"。

---

## 实验 A — 真·完整 League(缺失的 2×2 格,纯配置,~4h)

**这是 MAPPO 上加 PFSP,隔离出"PFSP×CTDE 交互"效应。**

```bash
cp algo/mappo/code/config.yaml algo/full_league/code/config.yaml
# 只改一行:pfsp_p: 0 → 2（打开 f_hard,其余与 MAPPO 完全一致 = league:true + use_mappo:true）
#   并把 checkpoint_dir 改成 algo/full_league/data/checkpoints
python main.py --config algo/full_league/code/config.yaml   # seed 42, ~3.5-4h
```
**唯一变量 = `pfsp_p`(0→2)**;env/reward/curriculum/seed 与 MAPPO 逐字节相同 → 干净测出"在 CTDE 之上加 PFSP 是否有增益"。

**看什么(逐轮,与三方报告同口径)**:`cum_red`、`eval_kr`、`adv_std`、`cmd_pl`、逐轮 `R/B/D`、`wr[opp]`。
**预期**:CTDE 的红方优势(~0.97 量级)+ PFSP 的稳定/鲁棒(adv_std 更低、打更难对手)。
**注意**:use_mappo=true 是 team-critic 路径——MAPPO arm 已证它在本重构代码里稳定(cum_red 0.97 无 alpha 崩),所以直接跑即可;若 alpha 有 schedule,确认 `alpha ≤ 0.5`(别让 team_adv 独占,`修改建议.md` F4)。

---

## 实验 B — cross-play 锦标赛 + 共同 held-out(唯一有效的"谁更强"裁决,~2-4h)

**复用现成基础设施**:`algo/_shared/self_play/payoff_matrix.py::evaluate_pair(red_trainer, blue_trainer, env, n_games)`(L86,"Evaluate win rate of red vs blue over multiple games")——它 reset 环境、跑 N 局、返回红蓝胜负。**这就是两策略对打原语。**

### B.1 加载各 arm 的最终策略
每个 final checkpoint = `(radar_ac.state_dict(), commander_ac.state_dict())`(train_laser L436-437 存池同构;final 存于各 arm 的 `checkpoint_dir`,L1704-1705)。写一个 `scripts/crossplay.py`:
```
for arm in [ippo, mappo, pfspfix, full_league (+classical_mpc)]:
    trainer[arm] = build 一个 trainer, load_state_dict(final radar_ac + commander_ac)
```
`classical_mpc` 用 `algo/_shared/baselines/classical_mpc.py`(**已存在**的规则控制器:aim=融合anchor、fire=min_dist<kr),作为"传统法"对照,包进锦标赛。

### B.2 round-robin 胜负矩阵(去先手优势)
对每个**有序对** (A,B):`evaluate_pair(red=A, blue=B, n_games=50, seed 集)` → 记 red_wins/blue_wins/draws。
**每对跑两次(A红B蓝 + B红A蓝)取平均**,消除红蓝不对称(这正是拆穿 cum_red=0.97 假象的关键——那 0.97 全是红方位置带来的)。产出 N×N 胜率矩阵 + 每方平均胜率/Elo。

### B.3 共同 held-out(鲁棒性,League 该赢的地方)
建一组**谁都没训过**的对手 K 个(3 选 1 或混合):
- (a) `classical_mpc` 控制器;
- (b) 用 **seed 43** 跑 IPPO 几轮,冻结中间快照 2-3 个(与所有 arm 的 seed-42 训练池不相交);
- (c) 各 arm 训练早期(iter 3/6)的快照**交叉施用**(A 的早期快照当 B 的 held-out)。
每个 final 对**同一组 held-out** 跑 evaluate_pair → **held-out 胜率 = 对未见对手的平均胜率**。**这是 League/PFSP 的设计目的,也是唯一能体现其价值的指标。**

### B.4 判据(命门)
| 结果 | 结论 |
|---|---|
| **round-robin**:PfspFix/FullLeague 头对头**赢** MAPPO/IPPO | League 真强,核心主张成立 |
| **held-out**:League 对未见对手胜率 **> MAPPO/IPPO** | League 鲁棒性优势坐实(即便 self-play cum_red 持平)|
| 两项都输 | 才可下"League 不占优",转 C1 工程洞察 或 "2×2 消融框架"叙事 |

---

## 执行顺序 + 产出
1. **实验 A** 先跑(补上缺失格,~4h)→ 得 FullLeague final;
2. **实验 B** cross-play + held-out(A 跑完后,4 个 final + classical 一起,~2-4h)；
3. 产出 `experiments/crossplay_matrix.md`(N×N 胜率矩阵 + held-out 胜率表 + 每方平均),这才是**paper 的 headline 表**(替代现在无效的三个自测数字)。

## 诚实缺口
- **有可能** A+B 跑完 League 仍不占优(CTDE 信息优势是真的,MAPPO 头对头也可能真强)——但**在跑这两个之前,"打不过"下不了结论**,现数据只证明了"CTDE 给红方位置优势"+"各方法过拟合各自对手";
- cross-play 要**两个方向都跑取平均**,否则又被红蓝不对称污染(现三方报告栽的就是这);
- held-out 集必须**与所有 arm 训练池不相交**,否则不算"未见";
- 全程用逐轮 R/B/D + policy_loss 判健康,不信 cum_red 单点(它是终身累积平均,滞后)。
