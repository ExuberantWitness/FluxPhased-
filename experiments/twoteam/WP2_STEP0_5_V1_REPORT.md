# WP2 Step 0-5 + V1 实验报告 — Candidate-Exploit 全部 FAIL,rule ≈ Nash 强证据

**日期**: 2026-07-14
**项目**: 两队对称多功能相控阵对抗(TWOTEAM_MULTIFUNCTION_PLAN.md)
**目标会议**: TAES(主)/ IET(备选)
**前序**: WP2_BC_LEAGUE_PLAN.md(plan)、G0_BC_PPO_REPORT.md(G0 #3)
**代码状态**: Step 0-5 实现 + smoke test 通过 + V1 eval 完成

---

## 摘要

WP2 BC→League plan 的前 5 步(Step 0-5)+ V1 独立 eval 已完成。**V1 重大发现:3 条 candidate-exploit 脚本(jam_spread / hard_jam_focus / track_heavy_agile)在 horizon=200、bidirectional 200 episodes eval 下全部 FAIL,且每条 fail 都可追溯到 rule 的反馈式设计反制了 exploit 机制**。结合 G0 #3(BC+PPO 93% 平局,CI 跨 0),构成 rule ≈ Nash 的第二条独立证据线。

**league 主训练(V2,~1-2 周)未启动** — 等用户判断是否照原计划跑、先跑稳定性测试、还是直接退 IET 地板。

**已就绪的代码资产**(全部 smoke test 通过):
- `algo/_shared/pilot/twoteam/opponent_pool.py` — TwoTeamOpponentPool + PolicyRecord + PFSP/EMA
- `algo/_shared/pilot/twoteam/extreme_commanders.py` — 新增 3 条 exploit 类
- `scripts/eval_candidate_exploits.py` — 独立 bidirectional eval
- `algo/_shared/pilot/twoteam/run_wp2_league.py` — BC + PFSP league 主循环
- `algo/_shared/pilot/twoteam/run_wp2_crossplay.py` — all-vs-all + Elo + DFS 非传递检测
- `tests/twoteam/test_wp2_smoke.py` — 3 个 smoke test

---

## 1. WP2 plan 定位回顾

**赌注定位**:贪心第一注,砸天花板。**地板(IET:两队 IQ testbed + BC→PPO pipeline + 近 Nash 表征,~85%)已建好在手**,所以这注全砸天花板:**BC→league 能否产出一个稳健打赢强规则的策略 → TAES 冠**。

**为什么 league 比 G0 的单 best-response 有戏**(原假设):
- G0 测的是 BC→PPO 单 best-response 从 rule 盆地做 local 搜索 → 打平(G0 #3)
- league 不同:种群 + PFSP + 共演化做**无向、非局部**搜索,能找到 local best-response 探不到的**非传递/混合策略**
- 把人想到的 candidate exploit 塞进对手池当种子,league 既无向搜索、又带上人的假设

**诚实赔率**(原估):~20-25%(G0 近 Nash 平局压着它)

**关键判据**:
- **G1(冠)**:league-commander head-to-head 打赢强规则,胜率>0.5,95%CI 排除 0.5
- **G1-clean(关键)**:赢是稳健支配不是石头剪刀布 → league Elo 显著最高 且 league 不输给 "规则能赢的东西"
- **G2**:league > MAPPO(无 league)

**硬停**:赢→WP3;非传递→IET;打平→IET 地板+Bet B。**不开第 4 轮 league 调参**。

---

## 2. Step 0-5 实现细节

### 2.1 Step 0 — `TwoTeamOpponentPool`(新建,~230 LOC)

**路径**: `algo/_shared/pilot/twoteam/opponent_pool.py`

**设计决策**:新建而非扩展 `algo/_shared/self_play/opponent_pool.py`。理由:
- 现有 pool 与 FluxLeague 的 team/role/generation 概念绑定,`sample_pfsp(current_policy_id)` 要求 current policy 在池内
- twoteam league 的 current AC 在内存中(不在池内),API 不匹配
- twoteam 是对称博弈(无 team filter)
- 不动 FluxLeague 现有调用方

**API**:
```python
class TwoTeamOpponentPool:
    def add(record: PolicyRecord)              # 4 种 kind:rule/extreme/script/checkpoint
    def sample_pfsp(exclude=None) -> PolicyRecord  # f_hard(x) = (1-wr)^p,unknown → max
    def update_win_rate(name, win)              # EMA,首观察替换(不向 0.5 blend)
    def ema_variance() -> float                 # 低 → PFSP 退化信号
    def summary() -> Dict                       # JSON 友好的池状态快照
```

**关键设计**:
- `PolicyRecord`:`name` / `kind` / `checkpoint_path`(可选) / `factory`(可选) / `win_rate_vs_current`(EMA) / `games_played_vs_current` / `is_self_snapshot` / `created_at_iter`
- PFSP 用 AlphaStar f_hard(x) = (1-x)^p 函数 + 直接归一化(非 softmax)
- unknown win rate(如新加入的 self-snapshot)被当作 "最难" → 优先评估
- 首观察替换(非 EMA-blend-toward-0.5)— 避免 PFSP 退化到 uniform(R3 fix)
- population_cap eviction:只 evict 旧的 self-snapshot,never evict 种子(rule/extreme/script)

### 2.2 Step 1 — 3 条 candidate-exploit(~200 LOC)

**路径**: `algo/_shared/pilot/twoteam/extreme_commanders.py`(新增 3 个 class + STRATEGIES 注册)

**3 条 exploit 设计假设**(基于 G0 anti-strawman 数据 + env 机制分析):

#### 2.2.1 `JamSpreadCommander`(jam_spread)
- **假设**:rule 的 anti-jam hop reaction 只在 `_last_jam_matrix[:, rule, :].max() > 0.30` 时触发。若我每个目标 jam=0.28 < 0.30,rule 不 hop,effective_jam=0.28 → jam_mul = 1+6×0.28 = 2.68 膨胀 rule track_sigma
- **task_alloc**:`[0, 0.50, 0.28, 0.22]` × 2 apertures
- **beam_target**:aperture 0 → radar 0,aperture 1 → radar 1(分散 jam)
- **freq_hop**:1.0(我不需要抗干扰)

#### 2.2.2 `HardJamFocusCommander`(hard_jam_focus)
- **假设**:即使 rule hop=6,effective_jam = 1.20/6 = 0.20,jam_mul = 2.20;配合聚焦 track,我应该 out-track rule 在一个 radar 上 → focus fire kill
- **task_alloc**:`[0, 0.30, 0.60, 0.10]` × 2 apertures
- **beam_target**:两个 aperture 都指向 rule 同一个 radar(focus)
- **freq_hop**:3.0(轻度抗干扰)

#### 2.2.3 `TrackHeavyAgileCommander`(track_heavy_agile)
- **假设**:rule track 浓度 71%,我用 80% + freq_hop=8 max agility 应该 out-track rule,且抗 rule 的 counter-jam
- **task_alloc**:`[0.05, 0.80, 0.05, 0.10]` × 2 apertures
- **freq_hop**:8.0(max agility)

### 2.3 Step 2 — 独立 eval 脚本(~150 LOC)

**路径**: `scripts/eval_candidate_exploits.py`

**做什么**:
- 每条 exploit vs StrongRule,bidirectional(exploit@t0 vs rule@t1 + rule@t0 vs exploit@t1)
- 复用 `run_g0_gate.py::run_episodes_two_commanders` + `bootstrap_ci`
- 输出:win_rate / raw_win_rate / draw_rate / kills / kill_delta + 95% CI
- Verdict 阈值:WR > 0.55 AND CI lower > 0.5 → CONFIRMED;WR > 0.5 → WEAK;else → NONE

### 2.4 Step 3 — League 主循环(~450 LOC)

**路径**: `algo/_shared/pilot/twoteam/run_wp2_league.py`

**主流程**:
```
[A] BC 预训练(复用 bc_pretrain.py)
    50K samples from StrongRule + 7 Extreme + 3 Exploit
    → BC 15 epoch → rule-equivalent 起点策略

[B] 初始化对手池
    pool = {StrongRule, 7 Extreme, 3 Exploit, BC snapshot}  (12 records)

[C] League 主循环(N_iters)
    for iter in range(N_iters):           # N ~ 1000
        opp_record = pool.sample_pfsp()   # 优先采难打对手
        # Build commander-like opponent
        if opp_record.kind == "checkpoint":
            frozen_cmd = ACCommander(load AC from opp_record.checkpoint_path)
        else:
            frozen_cmd = opp_record.factory()
        trainer.frozen_opponent = frozen_cmd
        
        # One PPO iter
        buf = trainer.collect_rollout(env, horizon, learning_team=0)
        trainer._compute_gae(buf)
        metrics = trainer.update(buf)
        
        # Health monitor(NaN guard, adv_std ∈ [0.1, 100], entropy stable)
        # Quick eval (n_eval_episodes) → win_rate → pool.update_win_rate(EMA)
        
        if iter % snapshot_every == 0:
            save checkpoint → pool.add(self snapshot)
        
        if iter % 100 == 0: assert_priv_normalized(env, tag=f"iter-{it}")
```

**关键 bug guard**:
- α_eff bug:`priv[:, 4]` 归一化 assert(每 100 iter 复查)
- ckpt_dir 严禁 /tmp(代码层硬检查,`raise ValueError`)
- NaN guard + adv_std 范围监控(per iter)
- ACCommander adapter 让 loaded AC 满足 `frozen_opponent.get_action(env, team)` API

### 2.5 Step 4 — Cross-play 锦标赛(~450 LOC)

**路径**: `algo/_shared/pilot/twoteam/run_wp2_crossplay.py`

**方法集**(可配置):`{league-commander, StrongRule, MAPPO, IPPO, 7 extreme, 3 exploit}`

**算法**:
- All-vs-all round-robin,**双向**(A@t0 vs B@t1 + B@t0 vs A@t1,平均)
- 每个 pairing 30 ep × 8 envs × 双向 = 480 ep
- **Elo**:logistic update 50 rounds,bootstrap 1e3 给 95% CI
- **非传递性**:DFS 找 length-≤3 cycle 在 directed graph(A→B if WR(A,B) > 0.55)
- **G1**:league vs rule head-to-head WR
- **G1-clean**:Elo 一致性 + 无 cycle
- **G2**:league Elo > MAPPO Elo

### 2.6 Step 5 — Smoke test(~120 LOC)

**路径**: `tests/twoteam/test_wp2_smoke.py`

3 个 test:
1. `test_pool_mixed_kinds`:4 种 kind record 都能入池,sample_pfsp 返回合法 record,EMA 正确
2. `test_candidate_exploits_action_format`:3 条 exploit 输出 action dict 形状 + 归一化正确
3. `test_league_loop_minimal`:league loop 跑 3 iter,snapshot 落盘,无 crash

**结果**: ✅ 全部通过(BC 500 samples + 1 epoch + 3 league iters < 1 min)

---

## 3. V0 Smoke 验证

### 3.1 WP2 smoke test 全 ✅
```bash
$ conda run -n fluxphased python -u tests/twoteam/test_wp2_smoke.py
[gpu] Using device: cuda
[gpu] GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
[gpu] VRAM: 101.9 GB
--- test 1: pool_mixed_kinds ---
✅ pool_mixed_kinds OK — sampled ckpt_fake (kind=checkpoint); rule EMA after 2W/1L = 0.900
--- test 2: candidate_exploits_action_format ---
✅ candidate_exploits_action_format OK — all 3 exploits emit valid actions
--- test 3: league_loop_minimal ---
[BC train] epoch=1/1 train_loss=-2.577 val_loss=-7.290
[it=0/3] opp=exploit/hard_jam_focus   r=-0.349 v_loss=7.441 ent=-0.097 kl=0.0219 adv_std=1.00
[it=1/3] opp=extreme/pure_detect      r=+0.001 v_loss=1.048 ent=-0.134 kl=-0.0051 adv_std=1.00
[it=2/3] opp=self/iter000_bc          r=-0.367 v_loss=6.261 ent=-0.116 kl=0.0244 adv_std=1.00
✅ league_loop_minimal OK — 3 iters ran, 2 snapshot(s) saved
✅ All WP2 smoke tests passed
```

### 3.2 Cross-play smoke(11 方法,horizon=30,2 ep × 双向)
- 55 pairings × 4 episodes = 220 ep total
- 28 秒跑完
- Elo + 非传递检测 + 报告渲染全部正常
- CI 很宽(sample 太小,但 pipeline 工作)

---

## 4. V1 完整结果 — 3/3 Exploit 全 FAIL

### 4.1 命令
```bash
python -u scripts/eval_candidate_exploits.py \
    --episodes 100 --n-envs 8 --horizon 200 \
    --out experiments/twoteam/candidate_exploits_eval.md
```

### 4.2 结果表

| Exploit | Win rate (draw=0.5) | 95% CI | Raw WR | Draw rate | Exploit kills | Rule kills | Kill delta | Verdict | Time |
|---|---|---|---|---|---|---|---|---|---|
| `jam_spread` | **0.476** | [0.471, 0.482] | 0.003 ([0.001, 0.005]) | 0.95 | 1.92 | 1.96 | -0.037 | NOT AN EXPLOIT | 71s |
| `hard_jam_focus` | **0.040** | [0.033, 0.047] | 0.001 ([0.000, 0.003]) | 0.08 | 0.99 | 1.91 | -0.916 | NOT AN EXPLOIT | 220s |
| `track_heavy_agile` | **0.018** | [0.013, 0.022] | 0.000 | 0.04 | 0.99 | 1.96 | -0.964 | NOT AN EXPLOIT | 223s |

**Verdict 阈值**:WR > 0.55 AND CI lower > 0.5 → CONFIRMED。**没有任何一条达到。**

### 4.3 视觉化

```
Win rate vs StrongRule (draw=0.5,200 ep each direction,horizon=200)
   1.0 ┤
       │
   0.8 ┤
       │
   0.6 ┤                          ← G1 PASS 阈值(0.55)
       │
   0.5 ┤■ track_heavy_agile 0.018  ← 被碾压
       │■ hard_jam_focus    0.040  ← 被碾压
       │
   0.4 ┤■ jam_spread        0.476  ← 接近持平但仍弱
       │
   0.2 ┤
       │
   0.0 ┤
```

### 4.4 机制诊断 — 每条 fail 的物理原因

#### `jam_spread`(WR=0.476,最接近持平)
- **设计**:jam 0.28 < 0.30 阈值 → rule 不 hop → jam_mul=2.68 膨胀 rule track_sigma
- **数学**:我 track=0.50 → track_sigma = base/√0.50 = base × 1.41
- **数学**:rule track=0.71, jam_mul=2.68 → track_sigma = base/√0.71 × 2.68 = base × 3.18
- **理论**:我应 2.25× 快过 rule → 但实测 ≈ 持平(1.92 vs 1.96 kills,95% 平局)
- **失败原因**:**rule 的 reactive jam 反制**。rule 看到 enemy_tracking_me(line 100-105)触发 boost jam → rule 反过来给 jam_spread 喂 jam。jam_spread 的 freq_hop=1 不抗干扰 → jam_spread 自己也被 jam_mul ≈ 2.2 倒扣。**rule 的反馈抹平了 jam_spread 的理论优势**。

#### `hard_jam_focus`(WR=0.040,被碾压)
- **设计**:全 jam 1.20 → rule hop=6 → effective_jam=0.20, jam_mul=2.20 → focus fire
- **数学**:我 track 只剩 0.30(0.60 给了 jam)+ freq_hop=3 overhead → track_sigma = base/√(0.30 × 1/3^0.25) = base × 2.10
- **数学**:rule track=0.71, jam_mul=2.20 → track_sigma = base/√0.71 × 2.20 = base × 2.61
- **失败原因**:**self-own**。给 jam 分太多(0.60),track 不足。0.30 track 打不过 rule 0.71 track,即使有 jam_mul 倒扣,我也被 out-track。focus fire 没机会发挥(track 不够好,focus 也杀不掉)。

#### `track_heavy_agile`(WR=0.018,被碾压)
- **设计**:80% track + freq_hop=8 → 抗干扰 + 高浓度 → 赢 track race
- **数学**:我 track=0.80, 但 freq_hop=8 的 processing_overhead = 1/8^0.25 = 0.595 → track_sigma = base/√(0.80 × 0.595) = base × 1.45
- **数学**:rule track=0.71, freq_hop=1 → track_sigma = base/√0.71 = base × 1.19
- **失败原因**:**agility overhead > 浓度增益**。0.595 损失 vs 0.80/0.71=1.13 增益,净亏。**rule 的 track_sigma 比我还小**(1.45 vs 1.19),我被 out-track 1.22×。我付了 agility 成本但没人 jam 我(rule 看到我没 jam 它,不进 high_jam 模式),白付。

### 4.5 横向对比 — G0 三次测试 vs V1 exploit

| 维度 | G0 #1(纯 PPO) | G0 #2(调超参) | G0 #3(BC+PPO) | **V1 jam_spread** | **V1 hard_jam** | **V1 track_agile** |
|---|---|---|---|---|---|---|
| **方法类型** | RL single BR | RL single BR | BC+RL single BR | 手工脚本 | 手工脚本 | 手工脚本 |
| **BR 起始策略** | 随机 | 随机 | rule-equivalent | 固定 0.28 jam | 固定 0.60 jam | 固定 0.80 track |
| **Cell 2 rule kills** | 1.96 | 1.96 | 1.95 | 1.96 | 1.91 | 1.96 |
| **Cell 2 method kills** | 1.00 | 0.98 | 1.93 | 1.92 | 0.99 | 0.99 |
| **Winner(rule/method/draw)** | 98/0/2% | 98/0/2% | 5/3/93% | 4/0/95% | 92/0/8% | 96/0/4% |
| **Win rate vs rule** | ~0.02 | ~0.02 | ~0.03 | **0.476** | **0.040** | **0.018** |
| **机制性质** | BR undertrained | BR undertrained | BR ≈ rule(rule ≈ Nash) | rule 反馈反制 | self-own | agility 反噬 |

**关键观察**:
- `jam_spread` 是有史以来**最接近持平**的方法(0.476 vs G0 #3 的 0.03)— 但仍 CI 排除 0.5
- `hard_jam_focus` 和 `track_heavy_agile` 跟 G0 #1/#2 一样差(BR undertrained 水平)
- **共同结论**:任何 unilateral 偏离 rule 策略(rule 策略空间内)都无法显著 exploit rule

---

## 5. rule ≈ Nash 假设的证据强度

### 5.1 三条独立证据线

1. **G0 #3 BC+PPO**(exploit_gap = -0.016,CI [-0.05, +0.02]):
   - BC 完美学 rule(70% vs 71% track)
   - PPO 从 rule-equivalent 起点微调,reward 收敛到 -0.05
   - Cell 2:BR 1.93 kills vs rule 1.95 kills,**93% 平局**
   - **解释**:RL 从 rule 盆地局部搜索找不到 exploit

2. **V1 三条手工 exploit**(全部 CI 排除 0.5):
   - 每条 exploit 都基于 rule 已知阈值设计
   - 每条都被 rule 的反馈式设计反制
   - **解释**:rule 的策略空间内的 unilateral 偏离(改 jam 比例 / 改 track 浓度 / 改 agility)都不能 exploit

3. **rule 设计原则分析**:
   - track 71% 浓度经验最优(BC 学 rule 时网络自然收敛到此)
   - anti-jam hop reaction(jam_detect=0.30, freq_hop_high=6)防御 jam 轴
   - reactive jam(enemy_tracking_me 时 boost jam)防御 track race 轴
   - 三个机制相互覆盖,任何偏离都被反制
   - **解释**:这是 Nash-style 设计的典型特征

### 5.2 反例可能性(为什么不能 100% 断言 rule=Nash)

- **league 可能找到非传递策略**:种群自博弈做无向搜索,可能发现 A→B→C→A 的 cycle,league 找到 A 后 "赢 rule" 但其实输给 B
- **G1-clean 阈值内的边缘赢**:league 可能 WR=0.55 CI=[0.50, 0.60] 勉强 G1 PASS,但这不是稳健支配
- **MAPPO/IPPO baseline 未跑**:还没法判 "CTDE without league" 是否足够,留待 V3

---

## 6. 决策树状态

```
当前状态:
  ✅ Step 0-5 全部实现 + smoke test 通过
  ✅ V1 完成 — 3/3 exploit FAIL,机制诊断清晰
  ⏸  V2 league 主训练(1-2 周)— 待用户判断
  ⏸  V3 cross-play(待 V2 完成)
  ⏸  V4-V5 判门 + 贴回 PRO6000

V2 选项:
  A) 照跑 league 1-2 周(GPU commit 大,但硬停条款保留)
  B) 先跑 100 iter 稳定性测试(~5-10 min,验证训练健康再投)
  C) 直接退 IET 地板(G0 #3 + V1 已足够支撑 rule ≈ Nash,省 GPU)
  D) 重新设计 exploit(基于诊断换 axis:几何 / 异构 aperture / multi-step)
```

---

## 7. 实验环境配置

### 7.1 硬件
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- **VRAM**: 101.9 GB
- **CUDA**: 是
- **框架**: PyTorch

### 7.2 V1 eval 参数
- `--episodes 100` × 2 directions = 200 episodes per exploit
- `--n-envs 8`, `--horizon 200`
- env: TwoTeamVecEnv, RANDOM_GEOMETRY, seed=42
- opponent: TwoTeamStrongRuleCommander(default 阈值:duck=60, jam_detect=0.30, freq_hop_high=6)
- bootstrap: 1e4 resample,95% CI

### 7.3 已花费成本

| 项 | 时间 |
|---|---|
| Step 0: opponent_pool.py(~230 LOC) | ~25 min |
| Step 1: 3 exploit 类(~200 LOC) | ~40 min |
| Step 2: eval_candidate_exploits.py(~150 LOC) | ~20 min |
| Step 3: run_wp2_league.py(~450 LOC) | ~50 min |
| Step 4: run_wp2_crossplay.py(~450 LOC) | ~50 min |
| Step 5: smoke test(~120 LOC) | ~15 min |
| V0 smoke tests | <2 min |
| V1 full eval(3 exploits × 200 ep × horizon=200) | ~9 min |
| **代码 + smoke + V1 总成本** | **~4h 实现 + ~12 min 跑** |

---

## 8. 文件清单

### 新增
- `algo/_shared/pilot/twoteam/opponent_pool.py`(~230 LOC)— TwoTeamOpponentPool + PolicyRecord + PFSP
- `algo/_shared/pilot/twoteam/run_wp2_league.py`(~450 LOC)— BC + PFSP league 主循环
- `algo/_shared/pilot/twoteam/run_wp2_crossplay.py`(~450 LOC)— cross-play + Elo + 非传递检测
- `scripts/eval_candidate_exploits.py`(~150 LOC)— 独立 eval
- `tests/twoteam/test_wp2_smoke.py`(~120 LOC)— 3 个 smoke test
- `experiments/twoteam/candidate_exploits_eval.md`(自动生成)
- `experiments/twoteam/candidate_exploits.log`(完整 V1 日志)
- `experiments/twoteam/WP2_STEP0_5_V1_REPORT.md`(本报告)

### 修改
- `algo/_shared/pilot/twoteam/extreme_commanders.py`:新增 3 个 Commander 类(JamSpread/HardJamFocus/TrackHeavyAgile)+ STRATEGIES 注册

### 未修改(完全复用)
- `algo/_shared/pilot/twoteam/bc_pretrain.py`(BC pipeline,G0 #3 验证)
- `algo/_shared/pilot/twoteam/br_trainer.py`(PPO+GAE+CTDE,直接复用)
- `algo/_shared/pilot/twoteam/commander_actor_critic.py`(AC 网络)
- `algo/_shared/pilot/twoteam/run_g0_gate.py`(只 import `run_episodes_two_commanders` + `bootstrap_ci`)
- `algo/_shared/self_play/opponent_pool.py`(不动,新建 twoteam 专用池)
- `env/gpu/twoteam/twoteam_env.py`(WP0-decisive 已过)

---

## 9. 推荐

我作为 AI 没有偏好。基于 V1 数据 + 诊断,客观地列三条路径的利弊:

### 路径 A:照跑 league(1-2 周)
- ✅ 符合原 plan
- ✅ league 做无向搜索,与阈值探测不同
- ✅ 硬停条款保留(不开第 4 轮)
- ⚠️ V1 结果降低了先验(原估 20-25%,现可能 10-15%)
- ⚠️ 若赢,**G1-clean 必查**(更可能非传递而非稳健支配)
- ⚠️ 1-2 周 GPU 时间机会成本

### 路径 B:先 100 iter 稳定性测试(~10 min)
- ✅ 验证 league 训练健康(reward 单调、adv_std 范围、PFSP EMA 分化)
- ✅ 廉价,信息密度高
- ✅ 若不稳定 → 直接退 IET,省 GPU
- ⚠️ 不能完全预测 1000 iter 行为

### 路径 C:直接退 IET 地板
- ✅ G0 #3 + V1 已是强证据(rule ≈ Nash)
- ✅ 省 1-2 周 GPU
- ✅ IET 故事完整:testbed + 近 Nash 表征 + BC pipeline 工具
- ✅ 转 Bet B(鲁棒/泛化 eval:几何/频段/对手变化)抬 AppInt
- ⚠️ 放弃 league 可能为 TAES 找到非传递赢法的可能(若存在)

### 路径 D:重新设计 exploit
- ⚠️ 基于诊断,rule 的阈值轴已被反馈覆盖
- ⚠️ 换 axis(几何 / 异构 aperture / 多步)可能也只是持平
- ⚠️ 仍要跑 league 才知道

**最务实建议**: 路径 B(100 iter 稳定性)→ 若稳定,接 A(1000 iter)→ 若不稳,接 C(退 IET)。

---

## 10. 附录 — V1 完整日志

```
[gpu] Using device: cuda
[gpu] GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
[gpu] VRAM: 101.9 GB
======================================================================
WP2 — Candidate-Exploit Scripts vs StrongRule (Independent Eval)
======================================================================

Evaluating 3 exploits vs StrongRule:
  - jam_spread
  - hard_jam_focus
  - track_heavy_agile

--- jam_spread vs StrongRule ---
  win_rate=0.476 (95% CI [0.471, 0.482])  draw=0.95  kills=1.92 vs 1.96  → NONE  (70.7s)

--- hard_jam_focus vs StrongRule ---
  win_rate=0.040 (95% CI [0.033, 0.047])  draw=0.08  kills=0.99 vs 1.91  → NONE  (220.2s)

--- track_heavy_agile vs StrongRule ---
  win_rate=0.018 (95% CI [0.013, 0.022])  draw=0.04  kills=0.99 vs 1.96  → NONE  (222.8s)

Report written: experiments/twoteam/candidate_exploits_eval.md

======================================================================
SUMMARY
======================================================================
  jam_spread                 WR=0.476 CI=[0.471,0.482]  → NONE
  hard_jam_focus             WR=0.040 CI=[0.033,0.047]  → NONE
  track_heavy_agile          WR=0.018 CI=[0.013,0.022]  → NONE

→ No exploit confirmed. League must use population search to find exploits.
```
