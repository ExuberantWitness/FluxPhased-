# G3-BSTA-v0 Clean-Successor Spec (Draft)

**Status**: DRAFT — for adoption owner review
**Origin**: This spec is derived from G3-BSTA PRO6000_AGENT_IMPLEMENTATION_SPEC (handoff docs branch `docs/g3-bsta-pro6000-handoff` @ `5cafbbfd`) **as informed by** static inspection of the quarantined orphan package (case `mfr-orphans-20260728T094154Z`).
**Adoption path**: if approved, implementation proceeds on a NEW branch `g3-bsta/clean-successor` from current HEAD `807588cab7d367bedd415b45efc85a72f2a38b89`. The orphan package is **reference-only**, not committed.

---

## 0. 不变量(实施前不许偏离)

1. orphan 文件**不**直接 commit。新代码可以参照 orphan 写法,但每次写入都基于本 spec,且通过本 spec 的不变量审查。
2. adoption commit 必须含 7 条 trailer(见 `ORPHAN_ADOPTION_ELIGIBILITY.md` §4)。
3. 新基准命名 `G3-BSTA-v0`;禁止 `recovered-M7` / `original-M7` / `fixed-G2a` / `mfr-restored` 等历史名义。
4. 原 G2'a 保持 **FAIL**。新基准的任何 PASS 都**不**回写成原 G2'a 修复。
5. 进入 P1 前必须完成 P0-Binding 八项物理绑定。

---

## 1. 物理绑定(P0-Binding,优先解决)

| # | 绑定项 | 候选值(orphan 线索) | 决策依据 | 必须 owner |
|---|---|---|---|---|
| 1 | 发射机数 | K=4 digital subarrays;30% 目标 emitter=True | RF chain 物理定义 | RF owner |
| 2 | per-emitter 峰功率 | 实验 factor 100W;无 cap | 平台 datasheet | RF owner |
| 3 | per-emitter episode 能量 | 无;`E_j,0 < P_fixed·T_active_max` 待定 | 平台 mission scenario | RF + mission owner |
| 4 | 能量池化 | 无 team pool | RF 共享电源/协同 cap 证据 | RF owner |
| 5 | 同时波束上限 | K_j=1 per emitter 或 K_team=1 | RF chain / mission | RF owner |
| 6 | service 选择性 | 无 frequency 维度 | 接收机/beam/range-Doppler/freq | RF owner |
| 7 | 雷达接收机绑定 | IMM-PDAF tracker 通用 | 接收机/dwell 物理对应 | RF + sim owner |
| 8 | cross-talk | 依赖 `iq_interference` 通用 | 跨 service 选择性 gate | sim owner |

**STOP 条件**:任何一项无法物理绑定,返回 `BLOCK_PHYSICS_SEMANTICS`,不进入 P1。

---

## 2. 资源状态(per-emitter,team pool 仅在 (4) 通过时)

```
E_{j,t+1} = E_{j,t} - Δt · P_{j,exec,t}
E_{j,t} ≥ 0
0 ≤ P_{j,exec,t} ≤ P_{j,max}
```

固定功率 `P_fixed` 时,排除 always-on 的硬约束:

```
E_{j,0} < P_{fixed,j} · T_{active,max,j}
```

(必须来自平台,不能为打破 always-on 后设)

---

## 3. Action 合同

```
a_t ∈ {0} ∪ {(emitter j, physically-addressable service i)}
```

- `0` = idle(永远合法)
- `(j,i)` = emitter j 以固定校准功率干扰物理可寻址 service i
- 单一等效 transmitter 时退化至 `idle-or-target`(必须 P0 证明)
- mask 为固定 categorical,**不**用 J 个独立 Bernoulli
- service i 必须对应**可验证选择性机制**:receiver / beam direction / range-Doppler-angle gate / frequency channel / 时间对准 dwell

**target-local action 物理不可定义**时:`BLOCK_TARGET_LOCALITY`,不得注入内部 task JNR。

---

## 4. 干扰与 SINR

```
J_{i,t} = Σ_j z_{j,t} · P_{j,jam} · G_{j→i,t} · ρ_{j,i,t} · L^{-1}_{j,i,t} · χ_{j→i,t}

SINR_{i,t} = S_{i,t} / (N_{i,t} + J_{i,t} + C_{i,t})
```

- 线性功率域计算
- `G`/`ρ`/`L`/`χ` 定义、单位、校准进入 `SYMBOL_MAP.md`
- 缺失项**不得**置为方便的 0/1 常量

---

## 5. Task-Specific Progress

### 5.1 Detect(主模式)

```
P_{d,i,t} = Q_N(√(2γ_{i,t}), √(η(P_fa, N)))
```

P0 后由 task semantics **冻结唯一 transition**(累计非中心参数 / expected service / 显式 detection draw 三选一,不留 executor 临场选)。

### 5.2 Track/Estimate(若 source 明确)

完整预测—检测—量测更新:

```
ΔJ_{i,t} = H^T_{i,t} R^{-1}_{i,t} H_{i,t}
P^- = F P F^T + Q
```

数据关联、状态坐标变换、covariance→task completion 映射必须实现;不可用 CRB 代替完整 tracker。

### 5.3 Legacy mode

`legacy_sqrt` 仅用于 regression/ablation;`eps` 只防除零,不造成物理 floor。

---

## 6. Observation Contract

**Actor 可用**:
- `remaining_energy / initial_energy`
- `time_remaining / episode_horizon`
- 每 candidate visibility/feasibility
- 延迟带噪 emission/carrier/bearing/gain proxy
- intercept confidence 与 age
- ESM/历史推断的 task urgency proxy
- previous executed action

**Actor 不可用**:
- 真值 queue length / exact progress / exact deadline(除非证明 jammer 可截获)
- 当前未观测的 radar hop
- 未来 task arrivals
- post-action detector outcome
- rule radar next action
- env RNG state

每个 observation field 记录:name / slice / shape / range / unit / causal source / latency。

**action mask = actor-visible observation/history 的确定函数**;不可从真值 alive / hidden geometry / 内部 task state 额外生成。

slot 系统定义 false alarm / miss / latency / slot birth-death / ID reuse。

---

## 7. Reward 与 metrics

### 7.1 主 metric(原始)

```
drop_ratio = dropped_tasks / eligible_arrivals
```

冻结 numerator / denominator / NA 处理 / reset / unfinished-at-horizon / termination / aggregation order。

### 7.2 training reward

固定一种,potential-based shaping only:

```
r'_t = r_t + γ Φ(s_{t+1}) - Φ(s_t)
```

terminal 处 Φ 正确归零。energy 是硬约束,不加 learner-only active penalty。

### 7.3 secondary metrics

`drop_per_kJ`(NA if energy=0)/ target switching / allocation entropy / energy used / constraint violations / requested-executed mismatch / return-drop correlation / `P_d` / SINR / drop-cause distributions。

secondary 不替代 raw-drop superiority。

---

## 8. 公平比较基线 family

所有基线使用同一 observation + action feasibility path:

1. `always_off`
2. `random_feasible`
3. `budgeted_barrage` / maximum-feasible fixed schedule
4. `budgeted_round_robin`
5. `periodic_blink`(预注册 duty/phase family)
6. `edf_or_threat_first`
7. `reactive_target_follower`
8. `marginal_information_per_joule`
9. `assignment_index`(Hungarian / knapsack)
10. `untrained_actor`
11. `trained_shuffled_observation`

每 family 完整超参网格 + scenario-macro selection score + deterministic tie-break 预注册;calibration 后每族保留一个 frozen finalist,生成 `BASELINE_FREEZE.json`。**确认性 gate 检验全部 frozen finalists**,不只 winner。

---

## 9. Oracle-first reachability

### 9.1 当前环境(on/off action)

```
g_s = U_s - B_{b*,s}    (U_s = exact/admissible pathwise upper bound)
gate: UCB95(E[g_s]) < 0.05  →  STOP_CURRENT_G2A_INFEASIBLE
```

否则 `INCONCLUSIVE`;不可凭 5 个测试策略声称 universal impossibility。

### 9.2 G3-BSTA environment

1. planner-dev split 上 reduced exact DP(2-3 services、绑定 beam cap、离散 energy、短 horizon)
2. full causal planner witness(beam search / MCTS / receding horizon)
3. clairvoyant planner 仅作 optimistic ceiling
4. planner/config/operating point 冻结后转 headroom-confirmation split
5. headroom 证据 = same-observation causal witness;belief update 同 actor

### 9.3 Pre-PPO headroom gate

```
min_{b∈B} LCB95(D_causal_witness - D_frozen_finalist_b) > 0.075
```

相邻 energy settings + detector calibration 两端:

```
LCB95(Δ) > 0.05
```

energy × detector-envelope × scenario-shift 的 sensitivity cells 预注册;baseline/planner 不在 cell 内重调。

---

## 10. Learnability 控制前 pilot

- trained actor vs frozen-init vs `random_feasible` 一致正差
- candidate features/history 与 slot/mask/action 同步预注册置换
- target choice 随 causal state 改变,非固定 action/duty
- action/state dependence 超出 permutation null
- reward 与 raw drop 方向一致

不可分(trained ≈ untrained ≈ random ≈ shuffled)→ "尚未证明学习",非"达到 optimum"。

---

## 11. 最终 G3-BSTA 统计 gate

- 独立 training seeds 是最高独立单位
- locked environment/scenario seeds 配对 learned 与 script
- stochastic policy action RNG 单独记录
- detector RNG 与 env RNG 单独记录
- **≥ 8 independent training seeds**
- 主 t inference 条件于固定 locked scenario suite;若推广至 scenario generator,另报 crossed random-effects sensitivity

```
H_{0,b}: E[d_{i,b}] ≤ 0.05      H_{1,b}: E[d_{i,b}] > 0.05
PASS: min_b LCB95_b(d) > 0.05   (intersection–union)
```

每 train seed 跑完整 `scenario × action-replicate × episode` 网格;exogenous randomness 用无状态 event key 对齐;action RNG 独立。

部署 sampled / deterministic 在训练前写 `EVAL_PROTOCOL.md`,**禁止**看过结果后切换。

---

## 12. Falsifiable claim boundary

**可声称**(全 gate 通过):
> 在预注册、物理绑定的 budgeted addressable-service MFR-IQ 新 benchmark G3-BSTA-v0、rule-radar opponent、causal observation、固定资源约束下,PPO jammer 在 N 个独立 training seeds 上相对每个冻结 competent scripted finalist 的 raw drop_ratio 提高超过 5pp;该结论条件于 locked scenario suite。

**不可声称**:
- PPO 普遍优于 scripted jammers
- resource allocation / PPO 本身新颖
- planner 是 exact upper bound
- 原 G2'a 被修复为 PASS
- simulation-only 结果代表真实系统
- "recovered M7" / "reproduced G2'a" / "original implementation"

---

## 13. P0-P9 阶段(简表)

| 阶段 | 内容 | 阻断条件 |
|---|---|---|
| P0 | provenance + symbol resolution | orphan 不可作 source;需 SOURCE_HANDOFF 或 B+ adoption |
| P0-B | resource + selectivity binding | 8 项物理绑定(本 spec §1) |
| P1 | legacy diagnostics + RNG separation | event-key CRN |
| P2 | IQ-calibrated progress | detector envelope 验证 |
| P3 | physically bound allocator | `E_j < P_fixed·T_active,max` 排除 always-on |
| P4 | causal observation | no-godview audit |
| P5 | masked categorical PPO | requested == executed |
| P6 | reward/metric alignment | drop_ratio numerator/denominator frozen |
| P7 | baselines + oracle | IUT 全 frozen finalists |
| A4/A5 | untouched robust headroom | 32 scenarios × energy × detector × shift |
| L0 | learnability pilot | 2 seeds only (go/no-go) |
| Power | variance pilot | N≥8 (power-based) |
| Confirm | full training | exactly N_train seeds |
| Locked G3-BSTA | locked gate | IUT PASS |

---

## 14. 与 orphan 的关系

orphan 提供的**参考线索**(非权威):
- 包结构(`env/gpu/mfr/`, `algo/_shared/pilot/mfr/`, `tests/mfr/`)可用作 namespace 命名约定
- σ-progress coupling 公式 `clamp(1/√(1+JNR), 0.1, 1.0)` 在 P2 阶段需 IQ 校准,不能直接搬用
- `tau_track=4.0` 在 P3 阶段需 RF owner 确认
- entrypoint 命名(`run_stage_a/b/b_jammer`, `league_eval`)可用,但 config 默认值必须 P0-B 后重写
- `action_mask` 在 mfr_env.py 的构造可作为 P4 audit 的对照参考,但需重写以满足本 spec §6 不变量

orphan **不提供**:
- 八项物理绑定的物理依据
- config template
- raw rows / checkpoint
- LICENSE
- author signature

---

## 15. Adoption 后的 commit 顺序

1. repo owner 在 `main` 上 commit `LICENSE`(选定 license)
2. adoption owner 在新 branch `g3-bsta/clean-successor`(base=807588c)上 commit:
   - `G3_BSTA_CLEAN_SUCCESSOR_SPEC.md`(本文件,从 draft 转 frozen)
   - `RESOURCE_AND_SELECTIVITY_CONTRACT.md`(P0-B 输出)
   - `SYMBOL_MAP.md`(stub,可在 P0 阶段填)
   - `OBSERVATION_SPEC.md`(stub,P4 阶段填)
   - `EVAL_PROTOCOL.md`(stub,P5 阶段填)
   - `ADOPTION_DECISION_PACKET.md`(签名页)
   - 7 条 trailer(Source-attribution 等)
3. P0 完成后再分阶段 commit P1-P9 实现
4. 任何阶段禁止 fast-forward 合入 `main` 或 `twoteam/bc-ppo`

---

## 16. 草案状态

```text
spec_status: DRAFT
awaiting_signoff_from:
  - repo_owner (LICENSE + adoption authorization)
  - rf_physics_owner (8 physics bindings)
  - experiment_owner (metrics + tests framework)
  - adoption_owner (current-version responsibility)
spec_freeze_prereq: all 4 signoffs recorded
implementation_start_prereq: spec frozen + P0-Binding complete
```
