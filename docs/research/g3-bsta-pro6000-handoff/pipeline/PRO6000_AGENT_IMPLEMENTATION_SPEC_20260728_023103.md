# PRO6000 Agent Implementation Specification

## 0. Authority, Scope, Gate Identity, and Current Blocker

这是给 PRO6000 agent 的实施工单。目标是实现 **MFR-IQ Budgeted Sparse Target Jammer v3**，不是继续调现有 PPO。

本环境和协议使用新的 gate ID：

```text
G3-BSTA
```

`BSTA` 表示 `Budgeted Service-Target Allocation`。原 `G2'a` 永久保持
`FAIL / unverified reachability`；任何 G3-BSTA 结果都不得回写为原 G2'a
PASS。代码、结果目录、schema 和 reason code 必须使用 `g3_bsta` /
`G3_BSTA_*`，避免把新问题误写成旧 gate 的修复结果。

审计机当前 checkout：

- 仓库：`/home/exuber/CODE/CORE/pythonProject1/FLUXPH/FluxPhased-`
- local HEAD：`fa485ad4cf6314df8a747498f4179d702a7c4923`
- public tracked branch 当前可见最新：`origin/twoteam/bc-ppo`
- 附件声称的 `env/gpu/mfr/*`、`algo/_shared/pilot/mfr/*`、M7 checkpoints、raw gate records 和四个 `/tmp` scripts 均不在可见 tree 中。

因此 PRO6000 **不得直接照猜测路径打 patch**。当前包不包含可执行 M7
源码；“PRO6000 另有未同步源码”不是可依赖的事实，必须先由可验证交接文件
证明。第一项输入必须是：

```text
SOURCE_HANDOFF.json
```

最少 schema：

```json
{
  "repository_path_or_uri": "",
  "verified_m7_commit": "",
  "source_tree_hash": "",
  "required_paths": [
    "env/gpu/mfr",
    "algo/_shared/pilot/mfr"
  ],
  "legacy_artifact_bundle_uri": null,
  "legacy_artifact_bundle_sha256": null,
  "resource_assumption_owner": "",
  "handoff_owner": "",
  "handoff_timestamp": ""
}
```

PRO6000 必须逐项解析并验证 commit、tree hash 和 required paths，不能只信
文件中的字符串。源未交接或 hash 不符时返回 `BLOCK_SOURCE_HANDOFF` 和
`BLOCK_PPO_PROVENANCE`，不得创建猜测实现。

必须遵守目标仓库根 `AGENTS.md`：

- 保持 shared code 结构；
- top-level `configs/` 是 legacy frozen，不要在那里新增或修改 v3 config；
- 新配置放 MFR pilot 自己的 config 目录或版本化 experiment directory；
- 不破坏用户已有 dirty changes；
- 不把唯一脚本或结果留在 `/tmp`。

P0 分为两个不同依赖：

- **P0-A source provenance**：实现所必需；缺失即阻塞全部代码工作；
- **P0-B legacy-result provenance**：用于审计原报告。若旧 raw rows、
  checkpoints 或 `/tmp` scripts 无法恢复，记录
  `LEGACY_CLAIM_UNVERIFIED`；这不授权重构原结果，也不得将原 G2'a
  写成 PASS。只有在 P0 人工批准明确允许“新环境从零预注册”时，才可继续
  G3-BSTA 的代码工作。

P0 结束后必须人工停点。PRO6000 生成 `P0_BINDING_PACKET.md` 和
`P0_BINDING_APPROVAL.json` 后返回
`AWAIT_P0_HUMAN_BINDING_APPROVAL`。批准文件必须冻结：

- verified M7 commit；
- jammer emitter 实体和 service selector 的物理语义；
- per-emitter power/energy/beam 资源值及来源；
- actor 可见 observation、association、latency 和 mask 字段；
- 唯一 primary transition 与唯一 primary training reward；
- G3-BSTA 统计 estimand、primary eval mode 和计算预算上限。

没有非空批准人、时间戳和所有绑定项的 hash，不得进入 P1。后续发现任一
绑定项必须改变时，作废批准、递增 protocol version，并重新人工停点。

## 1. Non-Negotiable Design Decisions

### Core v3

- 显式 jammer emitter 集合 `e=1,...,N_emitters`，不得把多个物理平台
  静默合并为可任意转移能量的虚拟池；
- `K_team=1` 仅在 P0 物理绑定批准后成立；同时保留每 emitter 的
  hard peak power、hard episode energy 和 beam 可用性约束；
- 每步动作只有 `idle` 或选择一个当前因果可观测且物理可寻址的
  `(emitter_id, service_slot)`；
- `service_slot` 是由 ESM/receiver processing 可定义的
  angle/range/Doppler/frequency/waveform service cell，不是隐藏
  env task ID、queue index 或真值 target ID；
- active 时使用该 emitter 的固定校准功率；
- masked categorical joint action；
- 经 receiver、selector 和 cross-service leakage 校准的
  emitter×service interference；
- IQ 校准的 detection/Fisher-information progress；
- learned、scripted、planner 共用一个 action/resource/physics path。

在写 allocator 前必须证明至少一种 target/service selectivity 机制：
angle beamforming、range/Doppler gate、frequency/waveform matching 或
经校准的等价处理。必须给出 cross-service coupling matrix；若只能得到
receiver-wide barrage JNR，或选择 slot 只是直接索引内部 task，则返回
`BLOCK_TARGET_LOCALITY`，不得以 target-local JNR 继续。

### Explicitly Out of Scope

- learner-only active-cost；
- frequency hopping/retuning；
- exposure、home-on-jam、nulling、thermal/cooldown；
- continuous power；
- multi-beam；
- co-learning radar；
- 后验 parity gate；
- 以 clairvoyant oracle 证明可部署 headroom；
- 用 `legacy_sqrt` 作为主结果。

任何 out-of-scope 项只能在 v3 core 被 falsify 后另开 proposal，不能在实现中悄悄加入。

## 2. Required Branch and Commit Strategy

从 **已验证含完整 M7 的 commit** 建新分支，例如：

```bash
git status --short --branch
git rev-parse HEAD
git switch -c feat/mfr-g3-bsta-v3 <verified-m7-commit>
```

如果 worktree 有未提交更改，先报告并避开冲突；不要 reset、checkout 覆盖或 stash 用户更改。

建议提交序列：

1. `chore(mfr): restore g2a provenance and version gate tools`
2. `test(mfr): add metric rng and saturation diagnostics`
3. `feat(mfr): add calibrated task progress behind feature flag`
4. `feat(mfr): add shared budgeted target allocation`
5. `feat(mfr): use masked categorical jammer policy`
6. `feat(mfr): add frozen baselines and reachability planner`
7. `exp(mfr): add headroom pilot and g3-bsta gate`

每个 commit 前运行该阶段 tests。不要在未通过 headroom gate 时提交 full-training “结果”。

## 3. Phase P0 — Source Handoff, Restore, Resolve, and Human Bind

### 3.0 Validate `SOURCE_HANDOFF.json`

在任何 branch、patch 或新文件之前：

1. 解析 `SOURCE_HANDOFF.json`；
2. 验证 repository 可访问、commit 存在、checkout tree 与
   `source_tree_hash` 一致；
3. 验证 required paths 和 M7 entry point 可达；
4. 记录 dirty state，但不修改、stash 或覆盖用户更改；
5. 将验证输出写入 `P0_SOURCE_HANDOFF_REPORT.md`。

任一步失败立即返回 `BLOCK_SOURCE_HANDOFF`。禁止通过搜索相似 symbol
后新建一套平行 MFR 实现来“补齐”缺失源码。

### 3.1 Locate Actual Code

运行：

```bash
rg -n "progress_add|sigma_scale|prog_factor|tgt_jnr" env algo tests experiments
rg -n "JAM_POLICY_NOISE|off_alive|reactive|blink" env algo tests experiments
rg -n "Bernoulli|logits|entropy|evaluate_actions|log_prob" algo tests
rg -n "drop_ratio|dropped|task_arrival|deadline" env algo tests experiments
rg -n "manual_seed|Generator|SeedSequence|seed" env/gpu/mfr algo/_shared/pilot/mfr
```

如果目录名已变化，记录实际 path，不要为了匹配附件而新建平行实现。

### 3.2 Create `SYMBOL_MAP.md`

放在：

```text
experiments/mfr_phaseB/g3_bsta/SYMBOL_MAP.md
```

至少填完：

| Concept | Actual path:symbol | Shape | Unit | Causal timing | Notes |
|---|---|---|---|---|---|
| env batch | | | | | |
| task slots | | | | | |
| task identity | | | | | |
| task progress | | | | | |
| task deadline | | | | | |
| drop numerator | | | | | |
| drop denominator | | | | | |
| jammer count | | | | | |
| emitter identity and platform | | | | | |
| old action | | | | | |
| jammer position/gain | | | | | |
| radar receiver identity | | | | | |
| observable service slot | | | | | |
| service selectivity mechanism | | | | | |
| cross-service leakage matrix | | | | | |
| carrier/channel | | | | | |
| `S/N/J/JNR/SINR` | | | | | |
| observation build time | | | | | |
| reward build time | | | | | |
| episode `dt/horizon` | | | | | |
| RNG streams | | | | | |

特别回答：

- observable service slot 的 causal association 是否跨 step 稳定；
- slot 是否来自 causal ESM association，而非真值 target/task ID；
- task urgency 哪些字段能被 jammer 物理截获；
- `100W` 是 per jammer、per beam 还是 team total；
- 能量是否能在 emitters 间转移；若不能，必须使用 per-emitter buckets；
- JNR 是 transmitter-side、receiver input 还是 post-processing；
- emitter 发射为何只影响某个 service；非选择 service 的 leakage 是多少；
- progress 是 dwell、标准差、累计能量、检测统计量还是纯抽象；
- unfinished-at-horizon 是否计 drop。

### 3.3 Restore Artifacts

将四个旧 `/tmp` scripts 恢复到：

```text
experiments/mfr_phaseB/scripts/legacy/
```

至少包括 scripted sweep、gate、logit inspection 和 sampled evaluation。保存：

- exact source；
- invocation；
- original output；
- config；
- raw rows；
- SHA-256。

大 checkpoint 可不直接进 Git，但必须有 `checkpoint_manifest.json`，包含 hash、size、storage path/URI、train seed 和 config hash。不要只保存 `final.pt` 文件名。

### 3.4 Freeze Binding Contracts

P0 必须额外交付：

```text
experiments/mfr_phaseB/g3_bsta/
├── P0_SOURCE_HANDOFF_REPORT.md
├── P0_PROVENANCE_REPORT.md
├── P0_BINDING_PACKET.md
├── P0_BINDING_APPROVAL.json
├── TRANSITION_CONTRACT.md
├── OBJECTIVE_CONTRACT.md
└── RESOURCE_ASSUMPTIONS.md
```

`TRANSITION_CONTRACT.md` 必须根据实际 task 语义冻结唯一 primary
transition。若多 task type 并存，可冻结逐 type 的确定映射，但每一 type
只能有一个主路径。必须在 `expected service`、`explicit detection draw`
和 tracker measurement/update 中选择并冻结，不能在看到 headroom 或 PPO
结果后切换。

`OBJECTIVE_CONTRACT.md` 必须冻结：

- `drop_ratio` numerator、denominator、zero-arrival、unfinished-at-horizon；
- per-episode/per-scenario/per-training-seed 聚合顺序；
- 唯一 primary training reward 及 fixed normalizer；
- reward 与 primary raw-drop estimand 的理由和回归测试；
- secondary metrics，且明确它们不替代 raw-drop。

`RESOURCE_ASSUMPTIONS.md` 必须逐 emitter 给出 power 是 conducted、
radiated、average 还是 EIRP，PA/duty-cycle 假设、episode energy、beam
数、值域来源和不确定区间。不得从 headroom/test seeds 反推资源值。

### P0 Exit — Mandatory Human Stop

source 无法恢复：`BLOCK_SOURCE_HANDOFF`。物理 task 语义无法绑定：
`BLOCK_PHYSICS_SEMANTICS`。service selectivity 无法成立：
`BLOCK_TARGET_LOCALITY`。

若 source 可用但旧结果 artifacts 不全，写 `LEGACY_CLAIM_UNVERIFIED`，
不伪造恢复。完成上述 packet 后必须返回
`AWAIT_P0_HUMAN_BINDING_APPROVAL`；只有批准文件验证通过才允许进入 P1。

## 4. Phase P1 — Instrument Legacy Behavior Without Changing It

### 4.1 Freeze Metric Definition

新增纯函数或明确入口：

```python
def compute_drop_ratio(
    dropped_count: Tensor,
    eligible_arrivals: Tensor,
) -> Tensor:
    ...
```

必须测试：

- numerator/denominator；
- zero arrivals；
- reset；
- vector-env aggregation；
- unfinished-at-horizon；
- per-episode → per-scenario → per-training-seed aggregation order。

### 4.2 Event-Keyed RNG Streams

至少分离：

- environment/arrival RNG；
- target/geometry RNG；
- detector RNG；
- policy action RNG；
- training minibatch RNG。

仅分离几个顺序 generator 不足：不同 policy 会改变分支和 draw 次数，从而
使后续 detector/arrival draws 错位。所有外生随机数必须由不可变 event key
派生，使用 counter-based RNG（如 Philox）或经过测试的 hash-to-seed
实现，不得依赖“上一次消耗了多少随机数”。

最小 key：

```text
(protocol_hash, "arrival", scenario_seed, episode_id, time_index, entity_id)
(protocol_hash, "geometry", scenario_seed, episode_id, entity_id, field_id)
(protocol_hash, "detector", scenario_seed, episode_id, time_index,
 service_id, look_id)
(protocol_hash, "action", train_seed, scenario_seed, episode_id,
 action_seed, time_index, env_id)
(protocol_hash, "minibatch", train_seed, update_index, epoch_index)
```

策略改变某一步是否触发 detector 时，其他 event key 的数值仍必须不变。
评估可生成版本化 `EXOGENOUS_TAPE_MANIFEST.json`；只保存 key/hash 即可，
但相同 key 必须跨 policy 复现相同 draw。policy action RNG 始终独立，不能
通过 common action uniforms 暗中耦合不同 stochastic policies。

新增 branch-invariance 测试：在两个策略走不同 action/event 分支后，所有
共享的未来 arrival、geometry 和同 key detector draws 仍逐项相同。失败返回
`FAIL_EVENT_KEY_RNG`。

### 4.3 Add Legacy Diagnostic Scripts

版本化：

```text
experiments/mfr_phaseB/scripts/
├── diagnose_saturation.py
├── sweep_fixed_duty.py
├── evaluate_untrained_actor.py
├── compare_sampled_argmax.py
└── reproduce_original_g2a.py
```

`sweep_fixed_duty.py` 至少扫 `p={0,.1,...,1}`，记录：

- active fraction；
- probability at least one jammer active；
- mean/quantiles of JNR；
- `prog_factor`；
- fraction hitting 0.1 floor；
- task arrivals/completions/drops；
- drop cause；
- reward；
- action RNG。

### P1 Exit

若 P0-B 恢复了 legacy artifacts，legacy feature flag 下结果必须与其在预注册
Monte Carlo tolerance 内一致；否则返回 `PROVENANCE_MISMATCH`。若 P0-B
未恢复且已获人工批准从零预注册，只能交付现有 source 的 duty/untrained/
saturation diagnostics，并持续标记 `LEGACY_CLAIM_UNVERIFIED`；不得声称
复现了原结果。

## 5. Phase P2 — Add Calibrated Task Progress

### 5.1 Proposed Files

实际路径以 P0 symbol map 为准。若附件路径成立：

```text
env/gpu/mfr/detector_progress.py       # new, pure physics helpers
env/gpu/mfr/mfr_env.py                 # call helper, own task state
algo/_shared/pilot/mfr/configs/
  jammer_budgeted_target_v3.yaml       # new; not top-level legacy configs/
```

### 5.2 Config Contract

```python
@dataclass(frozen=True)
class JammerProgressConfig:
    mode: Literal[
        "legacy_sqrt",
        "pd_accumulation",
        "fisher_information",
    ] = "legacy_sqrt"
    pfa: float = ...
    target_model: str = ...
    integration_mode: str = ...
    calibration_path: str | None = None
    numeric_eps: float = 1e-12
```

默认值先保持 legacy，直到回归与 calibration 全部通过。G3-BSTA config
必须与 P0 批准的 `TRANSITION_CONTRACT.md` hash 绑定并显式选择唯一主
mode；运行时 config 与 contract 不一致立即失败。不得根据 calibration、
headroom 或 PPO 结果在 `pd_accumulation` 与 `fisher_information` 间选择。

### 5.3 Pure Function Boundaries

```python
def compute_inband_jammer_power(
    allocation,
    emitter_to_receiver_path,
    receiver_gain_toward_emitter,
    spectral_overlap,
    service_selectivity,
) -> Tensor:
    """Return linear receiver-band J [W] for every receiver service."""

def compute_post_sinr(
    signal_power_w,
    noise_power_w,
    interference_power_w,
    clutter_power_w=None,
) -> Tensor:
    """All inputs linear W; return dimensionless SINR."""

def detection_probability(
    sinr,
    pfa,
    n_looks,
    target_model,
    calibration_table,
) -> Tensor:
    """Return Pd in [0,1]."""

def information_increment(
    sinr,
    waveform_features,
    geometry_features,
    calibration_table,
) -> Tensor:
    """Return nonnegative task information increment."""
```

emitter×service 主链路必须在 `TRANSITION_CONTRACT.md` 固定为可审计公式。
推荐张量边界：

\[
J_{r,q,t}=
\sum_{e,q'} a_{e,q',t} P_{e}
G^{tx}_{e,q'}L^{-1}_{e\rightarrow r}
G^{rx}_{r\leftarrow e}\rho_{e,r,q,t}
C_{r,q\leftarrow q'},
\]

其中 `r` 是实际 radar receiver，`q'` 是 jammer 选择的可观测 service，
`q` 是受影响 receiver service，`C` 是经 receiver/IQ 校准的 cross-service
coupling/leakage。所有功率在线性 W 域；每个 gain、path loss、spectral
overlap 和 `C` 的单位、范围、时间索引与因果来源必须进入 symbol map。
禁止把 `C` 实现为 `q == hidden_task_id` 的 one-hot shortcut。

禁止：

- 在 helper 中读取 policy type；
- 混入 reward；
- 将 dB 与 linear 静默混合；
- 用 0.1 之类的可见物理 floor；
- 通过 test performance 调 calibration。

### 5.4 Enforce the Frozen Progress Semantics

PRO6000 在 P0 `SYMBOL_MAP.md` 中确认，并在人工批准前完成选择：

- 若 task progress 本来表示检测服务：用 fixed-`Pfa` `Pd`/非中心参数；
- 若表示 track/estimate uncertainty：实现完整 prediction/update，包括
  state-transition、process noise、missed-detection、data-association 和
  covariance/information 到完成条件的映射；不能跨动态时刻裸加
  `H.T @ R^-1 @ H`；
- 若多 task type 并存：按 type 分派，不能用一条通用曲线。

如果 current task abstraction 无法映射到任何一种，返回 `BLOCK_PHYSICS_SEMANTICS`，不要自创公式。

对 detection service，`expected service`、显式 Bernoulli detection draw
或累计非中心参数是三个不同 transition，只能选择批准 contract 中的一种。
自然的 `Pd→Pfa`、`Pd→1` 渐近不是错误；禁止的是任意 clamp、lookup 截断
或未记录量化造成的人工平台。

### 5.5 IQ Calibration

新增：

```text
experiments/mfr_phaseB/scripts/calibrate_detector_progress.py
experiments/mfr_phaseB/g3_bsta/calibration/raw_iq_pd.jsonl
experiments/mfr_phaseB/g3_bsta/calibration/detector_fit.json
```

JNR 单独不能确定 SINR 或 `Pd`。预注册 calibration/validation grid 至少覆盖：

- baseline receiver SNR/SCR 的实际工作区间；
- receiver-band JNR `[-10,-5,0,5,10,15,20] dB` 或由 source 证明的范围；
- clutter/background model、实际 look counts、integration/correlation mode；
- 实际 target/Swerling models；
- emitter×receiver geometry、spectral overlap 和 selector cross-service
  leakage 的边界条件。

用真实 IQ injection → receiver/filter → integrator → detector。将 cells
预先拆为 fit 与 untouched validation；禁止在同一 cells 拟合并验收。每
validation cell 运行到预注册 CI half-width 或 cap；到 cap 仍未达精度时
该 cell 失败，不能静默接受。

验收：

- validation grid 上 fast model vs IQ 误差满足 P0 中由最终 5pp
  error budget 推导的阈值；`.03` 只能作为待批准上限，不能无依据硬编码；
- `J=0` no-jam completion regression；
- J 增大不改善 radar information；
- 无人工 clamp/lookup 截断平台；
- uncertainty envelope 存档，并传播到 headroom-confirm sensitivity。

### 5.6 Tests

```text
tests/mfr/test_detector_progress_units.py
tests/mfr/test_detector_iq_calibration.py
tests/mfr/test_progress_monotonicity.py
tests/mfr/test_no_jam_regression.py
tests/mfr/test_progress_no_artificial_clamp.py
tests/mfr/test_tracker_information_recursion.py
tests/mfr/test_transition_contract_hash.py
```

## 6. Phase P3 — Bound Emitter×Service Resource Allocator

### 6.1 Proposed File

```text
env/gpu/mfr/jammer_resource.py
```

该模块是 learned/scripted/planner 的唯一动作执行入口。

### 6.2 Config

```python
@dataclass(frozen=True)
class JammerResourceConfig:
    enabled: bool = False
    schema_version: str = "budgeted_emitter_service_v3"
    num_emitters: int = ...
    team_peak_power_w: float | None = None
    emitter_peak_power_w: tuple[float, ...] = ...
    fixed_beam_power_w: tuple[float, ...] = ...
    episode_energy_j: tuple[float, ...] = ...
    maximum_active_duration_s: tuple[float, ...] = ...
    max_team_active_beams: int | None = None
    max_emitter_active_beams: tuple[int, ...] = ...
    num_service_slots: int = ...
    step_dt_s: float = ...
    strict_invalid_action: bool = True
```

Constraints:

```python
if max_team_active_beams is not None:
    assert max_team_active_beams == 1  # only after P0 physical approval
assert len(fixed_beam_power_w) == num_emitters
assert len(episode_energy_j) == num_emitters
assert len(maximum_active_duration_s) == num_emitters
assert all(
    0 < fixed_beam_power_w[e] <= emitter_peak_power_w[e]
    for e in range(num_emitters)
)
if team_peak_power_w is not None:
    assert max(fixed_beam_power_w) <= team_peak_power_w
assert all(
    episode_energy_j[e]
    < fixed_beam_power_w[e] * maximum_active_duration_s[e]
    for e in range(num_emitters)
)
assert all(
    floor(episode_energy_j[e] /
          (fixed_beam_power_w[e] * step_dt_s)) < episode_horizon_steps
    for e in range(num_emitters)
)
```

最后两条才保证每个 emitter 的 fixed-power 全部可活动时段 always-on 不合法。
energy 不得在 emitters 间转移，除非 P0 提供真实共享电源/平台证据并另写
守恒式。实际值和相邻 sensitivity settings 均须来自
`RESOURCE_ASSUMPTIONS.md`，不得为 5pp 调甜点。

### 6.3 Static Template Table

固定 action dimension：

```text
template 0: idle
template 1: emitter 0 jams observable service slot 0 at emitter-0 fixed power
...
template A: every approved feasible (emitter, service slot) pair
```

service slot 必须由 causal ESM track/processing-cell association 创建。不得用
真值 target ID、env task index 或 queue index稳定 slot。association
birth/death/reacquisition、false track、latency 和 permutation 规则写入
`OBSERVATION_SPEC.md`；padding slot 不可用。

### 6.4 Pre-Action Mask

```python
def build_action_mask(
    actor_visible_service_fields,    # [B, S, F]
    actor_visible_resource_fields,   # [B, E, R]
    template_table,
    dt_s: float,
) -> BoolTensor:                     # [B, A]
    ...
```

要求：

- `mask[:,0] == True`；
- active actions 只能依赖 actor 已看到的 service track、causal reachability
  estimate、已知自身 emitter availability 和 remaining energy；
- 禁止读取真值 `alive`、真值 target/task ID、hidden exact geometry、
  queue/progress 或当前未截获的 mode；
- mask 必须是 `(obs_t, approved_static_template_table)` 的确定函数。任意两个
  hidden states 只要 `obs_t` 逐元素相同，mask 必须逐元素相同；
- 如果物理安全层知道 actor 不知道的约束，该约束必须先通过批准的 latency
  进入 observation；不能把 mask 当作隐式 privileged channel；
- mask 和 observation 一起保存到 rollout。

### 6.5 Execute

```python
allocation = allocator.decode(template_id, mask, strict=True)
assert allocation.requested_template_id == allocation.executed_template_id
energy_cost_by_emitter = allocation.power_w_by_emitter * dt_s
remaining_energy_by_emitter -= energy_cost_by_emitter
J_by_receiver_service = compute_inband_jammer_power(allocation, ...)
```

不能让 `JAM_POLICY_NOISE` 直接改 `off_alive` 或绕过 allocator。现有 scripts 改成 adapter，只输出合法 template ID。

训练和正式评估中不存在 invalid-action fallback。mask、decoder 或 stale-state
错误导致 requested/executed 不一致时，立即使该 rollout 和 run 失败并返回
`FAIL_ACTION_EXECUTION_MISMATCH`；不得把 requested action 的 log-prob 与
executed idle 的 transition 放入 PPO。仅离线故障注入测试可以验证错误路径。

### 6.6 Invariants and Tests

```text
tests/mfr/test_jammer_resource_conservation.py
tests/mfr/test_jammer_action_contract.py
tests/mfr/test_jammer_policy_path_parity.py
tests/mfr/test_emitter_service_coupling.py
tests/mfr/test_cross_service_leakage.py
tests/mfr/test_action_mask_causality.py
tests/mfr/test_mask_is_function_of_obs.py
tests/mfr/test_requested_equals_executed.py
```

验收：

- normalized energy conservation error `<=1e-6`；
- energy 永不为负；
- selected 与 non-selected services 的 J 必须匹配批准的 coupling/leakage
  matrix；只有经 IQ/receiver 校准为零时才可假设 non-selected J 不变；
- identical serialized state/template 由三类 adapter 执行时结果逐项相同；
- invalid action 在所有 mode 硬失败；
- requested action 与 executed action 永远逐项相等。

## 7. Phase P4 — Observation Contract

新增：

```text
env/gpu/mfr/jammer_observation.py
tests/mfr/test_jammer_obs_no_leak.py
experiments/mfr_phaseB/g3_bsta/OBSERVATION_SPEC.md
```

建议 observation layout：

```text
global:
  per_emitter_remaining_energy_frac
  per_emitter_known_availability
  time_remaining_frac
  previous_action_onehot_or_id

per_service_slot:
  visible
  association_confidence
  intercept_age_norm
  intercept_confidence
  bearing_sin, bearing_cos
  estimated_gain_or_range_bin
  emission_activity
  carrier_or_mode_likelihood
  causal_urgency_proxy
```

不要直接放 exact `queue`, `progress`, `deadline`，除非
`OBSERVATION_SPEC.md` 证明物理测量机制与 latency。service slot 必须来自
显式 ESM measurement/association pipeline，包含 missed intercept、false
track、birth/death/reacquisition 和 association uncertainty；不能从真值
target list 先建 slot 再只给特征加噪。

测试：

1. observation 在 `action_t` 前构造；
2. 改变 future arrivals 不改变 `obs_t`；
3. 改变 hidden exact progress、保持 intercept history 不变，不改变 `obs_t`；
4. 两个 hidden states 的 `obs_t` 相同时，action mask 必须相同；
5. shuffled service labels 同步 permutation observation/mask/action；
6. slot birth/death/reacquisition 不泄漏 truth identity；
7. normalization 不含 planner-dev、headroom-confirm 或 locked-test statistics。

## 8. Phase P5 — Replace Independent Bernoulli with Masked Categorical

### 8.1 Actual Edit Points

附件指向：

```text
algo/_shared/pilot/mfr/jammer_trainer.py
algo/_shared/pilot/mfr/run_stage_b_jammer.py
```

以 P0 实际 symbol 为准。

### 8.2 Recurrent Actor

由于 observation 明确包含延迟、带噪 intercept history，G3-BSTA primary
policy 使用 recurrent actor；不得让 causal planner 使用完整 history 而
PPO 仅看单帧。RNN 类型、hidden size、burn-in、sequence length 和 reset
语义在 P0 approval 中冻结。actor 和 critic 均只使用批准的 causal
observation/history；本版本不允许 privileged critic。

```python
logits, rnn_state_next = actor(obs, rnn_state, episode_start)  # [B, A]
assert action_mask[..., 0].all()
masked_logits = logits.masked_fill(~action_mask, -torch.inf)
dist = torch.distributions.Categorical(logits=masked_logits)
action = dist.sample(generator=action_gen)  # or project-specific seeded sampler
logp = dist.log_prob(action)
entropy = dist.entropy()
```

如果当前 PyTorch API 不支持 `generator`，使用明确的 seeded multinomial helper；不得退回全局 RNG。

Primary eval mode 在训练前写入：

```text
experiments/mfr_phaseB/g3_bsta/EVAL_PROTOCOL.md
```

若 stochastic sampling 是部署策略，`sampled` primary、`argmax` secondary；若实际部署要求 deterministic，则反过来。看过结果后不得切换。

### 8.3 Rollout Buffer

每步保存：

- `obs_t`；
- `action_mask_t`；
- `action_t`；
- `old_logp_t`；
- `rnn_state_t`、`episode_start_t`、terminated、truncated；
- requested/executed action（必须相等）；
- reward/value/done；
- event-key RNG identifiers。

PPO update：

```python
new_logp, entropy, value = policy.evaluate_sequence(
    obs_seq=batch.obs_seq,
    initial_rnn_state=batch.initial_rnn_state,
    episode_start_seq=batch.episode_start_seq,
    action_seq=batch.action_seq,
    action_mask_seq=batch.action_mask_seq,
    burn_in=approved_burn_in,
)
```

不能用 update 时重新生成的 mask。sequence minibatch 不得跨 episode
拼接 hidden state；terminated 与 truncated 的 GAE/bootstrap 语义必须分开。

### 8.4 Trainer Tests

```text
tests/mfr/test_masked_categorical.py
tests/mfr/test_rollout_mask_replay.py
tests/mfr/test_jammer_ppo_smoke.py
tests/mfr/test_action_rng_isolation.py
tests/mfr/test_recurrent_rollout_replay.py
tests/mfr/test_recurrent_hidden_reset.py
tests/mfr/test_gae_termination_truncation.py
tests/mfr/test_masked_ppo_ratio_kl.py
```

至少验证：

- illegal action probability 为 0；
- old/new policy 不变时 stored/recomputed logp 相等；
- entropy 不含 invalid actions；
- masked logits 的 invalid entries 不产生 policy gradient；
- only-idle row 不 NaN；
- requested/executed action mismatch 必须硬失败且样本不进入 buffer；
- sampled action 不改变任何共享 event-key RNG；
- recurrent hidden state 在真实 termination reset，在 truncation 按批准
  bootstrap contract 处理；
- sequence replay 的 logits/logp/value 与 rollout 时一致；
- 2–3 PPO updates CPU/GPU smoke 无 NaN；
- actor output dimension 与 schema version 一致，old checkpoints 明确拒绝或只允许 legacy eval。

## 9. Phase P6 — Reward Alignment

当前 `reward = -radar_team_reward` 只作 legacy 相关性审计。新增 raw episode logs：

- incremental dropped count；
- terminal drop ratio；
- training return；
- energy used；
- per-task drop cause。

G3-BSTA 只能使用 P0 人工批准并由 hash 锁定的唯一 primary reward。若批准
的是 incremental dropped count，定义必须完整写入：

```python
r_primary = newly_dropped_tasks / approved_fixed_normalizer
```

若批准的是 terminal raw metric，则不得在 pilot 后改为 count reward。变量
denominator 下 count 与 ratio 不自动等价；必须用冻结 scenario weighting
和单元测试证明 objective contract。zero-arrival episode 必须按 contract
标为 exclude/NA 或其他预注册语义，禁止默认写成 `drop_ratio=0`。

potential-based shaping 只有在 P0 approval 已包含公式、可用状态和
terminal telescoping proof 时才允许；不得因训练困难临时加入。不得加
learner-only active-cost。任何 reward/config hash 与
`OBJECTIVE_CONTRACT.md` 不符返回 `FAIL_OBJECTIVE_CONTRACT`。

新增：

```text
tests/mfr/test_reward_metric_alignment.py
tests/mfr/test_potential_shaping_telescope.py
tests/mfr/test_objective_contract_hash.py
tests/mfr/test_zero_arrival_metric.py
```

## 10. Phase P7 — Baselines and Reachability

### 10.0 Mutually Exclusive Development and Confirmation Splits

在任何 planner/baseline 运行前，将 exact scenario IDs 与生成器 hash 写入
`EVAL_PROTOCOL.md`：

```text
baseline-calibration     # only script family/threshold selection
planner-development      # planner code/config/debug only
headroom-confirmation    # untouched until both scripts and planner frozen
checkpoint-validation    # training checkpoint rule only
locked-final-test        # G3-BSTA only
```

五个 split 不得共享 scenario、event-key namespace 或由同一随机流切片。
`planner-development` 不能用于 headroom LCB；任何人在查看
`headroom-confirmation` rows 后更改 planner、resource operating point、
transition、baseline 或 scenario generator，必须废弃该 confirmation split、
递增 protocol hash 并重新人工批准。confirm/final scenario 数由预注册的
evaluation-noise power/precision analysis确定，不再硬编码为 32。

### 10.1 Files

```text
algo/_shared/pilot/mfr/baselines_v3.py
algo/_shared/pilot/mfr/reduced_oracle.py
experiments/mfr_phaseB/scripts/run_current_reachability.py
experiments/mfr_phaseB/scripts/run_reduced_oracle.py
experiments/mfr_phaseB/scripts/freeze_baselines.py
experiments/mfr_phaseB/scripts/run_headroom_gate.py
```

### 10.2 Required Baselines

- off；
- random feasible；
- budgeted barrage；
- round-robin；
- periodic blink duty/phase family；
- EDF/threat-first；
- reactive service follower；
- marginal information/drop per joule；
- short-horizon assignment/knapsack。

每个 baseline 函数签名统一：

```python
action_id = baseline.act(jammer_observation, action_mask)
```

禁止传 hidden env state。所有 baseline 只使用与 recurrent actor 相同的
observation history、association tracks 和 mask；需要记忆的 baseline 必须
显式维护自己的 causal state并在 episode reset。

### 10.3 Reduced Exact Oracle

最低 reduced case：

- observable services `2–3`；
- beam `K=1`；
- discrete energy；
- horizon `8–12` 或能够 exact solve 的最大值；
- real v3 transition helper；
- deterministic case exact DP；
- stochastic detector 的 exact DP 必须对完整 transition distribution
  做 Bellman expectation，并给出有限状态/离散化证书；
- 用 expected surrogate transition、single exogenous tape 或 common
  random numbers 优化得到的结果一律标为 approximate/sample-path，
  不能称 exact 或 causal upper bound。

分别输出：

- observation-history/belief causal policy；
- full-state policy；
- future-clairvoyant ceiling；
- exact/approximate 标记；
- explored state count；
- optimality certificate 或 gap。

causal planner 的 belief 只能由 approved observation history、known action
history 和冻结生成模型更新；入口不得接受 serialized hidden env state、
true queue/progress、future arrivals 或 detector tape。full-state 和
future-clairvoyant 结果仅作单独诊断。近似 causal planner 是 lower-bound
witness，不是 upper bound。

### 10.4 Nontriviality Gate

必须在预注册机制单元场景和代表性 planner-development distribution 中找到：

- 相同 time/resource、不同 causal observation history 的可达 belief states；
- 至少两个 service actions 的 ordering 反转；
- 最优与次优 action-value gap 超过批准的数值及实际意义 tolerance，而
  不是仅要求脆弱的“唯一最优”；
- causal policy 对最佳 open-loop schedule 的 held-out development
  差值达到 P0 批准的机制阈值。

该项只证明机制非平凡，不是 headroom 确认；不得通过调 resource/physics
参数专门打破 action ties。

### 10.5 Freeze

在 `baseline-calibration` 选择完整 script family/threshold，并在
`planner-development` 完成 planner 后分别写：

```text
BASELINE_FREEZE.json
PLANNER_FREEZE.json
```

二者包含 code/config/protocol/transition/resource hashes、使用过的 scenario
IDs 和 freeze timestamp。G3-BSTA 不选择单一 `best_script`；冻结 family
中的每个 competent script 都是必须单独超过的 comparator。

### 10.6 Headroom

只在未触碰的 `headroom-confirmation` split 上，对每个冻结 script `j`
计算：

```text
delta_j = D_causal_witness - D_frozen_script_j
```

主 headroom gate 是 intersection-union test（IUT）：

```text
min_j LCB95(delta_j) > approved_headroom_threshold
```

默认候选阈值 `.075` 只有在 P0 error budget 说明其覆盖 calibration、
planner-to-learner 和 evaluation uncertainty 时才能批准。相邻预注册 energy
settings、target-arrival/geometry shift 和 detector uncertainty envelope
两端也要求：

```text
min_j LCB95(delta_j) > .05
```

禁止使用 `max_j D_script_j` 的逐 scenario outcome-max 作为主 comparator；
它只可标记为 `oracle-script-ensemble sensitivity`。planner 和 scripts
必须在 confirmation 前冻结。任一 script 未通过、split 污染或阈值未满足，
返回 `STOP_G3_BSTA_NO_ROBUST_HEADROOM`，不要训练。

## 11. Phase P8 — Pilot Then Full Training

### 11.1 Pilot Only After Headroom

运行两个独立 train seeds：

```text
9000, 9001
```

比较：

- frozen-init actor；
- random feasible；
- trained actor；
- service features shuffled；
- temporal history shuffled。

同时记录：

- per-state action distribution；
- action/service mutual dependence；
- energy spending curve；
- service switches；
- invalid-action count；
- return/drop correlation。

shuffle 必须同步保持 service-label/action/mask 语义；不得把非法 action 或
破坏 association 当作“性能下降”。如果 trained 与 init/random 不可区分，
或合法 shuffle 不降低 state-dependent behavior/performance，返回
`STOP_LEARNING_NOT_DEMONSTRATED`。

这两个 seeds 仅是 smoke/go-no-go，不得用于估计 training-seed variance、
选择网络/超参、改变 transition/reward，或作为正式 claim。

### 11.2 Full Training

在任何正式 training seed 运行前，预注册：

```text
null_boundary = .05
alternative_mean = approved_mu_alt  # must be strictly > .05
target_power >= .80
alpha = .05 one-sided
N_train = max(8, noncentral_t_required(
    sigma=approved_conservative_training_seed_sd,
    null=.05,
    alternative=approved_mu_alt,
    alpha=.05,
    power=.80,
))
```

`.05` 只是 null boundary，不是 power analysis 的 effect size。方差必须来自
独立历史证据、保守上界，或在所有代码/超参/协议冻结后的至少 `6–8` 个
full-config/full-budget blinded variance-pilot seeds。两个 L0 seeds 不能单独
提供方差。若使用 blinded sample-size
re-estimation，规则、是否纳入最终分析、最大 `N_train` 和 GPU/存储预算必须
提前写入 `EVAL_PROTOCOL.md`。若所需 N 超过批准预算，返回
`STOP_INSUFFICIENT_POWER_BUDGET`，不得以 `N=8` 强行继续。

所有 seeds 固定：

- total training steps；
- checkpoint-selection rule；
- config；
- code；
- baseline；
- eval protocol。

checkpoint 只能在独立 `checkpoint-validation` split 按同一冻结规则选择。
不得从 headroom-confirmation/locked-final 选择 checkpoint、改变某些 seeds
的训练步数或只重跑失败 seeds。

## 12. Phase P9 — Locked G3-BSTA All-Script Gate

新增：

```text
experiments/mfr_phaseB/scripts/run_g3_bsta_gate.py
experiments/mfr_phaseB/scripts/stats_g3_bsta.py
```

### 12.1 Row Schema

`episode_rows.jsonl` 至少包含：

```json
{
  "schema_version": "g3-bsta.1",
  "protocol_hash": "",
  "train_seed": 0,
  "scenario_seed": 3000,
  "episode_id": 0,
  "action_seed": 0,
  "arrival_event_namespace": "",
  "geometry_event_namespace": "",
  "detector_event_namespace": "",
  "action_event_namespace": "",
  "policy_name": "ppo_seed0",
  "policy_mode": "sampled",
  "drop_numerator": 0,
  "drop_denominator": 0,
  "drop_ratio": null,
  "energy_used_j_by_emitter": [],
  "constraint_violations": 0,
  "requested_executed_mismatches": 0,
  "checkpoint_sha256": "",
  "config_sha256": "",
  "transition_contract_sha256": "",
  "objective_contract_sha256": "",
  "git_commit": ""
}
```

`drop_denominator==0` 时按 `OBJECTIVE_CONTRACT.md` 使用 `null/NA` 与冻结
排除/聚合规则，禁止填 `0.0`。每条 row 可由 manifest 恢复全部 event keys；
若为 stochastic script，也必须有独立 action namespace/seed。

### 12.2 Comparator

不存在单一 calibration-selected `best_script` 主 gate。所有
`BASELINE_FREEZE.json` 中的 competent scripts 均是 comparator。不得在
locked test 选一个有利的脚本，也不得用逐 scenario outcome-max 替代真实
可部署脚本。可额外报告：

- calibration-selected script；
- test-realized fixed-script ranking；
- oracle per-scenario ensemble sensitivity；

但三者都不改变 all-script 主 gate。

### 12.3 Correct Test

主 estimand 条件于冻结的 locked-final scenario/event tapes，以独立
training seed 为最高层重复。对每个 train seed `i` 和每个 frozen script
`j`，先按冻结顺序聚合 scenario/episode/action seeds：

```python
d_ij = learned_drop_i - frozen_script_drop_j
for j in frozen_scripts:
    se_j = std(d[:, j], ddof=1) / sqrt(n_train)
    t_j = (mean(d[:, j]) - 0.05) / se_j
    p_j = 1 - student_t.cdf(t_j, df=n_train - 1)
    lcb95_j = mean(d[:, j]) - student_t.ppf(
        0.95, n_train - 1
    ) * se_j
passed = all(lcb95_j > 0.05 for j in frozen_scripts)
```

这是对 union null“至少一个 script 的优势不超过 5pp”的 IUT。必须输出
每个 script 的 raw train-seed rows、LCB 和 p；不得只输出最有利 comparator。
另用预注册 crossed bootstrap/混合模型报告对 scenario population 的
secondary generalization uncertainty，但不得与上述 conditional primary
混写。

必须处理并测试 `se_j==0`、NaN、missing row、zero denominator 和 exact
boundary；只有逻辑和数值均严格满足 `> .05` 才 PASS。8 个 evaluation
seeds of one checkpoint 不能代替 training seeds。

Parity 如需研究，另开预注册 equivalence/non-inferiority protocol；`p>0.05` 不证明相同。

## 13. Required Artifact Layout

```text
experiments/mfr_phaseB/g3_bsta/
├── SOURCE_HANDOFF.json
├── P0_SOURCE_HANDOFF_REPORT.md
├── SYMBOL_MAP.md
├── P0_PROVENANCE_REPORT.md
├── P0_BINDING_PACKET.md
├── P0_BINDING_APPROVAL.json
├── TRANSITION_CONTRACT.md
├── OBJECTIVE_CONTRACT.md
├── RESOURCE_ASSUMPTIONS.md
├── OBSERVATION_SPEC.md
├── EVAL_PROTOCOL.md
├── BASELINE_FREEZE.json
├── PLANNER_FREEZE.json
├── RUN_MANIFEST.json
├── configs/
├── calibration/
├── diagnostics/
├── headroom/
├── pilot/
├── train/
├── gate/
└── reports/
```

`RUN_MANIFEST.json` 记录：

- git SHA 与 dirty state；
- Python/torch/CUDA/driver/GPU；
- exact commands；
- configs 与 hashes；
- all seed namespaces；
- input/output hashes；
- checkpoint URIs/hashes；
- approved CPU/GPU/storage budget、实际用量和 power-analysis inputs；
- started/finished timestamp；
- gate verdict/reason code。

## 14. Acceptance Test List

至少新增并通过：

```text
tests/mfr/test_source_handoff_schema.py
tests/mfr/test_binding_approval_hashes.py
tests/mfr/test_drop_ratio_definition.py
tests/mfr/test_zero_arrival_metric.py
tests/mfr/test_event_key_rng_branch_invariance.py
tests/mfr/test_detector_progress_units.py
tests/mfr/test_detector_iq_calibration.py
tests/mfr/test_progress_monotonicity.py
tests/mfr/test_progress_no_artificial_clamp.py
tests/mfr/test_tracker_information_recursion.py
tests/mfr/test_transition_contract_hash.py
tests/mfr/test_no_jam_regression.py
tests/mfr/test_jammer_resource_conservation.py
tests/mfr/test_jammer_action_contract.py
tests/mfr/test_jammer_policy_path_parity.py
tests/mfr/test_emitter_service_coupling.py
tests/mfr/test_cross_service_leakage.py
tests/mfr/test_action_mask_causality.py
tests/mfr/test_mask_is_function_of_obs.py
tests/mfr/test_requested_equals_executed.py
tests/mfr/test_jammer_obs_no_leak.py
tests/mfr/test_service_slot_identity_no_leak.py
tests/mfr/test_masked_categorical.py
tests/mfr/test_rollout_mask_replay.py
tests/mfr/test_action_rng_isolation.py
tests/mfr/test_recurrent_rollout_replay.py
tests/mfr/test_recurrent_hidden_reset.py
tests/mfr/test_gae_termination_truncation.py
tests/mfr/test_masked_ppo_ratio_kl.py
tests/mfr/test_reward_metric_alignment.py
tests/mfr/test_objective_contract_hash.py
tests/mfr/test_reduced_oracle_exactness.py
tests/mfr/test_causal_planner_no_hidden_state.py
tests/mfr/test_planner_confirmation_split.py
tests/mfr/test_g3_bsta_all_script_iut.py
tests/mfr/test_g3_bsta_null_coverage.py
tests/mfr/test_training_seed_power_analysis.py
```

统计测试必须用模拟 null/alternative 覆盖检查 Type-I error、power、zero
variance、missing/NaN、strict `>.05` boundary 和任一 script 失败即总 gate
FAIL。`test_planner_confirmation_split.py` 必须证明 freeze timestamp/hash
早于第一次 confirmation read，且两个 split 的 scenario/event namespaces
不相交。

建议命令：

```bash
pytest -q tests/mfr
pytest -q tests/twoteam tests/test_jammer_train_smoke.py
python experiments/mfr_phaseB/scripts/reproduce_original_g2a.py --config <legacy>
python experiments/mfr_phaseB/scripts/calibrate_detector_progress.py --config <v3-cal>
python experiments/mfr_phaseB/scripts/run_reduced_oracle.py --config <v3-reduced>
python experiments/mfr_phaseB/scripts/run_headroom_gate.py --config <v3-headroom>
```

最后两个命令只有前序 gate PASS 后运行。

## 15. Mandatory STOP Reasons

PRO6000 遇到以下任一条件必须停下并输出 reason code：

- `BLOCK_PPO_PROVENANCE`
- `BLOCK_SOURCE_HANDOFF`
- `AWAIT_P0_HUMAN_BINDING_APPROVAL`
- `LEGACY_CLAIM_UNVERIFIED`
- `PROVENANCE_MISMATCH`
- `BLOCK_PHYSICS_SEMANTICS`
- `BLOCK_TARGET_LOCALITY`
- `FAIL_IQ_CALIBRATION`
- `FAIL_RESOURCE_INVARIANT`
- `FAIL_POLICY_PATH_FAIRNESS`
- `FAIL_CAUSAL_OBSERVATION`
- `FAIL_ACTION_EXECUTION_MISMATCH`
- `FAIL_EVENT_KEY_RNG`
- `FAIL_OBJECTIVE_CONTRACT`
- `FAIL_PLANNER_CONFIRMATION_LEAKAGE`
- `STOP_CURRENT_G2A_INFEASIBLE`
- `INCONCLUSIVE_CURRENT_BOUND`
- `STOP_G3_BSTA_OPEN_LOOP_EQUIVALENT`
- `STOP_G3_BSTA_NO_ROBUST_HEADROOM`
- `STOP_LEARNING_NOT_DEMONSTRATED`
- `STOP_INSUFFICIENT_POWER_BUDGET`
- `G3_BSTA_FAIL_MARGIN`

`AWAIT_P0_HUMAN_BINDING_APPROVAL` 和 `LEGACY_CLAIM_UNVERIFIED` 是人工停点：
只有新的有效批准可恢复；其余失败码均按对应 phase 规则处理，不能由 agent
自行豁免。

“时间不够”“已经训练了很多”或“sampled mean 看起来一样”都不是跳过 STOP gate 的理由。

## 16. Required Progress Reports

每阶段只汇报证据：

```text
P0: SOURCE_HANDOFF verification + symbol map + binding packet + human stop
P1: legacy reproduction or LEGACY_CLAIM_UNVERIFIED + event-key diagnostics
P2: frozen transition hash + fit/held-out IQ calibration + unit/no-jam tests
P3/P4/P5: emitter×service/action/obs/recurrent-PPO invariant test output
P6/P7: objective hash + baseline/planner freezes + untouched confirmation rows
P8: pilot controls and go/stop verdict
P9: power analysis + full train manifest + locked all-script IUT
```

报告中不得把 planner result 写成 PPO result，不得把 evaluation seeds 写成 training seeds。

## 17. Copy-Paste Execution Directive for PRO6000

```text
Implement MFR-IQ Budgeted Emitter×Service Jammer v3 according to
PRO6000_AGENT_IMPLEMENTATION_SPEC_20260728_023103.md.

Start with P0 only. Require and verify SOURCE_HANDOFF.json before touching the
target repository. Preserve all existing user changes; create the source report,
SYMBOL_MAP.md, transition/objective/resource contracts and P0 binding packet.
Restore/version legacy artifacts when they actually exist; otherwise record
LEGACY_CLAIM_UNVERIFIED. Do not guess missing symbols and do not start GPU
training.

At the end of P0, stop with AWAIT_P0_HUMAN_BINDING_APPROVAL. Do not continue
until a valid approval freezes the source, emitter×service physical binding,
resources, causal observation/mask, one primary transition, one primary reward,
evaluation mode, estimand and compute budget. Later contract changes invalidate
the approval and require a new protocol version; never continue automatically
across a human binding point.

The v3 core is fixed after approval: explicit physical emitters, per-emitter
peak power and non-transferable episode energy, approved K_team=1, fixed
calibrated emitter power, idle-or-(emitter, observable service) recurrent masked
categorical action, calibrated receiver/service coupling and leakage, and one
frozen task transition. If service selectivity cannot be grounded in an
angle/range/Doppler/frequency/waveform mechanism, stop with
BLOCK_TARGET_LOCALITY. All policy types must share one
resource/action/physics path; requested and executed actions must always match.

Do not add active-cost, frequency hopping, exposure/thermal, continuous power,
multiple beams, co-learning radar, or a parity rewrite. Do not leave scripts or
sole results in /tmp. Keep original G2'a as FAIL. Do not claim G3-BSTA until the
pre-registered, adequately powered independent training seeds pass the all-script
intersection-union gate: every frozen script must have a one-sided 95% LCB for
delta strictly above 0.05.

If a required artifact, physical semantic, calibration, fairness invariant,
causal-decision test or robust headroom gate fails, stop with the prescribed
reason code and attach the exact evidence.
```

## 18. Final Definition of Done

实现完成不等于 gate PASS。Definition of Done 分三层：

1. **Code complete**：P0–P7 source、tests、configs、manifests 可复现；
2. **Environment admissible**：P0 人工批准有效，A0–A5 全 PASS，且物理、
   因果、公平与 untouched-confirmation robust headroom 证据完整；
3. **Scientific claim admissible**：L0 与 locked G3-BSTA all-script IUT
   在充分 power 的独立 training seeds 上 PASS。

若只完成第 1 或第 2 层，就按该层交付，不得升级措辞。
