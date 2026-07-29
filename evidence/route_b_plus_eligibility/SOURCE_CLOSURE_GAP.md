# Source Closure Gap — Orphan MFR Package

**Case**: mfr-orphans-20260728T094154Z
**Authoritative archive SHA-256**: `37bb3c9c13442d3eaf3e6d4fdeae99eb66b65925e70adea37489f4c7586d576a`
**Orphan package size**: 17 Python files (2 empty `__init__.py`, 15 substantive)
**HEAD reference**: `807588cab7d367bedd415b45efc85a72f2a38b89` (twoteam/bc-ppo)

---

## 1. 依赖闭包解析结果

| 依赖类型 | 数量 | 状态 |
|---|---|---|
| 标准库(`argparse` `csv` `json` `math` `os` `sys` `time` `__future__`) | 8 | OK — Python 内置 |
| 第三方库(`torch`) | 1 | OK — 已安装 in `fluxphased` conda env |
| FluxPhased 内部模块(`env.gpu.twoteam.*`) | 3 | **OK — HEAD 中已存在**(`detection.py`, `iq_interference.py`, `tracker.py`) |
| Orphan 内部互引(`env.gpu.mfr.*`, `algo._shared.pilot.mfr.*`) | 9 | OK — 自包含 |

**结论:依赖闭包可补齐**。没有任何 import 指向"既不在 HEAD 也不在 orphan 中"的模块。

---

## 2. 缺失的源码闭包组件(无法从 orphan 取得)

按 ORPHAN_MFR_QUARANTINE_PROTOCOL 与 G3-BSTA spec §P0 要求,以下闭包组件**缺失**:

### 2.1 配置文件(0 个随 orphan 提供)

orphan 内部引用 `config.json`,但**没有任何 config 文件**与 source 同时取证。仅在工作树 untracked `experiments/mfr*/` 下存在大量 `config.json`(已列在 `RELATED_EXPERIMENTS_DIR_LISTING.txt`),它们是**运行产物**而非可权威化的 source 配置。

| 缺失项 | 说明 |
|---|---|
| `mfr_default_config.json`(或等效 default config) | 缺失。`run_stage_*.py` 通过 argparse 命令行构造 config,但**默认值散布在源码字面量中** |
| `league_config.json`(template) | 缺失。`league_eval.py` 写出该文件,但没有 in-repo template |
| `pyproject.toml` / `setup.py` / `requirements.txt` mfr 部分 | 缺失。orphan 不带 dependency lock |

### 2.2 Metrics / raw rows(0 个随 orphan 提供)

orphan 引用 `metrics.csv`、`train_curve.csv`、`g2a_summary.json`,但**没有任何 raw rows 与 source 同时取证**。所有现存 csv/json 都是 untracked 运行产物。

| 缺失项 | 说明 |
|---|---|
| `metrics.csv` schema definition | 缺失。每 run 现场生成,无 schema 文档 |
| `train_curve.csv` schema | 缺失 |
| `g2a_summary.json` schema | 缺失 |
| raw per-episode/per-seed 数组 | 缺失。所有 csv 都是已聚合 |

### 2.3 Checkpoint / 种子 / assets

| 缺失项 | 说明 |
|---|---|
| 训练 checkpoint | untracked 运行产物(`jammer_sigma_100W_s0/final.pt` 等),不是 source |
| 训练 seed manifest | 缺失 |
| IQ 校准数据 / detector lookup | 缺失 |
| 资源(图形/几何)asset | 缺失 |

### 2.4 测试闭包(部分)

orphan 包含 3 个 `tests/mfr/test_*.py`,但**未带**:

| 缺失项 | 说明 |
|---|---|
| `conftest.py` mfr-specific | 缺失(可能依赖项目级 conftest) |
| `pytest.ini` / `pyproject.toml [tool.pytest]` mfr 节 | 缺失 |
| fixture data | 缺失 |
| baseline metrics snapshot | 缺失 |

---

## 3. 缺失的语义权威化(无法从 orphan 取得)

按 G3-BSTA spec §2.3 八项物理绑定,以下**UNKNOWN**不能从 orphan 文本中权威化:

| # | 物理绑定 | orphan 中线索 | 缺失 |
|---|---|---|---|
| 1 | 实际发射机数 | K=4 子阵;30% 目标 emitter=True | 是否 RF 意义"emitter" vs 任务系统标签 |
| 2 | per-emitter 峰功率/能量 | 实验 factor 含 100W;无峰值功率 cap | 平台依据(无) |
| 3 | 能量池化依据 | 无 team energy pool | 是否池化的 RF/mission 依据 |
| 4 | 同时波束上限 | K=4 各自独立 emission | K_team 是否 = 1 或 = 4 的依据 |
| 5 | service selectivity | 无 frequency/selectivity 维度 | target-local action 物理定义 |
| 6 | 雷达接收机绑定 | tracker 是 IMM-PDAF 通用 | receiver/dwell 绑定 |
| 7 | cross-talk 模型 | 依赖 `iq_interference` 通用接口 | cross-service 选择性 gate |
| 8 | detect/track/estimate 权威语义 | Wang 7 类任务;prog_factor=clamp(1/√(1+JNR),0.1,1) | detector calibration / IQ 验证 |

**所有 8 项都需要 adoption owner + RF 物理负责人明确填写,orphan 文本不能自证**。

---

## 4. 因果观察的静态可确定部分

| 关键语义 | 静态可确定? | 证据 |
|---|---|---|
| `drop_ratio` 定义 | 部分 | 在 8 个文件出现,但无 schema 文档(numerator/denominator 须 inferred from mfr_env.py) |
| `action_mask` 因果构造 | 部分 | 在 mfr_env.py 出现,但与 actor-visible obs 的关系需读源码确认 |
| `prog_factor` σ-progress coupling | 是 | 在 mfr_env.py:`(1.0 / sigma_scale).clamp(min=0.1, max=1.0)` |
| `tau_track=4.0` threshold | 是 | 在 mfr_env.py 出现 |
| `tracker_initialized` 因果 | 部分 | 在 tracker 模块;依赖 `env.gpu.twoteam.tracker` 既有定义 |
| `target_slot` identity | 部分 | 在 jammer.py 出现;false alarm / ID reuse 策略需 inferred |
| JNR → SINR 路径 | 部分 | 通过 `iq_interference.compute_jnr_matrix` |
| future-arrival / hidden-progress 泄漏 | **需静态审计** | mfr_env.py 需逐行检查 obs 构造 |

---

## 5. 闭包评估

| 维度 | 状态 |
|---|---|
| 依赖闭包(import 级别) | **CLOSABLE** — 所有 import 在 HEAD 或 orphan 中可 resolve |
| 配置闭包 | **NOT_CLOSABLE_FROM_ORPHAN** — 需 adoption owner 重建 default config |
| 测试闭包 | **PARTIALLY_CLOSABLE** — 3 个 test_*.py 存在,但 conftest/fixture 缺失 |
| 物理绑定闭包 | **NOT_CLOSABLE_FROM_ORPHAN** — 8 项需 owner 决策 |
| 历史复现闭包(raw rows/checkpoint) | **NOT_CLOSABLE_FROM_ORPHAN** — 全部是 untracked 运行产物 |
| 语义闭包(transition/reward/obs) | **STATIC_DERIVABLE** — AST + 源码阅读可推出主干,但 causal audit 须由 P4 阶段正式做 |

---

## 6. 不会自动修复的项

下列项目即使 adoption owner 签字也不会从 orphan 中"自动出现",必须在 P1-P3 实现阶段**新建**:

- 八项物理绑定 commit 化(必须以 `RESOURCE_AND_SELECTIVITY_CONTRACT.md` 形式落地)
- `SYMBOL_MAP.md` 完整定义(从 orphan 中只能列出符号,定义须由 implementation 阶段填写)
- causal observation audit(P4 阶段)
- masked categorical PPO invariant(P5 阶段)
- raw metrics schema(P6 阶段)
- legacy reproduction artifacts(完全无,**永远不能从 orphan 复现历史 G2'a raw rows**)

---

## 7. 对 claim 边界的影响

```
if adoption_owner_signs:
    allowed_claim_tier = NEW_BENCHMARK_ONLY
    allowed_claim = "G3-BSTA-v0 built from quarantined orphan seed at <new commit>"
    prohibited_claim = "recovered M7 / reproduced G2'a / original implementation"
else:
    allowed_claim_tier = NONE
    recommendation = CLEAN_ROOM_IMPLEMENTATION_FROM_APPROVED_SPEC
```
