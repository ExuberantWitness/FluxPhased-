# Orphan Adoption Eligibility — Route B+

**Case**: mfr-orphans-20260728T094154Z
**Orphan archive SHA-256 (authoritative)**: `37bb3c9c13442d3eaf3e6d4fdeae99eb66b65925e70adea37489f4c7586d576a`
**HEAD reference**: `807588cab7d367bedd415b45efc85a72f2a38b89`
**Generated**: 2026-07-29 (host clock node15)

---

## 1. Verdict

```text
adoption_eligibility: CONDITIONAL_PASS
historical_source: FAIL (per M7_RECOVERY_REPORT)
code_changes: none
status: AWAIT_ADOPTION_OWNER_APPROVAL
allowed_claim_tier_if_signed: NEW_BENCHMARK_ONLY
recommended_branch_name: g3-bsta/clean-successor
recommended_benchmark_name: G3-BSTA-v0
next_authorized_phase_without_signature: NONE
```

**核心结论**:orphan 包**技术条件上可以**作为 G3-BSTA-v0 新基准的 seed/spec,但**必须**满足 6 项前提才可签 adoption commit。无签名时禁止任何形式采纳。

---

## 2. B+ 必备条件逐项核对

| # | B+ PASS 要求 | 状态 | 证据 |
|---|---|---|---|
| a | 完整证据和 chain-of-custody | **PASS** | 取证包 SHA + SCENE + 双 hash + REFLOG + manifest 全在 |
| b | 无未解决 secrets 或恶意内容 | **PASS** | 无 AWS/GCP/Azure/PAT/private key 命中;无 eval/exec/shell=True 滥用;无网络调用 |
| c | 使用权和许可证可由有权负责人确认 | **PENDING** | FluxPhased- repo 整体无 LICENSE;需 repo owner 明确授权 |
| d | 有 adoption owner 愿意承担当前版本责任 | **PENDING** | 用户尚未指定 adoption owner;需明确人类签字 |
| e | 依赖闭包可补齐 | **PASS** | 8 stdlib + 1 torch + 3 twoteam(HEAD 中存在) + 9 内部互引;无外部 unresolved |
| f | 核心语义可确定 | **CONDITIONAL** | AST 全 OK;drop_ratio/action_mask/prog_factor 静态可见;但 §2.3 八项物理绑定需 owner 决策 |
| g | 能选定 verified base | **PASS** | HEAD `807588cab7d367bedd415b45efc85a72f2a38b89` 可作 base |
| h | 能建立当前时间的新 commit | **PASS** | 技术上可,需在新建 branch(不在 twoteam/bc-ppo) |
| i | commit 明确记录 attribution 状态 | **PENDING** | adoption commit 模板已准备,待 owner 签字后写入 |

**6 项 PASS / 3 项 PENDING / 0 项 FAIL**。在 3 项 PENDING 全部转 PASS 前,adoption 不得执行。

---

## 3. 静态审查总结(10 项,按用户指令 §五)

### 3.1 SHA-256 / blob hash / metadata 清单

完整 17 文件 manifest 见 `ORPHAN_EVIDENCE_MANIFEST.jsonl`。blob 匹配矩阵见 `ORPHAN_BLOB_MATCH.csv`。

- 17 文件 SHA-256 三值全等(pass A = pass B = copy)
- 16 个非空文件 git_blob_nofilter hash **全部不在 HEAD/main repo object db 也不在 Copy repo object db**
- 唯一命中的 `e69de29b` 是空 `__init__.py` 的通用 hash,与 MFR 无关

### 3.2 Symlink / hardlink / binary / secret / 恶意内容

| 检查 | 结果 |
|---|---|
| symlink | 0 |
| hardlink(link count > 1) | 0 |
| 真正 binary | **0**(file 命令把含中文 UTF-8 的 .py 误判 binary;mime-type 全部 `text/plain`) |
| AWS/GCP/Azure/Stripe/PAT key | 0 |
| private key markers | 0 |
| password / token assignment | 0 |
| 内部 IP / hostname | 0 |
| github_pat 引用 | 0 |
| eval / exec / os.system / shell=True | 0(仅有 nn.Module `.eval()`,正常) |
| 网络调用(urllib/requests/socket) | 0 |

### 3.3 Authorship / 第三方代码 / license

详见 `LICENSE_AND_AUTHORSHIP_RISK.md`。**无任何 license/author 标记**,无第三方借用,默认适用 repo 级 license 真空。

### 3.4 与 refs/reflogs/unreachable/镜像逐文件比较

| 比较源 | 命中 |
|---|---|
| `git log --all -- env/gpu/mfr/*` 等 | 0 |
| `git log --all --reflog -- env/gpu/mfr/*` 等 | 0 |
| 22 个 main repo unreachable commit trees | 0 |
| 30 个 Copy repo unreachable commit trees | 0 |
| GitHub 9 branches code search | 0 |
| GitHub code search `path:env/gpu/mfr` | 0 |

**所有 17 文件关系标签 = `NO_VERIFIABLE_HISTORY_MATCH`**(两个空 `__init__.py` 文件因 hash 通用,标签也是 `NO_VERIFIABLE_HISTORY_MATCH`,不能 attribute)。

### 3.5 关系标签使用

只用了允许的 4 个标签之一(`NO_VERIFIABLE_HISTORY_MATCH`)。无 `EXACT_PATH_AND_BLOB_MATCH`、无 `CONTENT_MATCH_DIFFERENT_PATH`、无 `DERIVED_PATCH_AGAINST` —— orphan 与任何 ref 都无关系。

### 3.6 Imports / dependency closure

- 第一方:`env.gpu.twoteam.{detection, iq_interference, tracker}`(全部在 HEAD 中)
- 第一方(orphan 互引):`env.gpu.mfr.*`, `algo._shared.pilot.mfr.*`(预期 missing in HEAD)
- 第三方:`torch`
- 标准库:`argparse` `csv` `json` `math` `os` `sys` `time` `__future__`

**无任何 import 指向不存在的模块**。

### 3.7 缺失闭包组件

详见 `SOURCE_CLOSURE_GAP.md`。核心缺失:
- 0 个 config template
- 0 个 raw row schema
- 0 个 checkpoint(只有 untracked 运行产物)
- 0 个 conftest.py mfr-specific
- 0 个 LICENSE/in-repo attribution

### 3.8 transition/reward/observation/action/drop metric/slot identity 静态可确定性

| 语义 | 静态可确定? | 关键文件 |
|---|---|---|
| `drop_ratio` | 部分(定义需 inferred) | mfr_env.py + tests/mfr/*.py |
| `action_mask` | 部分(构造规则需 P4 audit) | mfr_env.py |
| `prog_factor` σ-progress coupling | **是**(显式公式) | mfr_env.py |
| `tau_track=4.0` | **是**(literal) | mfr_env.py |
| `tracker_initialized` | 部分(依赖 twoteam.tracker) | mfr_env.py + tests |
| `target_slot` identity | 部分(false alarm/ID reuse 需 inferred) | jammer.py + task_layer.py |
| JNR/SINR 路径 | 部分(经 iq_interference) | mfr_env.py |

**结论**:主干语义静态可推出,但完整 causal observation audit 必须在 P4 阶段做。orphan 自身**不能**作为 audit 的权威依据。

### 3.9 隐藏使用 target ID / 违反 causal observation

- `target_id` 关键词在 orphan 中**0 hits**
- `target_slot` 在 `jammer.py` / `task_layer.py` 出现(预期,是 action 域)
- mfr_env.py 的 obs 构造需在 P4 阶段做 no-godview 测试(类似 WP-1 §2.6 测试)
- orphan 测试文件已含若干 no-godview 断言,但覆盖度未知

**红旗**:无隐藏使用 target ID 的直接证据,但需 P4 阶段做正式 audit。

### 3.10 八项物理绑定 UNKNOWN 状态

详见 `SOURCE_CLOSURE_GAP.md` §3。**8/8 都需要 adoption owner + RF 物理负责人填写,orphan 不能自证**。

---

## 4. Adoption commit 必须满足的 trailers

```text
Source-attribution: unknown
Adoption-status: adopted-new
Evidence-case: mfr-orphans-20260728T094154Z
Evidence-sha256: 37bb3c9c13442d3eaf3e6d4fdeae99eb66b65925e70adea37489f4c7586d576a
Historical-M7-recovery: false
License-inherited-from: <must-be-set-by-repo-owner>
Adoption-owner-signed: <human-name> <date>
```

---

## 5. 优先新命名

- **采纳**:`G3-BSTA-v0` / `g3-bsta/clean-successor` branch
- **禁止**:`recovered-M7` / `original-M7` / `fixed-G2a` / `mfr-restored`

---

## 6. Adoption 前必须人类签字的事项

```
[ ] FluxPhased- repo owner 签:repo-level LICENSE 选定
[ ] FluxPhased- repo owner 签:orphan 收养授权
[ ] RF / simulation physics owner 签:§2.3 八项物理绑定填写
[ ] Experiment owner 签:metrics schema + tests/mfr 框架定稿
[ ] Adoption owner 签:承担 G3-BSTA-v0 当前版本责任
```

任何一项缺失则停在 `AWAIT_ADOPTION_OWNER_APPROVAL`,不得创建 adoption commit。

---

## 7. 不会自动发生的事项

- 不会自动进入 P1 实现阶段
- 不会自动跑任何训练 / 测试
- 不会自动在原 worktree 创建任何 commit
- 不会自动 push 任何 branch
- 不会自动把 orphan 写入 SYMBOL_MAP
- 不会自动复现 G2'a 历史结果(无 raw rows)
- 不会自动把 G2'a 标为 PASS

---

## 8. 路径建议(决策表)

```
IF user provides SOURCE_HANDOFF.json with 40-hex commit + tree SHA + owner signatures:
    → route A revives with PASS, route B+ 不再需要
ELIF user signs all 5 human-sign-off items above:
    → adoption_eligibility PASS, may create adoption commit on new branch
    → still may NOT call orphan "historical M7"
    → still must do P0-Binding + P1-P9 stages before any G3-BSTA claim
ELSE:
    → stay at AWAIT_ADOPTION_OWNER_APPROVAL
    → recommendation: CLEAN_ROOM_IMPLEMENTATION_FROM_APPROVED_SPEC
    (即从 G3_BSTA_CLEAN_SUCCESSOR_SPEC.draft.md 出发,在新 branch 完全重新实现,
     orphan 仅作 reference,不被 commit)
```
