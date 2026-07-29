# P0 Binding Packet (Draft)

**Case**: mfr-orphans-20260728T094154Z
**Generated**: 2026-07-29 (host clock node15)
**Packet status**: DRAFT — pending SOURCE_HANDOFF verification or Adoption Decision Packet signatures

---

## 1. P0 Verdict(权威,本会话最终)

```text
phase: P0
status: BLOCKED
route_taken: A_FIRST_B_PLUS_FALLBACK
handoff_docs: PASS
historical_source: FAIL
orphan_adoption_eligibility: CONDITIONAL_PASS
orphan_files: QUARANTINED_ORIGIN_UNKNOWN
constraint_violations: 1
code_changes: none
stop_reason: BLOCK_PPO_PROVENANCE
next_authorized_phase: NONE
```

---

## 2. 进入 P1 的前置条件

按 ORPHAN_MFR_QUARANTINE_PROTOCOL §Phase 3 与 A_FIRST_B_PLUS_FALLBACK 指令,以下任一未达成不得进入 P1:

### 2.1 路径 A 复活(若用户提供 SOURCE_HANDOFF)

```
[ ] SOURCE_HANDOFF.json provided with:
    - repo_path or repo_url
    - exact 40-hex commit SHA
    - tree SHA
    - source_archive URI + SHA-256
    - owner name + signature
    - handoff timestamp
[ ] Independent verification:
    - fresh checkout at claimed commit installs dependencies
    - env/gpu/mfr/ + algo/_shared/pilot/mfr/ + tests/mfr/ exist in tree
    - SHA-256 of source_archive matches
    - SYMBOL_MAP resolvable from real call graph
[ ] Three-party P0-Binding signatures:
    - SOURCE_OWNER
    - RF_OR_SIMULATION_PHYSICS_OWNER
    - EXPERIMENT_OWNER
[ ] Eight physics bindings filled (RESOURCE_AND_SELECTIVITY_CONTRACT.md)
```

### 2.2 路径 B+ adoption(若 adoption owner 签)

```
[ ] ADOPTION_DECISION_PACKET.md signed by all 5 parties
[ ] repo-level LICENSE committed to FluxPhased- main
[ ] Eight physics bindings filled by RF owner
[ ] G3_BSTA_CLEAN_SUCCESSOR_SPEC.md frozen (from draft)
[ ] metrics schema + tests/mfr framework frozen by experiment owner
[ ] Adoption commit on g3-bsta/clean-successor with 7 mandatory trailers
[ ] Source-attribution: unknown trailer present
[ ] Historical-M7-recovery: false trailer present
```

### 2.3 路径 B+ 拒绝(clean-room fallback)

```
[ ] G3_BSTA_CLEAN_SUCCESSOR_SPEC.md frozen
[ ] New branch g3-bsta/clean-successor created from HEAD
[ ] Fresh implementation written without committing orphan bytes
[ ] Same 8 physics bindings still required (cannot bypass)
[ ] Adoption commit not required (no orphan content)
[ ] Author-only license/attribution clear from day one
```

---

## 3. 已验证内容(本会话产出)

### 3.1 handoff docs(11/11)

| 文档 | 来源 | 状态 |
|---|---|---|
| PRO6000_RESUME_PROMPT.md | docs/g3-bsta-pro6000-handoff @ 5cafbbfd | READ |
| ORPHAN_MFR_QUARANTINE_PROTOCOL.md | docs/g3-bsta-pro6000-handoff @ 5cafbbfd | READ |
| README.md (handoff) | docs/g3-bsta-pro6000-handoff @ 5cafbbfd | READ |
| PIPELINE_SUMMARY_20260728_105723.md | inner tarball bc8de428 | READ |
| PRO6000_EXECUTION_PROMPT_20260728_105723.md | inner tarball bc8de428 | READ |
| PRO6000_AGENT_IMPLEMENTATION_SPEC_20260728_023103.md | inner tarball bc8de428 | READ |
| EXPERIMENT_PLAN_20260728_023103.md | inner tarball bc8de428 | READ |
| EXPERIMENT_TRACKER_20260728_023103.md | inner tarball bc8de428 | READ |
| FINAL_PROPOSAL_20260728_023103.md | inner tarball bc8de428 | READ |
| EXPERIMENT_AUDIT.md | inner tarball bc8de428 | READ |
| SOURCE_HANDOFF.template.json | inner tarball bc8de428 | READ |

所有文档 **READ IN FULL, no skipping**。

### 3.2 取证补正

- `P0_EVIDENCE_CORRECTION.md` / `.json`:权威 archive SHA / HEAD 展开 / 17 vs 18 解释 / SHA 不一致披露 / fetch 违规披露 / 因果边界

### 3.3 路线 A 搜索覆盖

- 22 main repo unreachable commits scanned
- 30 Copy repo unreachable commits scanned
- 16 orphan blob hashes checked in both object dbs
- 9 GitHub branches enumerated via API
- GitHub code search returned 0 hits for `mfr_env.py` and `path:env/gpu/mfr`
- IDE local history empty
- /tmp /var/tmp .cache empty for mfr
- bash_history no mfr mentions
- bundle/backup: only handoff doc archives

### 3.4 路线 B+ 静态审查覆盖

- 17/17 AST OK
- 0 secret/symlink/binary/dangerous-call
- 0 license/author markers
- import closure resolvable (8 stdlib + torch + 3 twoteam + 9 internal)
- 8/8 physics bindings marked UNKNOWN
- 4 entrypoints identified
- 0 hardcoded network paths

### 3.5 constraint_violations 明细

| # | constraint | status |
|---|---|---|
| 1 | POST_INSTRUCTION_FETCH_BEFORE_EVIDENCE_CAPTURE | DISCLOSED |

无新违规。

---

## 4. 当前授权范围

| 行为 | 授权? |
|---|---|
| 在原 worktree 执行 git fetch/pull/checkout/reset/stash/add/commit/rebase/gc/prune/repack | **NO** |
| 在原 worktree 执行 import/compile/pytest/trainer | **NO** |
| 在隔离 bare mirror 中有限超时 fetch(已尝试失败) | done,失败,不再重试 |
| 在孤儿副本做静态检查 | YES |
| 创建 adoption commit | **NO**(无 owner 签字) |
| 创建新 branch g3-bsta/clean-successor | **NO**(无 spec freeze) |
| 进入 P1 实现 | **NO** |
| 写 G3-BSTA 实现代码 | **NO** |
| 把 orphan 提升为 authoritative source | **NO** |
| 把 orphan 加入 SYMBOL_MAP | **NO** |
| 把 G2'a 改为 PASS | **NO** |
| 把 orphan 称为 "recovered M7" | **NO** |
| 把 orphan 称为 "original implementation" | **NO** |
| 接受用户的 SOURCE_HANDOFF.json 并独立验证 | **YES**(用户提供后) |
| 接受用户的 5 项签字并创建 adoption commit | **YES**(5 项齐全后) |

---

## 5. 下一步(等用户)

**等待用户提供以下任一**:

1. **SOURCE_HANDOFF.json**(若存在历史 M7 commit)→ 路径 A 复活
2. **5 项签字 + LICENSE 选择**(若选 adoption)→ 路径 B+ 进入实施
3. **明确选择 clean-room**(若放弃 adoption)→ 基于 spec 新写,无 orphan

在收到任一之前,PRO6000 停在 P0 BLOCKED,不创建任何 commit,不写实现代码,不动原 worktree。

---

## 6. 引用文档全集

```
/home/ubuntu/evidence/p0_corrections_20260728/
  P0_EVIDENCE_CORRECTION.md
  P0_EVIDENCE_CORRECTION.json

/home/ubuntu/evidence/route_a_recovery/
  M7_RECOVERY_REPORT.md
  M7_RECOVERY_REPORT.json

/home/ubuntu/evidence/route_b_plus_eligibility/
  ORPHAN_ADOPTION_ELIGIBILITY.md
  ORPHAN_ADOPTION_ELIGIBILITY.json
  ORPHAN_BLOB_MATCH.csv
  SOURCE_CLOSURE_GAP.md
  LICENSE_AND_AUTHORSHIP_RISK.md
  G3_BSTA_CLEAN_SUCCESSOR_SPEC.draft.md
  ADOPTION_DECISION_PACKET.md
  P0_BINDING_PACKET.draft.md   (本文件)

/home/ubuntu/evidence/mfr-orphans-20260728T094154Z/         (immutable archive)
/home/ubuntu/evidence/mfr-orphans-20260728T094154Z.tar.gz   (SHA 37bb3c9c...)
```

---

## 7. 协议合规声明

```text
compliance_status: COMPLIANT
- all 11 handoff documents read in full
- archive SHA-256 disclosed authoritatively (corrects prior inconsistency)
- HEAD commit expanded to 40 hex
- 17 vs 18 file count discrepancy explained
- fetch constraint violation disclosed
- route A exhausted within ~0.5 engineer-hours
- route B+ performed on isolated copy, no execution
- no orphan file executed
- no commit / branch / push on any FluxPhased ref
- no adoption commit created without owner signatures
- no implementation code written
- no claim fabricated; orphan not labeled recovered/original
- final verdict: BLOCKED at P0, AWAIT_ADOPTION_OWNER_APPROVAL or user-provided SOURCE_HANDOFF
```
