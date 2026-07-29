# Adoption Decision Packet — Orphan MFR Package

**Case**: mfr-orphans-20260728T094154Z
**Orphan archive SHA-256 (authoritative)**: `37bb3c9c13442d3eaf3e6d4fdeae99eb66b65925e70adea37489f4c7586d576a`
**HEAD reference**: `807588cab7d367bedd415b45efc85a72f2a38b89`
**Generated**: 2026-07-29 (host clock node15)
**Packet status**: AWAITING_SIGNATURES

---

## 1. 决策摘要

| 项 | 值 |
|---|---|
| Route | A_FIRST_B_PLUS_FALLBACK |
| Route A 结果 | historical_source: FAIL (no candidate;详见 M7_RECOVERY_REPORT) |
| Route B+ 结果 | adoption_eligibility: CONDITIONAL_PASS |
| 当前状态 | AWAIT_ADOPTION_OWNER_APPROVAL |
| 推荐路径 | (a) 签 adoption 进入 G3-BSTA-v0;(b) 拒绝 adoption,改走 CLEAN_ROOM_IMPLEMENTATION_FROM_APPROVED_SPEC |
| Code changes made | **none** |
| Branches created | **none** |
| Commits created | **none** |
| Worktree mutations | **none**(constraint_violations: 1 在 fetch 阶段,已披露) |

---

## 2. 等待签字的 5 项

| # | 角色 | 签字事项 | 文档参考 |
|---|---|---|---|
| 1 | FluxPhased- repo owner | repo-level LICENSE 选定 + orphan 收养授权 | `LICENSE_AND_AUTHORSHIP_RISK.md` |
| 2 | RF / simulation physics owner | §2.3 八项物理绑定填写 | `G3_BSTA_CLEAN_SUCCESSOR_SPEC.draft.md` §1 |
| 3 | Experiment owner | metrics schema + tests/mfr 框架定稿 | `SOURCE_CLOSURE_GAP.md` §2 |
| 4 | Adoption owner | G3-BSTA-v0 当前版本责任承担 | `ORPHAN_ADOPTION_ELIGIBILITY.md` §6 |
| 5 | Source owner / author(若存在) | 明确否认或让渡 orphan 著作权(可选,但若有则强制) | (无 candidate author) |

任何一项未签 → 留在 AWAIT,**不得**创建 adoption commit。

---

## 3. 签字影响范围

### 3.1 若全部签字(adoption commit 创建)

```
branch: g3-bsta/clean-successor (NEW, base=807588cab7d367bedd415b45efc85a72f2a38b89)
benchmark name: G3-BSTA-v0
allowed_claim: NEW_BENCHMARK_ONLY
prohibited_claim:
  - recovered M7
  - reproduced G2'a
  - original implementation
  - G2'a PASS retroactively applied
adoption_commit_trailers (MANDATORY):
  Source-attribution: unknown
  Adoption-status: adopted-new
  Evidence-case: mfr-orphans-20260728T094154Z
  Evidence-sha256: 37bb3c9c13442d3eaf3e6d4fdeae99eb66b65925e70adea37489f4c7586d576a
  Historical-M7-recovery: false
  License-inherited-from: <chosen license>
  Adoption-owner-signed: <name> <date>
```

### 3.2 若任一拒绝签字

```
adoption_eligibility: FAIL
recommendation: CLEAN_ROOM_IMPLEMENTATION_FROM_APPROVED_SPEC
fallback_implementation_path:
  - 完全新写 G3-BSTA-v0,基于 G3_BSTA_CLEAN_SUCCESSOR_SPEC.draft.md
  - orphan 仅作研究参考,不被 commit
  - 新代码所有权属清晰,无 attribution 风险
```

### 3.3 不允许的中间状态

- **不得**只签部分就开始 commit
- **不得**先 commit 再补签
- **不得**让 PRO6000 agent 代签
- **不得**用口头同意代替书面签字

---

## 4. 取证记录与计算变更

### 4.1 取证包权威 SHA-256

```
37bb3c9c13442d3eaf3e6d4fdeae99eb66b65925e70adea37489f4c7586d576a
```

(ORPHAN_EVIDENCE_REPORT.md 与 PACKAGE_INFO.txt 中曾记录的两个 SHA 已在 `P0_EVIDENCE_CORRECTION.md` §4 中披露更正)

### 4.2 17 文件清单

见 `ORPHAN_EVIDENCE_MANIFEST.jsonl` 与 `ORPHAN_BLOB_MATCH.csv`。

### 4.3 constraint_violations

```
count: 1
constraint: POST_INSTRUCTION_FETCH_BEFORE_EVIDENCE_CAPTURE
command: timeout 30 git -C /home/ubuntu/CODE/FluxPhased- fetch origin
exit_code: 124
stderr: curl 28 / GnuTLS recv error (-110)
impact:
  - .git/FETCH_HEAD mtime touched (0 bytes)
  - reflog metadata may have been touched
  - working tree bytes UNAFFECTED
  - double-hash stability UNAFFECTED
```

详见 `P0_EVIDENCE_CORRECTION.md` §5。

---

## 5. 已生成文档清单

### 5.1 取证补正(`/home/ubuntu/evidence/p0_corrections_20260728/`)

- `P0_EVIDENCE_CORRECTION.md`
- `P0_EVIDENCE_CORRECTION.json`

### 5.2 路线 A(`/home/ubuntu/evidence/route_a_recovery/`)

- `M7_RECOVERY_REPORT.md`
- `M7_RECOVERY_REPORT.json`

### 5.3 路线 B+(`/home/ubuntu/evidence/route_b_plus_eligibility/`)

- `ORPHAN_ADOPTION_ELIGIBILITY.md`
- `ORPHAN_ADOPTION_ELIGIBILITY.json`
- `ORPHAN_BLOB_MATCH.csv`
- `SOURCE_CLOSURE_GAP.md`
- `LICENSE_AND_AUTHORSHIP_RISK.md`
- `G3_BSTA_CLEAN_SUCCESSOR_SPEC.draft.md`
- `ADOPTION_DECISION_PACKET.md`(本文件)
- `P0_BINDING_PACKET.draft.md`(下一文件)

---

## 6. 不会发生的事项(确认)

- 不会自动进入 P1 实现
- 不会自动跑训练 / 测试
- 不会自动创建 commit / branch
- 不会自动 push 任何远端
- 不会自动把 orphan 加入 SYMBOL_MAP
- 不会自动复现 G2'a 历史(无 raw rows)
- 不会自动改 G2'a 状态(保持 FAIL)
- 不会自动允许使用 orphan 在论文中作 "original implementation"
- 不会把 orphan 在 P1 前的任何阶段当作 authoritative source

---

## 7. 签字页(草稿,待填写)

```
REPO_OWNER:
  Name:    __________________________________
  Date:    __________________________________
  Signature: _________________________________
  Decision: [ ] approve adoption  [ ] deny → clean-room

RF_PHYSICS_OWNER:
  Name:    __________________________________
  Date:    __________________________________
  Signature: _________________________________
  8 physics bindings attached: [ ] yes  [ ] no

EXPERIMENT_OWNER:
  Name:    __________________________________
  Date:    __________________________________
  Signature: _________________________________
  metrics schema + tests framework frozen: [ ] yes  [ ] no

ADOPTION_OWNER:
  Name:    __________________________________
  Date:    __________________________________
  Signature: _________________________________
  assumes version responsibility for G3-BSTA-v0 at commit ___________

SOURCE_OWNER_OR_AUTHOR (if any):
  Name:    __________________________________
  Date:    __________________________________
  Signature: _________________________________
  relation to orphan: _______________________
  decision: [ ] asserts authorship  [ ] disclaims  [ ] unknown

PRO6000_AGENT (this conversation):
  Date: 2026-07-29
  Role: forensic collector + eligibility assessor; NOT a signatory
```

---

## 8. 最终路径建议

按 A_FIRST_B_PLUS_FALLBACK 与本 packet 评估,推荐顺序:

1. **优先**等待用户提供 SOURCE_HANDOFF.json(若有历史源,route A 复活)
2. 若确认无历史源 → 选 adoption 途径(B+):
   - 完成 5 项签字
   - 创建 `g3-bsta/clean-successor` branch
   - 在 branch 上按 spec 实现 G3-BSTA-v0
3. 若任一签字无法获得 → clean-room:
   - 用 spec 作起点
   - 完全新写,不 commit orphan
   - 命名 G3-BSTA-v0(新基准,与 orphan 无 attribution 关系)

无论 1/2/3,**不**进入 P1 直到 P0-Binding 完成。
