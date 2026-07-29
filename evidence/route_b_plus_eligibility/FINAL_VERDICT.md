# Final Verdict — A_FIRST_B_PLUS_FALLBACK

**Case**: mfr-orphans-20260728T094154Z
**Generated**: 2026-07-29 (host clock node15)
**Author**: PRO6000 agent

---

## 最终裁决

```text
result_class: B_PLUS_ELIGIBLE

route: B_PLUS
historical_source: FAIL
adoption_eligibility: CONDITIONAL_PASS
status: AWAIT_ADOPTION_OWNER_APPROVAL
allowed_claim_tier_if_signed: NEW_BENCHMARK_ONLY
code_changes: none
next_authorized_phase: NONE
```

## 三路径决策表

```
PATH_A_RECOVERED:
  triggered_by: user submits SOURCE_HANDOFF.json with verified 40-hex commit + tree SHA + owner signatures
  result: historical_source: PASS, status: AWAIT_SOURCE_OWNER_ATTESTATION
  (NOT triggered this session — route A exhausted; no historical source exists on node15 or GitHub remote)

PATH_B_PLUS_ADOPTION (currently eligible):
  triggered_by: 5 human signatures on ADOPTION_DECISION_PACKET.md
  + repo-level LICENSE committed
  + 8 physics bindings filled
  + G3_BSTA_CLEAN_SUCCESSOR_SPEC frozen
  result: adoption commit on g3-bsta/clean-successor branch; orphan bytes attributed as
          "Source-attribution: unknown / Adoption-status: adopted-new"
  prohibited post-adoption claims: recovered-M7 / reproduced-G2a / original-implementation

PATH_B_PLUS_CLEAN_ROOM (fallback):
  triggered_by: any of the 5 signatures denied; user chooses to implement fresh
  result: new branch g3-bsta/clean-successor with fresh implementation;
          orphan used as research reference only, never committed
```

## 取证补正(关键)

```text
evidence_archive_sha256_authoritative: 37bb3c9c13442d3eaf3e6d4fdeae99eb66b65925e70adea37489f4c7586d576a
  (corrects prior 32236db9... in ORPHAN_EVIDENCE_REPORT.md and 856d98bf... in PACKAGE_INFO.txt)

head_full_40hex: 807588cab7d367bedd415b45efc85a72f2a38b89

orphan_source_file_count_canonical: 17
  (early "约18" estimate explained as verbal; no 18th file conjured)

fetch_constraint_violation_count: 1
  (POST_INSTRUCTION_FETCH_BEFORE_EVIDENCE_CAPTURE; disclosed in P0_EVIDENCE_CORRECTION.md §5)
  working-tree bytes unaffected; double-hash stability holds

git_metadata_scene_freshness: POST_FETCH
  (cannot claim pristine; orphan byte stability unaffected)
```

## 静态审查无红旗

```
- 0 secret / token / private key / github_pat leakage
- 0 symlink / hardlink
- 0 true binary file (all 17 are text/plain Python; Chinese UTF-8 misled `file`)
- 0 dangerous eval/exec/os.system/shell=True (only nn.Module .eval())
- 0 network calls (no urllib/requests/socket/http imports)
- 17/17 AST parse OK
- 0 unresolved imports (all 3 first-party external imports resolve to HEAD env/gpu/twoteam/)
- 0 license/author markers
- 0 third-party code borrowing evidence
```

## 路线 A 搜索预算

```
calendar_days_used: 0 of 5
engineer_hours_used: ~0.5 of 16
search_surfaces_covered:
  - main repo reachable commits (5 local + 8 remote refs)
  - main repo unreachable commits (22)
  - main repo unreachable blobs (15 non-empty orphan blobs all ABSENT from object db)
  - main repo stash + reflog (no mfr entries)
  - Copy repo reachable + unreachable commits (4 + 30)
  - /tmp /var/tmp /home/ubuntu/.cache (no mfr files)
  - bash_history (no mfr mentions)
  - bundle/tar/zip backups (only handoff doc archives)
  - IDE local history (empty)
  - independent bare mirror ls-remote (TIMEOUT 60s, TLS issue persists)
  - GitHub REST API branches (9 enumerated, no mfr-bearing branch)
  - GitHub REST API code search (0 hits for mfr_env.py and path:env/gpu/mfr)
```

## 路线 B+ 静态审查覆盖

```
10 items per user directive §五 all completed:
  1. SHA / blob / metadata manifest                : COMPLETE
  2. symlink/hardlink/binary/secret/malicious      : PASS
  3. authorship/third-party/license                : CONDITIONAL (no markers; needs owner)
  4. per-file compare to refs/reflogs/unreachable/mirror : COMPLETE
  5. relation labels (4 allowed)                   : used = NO_VERIFIABLE_HISTORY_MATCH only
  6. imports/dependency closure                    : COMPLETE (closable)
  7. missing closure components                    : LISTED
  8. transition/reward/obs/action/drop/slot static : partial; main backbone visible
  9. hidden target_id / causal observation leak    : no static red flags; P4 audit required
 10. eight physics bindings UNKNOWN status         : all 8 UNKNOWN; need owner decision
```

## 生成文档总览

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
  P0_BINDING_PACKET.draft.md
  FINAL_VERDICT.md                          (this file)
```

## 不允许自动发生的事项

```
- no auto entry into P1 implementation phase
- no auto training or test execution
- no auto commit creation in original worktree
- no auto branch push to any remote
- no auto SYMBOL_MAP population from orphan as authoritative
- no auto claim of G2'a historical reproduction (raw rows absent)
- no auto claim of G2'a PASS retroactively
- no auto promotion of orphan beyond L1 frozen snapshot
```

## 等待用户决策

```text
decision_required_from_user:
  (a) submit SOURCE_HANDOFF.json → PATH_A_RECOVERED revival
  (b) sign 5 items → PATH_B_PLUS_ADOPTION
  (c) deny adoption → PATH_B_PLUS_CLEAN_ROOM
  (d) cancel G3-BSTA work entirely

until one of (a)/(b)/(c)/(d) is chosen:
  phase: P0
  status: BLOCKED
  verdict: AWAIT_USER_DECISION
```
