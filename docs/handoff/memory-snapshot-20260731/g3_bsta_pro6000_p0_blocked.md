---
name: g3-bsta-pro6000-p0-blocked
description: "2026-07-28 PRO6000 G3-BSTA 工程停在 P0 BLOCKED:handoff 文档 PASS,但 M7 源 provenance FAIL,17 orphan 文件已冻结取证但 origin UNKNOWN"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c76cde4-1c23-4df6-94f5-ee29ee81afbc
---

PRO6000 代理执行 G3-BSTA 任务停在 **P0 BLOCKED**,裁定为 `BLOCK_PPO_PROVENANCE`。

**Why:** 用户以独立授权代理身份委托,要求恢复原 G2'a 实验的 M7 源 provenance。GitHub `FluxPhased-` repo HEAD `807588c` (twoteam/bc-ppo) 与 57 文件 handoff 归档(基线 commit bc8de428 / fd1cfff5)中均无 `env/gpu/mfr/`、`algo/_shared/pilot/mfr/`、`tests/mfr/` 源码。原 worktree 中的 17 个 MFR 文件无任何 commit/tree SHA 背书,`origin_status=UNKNOWN`,不能视为权威 M7 源。用户规则明确:handoff 是工单载体不是源码基线,孤儿文件不得添加到 FluxPhased 分支,不得执行,不得用于填充 SYMBOL_MAP。

**How to apply:**
- 下次被要求"继续 G3-BSTA"或"恢复 MFR 工作"时,先确认是否收到了独立的 `SOURCE_HANDOFF.json`(40-hex commit + tree SHA + 三方持有者批准)。没有就停在 P0,不写任何实现代码。
- P0-Binding 解封条件:1) 提交并通过独立验证的 SOURCE_HANDOFF.json;2) §2.3 八项物理绑定(发射机数/per-emitter 峰功率与能量/能量池化依据/同时波束上限/service 选择性机制/雷达接收机绑定/cross-talk 模型/权威 detect/track 语义)获 RF + 实验 + 源码三方批准。
- 原 worktree (`/home/ubuntu/CODE/FluxPhased-/`)禁止再 fetch/checkout/stash/add/commit/import/test/trainer;之前的 `timeout 30 git fetch origin` 已记录为 constraint_violations: 1。
- orphan 证据包:`/home/ubuntu/evidence/mfr-orphans-20260728T094154Z.tar.gz`,**权威 SHA-256 = `37bb3c9c13442d3eaf3e6d4fdeae99eb66b65925e70adea37489f4c7586d576a`** (取证件内 PACKAGE_INFO / REPORT 中的两个 SHA 经实测更正;详见 `P0_EVIDENCE_CORRECTION.md`)。17 文件 double-hash 稳定性 PASS,捕获区间 `20260728T094154Z→20260728T094820Z`(host clock,非 mtime 证明)。
- handoff 文档位置:`/home/ubuntu/handoff/FluxPhased-g3-bsta-files/g3-bsta-pro6000-handoff/`(57 文件),外层 lite `/home/ubuntu/handoff/g3-bsta/fluxphased_g3_bsta_offline_handoff/`(SHA dbcc2201)。

**2026-07-29 A_FIRST_B_PLUS_FALLBACK 终裁 = `B_PLUS_ELIGIBLE`:**
- 路线 A 详尽搜索(本地 refs/reflog/22+30 unreachable/Copy/IDE/.cache/bash_history/独立 bare mirror/GitHub API 9 branches + code search)0 hit → `historical_source: FAIL`;MFR 源码从未被任何 git commit。
- 路线 B+ 静态审查 10 项全完成:17/17 AST OK;0 secret/symlink/binary/恶意调用;依赖闭包可补齐(8 stdlib + torch + 3 twoteam HEAD 模块);**无 license/author 标记**;§2.3 八项物理绑定全部 UNKNOWN。
- `adoption_eligibility: CONDITIONAL_PASS`;状态 `AWAIT_ADOPTION_OWNER_APPROVAL`;待 5 项签字(repo owner / RF owner / experiment owner / adoption owner / source owner 若有)。
- 三条解封路径:(a) 用户提交 SOURCE_HANDOFF.json;(b) 5 项签字 + repo LICENSE + 8 物理绑定 → 在新 branch `g3-bsta/clean-successor` 上创建带 7 条 trailer 的 adoption commit;(c) clean-room fallback(基于 spec 重写,orphan 不 commit)。
- 文档全集:`/home/ubuntu/evidence/{p0_corrections_20260728, route_a_recovery, route_b_plus_eligibility}/`。
- 相关:[[mfr-iq-m03-outcome]] G2'a 原始负结果;[[chinese-only-responses]]。
