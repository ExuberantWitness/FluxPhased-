# License and Authorship Risk — Orphan MFR Package

**Case**: mfr-orphans-20260728T094154Z
**Orphan package SHA-256 (authoritative)**: `37bb3c9c13442d3eaf3e6d4fdeae99eb66b65925e70adea37489f4c7586d576a`
**Report generated**: 2026-07-29 (host clock node15)

---

## 1. License markers scan

| Marker pattern | Hits in 17 files |
|---|---|
| `Copyright` (case-insensitive) | **0** |
| `License` / `LICENSE` | **0** |
| `MIT` / `GPL` / `Apache` / `BSD` (as license keyword) | **0** |
| `SPDX-License-Identifier` | **0** |
| `@author` / `Author:` | **0** |

**结论**:orphan 包**无任何 license 或 authorship 标记**。这与 FluxPhased- repo 整体无 top-level LICENSE 文件一致(repo level 也无 LICENSE)。

---

## 2. Authorship attribution

| 维度 | 状态 |
|---|---|
| Git blame | **不可用** — 文件从未 commit |
| Git log author | **不可用** — 无 commit |
| File mtime | 2026-07-23 ~ 2026-07-28(创建/修改时间,见 manifest) |
| File owner UID | 1000(ubuntu),与工作树所有者一致 |
| In-file author tag | 无 |
| 嵌入的 user/host path | `/home/ubuntu/CODE/FluxPhased-` 在 10 个文件中作 `sys.path.insert` |

**最合理推断**:由本机 `ubuntu@node15` 用户(运行 PRO6000 / Claude Code 会话的同一身份)在 2026-07-23 至 2026-07-28 期间直接写入工作树。**无第三方作者标记**。

---

## 3. 第三方代码借用检测

| 借用模式 | 检测结果 |
|---|---|
| 外部 repo 抄袭片段 | 无显著标记 |
| Stack Overflow 风格片段 | 无显著标记 |
| `# from` / `# adapted from` / `# based on` 注释 | 无 |
| 论文算法引用(Wang 2025 等) | docstring 中提及"Wang 表 1",但未给完整 citation;无 license 借用 |
| 第三方计算库 | `torch`(BSD-style)、Python stdlib — 无许可证风险 |

**结论**:orphan 是为 FluxPhased- 项目内部由本机用户创作的代码,无显著第三方代码借用。

---

## 4. Authoritative claim 边界

```
strongest_defensible_attribution:
  "17 Python files were written into the working tree of /home/ubuntu/CODE/FluxPhased-
   on host node15 by user uid=1000 (ubuntu) during 2026-07-23 to 2026-07-28
   (per file mtime; mtime is not creation proof). No third-party authorship markers
   are present. The files were never staged, committed, or pushed to any Git remote."

weaker_indefensible_attribution:
  "The orphan files are PRO6000 / Claude Code agent output from prior sessions."
  (no in-file evidence; cannot be claimed or denied from bytes alone)
```

---

## 5. License risk classification

| 风险类别 | 等级 | 说明 |
|---|---|---|
| 项目内部代码无 LICENSE | **MEDIUM** | FluxPhased- repo 整体无 LICENSE;默认适用作者独占版权(all rights reserved);内部使用 OK,外发受限 |
| 第三方代码借用 | LOW | 无显著借用 |
| 商业 SDK 依赖 | LOW | 仅 `torch`(BSD-style) |
| 学术算法引用 | LOW | Wang 2025 引用未完整 citation,但学术引用不是 license 风险 |
| AI 工具生成内容 | UNKNOWN | 无证据,但不能排除 |
| 内部机密信息 | LOW | 无 secret / token / private key 命中 |

---

## 6. Adoption 时的 license 要求

按 B+ clean-successor 准则,**adoption commit 必须明确**:

```text
Source-attribution: unknown
Adoption-status: adopted-new
Adoption-owner: <must be signed by human with authority>
License-inherited-from: FluxPhased- repo-level (currently NONE — must be set)
Historical-M7-recovery: false
```

强烈建议在 adoption 之前**先为 FluxPhased- repo 选定 LICENSE**(MIT / Apache-2.0 / 内部专用),否则 adoption commit 本身也处于 license 真空。

---

## 7. 不可做事项

- **不得**把 orphan 标为 "© 2026 PRO6000" 或类似
- **不得**把 orphan 标为 "originally authored by [name]"(无证据)
- **不得**把 orphan 与任何论文代码关联(无 evidence)
- **不得**在 adoption 前把 orphan 推到任何 GitHub fork(public/private)
- **不得**把 orphan 与 ExuberantWitness 之外的实体关联

---

## 8. 推荐 adoption 路径

```
1. Repo-level: 选定并 commit LICENSE 到 FluxPhased- main 分支(MIT / Apache-2.0 / proprietary)
2. Repo-level: 选定并 commit CONTRIBUTING.md 说明 AI-assisted work 的 attribution 规则
3. Adoption-commit:
   - on a NEW branch (e.g. g3-bsta/clean-successor), NOT on main / twoteam/bc-ppo
   - base = current HEAD (807588c) — verified by SHA
   - commit message contains the five required trailers (Source-attribution etc.)
   - same commit adds LICENSE notice to each adopted file header
4. Branch push requires adoption owner signature on ADOPTION_DECISION_PACKET.md
```

在 (1)-(4) 完成之前,orphan **不能**被任何下游使用、引用或宣称。
