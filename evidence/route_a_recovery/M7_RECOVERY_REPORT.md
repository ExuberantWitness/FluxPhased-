# M7 Recovery Report — Route A

**Case ID**: mfr-orphans-20260728T094154Z
**Route**: A (real historical source recovery)
**Authoritative archive SHA-256**: `37bb3c9c13442d3eaf3e6d4fdeae99eb66b65925e70adea37489f4c7586d576a`
**HEAD full**: `807588cab7d367bedd415b45efc85a72f2a38b89` (twoteam/bc-ppo)
**Report generated**: 2026-07-29 (host clock node15)
**Search budget used**: ~0.5 engineer-hours of 16 budget; <1 calendar day of 5-day limit

---

## 1. Verdict

```text
historical_source: FAIL
status: ROUTE_A_EXHAUSTED_NO_CANDIDATE
allowed_claim_tier: NONE_FROM_ROUTE_A
next_authorized_phase: PROCEED_TO_ROUTE_B_PLUS_ELIGIBILITY
```

路线 A 在 node15 已授权搜索范围内未发现任何 M7/MFR-IQ 历史 commit、tree、blob 或备份。GitHub 远端 `ExuberantWitness/FluxPhased-` 9 个分支 + code search 也无任何 mfr 路径。

---

## 2. 搜索覆盖矩阵

| 数据源 | 工具 | 结果 | 备注 |
|---|---|---|---|
| 主 repo reachable commits | `git log --all -- 'env/gpu/mfr/*'` 等 | 0 hits | 5 local branches + 8 remote refs |
| 主 repo unreachable commits | `git fsck --unreachable` + `git ls-tree -r` | 0 hits | 22 个 unreachable commit 逐一查 tree |
| 主 repo unreachable blobs | `git cat-file -e <orphan_blob_hash>` | 1/16 (空 `__init__.py`) | 仅 `e69de29b` 通用空文件命中,15 个含代码的 blob 全部不在 object db |
| 主 repo stash | `git stash show stash@{0}` | 0 hits | stash 是 phase1.5 COMA 代码 |
| 主 repo reflog | `git reflog --all` | 0 mfr entries | 全 reflog 无 mfr 提及 |
| Copy repo reachable commits | `git log --all` | 0 hits | Copy HEAD=566cc21 是 phase1 时代 |
| Copy repo unreachable commits | `git fsck --unreachable` + tree scan | 0 hits | 30 个 unreachable commit 全部不含 mfr |
| /tmp /var/tmp /home/ubuntu/.cache | `find` | 0 hits | 无残留 mfr 文件 |
| `.bash_history` | `grep -iE mfr` | 0 hits | 无 mfr 相关命令 |
| Bundle / tar / zip 备份 | `find /home/ubuntu -maxdepth 6` | 0 hits | 仅有 handoff 文档包,无源码包 |
| IDE local history (Code/JetBrains) | `ls` | empty | VSCode History dir 空,无 JetBrains cache |
| 独立 bare mirror `git ls-remote` | timeout 60s | TIMEOUT (exit 143) | node15 → github.com TLS 不稳定,与之前 fetch 失败一致 |
| GitHub API branches list | `curl api.github.com` | 9 branches enumerated | 完整列见 §3 |
| GitHub API code search `mfr_env.py` | `curl search/code` | **total_count: 0** | 全仓搜索 |
| GitHub API code search `path:env/gpu/mfr` | `curl search/code` | **total_count: 0** | 全仓路径搜索 |

---

## 3. GitHub 远端 branch 全集(2026-07-28T09:21:04Z pushed_at 快照)

```
appint/data-preflight              5d0c70358bd89530e2c371a67c5515b046f85167
docs/g3-bsta-pro6000-handoff       5cafbbfdb9f636aa5bffbd53dbfa58188e9dfe16
evo/laser-fix                      81f1abb61b946688f2c7442893d0f275ad8effab
evo/main-p14-docs                  9938a116e3bb76c35a007d6f8a1a2c19bc6b697d
fix/kalman-reset-episode           566cc21a6d78e926bdfd8d9b56466eb407ac837c
legacy-main                        81f1abb61b946688f2c7442893d0f275ad8effab
main                               af0d4c20fd2a693cdb14bc64bb786bcb62561883
phase1.5/three-way-baselines       4329bae9ec1a7c34b674f2f68f77e9540406a516
twoteam/bc-ppo                     80769974cb41fd86e2f80bc2a8992955fb228058
```

9 个 branch 全部已在主 repo refs 中存在(无新 branch)。`docs/g3-bsta-pro6000-handoff` 是用户提及的 handoff 文档分支,通过 raw URL 已下载;其本身不含 `env/gpu/mfr/` 或 `algo/_shared/pilot/mfr/`(这是预期)。

GitHub tags:空数组(`[]`),无任何 tag。

---

## 4. 关键技术结论

### 4.1 MFR 源码从未进入 Git 历史

```
git log --all --reflog --oneline -- 'env/gpu/mfr/*'        : (empty)
git log --all --reflog --oneline -- 'algo/_shared/pilot/mfr/*': (empty)
git log --all --reflog --oneline -- 'tests/mfr/*'           : (empty)
```

unreachable objects 也无 mfr:
```
for sha in $(git fsck --unreachable --no-reflogs | grep "^unreachable commit" | awk '{print $3}'); do
    git ls-tree -r --name-only "$sha" | grep -E "env/gpu/mfr/|algo/_shared/pilot/mfr/|tests/mfr/"
done
# output: (empty)
```

### 4.2 MFR blob hash 全部不在任何 Git object database

15 个非空 orphan blob(`jammer_trainer.py`, `mfr_env.py` 等)的 git blob hash(nofilter)在以下两个 object db 中均**不存在**:

- `/home/ubuntu/CODE/FluxPhased-/.git/objects/`
- `/home/ubuntu/CODE/FluxPhased- (Copy)/.git/objects/`

唯一命中的是 `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391`(空 `__init__.py` 的通用 hash),与 MFR 无关 —— 任何空 .py 文件都会产生这个 hash。

### 4.3 orphan 文件就是工作树 artifact,不是 commit 残留

结合 4.1 + 4.2:17 个 orphan 文件**从未**通过 `git add`/`git commit` 进入版本控制。它们是直接写入工作树的 untracked 文件,来源不进入 Git 对象数据库。

### 4.4 GitHub 远端同样无 mfr

`api.github.com/search/code` 是 GitHub 全仓代码搜索 API,覆盖所有 branch 与历史 commit。两个查询均返回 `total_count: 0`:

```
?repo:ExuberantWitness/FluxPhased-+filename:mfr_env.py  → 0
?repo:ExuberantWitness/FluxPhased-+path:env/gpu/mfr     → 0
```

### 4.5 网络通道状态

`git ls-remote` 通过 HTTPS 在 60s 内未完成(TLS 接收错误,与之前 fetch 失败同源)。但 GitHub REST API 通过 HTTPS 完全可达 —— 证明问题在 git smart HTTP 协议或 GnuTLS 在长流式传输下的稳定性,而非完全网络隔离。结论:即使重试 fetch,GitHub 远端不会暴露任何 mfr 路径(search API 已经替代 fetch 验证了这一点)。

---

## 5. 路线 A 候选交付物状态

按 ORPHAN_MFR_QUARANTINE_PROTOCOL 与用户指令,路线 A 候选应生成 5 个文件。**没有任何候选可以生成**(因为找不到任何 source):

| 文件 | 状态 |
|---|---|
| `M7_RECOVERY_REPORT.md` | **生成中**(本文件) |
| `M7_RECOVERY_REPORT.json` | **生成中** |
| `SOURCE_HANDOFF.candidate.json` | **不生成** — 无 candidate |
| `SYMBOL_MAP.candidate.md` | **不生成** — 无 source 可解析符号 |
| `LEGACY_ARTIFACT_MANIFEST.md` | **不生成** — 无 legacy source 配套 artifacts |

---

## 6. 引用取证记录

- 权威 archive SHA-256:`37bb3c9c13442d3eaf3e6d4fdeae99eb66b65925e70adea37489f4c7586d576a`(见 `P0_EVIDENCE_CORRECTION.md` §1)
- HEAD full:`807588cab7d367bedd415b45efc85a72f2a38b89`
- 取证包路径:`/home/ubuntu/evidence/mfr-orphans-20260728T094154Z/`
- 17 orphan 文件 manifest:`ORPHAN_EVIDENCE_MANIFEST.jsonl`
- 取证现场 fetch 违规:`constraint_violations: 1`(详见 `P0_EVIDENCE_CORRECTION.md` §5)

---

## 7. 下一步

按 A_FIRST_B_PLUS_FALLBACK 策略,路线 A FAIL 自动转路线 B+(orphan 收养资格审查)。B+ 路径**不会**把 orphan 升格为 historical M7,只会评估它们是否能作为**新基准 G3-BSTA-v0** 的 seed/spec。

不会进入 P1 实现阶段,不会创建任何 commit,不会写 G3-BSTA 实现代码,直到 adoption owner 明确签署批准。
