# Research Output Manifest

**Run ID**: `fluxphased-g2a-redesign-20260728`  
**Scope**: FluxPhased MFR-IQ G2'a structural diagnosis and PRO6000 implementation handoff  
**Current verdict**: `BLOCK_PPO_PROVENANCE`  
**New gate**: `G3-BSTA`  
**Original gate**: G2'a remains `FAIL / unverified reachability`  
**Target-repository mutations**: none  
**Review status**: `same-family / provisional`

## Primary handoff

| Stage | Artifact | Status | SHA-256 |
|---|---|---|---|
| summary | [PIPELINE_SUMMARY_20260728_105723.md](PIPELINE_SUMMARY_20260728_105723.md) | complete, unaccepted | `9e0fa614e90545bda055eb9160fc9be0ecae2384e14727a5d3ca172c1295e389` |
| executor directive | [PRO6000_EXECUTION_PROMPT_20260728_105723.md](PRO6000_EXECUTION_PROMPT_20260728_105723.md) | P0-only authorization | `3d73ef22ff62a45a8818a6eaa65b9f16ca483d4f6fdda0c82b7af509509712aa` |
| implementation spec | [PRO6000_AGENT_IMPLEMENTATION_SPEC_20260728_023103.md](PRO6000_AGENT_IMPLEMENTATION_SPEC_20260728_023103.md) | provisional after review | `a73a55683ba79db2bb3f89605e92d2fa9ade09303206af0adb2e169c6416ab42` |
| final proposal | [FINAL_PROPOSAL_20260728_023103.md](refine-logs/FINAL_PROPOSAL_20260728_023103.md) | provisional | `f6c2d54a390106b8edd8d0afc43f811d58032be640c3ca21911768048de2eb2e` |
| experiment plan | [EXPERIMENT_PLAN_20260728_023103.md](refine-logs/EXPERIMENT_PLAN_20260728_023103.md) | protocol drafted, not executed | `68ba832acedc6c7a115cf4c82b77d58699eecc985d2205576055fe7ff2287081` |
| execution tracker | [EXPERIMENT_TRACKER_20260728_023103.md](refine-logs/EXPERIMENT_TRACKER_20260728_023103.md) | blocked at P0 | `ef00d1d6a20dd471d2576d2a0de5ee1b812f12ceca39a5fad473537bba5cd37f` |
| source handoff template | [SOURCE_HANDOFF.template.json](SOURCE_HANDOFF.template.json) | required external input | `d940ec7f4bd931842a7ac6bf1f73c1f916d1f5d3b6231d84397395f0575b42b2` |
| source audit | [SOURCE_PROVENANCE_AUDIT_20260728_105723.md](SOURCE_PROVENANCE_AUDIT_20260728_105723.md) | blocker confirmed | `817798048b3490fb9a72e69ce9fa3c5397a1564acfcf9dfe2e107dfc32c9cc3c` |

## Research stages

| Stage | Artifact | Status | SHA-256 |
|---|---|---|---|
| scoped brief | [RESEARCH_BRIEF_20260728_102310.md](RESEARCH_BRIEF_20260728_102310.md) | complete | `c0262fc74f3b7bbe2395424a1ef725245ebc3e7dd496d032c7eb256983d35ae8` |
| literature | [LITERATURE_REPORT_20260728_023103.md](LITERATURE_REPORT_20260728_023103.md) | complete; novelty amended | `f2a47844db350286eea056a97607b9352a8d389995246c7a6fa36e11ade99e72` |
| idea generation | [IDEA_REPORT_20260728_023103.md](idea-stage/IDEA_REPORT_20260728_023103.md) | complete | `8b2327615e12b0bcd4a8f8ac3ca961cded672aad8e4a5c22d0cd91dd2b70fcb5` |
| independent jury | [JURY_DECISION_20260728_023103.md](idea-stage/JURY_DECISION_20260728_023103.md) | provisional | `6862c92c5be4344ddb64830b7c4f8d522fc65e412ebc86053617b3d9c4b9f5b4` |

## Independent reviews

| Review | Artifact | Verdict | SHA-256 |
|---|---|---|---|
| novelty | [NOVELTY_REVIEW_20260728_105723.md](review-traces/NOVELTY_REVIEW_20260728_105723.md) | low novelty, confidence .92 | `f06630982fa06d55ef38fd1217c714f08e2872673f3eb3d2878afa931935bcbc` |
| implementation/research | [RESEARCH_REVIEW_20260728_105723.md](review-traces/RESEARCH_REVIEW_20260728_105723.md) | BLOCK / major revision | `746bd5cd16df704ee9882a3dfda84e4ed573cf8761b07779fdd74237ca3f3bea` |
| statistics | [STATS_REVIEW_20260728_105723.md](review-traces/STATS_REVIEW_20260728_105723.md) | major revision | `22d028254d1baa5e8a0845d89ae299bd54dc411f20d0e5ae9845bf24d5922fb0` |
| amendment log | [REVIEW_RESPONSE_20260728_105723.md](review-traces/REVIEW_RESPONSE_20260728_105723.md) | amendments adopted | `7a0b85aa7c7d612c3ecd0fbf3ca835e0539da1e7aeea3ca83df1650959e0d89b` |

Review traces were also saved under:

```text
.aris/traces/novelty-check/2026-07-28_run01/
.aris/traces/research-review/2026-07-28_run01/
```

## Pipeline state

```text
idea-discovery     provisional
experiment-bridge provisional
auto-review-loop   provisional
summary            done / unaccepted
paper-writing      skipped (implementation handoff scope)
```

HTML rendering was intentionally skipped: the primary deliverable is a code-oriented Markdown/JSON handoff for PRO6000, not a narrative publication artifact.

