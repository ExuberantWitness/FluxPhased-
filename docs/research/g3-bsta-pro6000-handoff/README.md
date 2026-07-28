# G3-BSTA Research and PRO6000 Handoff

This directory is the complete versioned handoff for the FluxPhased MFR-IQ
G2'a investigation and the proposed `G3-BSTA` redesign.

## Current decision

```text
BLOCK_PPO_PROVENANCE
→ PRO6000 may execute P0 source/symbol discovery only
→ original G2'a remains FAIL / unverified reachability
→ G3-BSTA is a separate environment and gate
```

No environment, PPO or experiment code is changed by this branch.

## Start here

1. [Research-pipeline summary](pipeline/PIPELINE_SUMMARY_20260728_105723.md)
2. [Copy-paste PRO6000 directive](pipeline/PRO6000_EXECUTION_PROMPT_20260728_105723.md)
3. [Detailed implementation specification](pipeline/PRO6000_AGENT_IMPLEMENTATION_SPEC_20260728_023103.md)
4. [Experiment and statistical protocol](pipeline/refine-logs/EXPERIMENT_PLAN_20260728_023103.md)
5. [Execution tracker](pipeline/refine-logs/EXPERIMENT_TRACKER_20260728_023103.md)
6. [Source handoff template](pipeline/SOURCE_HANDOFF.template.json)
7. [Current sync-time source revalidation](SYNC_REVALIDATION.md)
8. [GitHub download and P0 resume prompt](offline/PRO6000_GITHUB_DOWNLOAD_PROMPT.md)
9. [Offline upload manifest and packages](offline/UPLOAD_MANIFEST.md)

PRO6000 must copy `SOURCE_HANDOFF.template.json` to `SOURCE_HANDOFF.json`, fill
and independently verify it, produce the P0 binding packet, and stop for
approval before implementation.

## Directory layout

```text
.
├── README.md
├── SYNC_REVALIDATION.md
├── inputs/
│   └── ORIGINAL_PROBLEM_STATEMENT.txt
├── pipeline/
│   ├── literature, idea, proposal and implementation artifacts
│   ├── experiment plan and tracker
│   ├── independent review prompts and responses
│   └── MANIFEST.md
├── prior-audit/
│   ├── EXPERIMENT_AUDIT.md
│   └── EXPERIMENT_AUDIT.json
└── provenance/
    ├── aris-traces/
    │   ├── experiment-audit/
    │   ├── idea-creator/
    │   ├── novelty-check/
    │   └── research-review/
    └── run-state/
```

## Provenance boundary

- Branch base: `origin/main` at
  `af0d4c20fd2a693cdb14bc64bb786bcb62561883`.
- The historical source audit recorded the then-visible
  `origin/twoteam/bc-ppo=983e1e5`.
- Immediately before this sync, the remote advanced to
  `80769974cb41fd86e2f80bc2a8992955fb228058`; the new commits were rechecked
  and still contained no MFR/M7 paths or reported G2'a symbols. See
  [SYNC_REVALIDATION.md](SYNC_REVALIDATION.md).
- Independent reviews are `same-family / provisional`; they are not external
  peer review or experimental validation.

Git records the exact content of this bundle. The primary artifact hashes
generated before synchronization are recorded in
[pipeline/MANIFEST.md](pipeline/MANIFEST.md).

Files below `provenance/aris-traces/` are immutable forensic records. Their
prompts/responses intentionally retain the absolute paths that existed during
the audit; those path-shaped Markdown links are historical evidence references,
not navigational links in the GitHub copy.
