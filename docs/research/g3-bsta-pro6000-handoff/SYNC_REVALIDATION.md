# Sync-Time Repository Revalidation

**Checked**: 2026-07-28 11:14:41 CST
**Repository**: `https://github.com/ExuberantWitness/FluxPhased-.git`
**Docs branch base**: `origin/main` /
`af0d4c20fd2a693cdb14bc64bb786bcb62561883`

## Why this revalidation exists

The pipeline source audit was completed when the newest visible tracked
twoteam commit was `983e1e5`. During synchronization, `git fetch --prune
origin` advanced:

```text
origin/twoteam/bc-ppo
983e1e530dc89c8755eb64b61f78630a32ce835f
→
80769974cb41fd86e2f80bc2a8992955fb228058
```

The newly fetched commits were therefore checked before publishing the
handoff.

## New remote content

The update consists of two WP-3.1 twoteam commits:

```text
8076997 feat(twoteam): WP-3.1 Fix D1+D2 — active-perception shaping + reverse curriculum
0202183 feat(twoteam): WP-3.1 Fix A+B+C — reward shaping + PFSP f_var + entropy gate + 80/20 self-play
```

Changed paths are limited to:

```text
algo/_shared/pilot/twoteam/
env/gpu/twoteam/
experiments/twoteam/
tests/twoteam/
```

## Checks

Read-only checks were run against the fetched remote tree and history:

```bash
git diff --name-only 983e1e5..origin/twoteam/bc-ppo
git ls-tree -r --name-only origin/twoteam/bc-ppo |
  rg '(^|/)(mfr|MFR)|g2a|G2.a|progress_add|tgt_jnr|JAM_POLICY_NOISE'
git grep -n -E \
  'progress_add|tgt_jnr|JAM_POLICY_NOISE|G2.a|g2a' \
  origin/twoteam/bc-ppo -- .
```

## Result

The updated remote still contains none of the required:

```text
env/gpu/mfr/
algo/_shared/pilot/mfr/
progress_add
tgt_jnr
JAM_POLICY_NOISE
G2'a raw records/checkpoints/scripts
```

The implementation status therefore remains:

```text
BLOCK_PPO_PROVENANCE
```

This conclusion is limited to the fetched Git refs. A separately stored source
archive or private commit may exist, but it must be supplied through the
versioned `SOURCE_HANDOFF.json` contract before PRO6000 changes code.
