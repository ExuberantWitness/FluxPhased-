# Source and Provenance Audit

**Audit time**: 2026-07-28 10:57 CST  
**Target repository**: `/home/exuber/CODE/CORE/pythonProject1/FLUXPH/FluxPhased-`  
**Local branch / HEAD**: `blind-spec-push` / `fa485ad4cf6314df8a747498f4179d702a7c4923`  
**Tracked upstream**: `origin/twoteam/bc-ppo` / `983e1e530dc89c8755eb64b61f78630a32ce835f`  
**Verdict**: `BLOCK_PPO_PROVENANCE`

## Read-only checks

The audit inspected:

- local heads, tags and all visible remote-tracking refs;
- all paths in the current working tree matching MFR/G2'a/jammer terms;
- filenames across all visible Git history matching MFR/G2'a and the reported implementation symbols;
- current branch status and exact commit identifiers.

## Findings

1. No visible ref contains the required `env/gpu/mfr/*` or `algo/_shared/pilot/mfr/*` source tree.
2. No visible history path contains the reported MFR/G2'a symbols such as `progress_add`, `tgt_jnr` or `JAM_POLICY_NOISE`.
3. The current tree has generic jammer material, notably `algo/_shared/pilot/taes/train_jammer.py`, `configs/wp4_ew_jam.yaml` and jammer smoke/results files. These are not evidence of the attached report's MFR-IQ implementation and must not be patched as substitutes.
4. The local branch is behind its tracked remote, but the newer visible upstream history is TAES/twoteam work and still does not supply the reported MFR source.
5. The prior G2'a report's `/tmp` scripts, raw episode rows, checkpoint/config hashes and source snapshot remain unavailable in the audited workspace.

## Consequence

PRO6000 must not infer filenames or transplant the redesign into the TAES code path. It may run P0 discovery and fill `SYMBOL_MAP.md`, but implementation begins only after a completed `SOURCE_HANDOFF.json` is cryptographically verified.

Missing legacy raw artifacts and missing source have different effects:

- missing source blocks implementation;
- missing legacy raw artifacts block reproduction and any claim about the original result, but do not by themselves prohibit a separately identified new benchmark once the actual source and physics contract are available.

