# Untracked MFR Files: Quarantine and Provenance Protocol

## Decision and scope

The approximately eighteen untracked MFR files reported by PRO6000 are
**unknown-origin orphan bytes**. They are not an authoritative M7 source, do
not belong to a known Git commit, and must not be added to any existing
FluxPhased branch.

The files were reported on a different host or worktree from the machine that
prepared this handoff. Evidence must therefore be captured on the original
PRO6000 host and original worktree. A snapshot made elsewhere cannot substitute
for it.

## Evidence levels

| Level | Allowed statement |
| --- | --- |
| L0 orphan | Untracked files were observed at the reported paths. |
| L1 frozen snapshot | The exact bytes were stable during capture interval `[t0,t1]` on host H. |
| L2 attributed candidate | The bytes/path match a verifiable object, or no verifiable match was found. |
| L3 adopted source | An authorized owner explicitly adopted the snapshot as new code at the current time. |
| L4 implementable M7 | Full source closure, semantics, tests, physical binding and handoff all passed. |

No operation may silently promote evidence between these levels.

## Phase 0: freeze the original scene

Before another fetch, checkout, formatter, import, test or training run:

1. stop PRO6000, trainers, IDE auto-save, formatters and file sync processes
   that may write the affected worktree;
2. prefer a filesystem or VM snapshot (Btrfs, ZFS, LVM or equivalent);
3. if no atomic snapshot exists, record UTC capture start and end and claim
   only a stable capture interval, not an instantaneous state;
4. record hostname, UID, kernel, filesystem/mount details, repository realpath,
   device/inode, current `HEAD`, branch, refs, reflog summary, remote URL, the
   failed fetch command, exit status and complete stderr;
5. record an exact NUL-safe status using
   `GIT_OPTIONAL_LOCKS=0 git status --porcelain=v2 --untracked-files=all -z`;
6. enumerate related ignored files separately using `git check-ignore -v`.

Do not retry fetch in the original repository. Fetch changes refs, reflogs and
`FETCH_HEAD`, even when it does not change the working tree. Network diagnosis
must use a separate bare mirror.

## Phase 1: capture bytes-at-time-t

Create a case directory outside the repository, mode `0700`, such as:

```text
/home/ubuntu/evidence/mfr-orphans-<UTC timestamp>/
```

Do not move or modify the originals. For the full untracked list and every
related ignored file:

- preserve the relative path without following symlinks;
- record type, symlink target, size, mode, UID/GID, inode, mtime, ctime,
  available birth time, ACLs and extended attributes;
- calculate SHA-256 and `git hash-object --no-filters` without `-w`;
- calculate hash pass A on the original;
- copy to the isolated case directory;
- calculate hash pass B on the original and a hash of the copy.

Accept a stable capture only when pass A, pass B and copy hashes match and the
path set/count is unchanged. If they differ, retain the failure logs, stop the
writer and recapture. Do not select whichever version looks most useful.

Package the copied bytes, metadata manifest, command versions and raw
stdout/stderr without dereferencing symlinks. Hash the package and manifest.
Have the collector sign them; a second signer and a trusted external timestamp
are preferable.

The report must say:

> These bytes remained stable during host-clock capture interval `[t0,t1]`.

mtime/ctime are not proof of creation time. A later timestamp proves only that
the manifest existed by that later time, not where the code originated.

Add these records:

```text
ORPHAN_EVIDENCE_MANIFEST.json
ORPHAN_EVIDENCE_REPORT.md
SHA256SUMS.txt
SOURCE_STATUS.txt
```

`SOURCE_STATUS.txt` must contain:

```text
origin_status=UNKNOWN
authority_status=NOT_APPROVED
permitted_use=FORENSIC_INSPECTION_ONLY
```

After the immutable archive and hashes exist, a separate standalone evidence
repository may anchor the snapshot. Use a current-time commit such as:

```text
evidence: snapshot untracked MFR files of unknown origin
```

That commit proves bytes observed at the capture interval. It does not recover
historical provenance.

## Phase 2: isolated inspection

Perform all inspection on a copy in a disposable clone/container with network
disabled and no credentials mounted. Never copy candidate files back into the
original evidence worktree.

Before importing, compiling, testing or executing anything:

1. inspect symlink escape, hardlinks, binaries, secrets, oversized files,
   conflict markers and unexpected generated output;
2. enumerate imports, entrypoints, configs, metrics, tests, locks and assets;
3. compare each raw SHA-256, blob ID and path against local refs, reflogs,
   read-only `git fsck --no-reflogs --unreachable`, a trusted bundle/fresh
   mirror and any owner-signed source archive;
4. record only one of:
   `EXACT_PATH_AND_BLOB_MATCH(commit X)`,
   `CONTENT_MATCH_DIFFERENT_PATH(commit X)`,
   `DERIVED_PATCH_AGAINST(base X)`, or
   `NO_VERIFIABLE_HISTORY_MATCH`.

Even an exact content/path match shows equivalence, not that the disk copy was
originally checked out from that commit.

The docs bundle in this package may be imported into a separate directory. Its
commit contains handoff material only and cannot promote any orphan file.

## Phase 3: promotion criteria

Formal `SOURCE_HANDOFF.json` and an implementation candidate are allowed only
after all of the following pass:

1. stable evidence package, double hashes, archive hash, signatures and
   chain-of-custody;
2. a verifiable commit/tree, or explicit current-time adoption by the source
   owner against a verified base;
3. no forged history: an adoption commit references the evidence case ID,
   snapshot SHA-256 and verified base and declares
   `Source-attribution: unknown / adopted-new`;
4. complete source closure, including imports, entrypoint, config, metric,
   tests, dependency locks and necessary assets;
5. secret, malicious-content, symlink, authorship, license and usage-right
   review;
6. clean-room build and tests from a fresh checkout of the candidate commit;
7. approval by the source owner, RF/simulation physics owner and experiment
   owner;
8. a fixed repository, 40-hex commit, tree hash, `dirty=false`, artifact
   manifest, platform physical binding and validation commands.

An adopted result may be described only as:

> newly adopted and verified M7 candidate source from an orphan snapshot

It must not be described as recovered historical M7 source.

If any source closure, owner, physical semantics or clean-room verification is
missing, retain `BLOCK_SOURCE_HANDOFF / BLOCK_PPO_PROVENANCE`.

## Prohibited before evidence capture

Do not run `clean`, `reset`, `restore`, `checkout`, `switch`, `stash`, `add`,
`commit`, `update-index`, `pull`, `merge`, `rebase`, fetch in the original
repository, `gc`, `prune`, `repack`, `fsck --lost-found`, or
`git hash-object -w`. Do not run an IDE formatter, import, test or trainer; do
not `mv`, `chmod`, `touch`, delete or overwrite the original orphan files; and
do not upload the archive before secret and license review.
