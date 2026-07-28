# PRO6000 Resume Prompt After Network Handoff Failure

**Transport precondition:** do not send only this text to PRO6000. First copy
the entire `fluxphased_g3_bsta_offline_handoff/` directory, or its portable
outer archive, onto the PRO6000 host. A path printed in a chat message is not a
cross-host file transfer.

After copying, replace `<HANDOFF_DIR>` below with the absolute directory that
actually exists on PRO6000. Do not assume `/home/exuber/...` exists there.

Copy the following block to PRO6000 only after the transfer is complete.

```text
Human resolution of your blocker:

T0. Transport gate — this must pass before any other phase:

   HANDOFF_DIR=<absolute directory physically present on this host>
   IMPORT_DIR=/home/ubuntu/handoff/FluxPhased-g3-bsta-docs

   test -d "$HANDOFF_DIR"
   test -r "$HANDOFF_DIR/SHA256SUMS.txt"
   test -r "$HANDOFF_DIR/PRO6000_RESUME_PROMPT.md"
   test -r "$HANDOFF_DIR/ORPHAN_MFR_QUARANTINE_PROTOCOL.md"
   test -r "$HANDOFF_DIR/FluxPhased-docs-g3-bsta-bc8de428.bundle" ||
     test -r "$HANDOFF_DIR/FluxPhased-g3-bsta-handoff-bc8de428.tar.gz"

   If any test fails, stop without searching unrelated hosts or claiming the
   handoff was imported:

     phase: T0
     status: BLOCKED
     stop_reason: EXECUTOR_ARTIFACT_NOT_COPIED
     next_required_action: SOURCE_HOST_TRANSFER
     next_authorized_phase: NONE

   A `/home/exuber/...` path mentioned by the human is a source-host path, not
   evidence that the same path is mounted on this host. Do not retry fetch or
   search the entire host as a substitute for a failed transport gate.

1. The remote branch and commit do exist and were independently revalidated:

   branch:
     docs/g3-bsta-pro6000-handoff
   commit:
     bc8de428d86a7f6e47123375c5a0a06a8eb4953f
   parent:
     af0d4c20fd2a693cdb14bc64bb786bcb62561883

   GitHub commit and raw README both returned HTTP 200. Your 134-second curl
   timeout is classified as EXECUTOR_NETWORK_PATH_FAILURE, not REMOTE_MISSING.

2. Record the following already-observed forensic limitation:

   A post-instruction `timeout 30 git fetch origin` was attempted in the
   original repository before orphan evidence capture. It timed out, but the
   attempt still violates the no-fetch-before-capture rule and may have touched
   Git metadata.

   Therefore:
   - report `constraint_violations >= 1`, not zero;
   - preserve the command, UTC time, exit status and stderr;
   - do not claim a pristine pre-fetch Git metadata scene;
   - perform no further fetch in the original repository.

   The orphan bytes may still reach L1 frozen-snapshot status if the required
   stable double-hash capture passes, with this limitation disclosed.

3. Before importing the docs package, freeze and capture the original
   worktree's orphan MFR evidence exactly as required by
   ORPHAN_MFR_QUARANTINE_PROTOCOL.md. The files exist on your host/worktree,
   so no snapshot made on another machine can substitute for yours.

   Until capture completes, do not retry fetch in the original repository and
   do not run checkout, stash, add, commit, formatter, import, tests or
   training. A fetch changes refs/reflogs/FETCH_HEAD and contaminates scene
   metadata even if the working tree is unchanged. If network diagnosis is
   needed, use a separate bare mirror.

4. Because your network path is unreliable, use the transferred offline
   handoff instead of retrying indefinitely.

   First run:

     cd "$HANDOFF_DIR"
     sha256sum --ignore-missing -c SHA256SUMS.txt

   All physically present files listed by the manifest must pass. Emit a
   transport receipt containing at least:

     destination_host=node15
     destination_path=<resolved HANDOFF_DIR>
     sha256_verification=PASS
     handoff_transport=VERIFIED_BUNDLE or VERIFIED_TAR
     carrier_sha256=<verified bundle or tar SHA-256>

   If the bundle is present, use the preferred Git import:

     git clone \
       -b docs/g3-bsta-pro6000-handoff \
       "$HANDOFF_DIR/FluxPhased-docs-g3-bsta-bc8de428.bundle" \
       "$IMPORT_DIR"

     test "$(
       git -C "$IMPORT_DIR" rev-parse HEAD
     )" = bc8de428d86a7f6e47123375c5a0a06a8eb4953f

   Set:

     DOCS_DIR="$IMPORT_DIR/docs/research/g3-bsta-pro6000-handoff"

   If the bundle is unavailable but the tar archive is present, use:

     IMPORT_DIR=/home/ubuntu/handoff/FluxPhased-g3-bsta-files
     mkdir -p "$IMPORT_DIR"
     tar -xzf \
       "$HANDOFF_DIR/FluxPhased-g3-bsta-handoff-bc8de428.tar.gz" \
       -C "$IMPORT_DIR"
     DOCS_DIR="$IMPORT_DIR/g3-bsta-pro6000-handoff"

   In that fallback case record `HANDOFF_TRANSPORT=VERIFIED_TAR`; do not claim a
   Git fetch or that a Git commit was locally checked out.

5. After successful import, read all mandatory handoff files from:

   $DOCS_DIR/

   The documentation-availability blocker is then resolved.

   The bundle is intentionally a documentation handoff carrier. It is expected
   not to contain `env/gpu/mfr/` or `algo/_shared/pilot/mfr/`. Do not report
   their absence from this bundle as a transport failure or treat the bundle as
   an authoritative M7 source.

6. The untracked MFR files in your original worktree remain non-authoritative.
   You are authorized to preserve and statically inspect them only under
   ORPHAN_MFR_QUARANTINE_PROTOCOL.md.

   Do not:
   - delete them;
   - execute them before snapshotting;
   - add them to an existing FluxPhased branch;
   - call them M7 source;
   - use them to populate SYMBOL_MAP as if verified.

   First create a stable double-hash snapshot outside the repository. Only
   after that immutable package exists may you create a separate evidence
   repository with origin_status=UNKNOWN and
   permitted_use=FORENSIC_INSPECTION_ONLY. A new evidence commit proves only
   bytes stable during capture interval [t0,t1], not historical origin.

7. Resume P0 only. Re-run the provenance report with these distinct results:

   HANDOFF_DOCS:
     PASS if bundle/tar hashes and fixed commit/content pass.

   AUTHORITATIVE_M7_SOURCE:
     PASS only if a separate SOURCE_HANDOFF.json identifies and verifies the
     real M7 repository/archive, complete source tree, hashes and owner.

   ORPHAN_MFR_FILES:
     QUARANTINED / ORIGIN_UNKNOWN until the promotion criteria pass.

8. Do not confuse successful documentation import with source recovery.
   If authoritative M7 source is still unavailable, finish with:

     phase: P0
     status: BLOCKED
     handoff_docs: PASS
     source_provenance: FAIL
     orphan_files: QUARANTINED_ORIGIN_UNKNOWN
     code_changes: none
     stop_reason: BLOCK_PPO_PROVENANCE
     next_authorized_phase: NONE

   If and only if authoritative source validation later passes, produce
   SYMBOL_MAP.md and P0_BINDING_PACKET.md from that verified source, then stop
   at AWAIT_P0_HUMAN_BINDING_APPROVAL. Do not enter P1 automatically.
```
