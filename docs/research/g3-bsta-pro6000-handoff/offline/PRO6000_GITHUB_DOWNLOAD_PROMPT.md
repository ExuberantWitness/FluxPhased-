# PRO6000 GitHub Download and P0 Resume Prompt

Copy the block below to PRO6000 on `node15`.

```text
Human resolution and authorization:

The complete G3-BSTA handoff has now been uploaded to GitHub. Use only the
fixed immutable artifact commit below; do not substitute a moving branch tip.

REPOSITORY:
  https://github.com/ExuberantWitness/FluxPhased-

ARTIFACT_COMMIT:
  fd1cfff51b2545d1fb1a2b4305a39f030f76c0c9

LITE_PACKAGE_SHA256:
  dbcc220166190931916000d87c1450fb1aad5cd7c22283552b1252700b0e12a1

Important boundaries:

1. GitHub upload does not prove that node15 can reach GitHub. Your earlier TLS
   timeout remains an executor network-path condition until a bounded download
   succeeds.
2. Do not fetch, checkout, stash, add, commit, import, test or train in the
   original FluxPhased worktree before orphan evidence capture.
3. A prohibited `timeout 30 git fetch origin` was already attempted before
   capture. Record it as a forensic limitation and report
   `constraint_violations >= 1`. Perform no further fetch in the original
   repository.
4. Download into a separate handoff directory. Do not use a repository-wide
   search or the original worktree as a transport mechanism.

T0 — bounded GitHub transport:

  mkdir -p /home/ubuntu/handoff/github-download
  cd /home/ubuntu/handoff/github-download

  PACKAGE_URL=https://raw.githubusercontent.com/ExuberantWitness/FluxPhased-/fd1cfff51b2545d1fb1a2b4305a39f030f76c0c9/docs/research/g3-bsta-pro6000-handoff/offline/packages/fluxphased_g3_bsta_offline_handoff-bc8de428-lite.tar.gz
  SHA_URL=https://raw.githubusercontent.com/ExuberantWitness/FluxPhased-/fd1cfff51b2545d1fb1a2b4305a39f030f76c0c9/docs/research/g3-bsta-pro6000-handoff/offline/packages/fluxphased_g3_bsta_offline_handoff-bc8de428-lite.tar.gz.sha256

  timeout 90 curl \
    --fail --location --connect-timeout 10 --max-time 75 --retry 1 \
    --output fluxphased_g3_bsta_offline_handoff-bc8de428-lite.tar.gz.part \
    "$PACKAGE_URL"

  On success only:

  mv \
    fluxphased_g3_bsta_offline_handoff-bc8de428-lite.tar.gz.part \
    fluxphased_g3_bsta_offline_handoff-bc8de428-lite.tar.gz

  timeout 30 curl \
    --fail --location --connect-timeout 10 --max-time 20 --retry 1 \
    --output fluxphased_g3_bsta_offline_handoff-bc8de428-lite.tar.gz.sha256 \
    "$SHA_URL"

  sha256sum -c \
    fluxphased_g3_bsta_offline_handoff-bc8de428-lite.tar.gz.sha256

Do not retry indefinitely. If either download or SHA verification fails, stop:

  phase: T0
  status: BLOCKED
  stop_reason: EXECUTOR_GITHUB_UNREACHABLE
  artifact_commit: fd1cfff51b2545d1fb1a2b4305a39f030f76c0c9
  code_changes: none
  next_required_action: OUT_OF_BAND_COPY_TO_NODE15
  next_authorized_phase: NONE

If T0 passes:

  mkdir -p /home/ubuntu/handoff/g3-bsta
  tar -xzf \
    /home/ubuntu/handoff/github-download/fluxphased_g3_bsta_offline_handoff-bc8de428-lite.tar.gz \
    -C /home/ubuntu/handoff/g3-bsta

  HANDOFF_DIR=/home/ubuntu/handoff/g3-bsta/fluxphased_g3_bsta_offline_handoff
  cd "$HANDOFF_DIR"
  sha256sum --ignore-missing -c SHA256SUMS.txt

Read these files completely, without skipping:

  $HANDOFF_DIR/ORPHAN_MFR_QUARANTINE_PROTOCOL.md
  $HANDOFF_DIR/PRO6000_RESUME_PROMPT.md
  $HANDOFF_DIR/README.md

Then execute PRO6000_RESUME_PROMPT.md exactly. Use its VERIFIED_TAR fallback;
the lite package intentionally does not contain the 23-MiB Git bundle.

The inner 57-file archive contains the handoff documentation, not authoritative
M7 source. Absence of `env/gpu/mfr/` and `algo/_shared/pilot/mfr/` from that
documentation archive is expected and must not be reported as a transport
failure.

Resume P0 only. If authoritative M7 provenance is still absent after reading
the handoff and quarantining the original-host orphan files, finish:

  phase: P0
  status: BLOCKED
  handoff_docs: PASS
  source_provenance: FAIL
  orphan_files: QUARANTINED_ORIGIN_UNKNOWN
  constraint_violations: 1
  code_changes: none
  stop_reason: BLOCK_PPO_PROVENANCE
  next_authorized_phase: NONE

Do not enter P1 unless a separate authoritative SOURCE_HANDOFF passes and a
human explicitly approves the P0 binding packet.
```
