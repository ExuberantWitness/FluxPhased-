# FluxPhased G3-BSTA Offline Handoff

This package resolves the network-specific handoff blocker reported by the
PRO6000 execution environment.

## Remote facts independently revalidated

```text
branch:
  refs/heads/docs/g3-bsta-pro6000-handoff
commit:
  bc8de428d86a7f6e47123375c5a0a06a8eb4953f
parent:
  af0d4c20fd2a693cdb14bc64bb786bcb62561883
```

At revalidation time:

- `git ls-remote` returned the branch and exact commit;
- the GitHub commit page returned HTTP 200;
- the raw README returned HTTP 200.

The PRO6000 timeout is therefore an executor-network-path failure, not evidence
that the branch or commit was never pushed.

## Files

### Small document-only archive

```text
FluxPhased-g3-bsta-handoff-bc8de428.tar.gz
SHA-256:
09249a625961e6c0a7fd6c72568bea2ab05f0be98ebb58b79591e7043c55d216
```

Contains the 57 handoff files only. It can be extracted without Git or network
access.

### Self-contained Git bundle

```text
FluxPhased-docs-g3-bsta-bc8de428.bundle
SHA-256:
5432112675ca28b351b9f07299d9edbc4b2816a87cf95f2a6e370567b70dc657
```

Contains the complete history required by
`docs/g3-bsta-pro6000-handoff`. `git bundle verify` passed and an actual offline
clone was checked out at the exact commit.

## Preferred offline import

The path containing this README is on the source host. It is not visible to
PRO6000 until an explicit cross-host copy succeeds. Transfer the entire
directory to the PRO6000 host, for example:

```text
/home/ubuntu/handoff/fluxphased_g3_bsta_offline_handoff/
```

Then:

```bash
cd /home/ubuntu/handoff/fluxphased_g3_bsta_offline_handoff
sha256sum -c SHA256SUMS.txt

git clone \
  -b docs/g3-bsta-pro6000-handoff \
  FluxPhased-docs-g3-bsta-bc8de428.bundle \
  /home/ubuntu/handoff/FluxPhased-g3-bsta-docs

git -C /home/ubuntu/handoff/FluxPhased-g3-bsta-docs \
  rev-parse HEAD
```

The final command must print:

```text
bc8de428d86a7f6e47123375c5a0a06a8eb4953f
```

If only the small archive is transferred:

```bash
mkdir -p /home/ubuntu/handoff/FluxPhased-g3-bsta-files
tar -xzf FluxPhased-g3-bsta-handoff-bc8de428.tar.gz \
  -C /home/ubuntu/handoff/FluxPhased-g3-bsta-files
```

The bundle is preferred because it preserves the Git commit and branch
provenance. The tar archive is a network-independent document fallback.

Before sending the resume prompt, verify on PRO6000 itself that the destination
directory and all files exist. `PRO6000_RESUME_PROMPT.md` now begins with a T0
transport gate and must stop with `EXECUTOR_ARTIFACT_NOT_COPIED` when they do
not.

If the source host cannot resolve `node15`, obtain an explicit node15 IP/FQDN
or use an already configured shared-storage path. Do not guess a network
address. Example source-host transfer after the address is known:

```bash
ssh ubuntu@<NODE15_IP_OR_FQDN> 'mkdir -p /home/ubuntu/handoff'
scp -r fluxphased_g3_bsta_offline_handoff \
  ubuntu@<NODE15_IP_OR_FQDN>:/home/ubuntu/handoff/
```

## Continue safely

After import, give PRO6000:

- `PRO6000_RESUME_PROMPT.md`;
- `ORPHAN_MFR_QUARANTINE_PROTOCOL.md`.

The offline handoff resolves only the documentation-availability blocker. It
does not make untracked MFR files authoritative and does not authorize PPO or
implementation beyond P0.
