# GitHub Offline-Handoff Upload Manifest

This directory mirrors every handoff and transport artifact prepared after the
PRO6000 `node15` network-path failure.

## Browser-readable operator files

| File | Purpose |
| --- | --- |
| `PRO6000_RESUME_PROMPT.md` | Transport-gated P0 resume instructions |
| `ORPHAN_MFR_QUARANTINE_PROTOCOL.md` | Original-host evidence capture and source-promotion rules |
| `README.md` | Offline package usage |
| `SHA256SUMS.txt` | Checksums used after package extraction |

## Portable packages

The `packages/` directory contains:

- a 120-KiB lite package using `HANDOFF_TRANSPORT=VERIFIED_TAR`;
- a 23-MiB complete package that also contains the Git bundle;
- a `.sha256` sidecar for each package.

The standalone bundle and inner 57-file archive are not duplicated as Git
blobs. They are present inside the complete package; the 57-file archive is
also present inside the lite package. This retains every handoff file while
avoiding an unnecessary second 23-MiB copy in repository history.

## Integrity anchors

```text
bundle:
5432112675ca28b351b9f07299d9edbc4b2816a87cf95f2a6e370567b70dc657

57-file documentation archive:
09249a625961e6c0a7fd6c72568bea2ab05f0be98ebb58b79591e7043c55d216

lite package:
dbcc220166190931916000d87c1450fb1aad5cd7c22283552b1252700b0e12a1

complete portable package:
451e070d6ff041642127c92148a9158abb99093d599a6ebc90142b88db2de804
```

## Provenance boundary

These artifacts resolve handoff-document transport only. The bundle is
intentionally a documentation carrier and does not contain authoritative MFR
source. Uploading or downloading it does not promote the untracked files
reported on `node15` and does not clear `BLOCK_PPO_PROVENANCE`.
