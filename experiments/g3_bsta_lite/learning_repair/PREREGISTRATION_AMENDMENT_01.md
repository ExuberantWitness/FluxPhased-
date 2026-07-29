# Preregistration Amendment 01 — Manifest Base-Seeds

```text
document:    PREREGISTRATION_AMENDMENT_01.md
branch:      g3-bsta/mfr-lite-learning-repair
base:        526ab938cfa991952ce5eb03f5d304a5dda789cf
parent_doc:  PREREGISTRATION.md
issued:      2026-07-29
scope:       §3 Scenario manifests (base_seed values only)
status:      BINDING (replaces §3 base_seed column until further amendment)
```

## 1. What changed

The base_seed values in PREREGISTRATION.md §3 are respaced. The
**sizes, purposes, and "must be disjoint" requirement are unchanged**;
only the seed offsets shift.

| manifest | old base_seed | new base_seed | size | reserved window |
|---|---|---|---|---|
| `dagger_train.json`          | 21000101 | **21000101** (unchanged) | 128 | [21000101, 21001000) |
| `ppo_train.json`             | 21000201 | **21001101**             |  64 | [21001101, 21002000) |
| `checkpoint_validation.json` | 21000301 | **21002101**             |  64 | [21002101, 21003000) |
| `locked_test.json`           | 21000401 | **21003101**             | 128 | [21003101, 21004000) |

The four reserved windows are 1000 seeds wide each and pairwise disjoint.
With `arrival_rate_per_service = 0.15`, `horizon = 64`, `n_services = 2`,
every seed is eligible (P(any arrival) ≈ 1 − (1 − 0.15)^128 ≈ 1), so
each manifest occupies a contiguous seed block well inside its own
window; collisions across windows are therefore impossible by
construction.

## 2. Why

The original §3 spacing was 100 seeds between consecutive base_seeds.
Under the frozen debug profile every seed is eligible, so `dagger_train`
(starting at 21000101, needing 128 eligible scenarios) scans seeds
`21000101..21000228`. `ppo_train` (starting at 21000201) collides with
`dagger_train` on seeds `21000201..21000228` — 28 overlapping seeds
detected by `MANIFEST_AUDIT.json` at first generation.

This is a procedural miss in seed accounting, not a research-result
sensitive parameter. The amendment is made before any model training
runs on this branch, before any R1/R2/R3 evaluation, and the four
manifests are generated **exactly once** under the amended table.

## 3. What did NOT change

- Sizes (128 / 64 / 64 / 128) — unchanged.
- Manifest purposes — unchanged.
- Pairwise-disjoint requirement — unchanged (still asserted in code and
  unit tests, recorded in `MANIFEST_AUDIT.json`).
- Legacy-range exclusion (`20260801..20260832`) — unchanged, still
  asserted.
- Per-scenario SHA-256 of the arrivals table — unchanged.
- Profile (horizon / n_services / arrival_rate / baseline_snr_db) —
  unchanged.
- Every threshold in §6 (R2 Gate 3), §7 (R3 PASS), §8 (R4 POMDP) —
  **unchanged and still frozen**.
- Every control, statistic, and forbidden action in §9, §10, §11 —
  unchanged.

## 4. Audit consequence

`MANIFEST_AUDIT.json` is regenerated under the amended table; its
`overall_verdict` MUST read `ALL_DISJOINT_AND_LEGACY_CLEAN` before R1B
is allowed to start. The audit records both the new pairwise
intersections (which must all be empty) and the legacy-range exclusion
check (which must be clean).

## 5. Forbidden by this amendment

- Re-spacing base_seeds a second time. Any further change requires a
  new `PREREGISTRATION_AMENDMENT_02.md` that names the conflict.
- Picking base_seeds outside the `2100xxxx` block to "make the audit
  pass" — only the four values listed above are authorised.
- Using any seed inside another manifest's reserved window for a
  purpose other than that manifest's stated role.
