# Compilation Report — FluxPhased TAES Draft

**Status:** SUCCESS
**PDF:** `paper/main.pdf`
**Pages:** 10 total (IEEE journal format; references included; within the TAES free-page limit)
**Undefined citations:** 0 after the final BibTeX pass (34 active entries)
**Undefined cross-references:** 0
**Overfull hboxes:** 0 in final compile log (fixed during the TAES pass: split array-factor equation; shortened drop-ratio subscripts; compacted link-budget, spaces, and hyperparameter tables; removed internal directory names from the protocol text)
**Fonts:** all embedded
**Figures:** 6 vector PDF figures; 4 tables (link budget, action/observation spaces, main comparison, scripted baselines, R5, hyperparameters)
**Numeric integrity gate:** `_check_paper_integrity.py` PASS — every quoted number re-derived from `paper/figures/results_table.py` (including the EDF oracle baseline); citation keys and bib entries are bijectively closed (34/34).

## Build

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex  -> 0
bibtex main                                                   -> 0
pdflatex -interaction=nonstopmode -halt-on-error main.tex  -> 0
pdflatex -interaction=nonstopmode -halt-on-error main.tex  -> 0
```

## Final checks

- `pdftotext` text-layer spot checks: Table I jvs 0.5294 and the baseline-table values confirmed in the PDF text layer (one vision-tool misread of a table digit was disproved by the text layer).
- Final compile log contains no `Citation ... undefined`, `Reference ... undefined`, or `There were undefined references` warnings.
- All figure labels and table labels resolve.
- Visual review of the changed pages (baseline table page, results pages, Figure 4 page): layout clean, no overlaps or clipping.
- S7 regression gates re-run after the revision: `test_array_face_s7.py` 12/12, `test_array_factor_s7.py` 6/6 (18/18 total).

## What changed in the 2026-08-29 revision

1. **Figure 4 unit bug fixed**: the S6 neutralization bar was plotted as the fraction 0.637 on a 0–75 axis and labeled "0.6%"; the text value 63.7% was always correct. All figures now import the single-source `paper/figures/results_table.py`; unit conversion exists only there.
2. **Continuation-window transcription fix**: third 2000–3000 window jvs mean corrected from 0.5119 to 0.5118 (re-derived from `val_metrics.jsonl`).
3. **References expanded 6 → 28**, every entry independently verified (publisher pages / Semantic Scholar / arXiv API); related work rewritten into four positioned subsections.
4. **Evaluation-only scripted baselines added** (protocol + results Table III + discussion): random/greedy radar and random/stare jammer views against the three converged teams. Both learned teams dominate random/stare opponents; the greedy mission-stare radar holds drop at 0.0889 against every trained jammer team, reframing h2h as an equilibrium quantity.
5. Abstract, introduction (five contributions), discussion, and limitations updated for the baseline findings and their boundary conditions.

## Audit state

- `PAPER_CLAIM_AUDIT.md`: PASS with the 2026-08-29 corrections recorded; numeric enforcement automated via the integrity gate.
- `CITATION_AUDIT.md`: PASS with three deliberate metadata gaps (two Bell DOIs, Jia author tail, IPPO arXiv-only) flagged for camera-ready.
- Remaining pre-submission work: SNR/jammer-power sensitivity retraining if a stronger external-validity claim is desired; R5 gradient-aligned multi-seed follow-up; camera-ready author/affiliation pass.
