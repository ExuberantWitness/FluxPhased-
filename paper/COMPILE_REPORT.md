# Compilation Report — FluxPhased TAES Draft

**Status:** SUCCESS
**PDF:** `paper/main.pdf`
**Pages:** 7 total (IEEE journal format; references included)
**Undefined citations:** 0 after the final BibTeX pass
**Undefined cross-references:** 0
**Overfull hboxes:** 0 in final compile log
**Fonts:** all embedded
**Figures:** 6 vector PDF figures, all present and rendered in the final PDF
**Visual gate:** PASS on all 7 rendered pages (latest source revision)

## Build

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex  -> 0
bibtex main                                                   -> 0
pdflatex -interaction=nonstopmode -halt-on-error main.tex  -> 0
pdflatex -interaction=nonstopmode -halt-on-error main.tex  -> 0
```

## Final checks

- `pdffonts main.pdf`: all fonts embedded.
- `pdfinfo main.pdf`: 7 pages.
- Final compile log contains no `Citation ... undefined`, `Reference ... undefined`, or `There were undefined references` warnings.
- All six figure labels and two table labels resolve.
- Final visual review of workspace-rendered pages `paper/visual_pages/page-1.png` through `page-7.png`: every page passed.

## Audit state

- `PAPER_CLAIM_AUDIT.md/json`: numeric values and claim scope corrected against raw result files; the primary S6 comparison uses two valid 12-dB seeds.
- `CITATION_AUDIT.md/json`: six active citations have matching complete BibTeX entries; earlier unverified radar discovery scaffolds were removed from active citations.
- Remaining pre-submission work: replace arXiv-only references with official publisher/proceedings versions where available, and consider a final human author/affiliation pass.
