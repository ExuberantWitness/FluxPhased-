# Citation Audit Report

**Paper:** *Scaling the Attack Breaks Defense Containment in Task-Level Radar--Jammer Self-Play*
**Active citations:** 6

## Verdict

**WARN — bibliography is structurally clean and the six active records resolve to identifiable canonical sources, but the final submission should retain the official publisher/proceedings metadata URLs in the author-maintained bibliography record.** No citation key is missing or uncited.

## Active entries

| Key | Source | Metadata status | Context status |
|---|---|---|---|
| `schulman2017proximal` | arXiv:1707.06347 | verified from arXiv; DOI included | supports PPO description |
| `yu2022surprising` | NeurIPS 2022 / arXiv:2103.01955 | verified from arXiv/OpenReview metadata | supports cooperative PPO/MAPPO context |
| `heinrich2016deep` | arXiv:1603.01121 | verified from arXiv | supports NFSP/self-play context |
| `vinyals2019alphastar` | Nature 575, 350--354 (2019), DOI 10.1038/s41586-019-1724-z | verified from DOI/publisher record | supports league/population comparison |
| `wang2020roma` | ICML 2020, PMLR 119, 9876--9886, arXiv:2003.08039 | verified from arXiv/PMLR metadata | supports emergent-role context |
| `wu2018moba` | arXiv:1812.07887 | verified from arXiv | supports hierarchical macro/micro coordination context |

## Checks

- Every `\\cite{...}` key in `paper/main.tex` and `paper/sections/*.tex` has exactly one matching BibTeX entry.
- No active BibTeX entry is uncited.
- The earlier incomplete radar-specific discovery records were removed from the active bibliography rather than cited with unverified metadata.
- The manuscript uses IEEE numeric citation commands and `IEEEtran.bst`.
- Citation contexts are deliberately bounded: the paper uses these sources for methodological context, not as evidence for FluxPhased's numerical results.

## Submission action

Before submission, replace any remaining arXiv-only record with its published version when an official proceedings or publisher record is available, and run this audit once more after that metadata change. No scientific claim depends on the removed unverified radar-specific records.
