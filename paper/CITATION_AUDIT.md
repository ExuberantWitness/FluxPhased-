# Citation Audit Report

**Paper:** *Scaling the Attack Breaks Defense Containment in Task-Level Radar--Jammer Self-Play*
**Active citations:** 34 (updated 2026-08-29, TAES revision)
**Target venue:** IEEE Transactions on Aerospace and Electronic Systems

## Verdict

**PASS (with one pre-submission action).** The bibliography was expanded 6 → 28 → 34 active entries on 2026-08-29, adding the TAES-native and textbook anchoring that the venue's reviewers expect (radar resource scheduling in this Transactions, the Moo & Ding RRM monograph, sensor-management survey, EW link-budget and array/radar handbooks). Every new entry was independently verified against publisher pages, Semantic Scholar records, or the arXiv API before being added; verification artifacts are `_verified_refs.json`, `_verified_arxiv.json`, and `_verified_radar_refs.json` at the repository root. No `\cite{...}` key is missing or uncited.

## Verification method by entry group

| Group | Entries | Verified against |
|---|---|---|
| Original 6 (PPO, MAPPO, NFSP, AlphaStar, ROMA, MOBA) | 6 | arXiv / Nature DOI (unchanged from previous audit) |
| Cognitive/adaptive radar | `haykin2006cognitive`, `bell2014far`, `bell2015jstsp`, `charlish2020development`, `martone2018spectrum`, `martone2021waveform`, `howard2022bandits` | IEEE Xplore doc/DOI records; S2 DOI lookup; publisher-page searches |
| TAES-native scheduling + textbooks | `akbar2025scheduling` (TAES 61(2):2434--2449), `moo2015book` (Elsevier), `neri2006book` (Artech House), `hero2011sensor` (IEEE Sensors J.), `vantrees2002` (Wiley), `skolnik2008` (McGraw-Hill) | IEEE Xplore / publisher book pages |
| Radar scheduling / anti-jamming DRL | `kosuru2022qlearn`, `jiang2023dsp`, `xie2023multijam`, `wang2024ietrsn`, `jia2025survey`, `han2017twodim` | S2 DOI lookup + publisher pages (Elsevier, Journal of Radars, Wiley/IET, IEEE) |
| MARL / self-play classics | `rashid2018qmix`, `lowe2017maddpg`, `foerster2018coma`, `lanctot2017psro`, `balduzzi2019mechanics`, `dewitt2020ippo`, `berner2019dota`, `foerster2018lola`, `hernandezleal2017survey` | arXiv API exact-ID lookups (title + full author lists cross-checked) |

## Known metadata gaps (deliberate)

- `bell2015jstsp` and `bell2014far`: DOI not printed in the verification sources; volume/pages verified via the published citation record. Add DOIs at camera-ready.
- `akbar2025scheduling`: volume/issue/pages verified via IEEE record; DOI not yet captured (IEEE doc 10726782). Add at camera-ready.
- `jia2025survey`: full author list beyond the first six recorded as `and others` (S2 record truncated); pages/volume verified via the ADS-style record `2025ICST...27.1798J`.
- `dewitt2020ippo`, `berner2019dota`: arXiv-only records (IPPO has a later IEEE TNNLS version; Dota 2 was never published outside arXiv). Replace IPPO with the published version at camera-ready if the citation remains.

## Checks

- Every `\cite{...}` key in `paper/main.tex` and `paper/sections/*.tex` has exactly one matching BibTeX entry (enforced by `_check_paper_integrity.py`, which also fails on any uncited bib entry).
- No unverified radar placeholder remains: the previous "identified but unverified" block was replaced by verified entries only.
- The manuscript uses IEEE numeric citation commands and `IEEEtran.bst`.

## Submission action

At camera-ready: fill the two Bell DOIs, complete the Jia author list, and swap `dewitt2020ippo` to its IEEE TNNLS version. No scientific claim depends on these metadata gaps.
