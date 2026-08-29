"""Manuscript integrity gate.

1. Citation closure: every \\cite key exists in references.bib and every bib
   entry is cited (no dead entries).
2. Numeric consistency: every headline number in the LaTeX sources is
   re-derived from ``paper/figures/results_table.py`` and its formatted form
   must appear verbatim in the text. If the data changes, the expected strings
   change and this gate fails until the text is updated.
3. The on-disk RESULTS_TABLE.json must match a fresh computation.
4. Known-stale claim strings must stay absent.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(r'E:\DATA\vscode\FluxPhased')
PAPER = REPO / 'paper'
sys.path.insert(0, str(PAPER / 'figures'))
from results_table import TABLE, FIG_DIR  # noqa: E402

text = '\n'.join(p.read_text(errors='ignore')
                 for p in [PAPER / 'main.tex', *sorted((PAPER / 'sections').glob('*.tex'))])

failures = []

# ---------- 1. citation closure ----------
keys = set()
for m in re.finditer(r'\\cite\{([^}]*)\}', text):
    keys.update(k.strip() for k in m.group(1).split(',') if k.strip())
bib = (PAPER / 'references.bib').read_text(errors='ignore')
bib_keys = set(re.findall(r'^@\w+\{([^,]+),', bib, flags=re.M))
if keys - bib_keys:
    failures.append(f'MISSING bib keys: {sorted(keys - bib_keys)}')
if bib_keys - keys:
    failures.append(f'UNCITED bib entries: {sorted(bib_keys - keys)}')

# ---------- 2. numeric consistency ----------
def expect(desc: str, s: str):
    if s not in text:
        failures.append(f'NUMBER MISMATCH [{desc}]: expected {s!r} not found in tex')

s6, s7 = TABLE['s6']['agg'], TABLE['s7']['agg']
col, xf = TABLE['colocated'], TABLE['crossfire_seed01']
per7 = [v for v in TABLE['s7']['per_seed'].values()]

def pm(v, sd, nd=4, pct=False):
    if pct:
        return f'{v:.1f}\\%\\pm{sd:.1f}\\%'
    return f'{v:.4f}\\pm{sd:.4f}'

# Main S6/S7 comparison
expect('S6 h2h', pm(s6['h2h'], s6['h2h_sd']))
expect('S6 jvs', pm(s6['jvs'], s6['jvs_sd']))
expect('S6 eta', pm(s6['eta_pct'], s6['eta_pct_sd'], pct=True))
expect('S7 h2h', pm(s7['h2h'], s7['h2h_sd']))
expect('S7 jvs', pm(s7['jvs'], s7['jvs_sd']))
expect('S7 eta', pm(s7['eta_pct'], s7['eta_pct_sd'], pct=True))
expect('S7 rad-idle', pm(s7['rad_idle'], s7['rad_idle_sd']))
expect('S6 rad-idle', pm(s6['rad_idle'], s6['rad_idle_sd']))

# Per-seed values quoted in prose
expect('S7 per-seed h2h list',
       ', '.join(f'{v["h2h"]:.4f}' for v in per7[:-1]) + f', and {per7[-1]["h2h"]:.4f}')
expect('S7 per-seed j1 list',
       ', '.join(f'{v["j1_only"]:.4f}' for v in per7[:-1]) + f', and {per7[-1]["j1_only"]:.4f}')
expect('S7 j1 mean', pm(s7['j1_only'], s7['j1_only_sd']))

# Co-located control vs cross-fire reference
expect('colocated h2h', pm(col['h2h'], col['h2h_sd']))
expect('colocated jvs', pm(col['jvs'], col['jvs_sd']))
expect('colocated eta', f'{col["eta_pct"]:.1f}\\%')
expect('crossfire eta', f'{xf["eta_pct"]:.1f}\\%')

# Derived scaling claims
expect('jvs relative increase', f'{TABLE["derived"]["jvs_relative_increase_pct"]:.0f}\\%')
removed = TABLE['derived']['containment_removed_fraction']
assert 0.6 < removed < 0.7, f'containment-removed fraction {removed} no longer ~two-thirds'

# R5 dose response: every table row in full; prose quotes only these values
def nz(v, nd):
    """format fraction, dropping the leading zero as the R5 table does"""
    s = f'{v:.{nd}f}'
    return s[1:] if s.startswith('0.') else s

q_fmt = {0.0: '0', 0.25: '.25', 0.5: '.50', 0.75: '.75'}
for row in TABLE['r5']:
    expect(f'R5 q={row["q"]} table row',
           f'{q_fmt[row["q"]]} & {nz(row["h2h"], 4)} & {nz(row["jvs"], 4)} & '
           f'{nz(row["j1_only"], 4)} & {nz(row["j1_over_jvs"], 3)} & '
           f'{row["eta_pct"]:.1f}\\%')
by_q = {row['q']: row for row in TABLE['r5']}
for q in (0.0, 0.75):  # prose quotes the j1/jvs endpoints only
    expect(f'R5 q={q} j1/jvs prose', f'{by_q[q]["j1_over_jvs"]:.3f}')
for q in (0.0, 0.5, 0.75):  # prose quotes these eta and h2h/jvs values
    expect(f'R5 q={q} eta prose', f'{by_q[q]["eta_pct"]:.1f}\\%')
    expect(f'R5 q={q} h2h/jvs prose', f'{by_q[q]["h2h_over_jvs"]:.3f}')

# Continuation stability windows and last-40 stats
w = TABLE['continuation']['windows']
expect('continuation h2h windows',
       ', '.join(f'{x["h2h"]:.4f}' for x in w[:-1]) + f', and {w[-1]["h2h"]:.4f}')
expect('continuation jvs windows',
       ', '.join(f'{x["jvs"]:.4f}' for x in w[:-1]) + f', and {w[-1]["jvs"]:.4f}')
c = TABLE['continuation']
expect('last40 h2h', pm(c['last40_h2h'], c['last40_h2h_sd']))
expect('last40 jvs', pm(c['last40_jvs'], c['last40_jvs_sd']))

# Baselines: every baseline table row + quoted values
if 'baselines' in TABLE:
    b = TABLE['baselines']['agg']
    expect('baseline random radar', pm(b['random_radar_vs_jam']['mean'],
                                        b['random_radar_vs_jam']['sd']))
    expect('baseline greedy radar', pm(b['greedy_radar_vs_jam']['mean'],
                                       b['greedy_radar_vs_jam']['sd']))
    expect('baseline edf radar', pm(b['edf_radar_vs_jam']['mean'],
                                    b['edf_radar_vs_jam']['sd']))
    expect('baseline random jammer', pm(b['random_jam_vs_rad']['mean'],
                                        b['random_jam_vs_rad']['sd']))
    expect('baseline stare jammer', pm(b['stare_jam_vs_rad']['mean'],
                                       b['stare_jam_vs_rad']['sd']))
    # greedy drop quoted without pm in prose ("drop of exactly 0.0889")
    expect('baseline greedy exact prose', f"{b['greedy_radar_vs_jam']['mean']:.4f}")
    # factor claim: h2h / greedy rounded to one decimal
    expect('baseline factor prose',
           f"{s7['h2h'] / b['greedy_radar_vs_jam']['mean']:.1f}")
else:
    failures.append('baselines missing from RESULTS_TABLE: run _s7_baseline_eval.py')

# ---------- 3. on-disk RESULTS_TABLE.json freshness ----------
disk = json.loads((FIG_DIR / 'RESULTS_TABLE.json').read_text())
if disk != json.loads(json.dumps(TABLE)):
    failures.append('RESULTS_TABLE.json is stale: rerun paper/figures/results_table.py')

# ---------- 4. stale claim strings ----------
for token in ['63.7%\\pm0.7%', 'three-seed S6', 'pre-registered', 'remains unchanged']:
    if token in text:
        failures.append(f'STALE claim string present: {token}')

print(f'cited keys: {len(keys)}; bib entries: {len(bib_keys)}; numeric checks run')
if failures:
    print('\nFAILURES:')
    for f in failures:
        print(' -', f)
    sys.exit(1)
print('ALL INTEGRITY CHECKS PASS')
