"""Single authoritative results table for the FluxPhased manuscript.

Every headline number in the paper text, tables, and figures is derived here
from the raw ``final_eval.json`` / ``val_metrics.jsonl`` files exactly once.
Figures import this module; the integrity checker re-derives the formatted
strings used in the LaTeX sources from this module. Manual transcription of
numbers between data files, figures, and text is what produced the original
Figure 4 unit bug (a fraction formatted as a percent), so no script may
re-implement the aggregation locally.

Conventions:
- drop ratios (h2h, jvs, j1, rad_idle, floor) are fractions in [0, 1];
- neutralization eta is stored as PERCENT (``*_eta_pct``) everywhere;
- across-training-seed aggregates are mean +/- sample stdev (statistics.stdev).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent
ROOT = FIG_DIR.parents[1]
S6_BASE = ROOT / 'experiments/array_face_s6/learning_repair'
S7_BASE = ROOT / 'experiments/array_face_s7/learning_repair'
ACTION_SEEDS = [4242, 777, 31337]

S6_SEEDS = [20260730, 20260731]  # valid 12-dB regime; 20260729 is 22-dB and excluded
S7_SEEDS = ['s7_continue2_output_seed20260801',
            's7_seed02_cont_output_seed20260802',
            's7_seed03_cont_output_seed20260803']


def eval_view(path) -> dict:
    """Per-file evaluation summary: mean over action seeds, plus floor."""
    d = json.loads(Path(path).read_text())
    h = [d[f'aseed_{a}']['h2h_drop'] for a in ACTION_SEEDS]
    j = [d[f'aseed_{a}']['jam_vs_sweep_drop'] for a in ACTION_SEEDS]
    ri = [1 - d[f'aseed_{a}']['rad_vs_idle_success'] for a in ACTION_SEEDS]
    j1 = [d[f'aseed_{a}'].get('j1_only_drop') for a in ACTION_SEEDS]
    out = {
        'h2h': statistics.mean(h), 'h2h_sd': statistics.stdev(h),
        'jvs': statistics.mean(j), 'jvs_sd': statistics.stdev(j),
        'rad_idle': statistics.mean(ri), 'rad_idle_sd': statistics.stdev(ri),
        'floor': d['sweep_vs_idle_floor']['drop'],
        'eta_pct': 100.0 * (1 - (statistics.mean(h) - statistics.mean(ri))
                            / (statistics.mean(j) - d['sweep_vs_idle_floor']['drop'])),
    }
    if all(v is not None for v in j1):
        out['j1_only'] = statistics.mean(j1)
        out['j1_only_sd'] = statistics.stdev(j1)
    return out


def _agg(views: list[dict]) -> dict:
    """Across-training-seed aggregate of per-seed evaluation summaries."""
    keys = ['h2h', 'jvs', 'rad_idle', 'eta_pct']
    keys += ['j1_only'] if all('j1_only' in v for v in views) else []
    out = {'n_seeds': len(views)}
    for k in keys:
        xs = [v[k] for v in views]
        out[k] = statistics.mean(xs)
        out[f'{k}_sd'] = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return out


def _continuation(dir_path) -> dict:
    """2000->3000 continuation windows and the last-40 validation points."""
    rows = [json.loads(line) for line in
            (Path(dir_path) / 'val_metrics.jsonl').read_text().splitlines() if line.strip()]
    # deduplicate resumed iterations, keeping the latest occurrence
    by_iter = {r['iter']: r for r in rows}
    rows = [by_iter[i] for i in sorted(by_iter)]
    windows = []
    for lo in range(2000, 3000, 200):
        w = [r for r in rows if lo < r['iter'] <= lo + 200]
        windows.append({
            'range': [lo, lo + 200],
            'h2h': statistics.mean(r['h2h_drop'] for r in w),
            'jvs': statistics.mean(r['jam_vs_sweep_drop'] for r in w),
        })
    last40 = rows[-40:]
    return {
        'windows': windows,
        'last40_h2h': statistics.mean(r['h2h_drop'] for r in last40),
        'last40_h2h_sd': statistics.stdev(r['h2h_drop'] for r in last40),
        'last40_jvs': statistics.mean(r['jam_vs_sweep_drop'] for r in last40),
        'last40_jvs_sd': statistics.stdev(r['jam_vs_sweep_drop'] for r in last40),
    }


def build_table() -> dict:
    s6_views = {s: eval_view(S6_BASE / f's6_selfplay_output_seed{s}/final_eval.json')
                for s in S6_SEEDS}
    s7_views = {name: eval_view(S7_BASE / name / 'final_eval.json') for name in S7_SEEDS}
    colocated = eval_view(S7_BASE / 's7_ablation_output_seed20260811/final_eval.json')
    r5_conditions = [
        ('s7_continue_output_seed20260801', 0.0),
        ('s7_r5_mix0p25_output_seed20260821', 0.25),
        ('s7_r5_mix0p5_output_seed20260822', 0.50),
        ('s7_r5_mix0p75_output_seed20260823', 0.75),
    ]
    r5 = []
    for name, q in r5_conditions:
        v = eval_view(S7_BASE / name / 'final_eval.json')
        r5.append({'q': q, 'h2h': v['h2h'], 'jvs': v['jvs'], 'j1_only': v['j1_only'],
                   'j1_over_jvs': v['j1_only'] / v['jvs'], 'eta_pct': v['eta_pct'],
                   'h2h_over_jvs': v['h2h'] / v['jvs']})

    s6_agg = _agg(list(s6_views.values()))
    s7_agg = _agg(list(s7_views.values()))
    return {
        's6': {'seeds': S6_SEEDS, 'per_seed': s6_views, 'agg': s6_agg},
        's7': {'runs': S7_SEEDS, 'per_seed': s7_views, 'agg': s7_agg},
        'colocated': colocated,
        'crossfire_seed01': s7_views[S7_SEEDS[0]],
        'r5': r5,
        'continuation': _continuation(S7_BASE / 's7_continue2_output_seed20260801'),
        'derived': {
            'jvs_relative_increase_pct':
                100.0 * (s7_agg['jvs'] - s6_agg['jvs']) / s6_agg['jvs'],
            'containment_removed_fraction':
                (s6_agg['eta_pct'] - s7_agg['eta_pct']) / s6_agg['eta_pct'],
        },
    }


def _baselines() -> dict:
    """Evaluation-only scripted baselines against the converged S7 teams.

    Aggregated per training seed over the three action seeds; the reported
    mean +/- sd is across the three training seeds. Returns {} until all
    three baseline_eval.json files exist.
    """
    per_seed = {}
    for name in S7_SEEDS:
        path = S7_BASE / name / 'baseline_eval.json'
        if not path.exists():
            return {}
        d = json.loads(path.read_text())
        views = {}
        for mode in ('random_radar_vs_jam', 'greedy_radar_vs_jam',
                     'edf_radar_vs_jam',
                     'random_jam_vs_rad', 'stare_jam_vs_rad'):
            xs = [d[f'aseed_{a}'][mode] for a in ACTION_SEEDS]
            views[mode] = statistics.mean(xs)
        per_seed[name] = views
    agg = {}
    for mode in per_seed[S7_SEEDS[0]]:
        xs = [v[mode] for v in per_seed.values()]
        agg[mode] = {'mean': statistics.mean(xs), 'sd': statistics.stdev(xs)}
    return {'per_seed': per_seed, 'agg': agg}


def _snr_reeval() -> dict:
    """Off-regime re-evaluation of converged teams at shifted SNR0.

    Policies remain trained at 12 dB; these are robustness readouts, not
    retrained sensitivity. Returns {} until both snr_reeval.json files exist.
    """
    out = {}
    for name, label in ((S7_SEEDS[0], 'crossfire'),
                        ('s7_ablation_output_seed20260811', 'colocated')):
        path = S7_BASE / name / 'snr_reeval.json'
        if not path.exists():
            return {}
        out[label] = json.loads(path.read_text())
    return out


TABLE = build_table()
BASELINES = _baselines()
if BASELINES:
    TABLE['baselines'] = BASELINES
SNR_REEVAL = _snr_reeval()
if SNR_REEVAL:
    TABLE['snr_reeval'] = SNR_REEVAL

if __name__ == '__main__':
    target = FIG_DIR / 'RESULTS_TABLE.json'
    target.write_text(json.dumps(TABLE, indent=1))
    print(f'wrote {target}')
    print('S6 eta %:', round(TABLE['s6']['agg']['eta_pct'], 2),
          '+/-', round(TABLE['s6']['agg']['eta_pct_sd'], 2))
    print('S7 eta %:', round(TABLE['s7']['agg']['eta_pct'], 2),
          '+/-', round(TABLE['s7']['agg']['eta_pct_sd'], 2))
    print('colocated eta %:', round(TABLE['colocated']['eta_pct'], 2))
    print('crossfire eta %:', round(TABLE['crossfire_seed01']['eta_pct'], 2))
    print('jvs relative increase %:', round(TABLE['derived']['jvs_relative_increase_pct'], 1))
    print('containment removed fraction:',
          round(TABLE['derived']['containment_removed_fraction'], 3))
    if 'baselines' in TABLE:
        for k, v in TABLE['baselines']['agg'].items():
            print(f'baseline {k}: {v["mean"]:.4f} +/- {v["sd"]:.4f}')
    if 'snr_reeval' in TABLE:
        for g, pts in TABLE['snr_reeval'].items():
            for k, v in pts.items():
                print(f'snr_reeval {g} {k}: eta {v["eta_pct"]:.1f}%')
