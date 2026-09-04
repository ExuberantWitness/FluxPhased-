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
import sys

FIG_DIR = Path(__file__).resolve().parent
ROOT = FIG_DIR.parents[1]
if str(FIG_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_DIR))
from final_eval_schema import load_final_eval, FinalEvalSchemaError
S6_BASE = ROOT / 'experiments/array_face_s6/learning_repair'
S7_BASE = ROOT / 'experiments/array_face_s7/learning_repair'
ACTION_SEEDS = [4242, 777, 31337]

S6_SEEDS = [20260730, 20260731, 20260732]  # valid 12-dB regime; 20260729 is 22-dB and excluded
S7_SEEDS = ['s7_continue2_output_seed20260801',
            's7_seed02_cont_output_seed20260802',
            's7_seed03_cont_output_seed20260803']

# Canonical n-scale comparison: all policies terminate at the same 2000-step
# endpoint, use three training seeds, and use the same resolved geometry.
N2_STRICT = {
    20260801: S7_BASE / 's7_continue_output_seed20260801/final_eval_v2.json',
    20260802: S7_BASE / 's7_strict_n2_output_seed20260802/final_eval.json',
    20260803: S7_BASE / 's7_strict_n2_output_seed20260803/final_eval.json',
}
N3_STRICT = {sd: S7_BASE / f's9_n3_output_seed{sd}/final_eval_v2.json'
             for sd in (20261011, 20261012, 20261013)}
N4_STRICT = {
    20261021: S7_BASE / 's9_n4_output_seed20261021/final_eval_v2.json',
    20261022: S7_BASE / 's9_strict_n4_output_seed20261022/final_eval_v2.json',
    20261023: S7_BASE / 's9_strict_n4_output_seed20261023/final_eval_v2.json',
}


def _expected(*, n_jammers, jammer_az_deg, radar_az_deg,
              baseline_snr_db=12.0, checkpoint_iteration=1999,
              algorithm='mappo') -> dict:
    return {
        'algorithm': algorithm, 'n_jammers': n_jammers, 'n_radars': 2,
        'jammer_az_deg': list(jammer_az_deg), 'radar_az_deg': list(radar_az_deg),
        'baseline_snr_db': baseline_snr_db, 'P_jam_W': 0.1,
        'active_budget_steps': 63, 'horizon': 64, 'validation_seed_count': 64,
        'action_seeds': ACTION_SEEDS, 'n_action_reps': 1,
        'checkpoint_iteration': checkpoint_iteration,
    }


def eval_view(path, *, expected=None, terminal=False) -> dict:
    """Per-file summary from schema-v2 canonical evidence only."""
    d = load_final_eval(path, expected=expected, require_terminal=terminal)
    return _eval_view_data(d)


def _eval_view_data(d: dict) -> dict:
    """Compute the summary after schema validation has already happened."""
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


def legacy_eval_view(path) -> dict:
    """Read a legacy result only for explicitly non-canonical context data.

    Main headline and n-scale results must use ``eval_view``. This helper is
    retained for R5 and baseline files until those evaluators are also upgraded
    to schema v2; it is intentionally not used by the strict n-scale allowlist.
    """
    d = json.loads(Path(path).read_text())
    return _eval_view_data(d)


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
    # Historical sections retain their explicitly labelled legacy inputs until
    # all evaluator outputs have been regenerated as schema-v2 artifacts.
    s6_views = {s: legacy_eval_view(S6_BASE / f's6_selfplay_output_seed{s}/final_eval.json')
                for s in S6_SEEDS}
    s6_3 = S6_SEEDS
    s6_views_3 = dict(s6_views)
    s7_views = {name: legacy_eval_view(S7_BASE / name / 'final_eval.json')
                for name in S7_SEEDS}
    colocated = legacy_eval_view(S7_BASE / 's7_ablation_output_seed20260811/final_eval.json')
    r5_conditions = [
        ('s7_continue_output_seed20260801', 0.0),
        ('s7_r5_mix0p25_output_seed20260821', 0.25),
        ('s7_r5_mix0p5_output_seed20260822', 0.50),
        ('s7_r5_mix0p75_output_seed20260823', 0.75),
    ]
    r5 = []
    for name, q in r5_conditions:
        v = legacy_eval_view(S7_BASE / name / 'final_eval.json')
        r5.append({'q': q, 'h2h': v['h2h'], 'jvs': v['jvs'], 'j1_only': v['j1_only'],
                   'j1_over_jvs': v['j1_only'] / v['jvs'], 'eta_pct': v['eta_pct'],
                   'h2h_over_jvs': v['h2h'] / v['jvs']})

    s6_agg = _agg(list(s6_views_3.values()))
    s7_agg = _agg(list(s7_views.values()))
    return {
        's6': {'seeds': s6_3, 'per_seed': s6_views_3, 'agg': s6_agg},
        's6_two_seed': {'seeds': S6_SEEDS[:2], 'per_seed': {k: s6_views[k] for k in S6_SEEDS[:2]},
                        'agg': _agg([s6_views[k] for k in S6_SEEDS[:2]])},
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


def _strict_group(mapping, *, n_jammers, jammer_az_deg, checkpoint_iteration=1999):
    """Load an allowlisted, matched n-scale group; fail closed on any gap."""
    per = {}
    expected = _expected(
        n_jammers=n_jammers, jammer_az_deg=jammer_az_deg,
        radar_az_deg=(20.0, -20.0), checkpoint_iteration=checkpoint_iteration,
    )
    for seed, path in mapping.items():
        per[seed] = eval_view(path, expected=expected, terminal=True)
    return {'per_seed': per, 'agg': _agg(list(per.values()))}


def _strict_nscale() -> dict:
    """Load the matched 2000-iteration n=2/3/4 curve, or return empty until
    every allowlisted canonical artifact has been produced."""
    mappings = (
        ('n2', N2_STRICT, 2, (60.0, -60.0)),
        ('n3', N3_STRICT, 3, (60.0, 0.0, -60.0)),
        ('n4', N4_STRICT, 4, (60.0, 20.0, -20.0, -60.0)),
    )
    if any(not path.exists() for _, mapping, _, _ in mappings for path in mapping.values()):
        return {}
    out = {}
    for name, mapping, n, jaz in mappings:
        out[name] = _strict_group(mapping, n_jammers=n, jammer_az_deg=jaz)
    return out


def _snr_retrain() -> dict:
    """Retrained SNR regimes (9/15 dB, 2000-iter protocol)."""
    out = {}
    for snr, sd in ((9, 20260911), (15, 20260912)):
        p = S7_BASE / f's7_snr{snr}db_output_seed{sd}' / 'final_eval.json'
        if p.exists():
            v = eval_view(p)
            out[f'snr_{snr}db'] = v
    return out


def _greedy_counter() -> dict:
    """Final greedy_vs_jam_drop of the counter-adaptation run (last val row)."""
    p = S7_BASE / 's7_greedycounter_output_seed20260921' / 'val_metrics.jsonl'
    if not p.exists():
        return {}
    import re
    rows = []
    for m in re.finditer(r'\{[^{}]*\}', p.read_text()):
        try:
            r = json.loads(m.group(0))
            if 'greedy_vs_jam_drop' in r:
                rows.append(r)
        except Exception:
            pass
    if not rows:
        return {}
    last = rows[-1]
    return {'final_iter': last['iter'], 'final_greedy_vs_jam': last['greedy_vs_jam_drop'],
            'baseline_selfplay': 0.0889}


def _colocated_seeds() -> dict:
    """Co-located mechanism control across its three training seeds."""
    per = {}
    for name, sd in (('s7_ablation_output_seed20260811', 20260811),
                     ('s7_ablation_output_seed20260812', 20260812),
                     ('s7_ablation_output_seed20260813', 20260813)):
        p = S7_BASE / name / 'final_eval.json'
        if not p.exists():
            return {}
        per[sd] = eval_view(p)
    return {'per_seed': per, 'agg': _agg(list(per.values()))}


def _s6_three_seed() -> dict:
    """S6 baseline with the recovered third valid 12-dB seed."""
    seeds = [20260730, 20260731, 20260732]
    per = {}
    for sd in seeds:
        p = S6_BASE / f's6_selfplay_output_seed{sd}' / 'final_eval.json'
        if not p.exists():
            return {}
        per[sd] = eval_view(p)
    return {'per_seed': per, 'agg': _agg(list(per.values()))}


TABLE = build_table()
BASELINES = _baselines()
if BASELINES:
    TABLE['baselines'] = BASELINES
SNR_REEVAL = _snr_reeval()
if SNR_REEVAL:
    TABLE['snr_reeval'] = SNR_REEVAL
for name, fn in (('nscale', _strict_nscale), ('snr_retrain', _snr_retrain),
                 ('greedy_counter', _greedy_counter),
                 ('colocated_seeds', _colocated_seeds),
                 ('s6_three_seed', _s6_three_seed)):
    v = fn()
    if v:
        TABLE[name] = v

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
