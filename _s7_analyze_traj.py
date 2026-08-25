"""Parse S7 validation trajectories (run.log) into convergence statistics."""
import re, statistics, os

def parse(path):
    txt = open(path, 'rb').read()
    for enc in ('utf-16', 'utf-8'):
        try:
            txt = txt.decode(enc); break
        except UnicodeDecodeError:
            continue
    rows = {}
    pat = re.compile(
        r"iter\s+(\d+)\s+h2h_drop=([\d.]+)\s+jam_vs_sweep=([\d.]+)\s+"
        r"rad_vs_idle_succ=([\d.]+)\s+j1_only=([\d.]+)\s+j_ent=([\d.]+)\s+r_ent=([\d.]+)")
    for m in pat.finditer(txt):
        it = int(m.group(1))
        rows[it] = dict(h2h=float(m.group(2)), jvs=float(m.group(3)), rvs=float(m.group(4)),
                        j1=float(m.group(5)), jent=float(m.group(6)), rent=float(m.group(7)))
    return rows

SEED_DIRS = {
    20260801: "s7_selfplay_output_seed20260801",
    20260802: "s7_selfplay_output_seed20260802",
    20260803: "s7_selfplay_output_seed20260803",
    "20260801_cont": "s7_continue_output_seed20260801",
    "20260801_cont2": "s7_continue2_output_seed20260801",
}
for seed, dirname in SEED_DIRS.items():
    log = f"experiments/array_face_s7/learning_repair/{dirname}/run.log"
    if not os.path.exists(log):
        print(f"=== seed {seed}: not started yet")
        continue
    rows = parse(log)
    if not rows:
        print(f"=== seed {seed}: no data")
        continue
    its = sorted(rows)
    print(f"=== seed {seed}: {len(its)} validation points, iters {its[0]}..{its[-1]}")
    for q in range(10):
        lo, hi = q*200, q*200+199
        sel = [rows[i] for i in its if lo <= i <= hi]
        if not sel:
            continue
        def m(k): return statistics.mean(r[k] for r in sel)
        print(f"  q{q} [{lo}-{hi}]: h2h={m('h2h'):.4f} jvs={m('jvs'):.4f} j1={m('j1'):.4f} "
              f"rvs={m('rvs'):.4f}  gap(jvs-h2h)={m('jvs')-m('h2h'):.4f}  j1_marg={m('jvs')-m('j1'):.4f}")
    last40 = [rows[i] for i in its[-40:]]
    def m(k): return statistics.mean(r[k] for r in last40)
    def s(k): return statistics.stdev(r[k] for r in last40)
    print(f"  last-40: h2h={m('h2h'):.4f}±{s('h2h'):.4f} jvs={m('jvs'):.4f}±{s('jvs'):.4f} "
          f"j1={m('j1'):.4f}±{s('j1'):.4f} rvs={m('rvs'):.4f}±{s('rvs'):.4f}")
    print(f"  final: h2h={rows[its[-1]]['h2h']:.4f} jvs={rows[its[-1]]['jvs']:.4f} "
          f"j1={rows[its[-1]]['j1']:.4f} rvs={rows[its[-1]]['rvs']:.4f} "
          f"j_ent={rows[its[-1]]['jent']:.2f} r_ent={rows[its[-1]]['rent']:.2f}")
