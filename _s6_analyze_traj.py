import re, statistics
def parse(path):
    txt = open(path, 'rb').read()
    for enc in ('utf-16', 'utf-8'):
        try:
            txt = txt.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    rows = {}
    pat = re.compile(r"iter\s+(\d+)\s+h2h_drop=([\d.]+)\s+jam_vs_sweep=([\d.]+)\s+rad_vs_idle_succ=([\d.]+)\s+j_ent=([\d.]+)\s+r_ent=([\d.]+)")
    for m in pat.finditer(txt):
        it = int(m.group(1))
        rows[it] = dict(h2h=float(m.group(2)), jvs=float(m.group(3)), rvs=float(m.group(4)),
                        jent=float(m.group(5)), rent=float(m.group(6)))
    return rows

for seed in (20260730, 20260731):
    rows = parse(f"experiments/array_face_s6/learning_repair/s6_selfplay_output_seed{seed}/run.log")
    its = sorted(rows)
    print(f"=== seed {seed}: {len(its)} validation points, iters {its[0]}..{its[-1]}")
    # quarters (0-999 in 200-iter blocks)
    for q in range(5):
        lo, hi = q*200, q*200+199
        sel = [rows[i] for i in its if lo <= i <= hi]
        if not sel:
            continue
        def m(k): return statistics.mean(r[k] for r in sel)
        print(f"  q{q} [{lo}-{hi}]: h2h={m('h2h'):.4f} jvs={m('jvs'):.4f} rvs={m('rvs'):.4f}  gap(jvs-h2h)={m('jvs')-m('h2h'):.4f}")
    last40 = [rows[i] for i in its[-40:]]
    def m(k): return statistics.mean(r[k] for r in last40)
    def s(k): return statistics.stdev(r[k] for r in last40)
    print(f"  last-40: h2h={m('h2h'):.4f}±{s('h2h'):.4f} jvs={m('jvs'):.4f}±{s('jvs'):.4f} rvs={m('rvs'):.4f}±{s('rvs'):.4f}")
    print(f"  final: h2h={rows[its[-1]]['h2h']:.4f} jvs={rows[its[-1]]['jvs']:.4f} rvs={rows[its[-1]]['rvs']:.4f}  j_ent={rows[its[-1]]['jent']:.2f} r_ent={rows[its[-1]]['rent']:.2f}")
