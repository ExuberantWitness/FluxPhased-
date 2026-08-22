"""Merge S7 final-eval stats across the 3 seeds + compare vs S6 snr=12 baseline."""
import json, statistics

base = "experiments/array_face_s7/learning_repair/s7_selfplay_output_seed"
seeds = [20260801, 20260802, 20260803]
data = {}
for s in seeds:
    with open(f"{base}{s}/final_eval.json") as f:
        data[s] = json.load(f)

def row_stats(d, key):
    vals = [d[f"aseed_{a}"][key] for a in (4242, 777, 31337)]
    return statistics.mean(vals), statistics.stdev(vals)

print(f"{'seed':<10}{'h2h':<16}{'jam_vs_sweep':<18}{'j1_only':<16}{'rad_vs_idle':<16}{'floor':<10}")
for s in seeds:
    d = data[s]
    h2h_m, h2h_s = row_stats(d, "h2h_drop")
    jvs_m, jvs_s = row_stats(d, "jam_vs_sweep_drop")
    j1_m, j1_s = row_stats(d, "j1_only_drop")
    rvi = statistics.mean([d[f"aseed_{a}"]["rad_vs_idle_success"] for a in (4242, 777, 31337)])
    floor = d["sweep_vs_idle_floor"]["drop"]
    print(f"{s:<10}{h2h_m:.4f}+-{h2h_s:.4f}   {jvs_m:.4f}+-{jvs_s:.4f}   {j1_m:.4f}+-{j1_s:.4f}   {(1-rvi):.4f}        {floor:.4f}")

print("\n=== merged across seeds ===")
for key, label in [("h2h_drop", "h2h"), ("jam_vs_sweep_drop", "jam_vs_sweep"),
                   ("j1_only_drop", "j1_only")]:
    per_seed = [row_stats(data[s], key)[0] for s in seeds]
    print(f"{label}: per-seed {['%.4f' % v for v in per_seed]}  mean {statistics.mean(per_seed):.4f}  sd {statistics.stdev(per_seed):.4f}")

# neutralization (floor-adjusted) per seed and merged
print("\n=== floor-adjusted neutralization (1 - (h2h - rad_idle)/(jvs - floor)) ===")
neut = []
for s in seeds:
    d = data[s]
    h2h = row_stats(d, "h2h_drop")[0]
    jvs = row_stats(d, "jam_vs_sweep_drop")[0]
    j1 = row_stats(d, "j1_only_drop")[0]
    rvi_drop = 1 - statistics.mean([d[f"aseed_{a}"]["rad_vs_idle_success"] for a in (4242, 777, 31337)])
    floor = d["sweep_vs_idle_floor"]["drop"]
    n = (1 - (h2h - rvi_drop) / (jvs - floor)) * 100 if jvs > floor else float("nan")
    neut.append(n)
    # cross-beam marginal: 2-jammer raw power vs 1-jammer raw power (same env)
    cb = (jvs - j1) / max(j1, 1e-9) * 100
    print(f"seed {s}: neutralization {n:.1f}%   cross-beam marginal (jvs-j1) {cb:.1f}% of j1")
print(f"merged neutralization: {statistics.mean(neut):.1f}% ± {statistics.stdev(neut):.1f}%")

print("\n=== vs S6 snr=12 baseline (h2h 0.0888, jvs 0.2751, neut 63.7%) ===")
h2h_m = statistics.mean([row_stats(data[s], "h2h_drop")[0] for s in seeds])
jvs_m = statistics.mean([row_stats(data[s], "jam_vs_sweep_drop")[0] for s in seeds])
j1_m = statistics.mean([row_stats(data[s], "j1_only_drop")[0] for s in seeds])
print(f"S7 h2h {h2h_m:.4f} vs S6 h2h 0.0888  (defense dominance survives? {h2h_m < 0.15})")
print(f"S7 jvs {jvs_m:.4f} vs S6 jvs 0.2751")
print(f"S7 j1_only {j1_m:.4f} — the S7-env 1v2 control (vs S6 h2h 0.0888)")
