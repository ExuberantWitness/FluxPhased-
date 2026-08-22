import json, statistics

base = "experiments/array_face_s6/learning_repair/s6_selfplay_output_seed"
seeds = [20260729, 20260730, 20260731]
regime = {20260729: "snr=22", 20260730: "snr=12", 20260731: "snr=12"}

data = {}
for s in seeds:
    with open(f"{base}{s}/final_eval.json") as f:
        data[s] = json.load(f)

def row_stats(d, key):
    vals = [d[f"aseed_{a}"][key] for a in (4242, 777, 31337)]
    return statistics.mean(vals), statistics.stdev(vals)

print(f"{'seed':<10}{'regime':<8}{'h2h':<16}{'jam_vs_sweep':<18}{'rad_vs_idle':<16}{'floor':<10}{'neutral%':<10}")
for s in seeds:
    d = data[s]
    h2h_m, h2h_s = row_stats(d, "h2h_drop")
    jvs_m, jvs_s = row_stats(d, "jam_vs_sweep_drop")
    rvi = statistics.mean([d[f"aseed_{a}"]["rad_vs_idle_success"] for a in (4242, 777, 31337)])
    floor = d["sweep_vs_idle_floor"]["drop"]
    marg_sweep = jvs_m - floor
    marg_learned = h2h_m - (1 - rvi)
    neutral = (marg_sweep - marg_learned) / marg_sweep * 100 if marg_sweep > 0 else float("nan")
    print(f"{s:<10}{regime[s]:<8}{h2h_m:.4f}+-{h2h_s:.4f}   {jvs_m:.4f}+-{jvs_s:.4f}   {(1-rvi):.4f}        {floor:.4f}     {neutral:.1f}")

# merged statistics within snr=12
s12 = [s for s in seeds if regime[s] == "snr=12"]
print("\n=== merged snr=12 (2 seeds) ===")
for key, label in [("h2h_drop", "h2h"), ("jam_vs_sweep_drop", "jam_vs_sweep")]:
    per_seed = [row_stats(data[s], key)[0] for s in s12]
    print(f"{label}: per-seed means {['%.4f' % v for v in per_seed]}, across-seed mean {statistics.mean(per_seed):.4f} sd {statistics.stdev(per_seed):.4f}")

# cross-regime contrast
print("\n=== regime contrast ===")
for key, label in [("h2h_drop", "h2h"), ("jam_vs_sweep_drop", "jam_vs_sweep")]:
    v22 = row_stats(data[20260729], key)[0]
    v12 = [row_stats(data[s], key)[0] for s in s12]
    print(f"{label}: snr22 {v22:.4f} vs snr12 {statistics.mean(v12):.4f} (seeds {['%.4f' % x for x in v12]})")
