import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def parse(path):
    txt = open(path, 'rb').read()
    for enc in ('utf-16', 'utf-8'):
        try:
            txt = txt.decode(enc); break
        except UnicodeDecodeError:
            continue
    rows = {}
    pat = re.compile(r"iter\s+(\d+)\s+h2h_drop=([\d.]+)\s+jam_vs_sweep=([\d.]+)\s+rad_vs_idle_succ=([\d.]+)\s+j1_only=([\d.]+)")
    for m in pat.finditer(txt):
        rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), 1.0 - float(m.group(4)), float(m.group(5)))
    its = sorted(rows)
    return (np.array(its, dtype=float),
            np.array([rows[i][0] for i in its]), np.array([rows[i][1] for i in its]),
            np.array([rows[i][2] for i in its]), np.array([rows[i][3] for i in its]))

base = "experiments/array_face_s7/learning_repair"
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, seed in zip(axes, (20260801, 20260802, 20260803)):
    x, h2h, jvs, rvi, j1 = parse(f"{base}/s7_selfplay_output_seed{seed}/run.log")
    ax.plot(x, h2h, label="h2h (2 jam vs 2 rad)", lw=1.6)
    ax.plot(x, jvs, label="jam vs sweep (raw jammer power)", lw=1.6)
    ax.plot(x, j1, label="j1_only (1 jam vs learned radars)", lw=1.6)
    ax.plot(x, rvi, label="rad vs idle (radar competence floor)", lw=1.6)
    ax.set_title(f"seed {seed} (snr=12)", fontsize=10)
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mission drop ratio")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("S7 four-view arms-race curves — 2v2 MAPPO (snr=12)", fontsize=12)
fig.tight_layout()
out = "experiments/array_face_s7/arms_race_curves_s7.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
