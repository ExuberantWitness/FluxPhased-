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
    pat = re.compile(r"iter\s+(\d+)\s+h2h_drop=([\d.]+)\s+jam_vs_sweep=([\d.]+)\s+rad_vs_idle_succ=([\d.]+)")
    for m in pat.finditer(txt):
        rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), 1.0 - float(m.group(4)))
    its = sorted(rows)
    x = np.array(its, dtype=float)
    h2h = np.array([rows[i][0] for i in its])
    jvs = np.array([rows[i][1] for i in its])
    rvi = np.array([rows[i][2] for i in its])
    return x, h2h, jvs, rvi

base = "experiments/array_face_s6/learning_repair"
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, seed in zip(axes, (20260730, 20260731)):
    x, h2h, jvs, rvi = parse(f"{base}/s6_selfplay_output_seed{seed}/run.log")
    ax.plot(x, h2h, label="h2h (learned jammer vs learned radars)", lw=1.6)
    ax.plot(x, jvs, label="jam vs sweep (raw jammer power)", lw=1.6)
    ax.plot(x, rvi, label="rad vs idle (radar competence floor)", lw=1.6)
    ax.set_title(f"seed {seed} (snr=12)", fontsize=10)
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mission drop ratio")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("S6 three-view arms-race curves — snr=12 replication seeds", fontsize=12)
fig.tight_layout()
out = "experiments/array_face_s6/arms_race_curves_seeds12.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
