"""Full S7 training-curve figure: seed 01 concatenated 9..1999 (1000-iter run +
2000-iter continuation) with the four views, S6 reference levels, and the
anneal-freeze boundary; second panel: 3-seed h2h/jvs comparison."""
import re, os
import matplotlib
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

# ---- panel 1: seed 01 full curve 9..1999 (original + continuation) ----
x0, h2h0, jvs0, rvi0, j10 = parse(f"{base}/s7_selfplay_output_seed20260801/run.log")
x1, h2h1, jvs1, rvi1, j11 = parse(f"{base}/s7_continue_output_seed20260801/run.log")
x = np.concatenate([x0, x1]); h2h = np.concatenate([h2h0, h2h1]); jvs = np.concatenate([jvs0, jvs1])
rvi = np.concatenate([rvi0, rvi1]); j1 = np.concatenate([j10, j11])

fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
ax = axes[0]
ax.plot(x, h2h, label="h2h (2 learned jammers vs 2 learned radars)", lw=1.6, color="#d62728")
ax.plot(x, jvs, label="jam_vs_sweep (raw pair firepower)", lw=1.6, color="#1f77b4")
ax.plot(x, j1, label="j1_only (1 jammer vs learned radars)", lw=1.4, color="#9467bd")
ax.plot(x, rvi, label="rad_vs_idle drop (radar competence)", lw=1.4, color="#2ca02c")
ax.axvline(1000, color="grey", ls="--", lw=1)
ax.text(1010, 0.50, "continuation\n(anneal frozen)", fontsize=8, color="grey")
# S6 snr=12 reference levels
ax.axhline(0.0888, color="red", ls=":", lw=1); ax.text(5, 0.10, "S6 h2h 0.089", fontsize=8, color="red")
ax.axhline(0.2751, color="blue", ls=":", lw=1); ax.text(5, 0.29, "S6 jvs 0.275", fontsize=8, color="blue")
# converged plateau band
ax.axhspan(0.343-0.015, 0.343+0.015, color="grey", alpha=0.15)
ax.text(1600, 0.373, "converged plateau 0.343±0.015", fontsize=8, color="grey")
ax.set_title("S7 seed 20260801 — full 2000-iter trajectory (4 views)", fontsize=10)
ax.set_xlabel("iteration"); ax.set_ylabel("mission drop ratio")
ax.set_ylim(-0.02, 0.56); ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=7.5)

# ---- panel 2: 3-seed comparison ----
ax = axes[1]
for seed, d, c, ls in [(20260801, "s7_selfplay_output_seed20260801", "#d62728", "-"),
                       (20260802, "s7_selfplay_output_seed20260802", "#1f77b4", "-"),
                       (20260803, "s7_selfplay_output_seed20260803", "#2ca02c", "-")]:
    p = f"{base}/{d}/run.log"
    if not os.path.exists(p):
        continue
    xr, h2hr, jvsr, rvir, j1r = parse(p)
    ax.plot(xr, h2hr, color=c, ls=ls, lw=1.4, label=f"seed {seed} h2h")
    ax.plot(xr, jvsr, color=c, ls=":", lw=1.2, alpha=0.75, label=f"seed {seed} jvs")
ax.axvline(1000, color="grey", ls="--", lw=1)
ax.axhline(0.0888, color="red", ls=":", lw=1)
ax.text(5, 0.10, "S6 h2h", fontsize=8, color="red")
ax.set_title("Three seeds — h2h (solid) / jam_vs_sweep (dotted), 1000-iter protocol", fontsize=10)
ax.set_xlabel("iteration"); ax.set_ylabel("mission drop ratio")
ax.set_ylim(-0.02, 0.56); ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=7.5)

fig.suptitle("S7 — performance vs training iteration (2 jammers vs 2 radars, snr=12)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = "experiments/array_face_s7/arms_race_curves_s7_full.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
