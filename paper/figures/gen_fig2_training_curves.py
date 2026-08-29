import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from paper_plot_style import COLORS, save, FIG_DIR

ROOT = FIG_DIR.parents[1]
base = ROOT / 'experiments/array_face_s7/learning_repair'

def parse(path):
    b = Path(path).read_bytes()
    try: txt = b.decode('utf-16')
    except UnicodeDecodeError: txt = b.decode('utf-8', 'replace')
    rows = {}
    pat = re.compile(r'iter\s+(\d+)\s+h2h_drop=([\d.]+)\s+jam_vs_sweep=([\d.]+)\s+rad_vs_idle_succ=([\d.]+)\s+j1_only=([\d.]+)')
    for m in pat.finditer(txt): rows[int(m[1])] = tuple(float(m[i]) for i in range(2,6))
    return rows

r0 = parse(base/'s7_selfplay_output_seed20260801/run.log')
r1 = parse(base/'s7_continue_output_seed20260801/run.log')
r2 = parse(base/'s7_continue2_output_seed20260801/run.log')
segments = [(r0, 'normal anneal'), (r1, 'continuation'), (r2, 'stage 2')]
fig, ax = plt.subplots(figsize=(3.45,2.45))
for key,color,label in [(0,COLORS['red'],'h2h'),(1,COLORS['blue'],'jam vs sweep'),(3,COLORS['purple'],'j1-only'),(2,COLORS['green'],'radar floor')]:
    for rows,_ in segments:
        xs=sorted(rows); ax.plot(xs,[rows[i][key] if key != 2 else 1-rows[i][key] for i in xs],color=color,lw=1.2,alpha=.85)
    ax.plot([],[],color=color,lw=1.4,label=label)
for x in [1000,2000]: ax.axvline(x,color=COLORS['grey'],ls='--',lw=.8)
ax.axhspan(.343-.015,.343+.015,color=COLORS['grey'],alpha=.16)
ax.axhline(.0888,color=COLORS['red'],ls=':',lw=.7)
ax.set_xlabel('Training iteration'); ax.set_ylabel('Mission drop ratio')
ax.set_xlim(0,3000); ax.set_ylim(-.02,.58); ax.legend(frameon=False,fontsize=6.4,loc='upper left',ncol=2)
ax.text(1015,.545,'anneal frozen',fontsize=6.3,color=COLORS['grey'])
ax.text(2015,.545,'anneal frozen',fontsize=6.3,color=COLORS['grey'])
ax.grid(alpha=.22)
save(fig,'fig2_training_curves')
