import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from paper_plot_style import COLORS, save, FIG_DIR

ROOT=FIG_DIR.parents[1]
def summary(path):
 d=json.loads(Path(path).read_text()); a=[4242,777,31337]
 h=[d[f'aseed_{x}']['h2h_drop'] for x in a]
 j=[d[f'aseed_{x}']['jam_vs_sweep_drop'] for x in a]
 ri=[1-d[f'aseed_{x}']['rad_vs_idle_success'] for x in a]
 f=d['sweep_vs_idle_floor']['drop']
 return statistics.mean(h), statistics.stdev(h), statistics.mean(j), statistics.stdev(j), statistics.mean(ri), f
s6base=ROOT/'experiments/array_face_s6/learning_repair'
s6=[summary(s6base/f's6_selfplay_output_seed{s}/final_eval.json') for s in [20260730,20260731]]
s6h=statistics.mean([x[0] for x in s6]); s6j=statistics.mean([x[2] for x in s6]); s6ri=statistics.mean([x[4] for x in s6]); s6f=statistics.mean([x[5] for x in s6])
# S6 uses only the two valid 12-dB training seeds; the 22-dB seed is excluded.
s6eta_values = [100 * (1 - (x[0]-x[4])/(x[2]-x[5])) for x in s6]
s6_eta = statistics.mean(s6eta_values)
s6_eta_sd = statistics.stdev(s6eta_values)
s7base=ROOT/'experiments/array_face_s7/learning_repair'
s7paths=[s7base/'s7_continue2_output_seed20260801/final_eval.json',s7base/'s7_seed02_cont_output_seed20260802/final_eval.json',s7base/'s7_seed03_cont_output_seed20260803/final_eval.json']
s7=[]
for p in s7paths:
 x=summary(p); eta=100*(1-(x[0]-x[4])/(x[2]-x[5])); s7.append((x,eta))
s7eta=[x[1] for x in s7]
fig,ax=plt.subplots(figsize=(3.35,2.45))
labels=['S6\n1 jammer','S7\n2 jammers']; means=[s6_eta,statistics.mean(s7eta)]; errs=[s6_eta_sd,statistics.stdev(s7eta)]
bars=ax.bar(labels,means,yerr=errs,capsize=3,width=.58,color=[COLORS['blue'],COLORS['red']],edgecolor=COLORS['black'],linewidth=.5)
for b,v in zip(bars,means): ax.text(b.get_x()+b.get_width()/2,v+3,f'{v:.1f}%',ha='center',fontsize=8)
ax.scatter([1-.18,1,1+.18],s7eta,color=COLORS['black'],s=15,zorder=3,label='S7 seeds')
ax.set_ylabel('Floor-adjusted neutralization (%)'); ax.set_ylim(0,80); ax.legend(frameon=False,fontsize=6.5); ax.grid(axis='y',alpha=.25); ax.set_axisbelow(True)
save(fig,'fig3_neutralization')
