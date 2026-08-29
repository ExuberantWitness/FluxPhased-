import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from paper_plot_style import COLORS, save, FIG_DIR
ROOT=FIG_DIR.parents[1]
def read(path):
 d=json.loads(Path(path).read_text()); a=[4242,777,31337]
 h=[d[f'aseed_{x}']['h2h_drop'] for x in a]; j=[d[f'aseed_{x}']['jam_vs_sweep_drop'] for x in a]
 ri=[1-d[f'aseed_{x}']['rad_vs_idle_success'] for x in a]; f=d['sweep_vs_idle_floor']['drop']
 hm,jm,rim=statistics.mean(h),statistics.mean(j),statistics.mean(ri)
 return hm,jm,1-(hm-rim)/(jm-f)
s6=ROOT/'experiments/array_face_s6/learning_repair'
s6files=[s6/f's6_selfplay_output_seed{s}/final_eval.json' for s in [20260730,20260731]]
s6v=[read(p) for p in s6files]; s6eta=statistics.mean([x[2] for x in s6v])
col=read(ROOT/'experiments/array_face_s7/learning_repair/s7_ablation_output_seed20260811/final_eval.json')
cross=read(ROOT/'experiments/array_face_s7/learning_repair/s7_continue2_output_seed20260801/final_eval.json')
# paired panels: containment and h2h, with direct measured values
fig,axs=plt.subplots(1,2,figsize=(6.9,2.45))
labels=['S6\n1 jammer','S7 co-located\n2 jammers','S7 cross-fire\n2 jammers']
vals=[s6eta,100*col[2],100*cross[2]]
colors=[COLORS['blue'],COLORS['yellow'],COLORS['red']]
b=axs[0].bar(labels,vals,color=colors,edgecolor=COLORS['black'],linewidth=.5,width=.62)
for x,v in zip(b,vals): axs[0].text(x.get_x()+x.get_width()/2,v+2,f'{v:.1f}%',ha='center',fontsize=7.5)
axs[0].set_ylabel('Neutralization (%)'); axs[0].set_ylim(0,75); axs[0].grid(axis='y',alpha=.25)
# h2h values, show action-seed SD for controls
hm=[statistics.mean([read(p)[0] for p in s6files]),col[0],cross[0]]
axs[1].bar(labels,hm,color=colors,edgecolor=COLORS['black'],linewidth=.5,width=.62)
for x,v in zip(axs[1].patches,hm): axs[1].text(x.get_x()+x.get_width()/2,v+.02,f'{v:.3f}',ha='center',fontsize=7.5)
axs[1].set_ylabel('Head-to-head mission drop'); axs[1].set_ylim(0,.42); axs[1].grid(axis='y',alpha=.25)
for ax in axs: ax.tick_params(axis='x',labelsize=7); ax.set_axisbelow(True)
fig.tight_layout()
save(fig,'fig4_mechanism_ablation')
