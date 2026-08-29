import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from paper_plot_style import COLORS, save, FIG_DIR
ROOT=FIG_DIR.parents[1]
def read(path):
 d=json.loads(Path(path).read_text()); a=[4242,777,31337]
 h=statistics.mean([d[f'aseed_{x}']['h2h_drop'] for x in a]); j=statistics.mean([d[f'aseed_{x}']['jam_vs_sweep_drop'] for x in a]); j1=statistics.mean([d[f'aseed_{x}']['j1_only_drop'] for x in a]); ri=1-statistics.mean([d[f'aseed_{x}']['rad_vs_idle_success'] for x in a]); f=d['sweep_vs_idle_floor']['drop']; eta=100*(1-(h-ri)/(j-f)); return h,j,j1,eta
base=ROOT/'experiments/array_face_s7/learning_repair'
paths=[base/'s7_continue_output_seed20260801/final_eval.json',base/'s7_r5_mix0p25_output_seed20260821/final_eval.json',base/'s7_r5_mix0p5_output_seed20260822/final_eval.json',base/'s7_r5_mix0p75_output_seed20260823/final_eval.json']
rows=[read(p) for p in paths]; mix=[0,.25,.5,.75]
ratio=[r[2]/r[1] for r in rows]; eta=[r[3] for r in rows]
fig,axs=plt.subplots(1,2,figsize=(6.9,2.45))
axs[0].plot(mix,ratio,'o-',color=COLORS['purple'],lw=1.8,ms=5); axs[0].set_xlabel('Singleton exposure in training'); axs[0].set_ylabel('$d_{j1}/d_{jvs}$'); axs[0].set_xticks(mix); axs[0].set_ylim(0,.58); axs[0].grid(alpha=.25)
for x,y in zip(mix,ratio): axs[0].text(x,y+.025,f'{y:.3f}',ha='center',fontsize=7)
axs[1].plot(mix,eta,'o-',color=COLORS['green'],lw=1.8,ms=5); axs[1].set_xlabel('Singleton exposure in training'); axs[1].set_ylabel('Neutralization $\eta$ (%)'); axs[1].set_xticks(mix); axs[1].set_ylim(15,30); axs[1].grid(alpha=.25)
for x,y in zip(mix,eta): axs[1].text(x,y+.7,f'{y:.1f}',ha='center',fontsize=7)
for ax in axs: ax.set_axisbelow(True)
fig.tight_layout(); save(fig,'fig6_r5_dose_response')
