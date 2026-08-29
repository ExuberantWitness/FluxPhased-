import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from paper_plot_style import COLORS, save, FIG_DIR

# Values from the final S7 behavior extraction (seed 20260801, 16 validation
# seeds x 64 steps); the extraction script remains the reproducible source.
rad_az = [0.0, 0.488, 0.0, 0.512, 0.0]
rad_beams = [0.0, 0.488, 0.0, 0.512, 0.0]
j0_az = [0.256,0.138,0.184,0.200,0.223]
j1_az = [0.244,0.142,0.183,0.202,0.229]
assign = np.array([[22.2,22.6],[21.9,12.3]])

fig=plt.figure(figsize=(6.9,2.65))
ax=fig.add_axes([.07,.18,.25,.70])
x=np.arange(5); w=.36
ax.bar(x-w/2,j0_az,w,color=COLORS['red'],label='jammer 0')
ax.bar(x+w/2,j1_az,w,color=COLORS['blue'],label='jammer 1')
ax.set_xticks(x); ax.set_xticklabels(['$-60^\circ$','$-30^\circ$','$0^\circ$','$30^\circ$','$60^\circ$'])
ax.set_xlabel('Jammer beam azimuth'); ax.set_ylabel('Stochastic beam mass'); ax.legend(frameon=False,fontsize=6.5); ax.grid(axis='y',alpha=.25); ax.set_ylim(0,.32)
ax=fig.add_axes([.40,.18,.25,.70])
ax.bar(x,rad_az,color=COLORS['green'],width=.62)
ax.set_xticks(x); ax.set_xticklabels(['$-60^\circ$','$-30^\circ$','$0^\circ$','$30^\circ$','$60^\circ$'])
ax.set_xlabel('Radar beam azimuth'); ax.set_ylabel('Beam mass'); ax.set_ylim(0,.62); ax.grid(axis='y',alpha=.25)
ax=fig.add_axes([.73,.18,.23,.70])
im=ax.imshow(assign,cmap='YlOrRd',vmin=0,vmax=25,aspect='auto')
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(['radar 0','radar 1'],fontsize=7); ax.set_yticklabels(['jammer 0','jammer 1'],fontsize=7)
ax.set_xlabel('JNR contribution (dB)',fontsize=7)
for i in range(2):
 for j in range(2): ax.text(j,i,f'{assign[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if assign[i,j]>15 else COLORS['black'])
fig.colorbar(im,ax=ax,fraction=.045,pad=.03,label='dB')
fig.text(.5,.03,'Parameter-shared radars divide azimuth sectors; jammers use a symmetric mixed policy.',ha='center',fontsize=7.5)
save(fig,'fig5_behavior')
