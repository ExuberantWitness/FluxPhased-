"""Fig. 6: R5-lite opponent-class mixing dose response."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from paper_plot_style import COLORS, save
from results_table import TABLE

r5 = TABLE['r5']
mix = [row['q'] for row in r5]
ratio = [row['j1_over_jvs'] for row in r5]
eta = [row['eta_pct'] for row in r5]

fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.45))
axs[0].plot(mix, ratio, 'o-', color=COLORS['purple'], lw=1.8, ms=5)
axs[0].set_xlabel('Singleton exposure in training')
axs[0].set_ylabel('$d_{j1}/d_{jvs}$')
axs[0].set_xticks(mix)
axs[0].set_ylim(0, .58)
axs[0].grid(alpha=.25)
for x, y in zip(mix, ratio):
    axs[0].text(x, y + .025, f'{y:.3f}', ha='center', fontsize=7)

axs[1].plot(mix, eta, 'o-', color=COLORS['green'], lw=1.8, ms=5)
axs[1].set_xlabel('Singleton exposure in training')
axs[1].set_ylabel('Neutralization $\\eta$ (%)')
axs[1].set_xticks(mix)
axs[1].set_ylim(15, 30)
axs[1].grid(alpha=.25)
for x, y in zip(mix, eta):
    axs[1].text(x, y + .7, f'{y:.1f}', ha='center', fontsize=7)

for ax in axs:
    ax.set_axisbelow(True)
fig.tight_layout()
save(fig, 'fig6_r5_dose_response')
