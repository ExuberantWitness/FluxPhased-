"""Fig. 4: mechanism decomposition (attacker count vs cross-fire geometry).

All values come from the authoritative ``results_table`` module; eta is stored
there as a percent, so no unit conversion happens in this script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from paper_plot_style import COLORS, save
from results_table import TABLE

fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.45))
labels = ['S6\n1 jammer', 'S7 co-located\n2 jammers', 'S7 cross-fire\n2 jammers']
eta_vals = [TABLE['s6']['agg']['eta_pct'],
            TABLE['colocated']['eta_pct'],
            TABLE['crossfire_seed01']['eta_pct']]
colors = [COLORS['blue'], COLORS['yellow'], COLORS['red']]

b = axs[0].bar(labels, eta_vals, color=colors, edgecolor=COLORS['black'],
               linewidth=.5, width=.62)
for x, v in zip(b, eta_vals):
    axs[0].text(x.get_x() + x.get_width() / 2, v + 2, f'{v:.1f}%',
                ha='center', fontsize=7.5)
axs[0].set_ylabel('Neutralization (%)')
axs[0].set_ylim(0, 75)
axs[0].grid(axis='y', alpha=.25)

h2h_vals = [TABLE['s6']['agg']['h2h'],
            TABLE['colocated']['h2h'],
            TABLE['crossfire_seed01']['h2h']]
axs[1].bar(labels, h2h_vals, color=colors, edgecolor=COLORS['black'],
           linewidth=.5, width=.62)
for x, v in zip(axs[1].patches, h2h_vals):
    axs[1].text(x.get_x() + x.get_width() / 2, v + .02, f'{v:.3f}',
                ha='center', fontsize=7.5)
axs[1].set_ylabel('Head-to-head mission drop')
axs[1].set_ylim(0, .42)
axs[1].grid(axis='y', alpha=.25)

for ax in axs:
    ax.tick_params(axis='x', labelsize=7)
    ax.set_axisbelow(True)
fig.tight_layout()
save(fig, 'fig4_mechanism_ablation')
