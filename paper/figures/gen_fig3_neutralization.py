"""Fig. 3: floor-adjusted neutralization, S6 vs S7, across training seeds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from paper_plot_style import COLORS, save
from results_table import TABLE

s6_agg = TABLE['s6']['agg']
s7_agg = TABLE['s7']['agg']
s7_seed_etas = [v['eta_pct'] for v in TABLE['s7']['per_seed'].values()]

fig, ax = plt.subplots(figsize=(3.35, 2.45))
labels = ['S6\n1 jammer', 'S7\n2 jammers']
means = [s6_agg['eta_pct'], s7_agg['eta_pct']]
errs = [s6_agg['eta_pct_sd'], s7_agg['eta_pct_sd']]
bars = ax.bar(labels, means, yerr=errs, capsize=3, width=.58,
              color=[COLORS['blue'], COLORS['red']],
              edgecolor=COLORS['black'], linewidth=.5)
for b, v in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, v + 3, f'{v:.1f}%',
            ha='center', fontsize=8)
ax.scatter([1 - .18, 1, 1 + .18], s7_seed_etas, color=COLORS['black'],
           s=15, zorder=3, label='S7 seeds')
ax.set_ylabel('Floor-adjusted neutralization (%)')
ax.set_ylim(0, 80)
ax.legend(frameon=False, fontsize=6.5)
ax.grid(axis='y', alpha=.25)
ax.set_axisbelow(True)
save(fig, 'fig3_neutralization')
