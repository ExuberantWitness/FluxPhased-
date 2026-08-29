from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

FIG_DIR = Path(__file__).resolve().parent
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})
# Okabe-Ito palette: print- and color-vision friendly.
COLORS = {
    'red': '#D55E00',
    'blue': '#0072B2',
    'green': '#009E73',
    'purple': '#CC79A7',
    'yellow': '#E69F00',
    'black': '#222222',
    'grey': '#777777',
    'lightgrey': '#D9D9D9',
}

def save(fig, name):
    fig.savefig(FIG_DIR / f'{name}.pdf', format='pdf')
    plt.close(fig)
