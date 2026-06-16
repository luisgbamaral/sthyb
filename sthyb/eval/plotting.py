"""
plotting.py — standardized NeurIPS-style figure config for sthyb.

Import and call `set_style()` once before plotting; use `method_color()` /
`PALETTE` for consistent per-method colors and `save_fig()` for vector-friendly
output. Keeps all paper figures visually uniform (fonts, sizes, palette).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import join as pjoin

# ── consistent colors per method family ──────────────────────────────────────
PALETTE = {
    'base':     '#7f7f7f',   # grey   — single model
    'anscombe': '#1f77b4',   # blue   — Anscombe hybrid
    'log1p':    '#2ca02c',   # green  — log1p hybrid
    'additive': '#ff7f0e',   # orange — additive (level) hybrid
    'guardia':  '#d62728',   # red    — GUARD IA
}


def set_style():
    """Apply NeurIPS-style rcParams (serif, compact, vector-editable text)."""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 9,
        'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.3, 'lines.markersize': 4,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.4,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed editable fonts
    })


def method_color(name):
    """Map a method name (hybrid-*, hybrid-add-*, hybrid-log-*, guardia*, base*) to a color."""
    n = name.lower()
    if n.startswith('guardia'):              return PALETTE['guardia']
    if 'add' in n:                           return PALETTE['additive']
    if 'log' in n:                           return PALETTE['log1p']
    if n.startswith('hybrid'):               return PALETTE['anscombe']
    return PALETTE['base']


def save_fig(fig, out_dir, name):
    """Save a figure as both PDF (paper) and PNG (preview)."""
    for ext in ('pdf', 'png'):
        fig.savefig(pjoin(out_dir, f'{name}.{ext}'))
    plt.close(fig)
