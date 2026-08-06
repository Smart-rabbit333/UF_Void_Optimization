"""
Shared baseline style + cache loader for figure scripts.

Import what you need in each figscripts/fig*.py, then feel free to override
anything locally (colors, figsize, cmap, ...) right in that file — nothing
here is binding, it's just a sane default so all figures start consistent.
"""
import os, joblib
import matplotlib.pyplot as plt  # no Agg backend -> uses the system's interactive backend, so plt.show() opens a window

BASE = 'C:/Workspace/UF_Optimization/Optimization_Code/'
RESULTS_PATH = BASE + 'results/results.joblib'
FD = BASE + 'figs/'  # kept around in case you want to plt.savefig() manually while iterating
os.makedirs(FD, exist_ok=True)

# Baseline rcParams applied when this module is imported. Override after
# import in a specific fig script if you want that figure to look different,
# e.g.: plt.rcParams.update({'font.size': 13})
# figure.dpi controls both on-screen rendering resolution and the default
# savefig resolution, so 300 here keeps the two consistent.
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 300})

# A few named colors reused across figures — tweak here to restyle everywhere
# at once, or just hardcode a different color inside one fig script.
COLORS = {
    'trees': '#2166ac',
    'depth': '#b2182b',
    'leaf':  '#1b7837',
    'mae_bar': '#888888',
    'r2_bar': '#d6604d',
}


def load_results():
    """Load the cached computation results produced by compute.py."""
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f'{RESULTS_PATH} not found — run `python compute.py` first.'
        )
    return joblib.load(RESULTS_PATH)
