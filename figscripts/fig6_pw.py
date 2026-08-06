"""Fig 6: process window maps (P x t), 2x3 — one panel per initial void fraction,
with a vertical line marking the pressure of minimum predicted remaining void."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from style import load_results

r = load_results()
final, tab = r['final'], r['tab']

# ---- style knobs for this figure ----
FIGSIZE = (12.5, 6.4)
CMAP = 'RdYlGn_r'
VMIN, VMAX = 0, 16
N_GRID = 80
VINIT_PANELS = [5, 10, 15, 20, 25, 30]
MARK_COLOR = 'navy'
# --------------------------------------

pg = np.linspace(0.03, 0.09, N_GRID)
tg = np.linspace(1, 5, N_GRID)

fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex=True, sharey=True)
for idx, (vi, ax) in enumerate(zip(VINIT_PANELS, axes.ravel())):
    PP2, TT2 = np.meshgrid(pg, tg)
    Z2 = final.predict(np.column_stack([PP2.ravel(), TT2.ravel(), np.full(PP2.size, float(vi))])).reshape(PP2.shape)
    cs = ax.contourf(PP2, TT2, Z2, levels=20, cmap=CMAP, vmin=VMIN, vmax=VMAX)
    pr, _ = tab[vi]
    ax.axvline(pr, color=MARK_COLOR, ls='--', lw=1.8)
    lab = chr(ord('a') + idx)
    ax.set_title(f'({lab}) Initial void = {vi}%', loc='left', fontweight='bold', fontsize=11)
for ax in axes[1]:
    ax.set_xlabel('Vacuum pressure (MPa)')
for ax in axes[:, 0]:
    ax.set_ylabel('Holding time (min)')
fig.colorbar(cs, ax=axes, label='Predicted remaining void (%)', shrink=0.85)

plt.show()
