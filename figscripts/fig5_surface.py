"""Fig 5: response surface (fixed t = 3 min) — 2D contour + 1D cross-sections."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from style import load_results

r = load_results()
df, final = r['df'], r['final']

# ---- style knobs for this figure ----
FIGSIZE = (12.5, 4.4)
CMAP = 'RdYlGn_r'
N_GRID = 120
PAD_P, PAD_V = 0.005, 2.5
LINE_CMAP = plt.cm.plasma
VINIT_LINES = [5, 10, 15, 20, 25, 30]
T_FIXED = 3.0
# --------------------------------------

pg = np.linspace(0.03, 0.09, N_GRID)
vg = np.linspace(2, 33, N_GRID)
pg_ext = np.linspace(0.03 - PAD_P, 0.09 + PAD_P, N_GRID)
vg_ext = np.linspace(2 - PAD_V, 33 + PAD_V, N_GRID)
PPe, VVe = np.meshgrid(pg_ext, vg_ext)
Ze = final.predict(np.column_stack([PPe.ravel(), np.full(PPe.size, T_FIXED), VVe.ravel()])).reshape(PPe.shape)

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

cs = axes[0].contourf(PPe, VVe, Ze, levels=20, cmap=CMAP)
axes[0].scatter(df['P'], df['vinit'], c='white', s=32, edgecolor='k', linewidth=0.8, zorder=3)
axes[0].set_xlim(0.03 - PAD_P, 0.09 + PAD_P); axes[0].set_ylim(2 - PAD_V, 33 + PAD_V)
plt.colorbar(cs, ax=axes[0], label='Predicted remaining void (%)')
axes[0].set_xlabel('Vacuum pressure (MPa)'); axes[0].set_ylabel('Initial void fraction (%)')
axes[0].set_title('(a) Response surface (t = 3 min)', loc='left', fontweight='bold')

colors = LINE_CMAP(np.linspace(0.1, 0.85, len(VINIT_LINES)))
for vi, ccol in zip(VINIT_LINES, colors):
    zz = final.predict(np.column_stack([pg, np.full_like(pg, T_FIXED), np.full_like(pg, vi)]))
    axes[1].plot(pg, zz, color=ccol, label=f'V_init = {vi}%')
axes[1].scatter(df['P'], df['vrem'], c='k', s=14, alpha=0.5, zorder=3)
axes[1].set_xlabel('Vacuum pressure (MPa)'); axes[1].set_ylabel('Remaining void (%)')
axes[1].set_title('(b) 1D cross-sections', loc='left', fontweight='bold'); axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()
