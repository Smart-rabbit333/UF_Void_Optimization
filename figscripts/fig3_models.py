"""Fig 3: model comparison — MAE (bar, left axis) vs R2 (bar, right axis)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from style import load_results, COLORS

r = load_results()
results = r['results']

# ---- style knobs for this figure ----
FIGSIZE = (8, 4.5)
BAR_WIDTH = 0.35
MAE_COLOR = COLORS['mae_bar']
R2_COLOR = COLORS['r2_bar']
TITLE = 'Model comparison — LOOCV (n = 45)'
# --------------------------------------

names = list(results.keys())
maes = [results[n][1] for n in names]
r2s = [results[n][0] for n in names]

fig, ax1 = plt.subplots(figsize=FIGSIZE)
xp = np.arange(len(names)); w = BAR_WIDTH
ax1.bar(xp - w/2, maes, w, color=MAE_COLOR, label='MAE')
ax1.set_ylabel('MAE (%)')
ax1.set_xticks(xp); ax1.set_xticklabels(names)
ax1.set_ylim(0, max(maes)*1.35)  # headroom for legend inside the axes

ax2 = ax1.twinx(); ax2.grid(False)
ax2.bar(xp + w/2, r2s, w, color=R2_COLOR, label='R$^2$')
ax2.set_ylabel('R$^2$')
ax2.set_ylim(0, 1.18)

for i, (m, rr) in enumerate(zip(maes, r2s)):
    ax1.text(i - w/2, m + max(maes)*0.03, f'{m:.2f}', ha='center', fontweight='bold', fontsize=10)
    ax2.text(i + w/2, rr + 0.025, f'{rr:.3f}', ha='center', fontweight='bold', fontsize=10)

ax1.set_title(TITLE)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper center', ncol=2, frameon=True)

plt.tight_layout()
plt.show()
