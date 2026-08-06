"""Fig 4: parity plot — actual vs predicted remaining void (Random Forest, LOOCV)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from style import load_results

r = load_results()
df, y = r['df'], r['y']
yp_rf = r['preds']['Random Forest']
r2_rf = r['results']['Random Forest'][0]

# ---- style knobs for this figure ----
FIGSIZE = (5.2, 4.6)
LIM = [0, 17]
BAND = 1          # +/- band width (percentage points)
CMAP = 'viridis'
POINT_SIZE = 70
# --------------------------------------

fig, ax = plt.subplots(figsize=FIGSIZE)
sc = ax.scatter(y, yp_rf, c=df['P'], cmap=CMAP, s=POINT_SIZE, edgecolor='k', zorder=3)
ax.plot(LIM, LIM, '--', color='gray')
ax.fill_between(LIM, [l - BAND for l in LIM], [l + BAND for l in LIM],
                 color='gray', alpha=0.15, label=f'±{BAND} %p band')
ax.set_xlim(LIM); ax.set_ylim(LIM)
ax.set_xlabel('Actual remaining void (%)'); ax.set_ylabel('Predicted remaining void (%)')
ax.set_title(f'Random Forest — LOOCV (R² = {r2_rf:.3f})')
cb = plt.colorbar(sc); cb.set_label('Vacuum pressure (MPa)')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()
