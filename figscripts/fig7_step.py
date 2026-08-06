"""Fig 7: RF prediction vs pressure at fixed vinit/t — shows the step/plateau
structure inherent to a tree-based model, plus a printed plateau breakdown."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from style import load_results

r = load_results()
final = r['final']

# ---- style knobs for this figure ----
FIGSIZE = (6.5, 4.4)
POINT_SIZE = 14
POINT_COLOR = '#2166ac'
VI_DEMO, T_DEMO = 20.0, 3.0
N_GRID = 300
# --------------------------------------

Pfine = np.linspace(0.03, 0.09, N_GRID)
pred_fine = final.predict(np.column_stack([Pfine, np.full_like(Pfine, T_DEMO), np.full_like(Pfine, VI_DEMO)]))

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.scatter(Pfine, pred_fine, s=POINT_SIZE, color=POINT_COLOR, zorder=3)
ax.set_xlabel('Vacuum pressure (MPa)'); ax.set_ylabel('Predicted remaining void (%)')
ax.set_title(f'RF prediction vs pressure (V_init = {VI_DEMO:.0f}%, t = {T_DEMO:.0f} min)')

plt.tight_layout()

# print diagnostics before show() since show() blocks until the window is closed
print(f'\nRF prediction vs pressure (V_init = {VI_DEMO:.0f}%, t = {T_DEMO:.0f} min)')
print(f'{"P (MPa)":>10} | {"pred (%)":>9}')
for p_show in np.arange(0.030, 0.0901, 0.005):
    pv = final.predict([[p_show, T_DEMO, VI_DEMO]])[0]
    print(f'{p_show:10.3f} | {pv:9.3f}')

change = np.where(np.abs(np.diff(pred_fine)) > 1e-6)[0]
bounds = [0] + (change + 1).tolist() + [len(Pfine) - 1]
print('\nDetected plateaus (leaf boundaries):')
for i in range(len(bounds) - 1):
    a, b = bounds[i], bounds[i + 1] - 1 if i < len(bounds) - 2 else bounds[i + 1]
    print(f'  P in [{Pfine[a]:.3f}, {Pfine[b]:.3f}] -> pred = {pred_fine[a]:.3f}%')

plt.show()
