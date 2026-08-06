"""Fig 2: hyperparameter tuning (1x3) — LOOCV R2 vs n_trees / max_depth / min_leaf."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from style import load_results, COLORS

r = load_results()
trees_v, r2_trees = r['trees_v'], r['r2_trees']
depth_v, r2_depth = r['depth_v'], r['r2_depth']
leaf_v,  r2_leaf  = r['leaf_v'],  r['r2_leaf']

# ---- style knobs for this figure ----
FIGSIZE = (12.5, 3.0)
YLIM = (0.3, 1.0)
# --------------------------------------

fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
axes[0].plot(trees_v, r2_trees, 'o-', color=COLORS['trees'])
axes[0].set_xlabel('Number of trees'); axes[0].set_ylabel('LOOCV R²')
axes[0].set_title('(a)', loc='left', fontweight='bold')

dl = [str(d) if d else 'None' for d in depth_v]
axes[1].plot(range(len(depth_v)), r2_depth, 's-', color=COLORS['depth'])
axes[1].set_xticks(range(len(depth_v))); axes[1].set_xticklabels(dl)
axes[1].set_xlabel('Maximum depth'); axes[1].set_title('(b)', loc='left', fontweight='bold')

axes[2].plot(leaf_v, r2_leaf, '^-', color=COLORS['leaf'])
axes[2].set_xlabel('Minimum samples per leaf'); axes[2].set_title('(c)', loc='left', fontweight='bold')

for a in axes:
    a.set_ylim(*YLIM)

plt.tight_layout()
plt.show()
