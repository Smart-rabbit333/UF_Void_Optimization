"""Display all 6 figures from the cached results, one window at a time (run
compute.py first if the cache is missing or stale). Each window is shown via
plt.show(), which blocks until you close it, then the next figure opens.
Each figscripts/fig*.py can also be run standalone for a single figure."""
import runpy, os

FIGSCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figscripts')
SCRIPTS = ['fig2_hp.py', 'fig3_models.py', 'fig4_parity.py', 'fig5_surface.py', 'fig6_pw.py', 'fig7_step.py']

for name in SCRIPTS:
    runpy.run_path(os.path.join(FIGSCRIPTS, name), run_name='__main__')
