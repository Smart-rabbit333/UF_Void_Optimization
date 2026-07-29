import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

BASE = 'C:/Workspace/UF_Optimization/Optimization_Code/'

RS = 42
data = [
(1,28.1,0.09,5,2.5),(2,31,0.09,5,2.1),(3,24,0.09,5,1.7),(4,13.2,0.09,5,1),(5,16,0.09,5,1.2),
(6,11,0.09,5,1.3),(7,2,0.09,5,0.7),(8,8.3,0.09,5,1.4),(9,4.1,0.09,5,0.8),
(10,23.9,0.09,3,2),(11,26,0.09,3,1.8),(12,28,0.09,3,2.3),(13,15.8,0.09,3,1.8),(14,12,0.09,3,1.5),
(15,17.5,0.09,3,1.2),(16,3,0.09,3,1.1),(17,4.3,0.09,3,1),(18,5.5,0.09,3,1.2),
(19,33,0.09,1,1.9),(20,25,0.09,1,2.1),(21,20.4,0.09,1,2.2),(22,17.9,0.09,1,0.9),(23,14.6,0.09,1,1.2),
(24,16.2,0.09,1,1.3),(25,2.5,0.09,1,0.9),(26,2,0.09,1,0.7),(27,7.3,0.09,1,0.8),
(28,22.3,0.06,3,8.6),(29,24.6,0.06,3,8.1),(30,27.9,0.06,3,6.6),(31,15.1,0.06,3,6.7),(32,11.8,0.06,3,6.9),
(33,14.9,0.06,3,7.5),(34,4.7,0.06,3,2.5),(35,2.9,0.06,3,2),(36,6.8,0.06,3,1.8),
(37,24.6,0.03,3,16.1),(38,22.3,0.03,3,15),(39,21,0.03,3,14.2),(40,10.2,0.03,3,7),(41,15,0.03,3,8.1),
(42,17.2,0.03,3,6.8),(43,3.2,0.03,3,2.6),(44,7.6,0.03,3,2.1),(45,2.4,0.03,3,2),
]
df = pd.DataFrame(data, columns=['run','vinit','P','t','vrem'])
df.to_csv(BASE+'raw45.csv', index=False)
X = df[['P','t','vinit']].values
y = df['vrem'].values
loo = LeaveOneOut()

def rf(**kw):
    p = dict(n_estimators=50, max_depth=4, min_samples_leaf=2, random_state=RS)
    p.update(kw)
    return make_pipeline(StandardScaler(), RandomForestRegressor(**p))

def metrics(yt, yp):
    return r2_score(yt, yp), mean_absolute_error(yt, yp), np.sqrt(mean_squared_error(yt, yp))

# ===== Models under LOOCV =====
models = {
 'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
 'Neural Network': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(15,8), activation='relu',
        solver='adam', max_iter=5000, random_state=RS)),
 'Random Forest': rf(),
}
results, preds = {}, {}
for name, m in models.items():
    yp = cross_val_predict(m, X, y, cv=loo)
    preds[name] = yp
    results[name] = metrics(y, yp)
    print(f"{name:18s} R2={results[name][0]:.4f}  MAE={results[name][1]:.3f}  RMSE={results[name][2]:.3f}")

# ===== Hyperparameter sweeps (LOOCV R2) =====
def sweep(param, values):
    out = []
    for v in values:
        yp = cross_val_predict(rf(**{param:v}), X, y, cv=loo)
        out.append(r2_score(y, yp))
    return out
trees_v = [1,5,10,20,30,50,100,200]
depth_v = [1,2,3,4,5,6,8,None]
leaf_v  = [1,2,3,4,5]
r2_trees = sweep('n_estimators', trees_v)
r2_depth = sweep('max_depth', depth_v)
r2_leaf  = sweep('min_samples_leaf', leaf_v)
print('trees:', dict(zip(trees_v, [round(v,3) for v in r2_trees])))
print('depth:', dict(zip([str(d) for d in depth_v], [round(v,3) for v in r2_depth])))
print('leaf :', dict(zip(leaf_v, [round(v,3) for v in r2_leaf])))

# ===== Permutation importance (final model, full data, RMSE increase) =====
final = rf().fit(X, y)
pi = permutation_importance(final, X, y, scoring='neg_root_mean_squared_error', n_repeats=50, random_state=RS)
print('perm importance dRMSE: P=%.3f t=%.3f vinit=%.3f' % tuple(pi.importances_mean))

# ===== Per-regime parity diagnostics =====
yp_rf = preds['Random Forest']
for p in [0.09,0.06,0.03]:
    m = df['P']==p
    err = yp_rf[m]-y[m]
    print(f"P={p}: n={m.sum()} MAE={np.abs(err).mean():.2f} bias={err.mean():+.2f} maxabs={np.abs(err).max():.2f}")
resid = yp_rf - y
print(f"overall residual mean={resid.mean():+.3f} std={resid.std():.3f}")

# ===== Recommended pressure table (minimum predicted remaining void, t=3 min) =====
Pgrid = np.linspace(0.03, 0.09, 601)
def p_opt(vinit):
    Xg = np.column_stack([Pgrid, np.full_like(Pgrid,3.0), np.full_like(Pgrid,vinit)])
    pred = final.predict(Xg)
    idx = np.argmin(pred)
    return Pgrid[idx], pred[idx]
tab = {}
for vi in [5,10,15,20,25,30]:
    tab[vi] = p_opt(vi)
print('P_opt table (P at min predicted void):', {k:(round(p,3), round(v,2)) for k,(p,v) in tab.items()})

# ============ FIGURES ============
plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':0.3})
FD=BASE+'figs/'
import os; os.makedirs(FD, exist_ok=True)

# --- Fig 2: hyperparameter tuning (1x3) ---
fig, axes = plt.subplots(1,3, figsize=(12.5,3.0))
axes[0].plot(trees_v, r2_trees,'o-',color='#2166ac'); axes[0].set_xlabel('Number of trees'); axes[0].set_ylabel('LOOCV R²'); axes[0].set_title('(a)', loc='left', fontweight='bold')
dl = [str(d) if d else 'None' for d in depth_v]
axes[1].plot(range(len(depth_v)), r2_depth,'s-',color='#b2182b'); axes[1].set_xticks(range(len(depth_v))); axes[1].set_xticklabels(dl); axes[1].set_xlabel('Maximum depth'); axes[1].set_title('(b)', loc='left', fontweight='bold')
axes[2].plot(leaf_v, r2_leaf,'^-',color='#1b7837'); axes[2].set_xlabel('Minimum samples per leaf'); axes[2].set_title('(c)', loc='left', fontweight='bold')
for a in axes: a.set_ylim(0.3,1.0)
plt.tight_layout(); plt.savefig(FD+'fig2_hp.png', dpi=300); plt.close()

# --- Fig 3: model comparison ---
names = list(results.keys())
maes = [results[n][1] for n in names]; r2s = [results[n][0] for n in names]
fig, ax1 = plt.subplots(figsize=(8,4.5))
xp = np.arange(3); w=0.35
b1 = ax1.bar(xp-w/2, maes, w, color='#888', label='MAE (%p)')
ax1.set_ylabel('MAE (remaining void, %p)'); ax1.set_xticks(xp); ax1.set_xticklabels(names)
ax1.set_ylim(0, max(maes)*1.35)
ax2 = ax1.twinx(); ax2.grid(False)
b2 = ax2.bar(xp+w/2, r2s, w, color='#d6604d', label='R²')
ax2.set_ylabel('R² (LOOCV)'); ax2.set_ylim(0,1.18)
for i,(m,r) in enumerate(zip(maes,r2s)):
    ax1.text(i-w/2, m+max(maes)*0.03, f'{m:.2f}', ha='center', fontweight='bold', fontsize=10)
    ax2.text(i+w/2, r+0.025, f'{r:.3f}', ha='center', fontweight='bold', fontsize=10)
ax1.set_title('Model comparison — LOOCV (n = 45)', pad=34)
h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5,1.14), ncol=2, frameon=False)
plt.tight_layout(); plt.savefig(FD+'fig3_models.png', dpi=300); plt.close()

# --- Fig 4: parity plot ---
fig, ax = plt.subplots(figsize=(5.2,4.6))
sc = ax.scatter(y, yp_rf, c=df['P'], cmap='viridis', s=70, edgecolor='k', zorder=3)
lim=[0,17]; ax.plot(lim,lim,'--',color='gray'); ax.fill_between(lim,[l-1 for l in lim],[l+1 for l in lim],color='gray',alpha=0.15,label='±1 %p band')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Actual remaining void (%)'); ax.set_ylabel('Predicted remaining void (%)')
ax.set_title(f'Random Forest — LOOCV (R² = {results["Random Forest"][0]:.3f})')
cb = plt.colorbar(sc); cb.set_label('Vacuum pressure (MPa)')
ax.legend(loc='upper left')
plt.tight_layout(); plt.savefig(FD+'fig4_parity.png', dpi=300); plt.close()

# --- Fig 5: response surface (fixed t = 3 min) ---
pg = np.linspace(0.03,0.09,120); vg = np.linspace(2,33,120)
PP, VV = np.meshgrid(pg, vg)
Z = final.predict(np.column_stack([PP.ravel(), np.full(PP.size,3.0), VV.ravel()])).reshape(PP.shape)
pad_p = 0.005; pad_v = 2.5
pg_ext = np.linspace(0.03-pad_p, 0.09+pad_p, 120); vg_ext = np.linspace(2-pad_v, 33+pad_v, 120)
PPe, VVe = np.meshgrid(pg_ext, vg_ext)
Ze = final.predict(np.column_stack([PPe.ravel(), np.full(PPe.size,3.0), VVe.ravel()])).reshape(PPe.shape)
fig, axes = plt.subplots(1,2, figsize=(12.5,4.4))
cs = axes[0].contourf(PPe, VVe, Ze, levels=20, cmap='RdYlGn_r')
axes[0].scatter(df['P'], df['vinit'], c='white', s=32, edgecolor='k', linewidth=0.8, zorder=3)
axes[0].set_xlim(0.03-pad_p, 0.09+pad_p); axes[0].set_ylim(2-pad_v, 33+pad_v)
plt.colorbar(cs, ax=axes[0], label='Predicted remaining void (%)')
axes[0].set_xlabel('Vacuum pressure (MPa)'); axes[0].set_ylabel('Initial void fraction (%)')
axes[0].set_title('(a) Response surface (t = 3 min)', loc='left', fontweight='bold')
colors = plt.cm.plasma(np.linspace(0.1,0.85,6))
for vi,ccol in zip([5,10,15,20,25,30],colors):
    zz = final.predict(np.column_stack([pg, np.full_like(pg,3.0), np.full_like(pg,vi)]))
    axes[1].plot(pg, zz, color=ccol, label=f'V_init = {vi}%')
axes[1].scatter(df['P'], df['vrem'], c='k', s=14, alpha=0.5, zorder=3)
axes[1].set_xlabel('Vacuum pressure (MPa)'); axes[1].set_ylabel('Remaining void (%)')
axes[1].set_title('(b) 1D cross-sections', loc='left', fontweight='bold'); axes[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(FD+'fig5_surface.png', dpi=300); plt.close()

# --- Fig 6: process window maps (P × t), 2x3, with P at minimum predicted void ---
tg = np.linspace(1,5,80)
fig, axes = plt.subplots(2,3, figsize=(12.5,6.4), sharex=True, sharey=True)
for idx,(vi,ax) in enumerate(zip([5,10,15,20,25,30], axes.ravel())):
    PP2, TT2 = np.meshgrid(pg, tg)
    Z2 = final.predict(np.column_stack([PP2.ravel(), TT2.ravel(), np.full(PP2.size,float(vi))])).reshape(PP2.shape)
    cs = ax.contourf(PP2, TT2, Z2, levels=20, cmap='RdYlGn_r', vmin=0, vmax=16)
    pr, _ = tab[vi]
    ax.axvline(pr, color='navy', ls='--', lw=1.8)
    lab = chr(ord('a')+idx)
    ax.set_title(f'({lab}) Initial void = {vi}%', loc='left', fontweight='bold', fontsize=11)
for ax in axes[1]: ax.set_xlabel('Vacuum pressure (MPa)')
for ax in axes[:,0]: ax.set_ylabel('Holding time (min)')
fig.colorbar(cs, ax=axes, label='Predicted remaining void (%)', shrink=0.85)
plt.savefig(FD+'fig6_pw.png', dpi=300, bbox_inches='tight'); plt.close()

# --- Fig 7: RF prediction vs pressure at vinit=20%, t=3 min (shows the step/plateau structure) ---
vi_demo, t_demo = 20.0, 3.0
Pfine = np.linspace(0.03, 0.09, 300)
pred_fine = final.predict(np.column_stack([Pfine, np.full_like(Pfine,t_demo), np.full_like(Pfine,vi_demo)]))
fig, ax = plt.subplots(figsize=(6.5,4.4))
ax.scatter(Pfine, pred_fine, s=14, color='#2166ac', zorder=3)
ax.set_xlabel('Vacuum pressure (MPa)'); ax.set_ylabel('Predicted remaining void (%)')
ax.set_title(f'RF prediction vs pressure (V_init = {vi_demo:.0f}%, t = {t_demo:.0f} min)')
plt.tight_layout(); plt.savefig(FD+'fig7_stepfunction.png', dpi=300); plt.close()

print(f'\nRF prediction vs pressure (V_init = {vi_demo:.0f}%, t = {t_demo:.0f} min)')
print(f'{"P (MPa)":>10} | {"pred (%)":>9}')
for p_show in np.arange(0.030, 0.0901, 0.005):
    pv = final.predict([[p_show, t_demo, vi_demo]])[0]
    print(f'{p_show:10.3f} | {pv:9.3f}')

change = np.where(np.abs(np.diff(pred_fine)) > 1e-6)[0]
bounds = [0] + (change+1).tolist() + [len(Pfine)-1]
print('\nDetected plateaus (leaf boundaries):')
for i in range(len(bounds)-1):
    a,b = bounds[i], bounds[i+1]-1 if i < len(bounds)-2 else bounds[i+1]
    print(f'  P in [{Pfine[a]:.3f}, {Pfine[b]:.3f}] -> pred = {pred_fine[a]:.3f}%')
print('figures saved')
