"""
Heavy computation for the void-fraction analysis: data, models, LOOCV,
hyperparameter sweeps, permutation importance, final fitted model, and the
recommended-pressure table.

Run this ONCE (or whenever the data/model logic changes). Results are cached
to results/results.joblib. Figure scripts in figscripts/ just load the cache
and plot, so tweaking a figure's style never requires re-fitting anything.
"""
import numpy as np, pandas as pd, joblib, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

BASE = 'C:/Workspace/UF_Optimization/Optimization_Code/'
RESULTS_PATH = BASE + 'results/results.joblib'

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

def main():
    # ===== Models under LOOCV =====
    models = {
     'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
     'Neural Network': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(15,8), activation='relu',
            solver='lbfgs', max_iter=5000, random_state=RS)),
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

    os.makedirs(BASE+'results', exist_ok=True)
    joblib.dump(dict(
        df=df, X=X, y=y,
        results=results, preds=preds,
        trees_v=trees_v, r2_trees=r2_trees,
        depth_v=depth_v, r2_depth=r2_depth,
        leaf_v=leaf_v, r2_leaf=r2_leaf,
        pi=pi, final=final, tab=tab,
    ), RESULTS_PATH)
    print(f'\nresults cached -> {RESULTS_PATH}')

if __name__ == '__main__':
    main()
