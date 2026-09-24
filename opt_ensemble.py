"""Optimize old-regime ensemble on the 680-exped holdout (5842-6521).
Train on 1000-5799. Candidates: ridge, lgbm x2 seeds, lgbm-tuned. Grid-search weights."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb
from scipy.stats import norm

tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
fcols = [c for c in tr.columns if c.startswith("feature")]
expeds = sorted(tr["exped"].unique())
tr["eord"] = tr["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
dtr = tr[(tr["eord"] >= 1000) & (tr["eord"] <= 5799)]
dte = tr[(tr["eord"] >= 5842) & (tr["eord"] <= 6521)]
print(f"train {len(dtr)} test {len(dte)}", flush=True)
ytr = dtr["target_everest"].values.astype(np.float32)
yte = dte["target_everest"].values.astype(np.float32)

a = dtr[fcols].values.astype(np.float32); a[a == -1] = np.nan
med = np.nanmedian(a, axis=0); med = np.where(np.isnan(med), 0.0, med)
def imp(X):
    b = X.astype(np.float64); b[b == -1] = np.nan
    return np.where(np.isnan(b), med, b)
Xtr_i = imp(dtr[fcols].values.astype(np.float32))
Xte_i = imp(dte[fcols].values.astype(np.float32))
Xtr_n = pd.DataFrame(dtr[fcols].values.astype(np.float32)).replace(-1, np.nan)
Xte_n = pd.DataFrame(dte[fcols].values.astype(np.float32)).replace(-1, np.nan)

preds = {}
m = Ridge(alpha=1.0); m.fit(Xtr_i, ytr); preds["ridge"] = m.predict(Xte_i)
for tag, params, seed in [
    ("lgbm_a", dict(objective="regression", num_leaves=31, learning_rate=0.05,
                    min_child_samples=200, verbosity=-1), 0),
    ("lgbm_b", dict(objective="regression", num_leaves=63, learning_rate=0.03,
                    min_child_samples=100, verbosity=-1, seed=1), 1),
    ("lgbm_c", dict(objective="regression", num_leaves=31, learning_rate=0.02,
                    min_child_samples=200, verbosity=-1, seed=2), 2)]:
    d = lgb.Dataset(Xtr_n, label=ytr)
    bm = lgb.train(params, d, 300)
    preds[tag] = bm.predict(Xte_n).astype(np.float64)
    print(f"{tag} done", flush=True)

def rgauss(x):
    r = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").values
    return norm.ppf(np.clip((r - 0.5) / len(x), 1e-12, 1 - 1e-12))
def corr20(p):
    cs = []
    for _, pos in dte.groupby("exped", sort=False).indices.items():
        pp, tt = p[pos], yte[pos].astype(np.float64)
        if np.std(pp) > 0 and np.std(tt) > 0:
            g = rgauss(pp); tc = tt - tt.mean()
            cs.append(np.corrcoef(np.sign(g)*np.abs(g)**1.5,
                                 np.sign(tc)*np.abs(tc)**1.5)[0, 1])
    return float(np.mean(cs))

names = list(preds.keys())
P = np.column_stack([preds[n] for n in names])
print("individual:", {n: round(corr20(preds[n]), 4) for n in names}, flush=True)
best = (0, None)
import itertools
for w in itertools.product([0, 0.25, 0.5, 0.75, 1.0], repeat=len(names)):
    if sum(w) == 0: continue
    w = np.array(w) / sum(w)
    c = corr20(P @ w)
    if c > best[0]: best = (c, dict(zip(names, w.round(2))))
print("BEST:", round(best[0], 4), best[1], flush=True)
# pairwise with ridge+lgbm_a (E1-style) weight sweep
for wr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    c = corr20(wr * preds["ridge"] + (1 - wr) * preds["lgbm_a"])
    print(f"ridge {wr:.1f} + lgbm_a: {c:.4f}", flush=True)
