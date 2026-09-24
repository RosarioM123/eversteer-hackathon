"""Hyperparameter sweep for old-regime LGBM. Train 1000-5799, test 5842-6521 holdout."""
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import norm

tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
fcols = [c for c in tr.columns if c.startswith("feature")]
expeds = sorted(tr["exped"].unique())
tr["eord"] = tr["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
dtr = tr[(tr["eord"] >= 1000) & (tr["eord"] <= 5799)]
dte = tr[(tr["eord"] >= 5842) & (tr["eord"] <= 6521)]
ytr = dtr["target_everest"].values.astype(np.float32)
yte = dte["target_everest"].values.astype(np.float32)
Xtr = pd.DataFrame(dtr[fcols].values.astype(np.float32)).replace(-1, np.nan)
Xte = pd.DataFrame(dte[fcols].values.astype(np.float32)).replace(-1, np.nan)
dset = lgb.Dataset(Xtr, label=ytr)

def rgauss(x):
    r = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").values
    return norm.ppf(np.clip((r - 0.5) / len(x), 1e-12, 1 - 1e-12))
def corr20(p):
    cs = []
    for _, pos in dte.groupby("exped", sort=False).indices.items():
        pp, tt = np.asarray(p)[pos], yte[pos].astype(np.float64)
        if np.std(pp) > 0 and np.std(tt) > 0:
            g = rgauss(pp); tc = tt - tt.mean()
            cs.append(np.corrcoef(np.sign(g)*np.abs(g)**1.5,
                                 np.sign(tc)*np.abs(tc)**1.5)[0, 1])
    return float(np.mean(cs))

configs = [
    ("c_baseline", dict(objective="regression", num_leaves=31, learning_rate=0.02,
                        min_child_samples=200, verbosity=-1, seed=2), 300),
    ("lr001_500", dict(objective="regression", num_leaves=31, learning_rate=0.01,
                       min_child_samples=200, verbosity=-1, seed=3), 500),
    ("leaves63", dict(objective="regression", num_leaves=63, learning_rate=0.02,
                      min_child_samples=200, verbosity=-1, seed=4), 300),
    ("subsample", dict(objective="regression", num_leaves=31, learning_rate=0.02,
                       min_child_samples=200, feature_fraction=0.7,
                       bagging_fraction=0.8, bagging_freq=1, verbosity=-1, seed=5), 300),
    ("lr015_400", dict(objective="regression", num_leaves=47, learning_rate=0.015,
                       min_child_samples=150, verbosity=-1, seed=6), 400),
]
for tag, params, rounds in configs:
    m = lgb.train(params, dset, rounds)
    c = corr20(m.predict(Xte))
    print(f"{tag}: CORR={c:.4f}", flush=True)
