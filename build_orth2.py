"""Memory-efficient AIMC model: orthogonalize target vs v1_sherpa, train LGBM, negate."""
import numpy as np, pandas as pd, cloudpickle, time, gc
import lightgbm as lgb
ts = time.time()
tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet",
                      columns=[c for c in pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet").columns if c.startswith("feature") or c in ("exped","target_everest")])
# Simpler: load full then drop
tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
bm = pd.read_parquet("/home/hatch/workspace/user/files/eiq_train_benchmark_models.parquet")
fcols = [c for c in tr.columns if c.startswith("feature")]
expeds = sorted(tr["exped"].unique())
eord = tr["exped"].map({e: i for i, e in enumerate(expeds)}).values
mask = eord >= 1000
dtr_exp = tr.loc[mask, "exped"].values
ytr = tr.loc[mask, "target_everest"].values.astype(np.float32)
sherpa = bm.set_index("exped").loc[dtr_exp, "v1_sherpa"].values.astype(np.float64)
del tr, bm; gc.collect()
# Per-exped orthogonalize y vs sherpa using numpy (no groupby copies)
uniq = np.unique(dtr_exp)
y_orth = np.empty_like(ytr, dtype=np.float64)
for e in uniq:
    m = dtr_exp == e
    s = sherpa[m]; y = ytr[m].astype(np.float64)
    A = np.column_stack([np.ones(m.sum()), s])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_orth[m] = y - (c[0] + c[1] * s)
print(f"y_orth corr y: {np.corrcoef(y_orth, ytr)[0,1]:.4f}", flush=True)
del dtr_exp, sherpa, ytr; gc.collect()
# Reload X (we deleted tr; reload features only)
trf = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet", columns=fcols+["exped"])
eord2 = trf["exped"].map({e: i for i, e in enumerate(expeds)}).values
Xtr = trf.loc[eord2 >= 1000, fcols].values.astype(np.float32)
del trf; gc.collect()
Xtr = np.where(Xtr == -1, np.nan, Xtr)
# median impute
med = np.nanmedian(Xtr, axis=0); med[np.isnan(med)] = 0
inds = np.where(np.isnan(Xtr))
Xtr[inds] = np.take(med, inds[1])
del med, inds; gc.collect()
va = pd.read_parquet("/home/hatch/workspace/user/files/eiq_validation.parquet")
Xva = va[fcols].values.astype(np.float32); del va; gc.collect()
Xva = np.where(Xva == -1, np.nan, Xva)
medv = np.nanmedian(Xva, axis=0); medv[np.isnan(medv)] = 0
indsv = np.where(np.isnan(Xva)); Xva[indsv] = np.take(medv, indsv[indsv[0] if False else 1] if False else indsv[1])
# Fix: proper impute
Xva = va[fcols].values.astype(np.float32) if 'va' in dir() else Xva
print("shapes", Xtr.shape, flush=True)
dset = lgb.Dataset(Xtr, label=y_orth.astype(np.float32))
del Xtr, y_orth; gc.collect()
params = dict(objective="regression", num_leaves=31, learning_rate=0.01,
              min_child_samples=200, verbosity=-1, seed=3)
_model = lgb.train(params, dset, 500)
print(f"trained ({time.time()-ts:.0f}s)", flush=True)
