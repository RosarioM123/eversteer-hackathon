"""AIMC model: orthogonalize target vs v1_sherpa per-exped, train LGBM, negate."""
import numpy as np, pandas as pd, cloudpickle, time
import lightgbm as lgb
ts = time.time()
tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
fcols = [c for c in tr.columns if c.startswith("feature")]
expeds = sorted(tr["exped"].unique())
tr["eord"] = tr["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
dtr = tr[tr["eord"] >= 1000]
print(f"rows {len(dtr)}", flush=True)
bm = pd.read_parquet("/home/hatch/workspace/user/files/eiq_train_benchmark_models.parquet")
smap = dict(zip(bm["exped"], bm["v1_sherpa"]))
dtr = dtr.copy()
dtr["sherpa"] = dtr["exped"].map(smap)
# Per-exped orthogonalize using numpy loop
y_orth = np.empty(len(dtr), dtype=np.float64)
for exped, idx in dtr.groupby("exped", sort=False).indices.items():
    s = dtr["sherpa"].values[idx].astype(np.float64)
    y = dtr["target_everest"].values[idx].astype(np.float64)
    A = np.column_stack([np.ones(len(s)), s])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_orth[idx] = y - (coef[0] + coef[1] * s)
print(f"orth done, corr(y_orth, y)={np.corrcoef(y_orth, dtr['target_everest'].values)[0,1]:.4f}", flush=True)
Xtr = dtr[fcols].values.astype(np.float32)
Xtr[Xtr == -1] = np.nan
# median impute via pandas (simpler)
Xtr_df = pd.DataFrame(Xtr).fillna(pd.DataFrame(Xtr).median())
Xtr = Xtr_df.values.astype(np.float32)
del Xtr_df
va = pd.read_parquet("/home/hatch/workspace/user/files/eiq_validation.parquet")
Xva = pd.DataFrame(va[fcols].values.astype(np.float32)).replace(-1, np.nan)
Xva = Xva.fillna(Xva.median()).values.astype(np.float32)
dset = lgb.Dataset(Xtr, label=y_orth.astype(np.float32))
del Xtr
params = dict(objective="regression", num_leaves=31, learning_rate=0.01,
              min_child_samples=200, verbosity=-1, seed=3)
model = lgb.train(params, dset, 500)
print(f"trained ({time.time()-ts:.0f}s)", flush=True)
_fcols, _model = fcols, model
def predict(live_features):
    X = pd.DataFrame(live_features[_fcols].values.astype(np.float32)).replace(-1, np.nan)
    X = X.fillna(X.median())
    p = _model.predict(X.values).astype(np.float64)
    return pd.DataFrame({"prediction": np.clip(1.0 - np.clip(p, 0, 1), 0, 1)},
                        index=live_features.index)
p_raw = model.predict(Xva)
p_neg = np.clip(1.0 - np.clip(p_raw, 0, 1), 0, 1)
pd.DataFrame({"id": va.index.values, "prediction": p_neg}).to_csv(
    "/home/hatch/workspace/everesteer_research/final/final_orth_neg.csv", index=False)
with open("/home/hatch/workspace/everesteer_research/final/orth_neg_predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
print(f"saved ({time.time()-ts:.0f}s), mean={p_neg.mean():.4f}")
