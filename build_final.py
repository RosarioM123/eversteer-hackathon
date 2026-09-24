"""FINAL: train best old-regime LGBM (lr 0.01, 500 rounds) on train 1000-6521,
predict validation, negate for new regime, build pkl."""
import numpy as np, pandas as pd, cloudpickle, time
import lightgbm as lgb
ts = time.time()
tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
va = pd.read_parquet("/home/hatch/workspace/user/files/eiq_validation.parquet")
fcols = [c for c in tr.columns if c.startswith("feature")]
expeds = sorted(tr["exped"].unique())
tr["eord"] = tr["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
dtr = tr[tr["eord"] >= 1000]
print(f"train {len(dtr)}", flush=True)
ytr = dtr["target_everest"].values.astype(np.float32)
Xtr = pd.DataFrame(dtr[fcols].values.astype(np.float32)).replace(-1, np.nan)
Xva = pd.DataFrame(va[fcols].values.astype(np.float32)).replace(-1, np.nan)
dset = lgb.Dataset(Xtr, label=ytr)
params = dict(objective="regression", num_leaves=31, learning_rate=0.01,
              min_child_samples=200, verbosity=-1, seed=3)
_model = lgb.train(params, dset, 500)
print(f"trained ({time.time()-ts:.0f}s)", flush=True)
_fcols = fcols
def predict(live_features):
    """Negated low-lr LGBM for inverted regime."""
    X = pd.DataFrame(live_features[_fcols].values.astype(np.float32)).replace(-1, np.nan)
    p = _model.predict(X).astype(np.float64)
    neg = 1.0 - np.clip(p, 0, 1)  # negate to [0,1]
    # Actually: negate raw then clip. Simpler: 1 - clip(p,0,1)
    return pd.DataFrame({"prediction": np.clip(neg, 0, 1)}, index=live_features.index)
# predict validation for diagnostic CSV
p_raw = _model.predict(Xva)
p_neg = np.clip(1.0 - np.clip(p_raw, 0, 1), 0, 1)
pd.DataFrame({"id": va.index.values, "prediction": p_neg}).to_csv(
    "/home/hatch/workspace/everesteer_research/final/final_lgbm_neg.csv", index=False)
with open("/home/hatch/workspace/everesteer_research/final/lgbm_neg_predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
print(f"wrote CSV + pkl ({time.time()-ts:.0f}s), pred mean={p_neg.mean():.4f} std={p_neg.std():.4f}")
