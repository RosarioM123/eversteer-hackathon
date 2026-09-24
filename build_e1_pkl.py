"""Retrain E1 (B1 ridge + B2 LightGBM) on train orders 1000+ (same recipe as
final_build.py Stage 2) and build the cloudpickled predict callable."""
import numpy as np, pandas as pd, pickle, cloudpickle, time
from sklearn.linear_model import Ridge
import lightgbm as lgb
ts = time.time()

TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
df = pd.read_parquet(TRAIN)
fcols = [c for c in df.columns if c.startswith("feature")]
expeds = sorted(df["exped"].unique())
df["eord"] = df["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
tr = df[df["eord"] >= 1000]
print(f"train rows={len(tr)} features={len(fcols)}", flush=True)

# median imputer (fit on train)
a = tr[fcols].values.astype(np.float32); a[a == -1] = np.nan
_med = np.nanmedian(a, axis=0); _med = np.where(np.isnan(_med), 0.0, _med)

def _imp(X_raw):
    b = X_raw.astype(np.float64); b[b == -1] = np.nan
    return np.where(np.isnan(b), _med, b)

y = tr["target_everest"].values.astype(np.float32)
_ridge = Ridge(alpha=1.0)
_ridge.fit(_imp(tr[fcols].values.astype(np.float32)), y)
print(f"ridge done ({time.time()-ts:.0f}s)", flush=True)

dtr = lgb.Dataset(pd.DataFrame(tr[fcols].values.astype(np.float32)).replace(-1, np.nan), label=y)
_lgbm = lgb.train({"objective": "regression", "num_leaves": 31, "learning_rate": 0.05,
                    "min_child_samples": 200, "verbosity": -1}, dtr, 200)
print(f"lgbm done ({time.time()-ts:.0f}s)", flush=True)
_fcols = fcols

def predict(live_features):
    """E1: 0.5*ridge + 0.5*LightGBM, clipped [0,1]."""
    X = live_features[_fcols].values.astype(np.float32)
    p1 = _ridge.predict(_imp(X)).astype(np.float64)
    p2 = _lgbm.predict(pd.DataFrame(X).replace(-1, np.nan)).astype(np.float64)
    blend = np.clip(0.5 * p1 + 0.5 * p2, 0.0, 1.0)
    return pd.DataFrame({"prediction": blend}, index=live_features.index)

with open("/home/hatch/workspace/everesteer_research/final/e1_predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
print(f"wrote e1_predict.pkl ({time.time()-ts:.0f}s)")
