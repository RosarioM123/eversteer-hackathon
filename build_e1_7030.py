"""E1 70/30 RIDGE-HEAVY blend (training-direction, for live rounds).
0.7*ridge + 0.3*LightGBM, per-exped rank(pct), NO negation.
Memory-lean float32 pipeline (OOM fix): single float32 copy of train features,
ridge on float32, LGBM on float32 numpy with NaN. Recipe matches final_build.py
Stage 2 (Ridge alpha=1.0; LGBM 200 rounds / lr 0.05 / leaves 31 / min_child 200),
only the blend weights differ.
"""
import gc
import numpy as np, pandas as pd, cloudpickle, time
from sklearn.linear_model import Ridge
import lightgbm as lgb
ts = time.time()

TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
VAL = "/home/hatch/workspace/user/files/eiq_validation.parquet"
OUTDIR = "/home/hatch/workspace/everesteer_research/final"

print("loading train...", flush=True)
df = pd.read_parquet(TRAIN)
fcols = [c for c in df.columns if c.startswith("feature")]
expeds = sorted(df["exped"].unique())
eord = df["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
keep = (eord.values >= 1000)
del eord
Xraw = df.loc[keep, fcols].values.astype(np.float32)   # 418k x 178 float32 (~298MB)
y = df.loc[keep, "target_everest"].values.astype(np.float32)
del df
gc.collect()
print(f"train rows={Xraw.shape[0]} features={len(fcols)}", flush=True)

# median imputer (fit on train), float32 throughout
a = Xraw.copy(); a[a == -1] = np.nan
_med = np.nanmedian(a, axis=0).astype(np.float32)
_med = np.where(np.isnan(_med), 0.0, _med)
del a; gc.collect()

def _imp(X_raw32):
    b = X_raw32.astype(np.float32, copy=True)
    b[b == -1] = np.nan
    nancol = np.isnan(b)
    b[nancol] = np.take(_med, np.nonzero(nancol)[1])
    return b

_ridge = Ridge(alpha=1.0)
_ridge.fit(_imp(Xraw), y)
print(f"ridge done ({time.time()-ts:.0f}s)", flush=True)

Xn = Xraw.astype(np.float32, copy=True); Xn[Xn == -1] = np.nan
dtr = lgb.Dataset(Xn, label=y)
del Xn; gc.collect()
_lgbm = lgb.train({"objective": "regression", "num_leaves": 31, "learning_rate": 0.05,
                    "min_child_samples": 200, "verbosity": -1}, dtr, 200)
del dtr, Xraw, y; gc.collect()
print(f"lgbm done ({time.time()-ts:.0f}s)", flush=True)
_fcols = fcols

def predict(live_features):
    """E1 70/30: 0.7*ridge + 0.3*LightGBM, clipped [0,1], per-exped rank(pct), NO negation."""
    X = live_features[_fcols].values.astype(np.float32)
    p1 = _ridge.predict(_imp(X)).astype(np.float64)
    Xn2 = X.astype(np.float32, copy=False); Xn2 = np.where(Xn2 == -1, np.nan, Xn2)
    p2 = _lgbm.predict(Xn2).astype(np.float64)
    blend = np.clip(0.7 * p1 + 0.3 * p2, 0.0, 1.0)
    s = pd.Series(blend, index=live_features.index)
    if "exped" in live_features.columns:
        ranks = s.groupby(live_features["exped"].values, sort=False).rank(pct=True).values
    else:
        ranks = s.rank(pct=True).values
    out = np.clip(ranks, 0.0, 1.0)
    return pd.DataFrame({"prediction": out}, index=live_features.index)

with open(f"{OUTDIR}/e1_7030_predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
print(f"wrote e1_7030_predict.pkl ({time.time()-ts:.0f}s)", flush=True)

print("loading validation + generating predictions via callable...", flush=True)
val = pd.read_parquet(VAL)
print(f"val rows={len(val)} expeds={val['exped'].nunique()}", flush=True)
out = predict(val)
assert len(out) == len(val) == 23886, f"row count mismatch: {len(out)} vs {len(val)}"
assert out["prediction"].notna().all(), "NaNs found"
assert ((out["prediction"] >= 0) & (out["prediction"] <= 1)).all(), "out of [0,1]"
sub = pd.DataFrame({"id": val.index.values, "prediction": out["prediction"].values})
assert sub["id"].is_unique and len(sub) == len(val)
sub.to_csv(f"{OUTDIR}/e1_7030_raw.csv", index=False)
print(f"wrote e1_7030_raw.csv n={len(sub)} "
      f"mean={sub['prediction'].mean():.4f} std={sub['prediction'].std():.4f} "
      f"min={sub['prediction'].min():.4f} max={sub['prediction'].max():.4f}", flush=True)
print(f"DONE ({time.time()-ts:.0f}s)")
