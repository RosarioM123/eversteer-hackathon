"""Build NEUTRALIZED E1 (training-direction) for live rounds.

E1 = 0.5*ridge + 0.5*LightGBM, same recipe as build_e1_pkl.py, but predictions
are per-exped neutralized against the top-5 benchmark-correlated features
(Agouti, Tazzarine, Takerkoust, Tagoudiche, Imenane) before ranking.
NO negation (training-direction for live rounds).

Outputs:
  final/e1_neut_raw.csv       (validation predictions)
  final/e1_neut_predict.pkl   (cloudpickled predict callable)
"""
import numpy as np, pandas as pd, pickle, cloudpickle, time
from sklearn.linear_model import Ridge
import lightgbm as lgb
ts = time.time()

TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
VAL = "/home/hatch/workspace/user/files/eiq_validation.parquet"
OUT_CSV = "/home/hatch/workspace/everesteer_research/final/e1_neut_raw.csv"
OUT_PKL = "/home/hatch/workspace/everesteer_research/final/e1_neut_predict.pkl"
NEUT_FEATS = ["feature_Agouti", "feature_Tazzarine", "feature_Takerkoust",
              "feature_Tagoudiche", "feature_Imenane"]

df = pd.read_parquet(TRAIN)
fcols = [c for c in df.columns if c.startswith("feature")]
expeds = sorted(df["exped"].unique())
df["eord"] = df["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
tr = df[df["eord"] >= 1000]
print(f"train rows={len(tr)} features={len(fcols)}", flush=True)

# median imputer fit on train (NaN median -> 0.0), same as E1
a = tr[fcols].values.astype(np.float32); a[a == -1] = np.nan
_med = np.nanmedian(a, axis=0); _med = np.where(np.isnan(_med), 0.0, _med).astype(np.float64)

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

def _e1_blend(live_features):
    """Raw E1: 0.5*ridge + 0.5*LightGBM, no clip here (rank later)."""
    X = live_features[_fcols].values.astype(np.float32)
    p1 = _ridge.predict(_imp(X)).astype(np.float64)
    p2 = _lgbm.predict(pd.DataFrame(X).replace(-1, np.nan)).astype(np.float64)
    return 0.5 * p1 + 0.5 * p2

def _neutralize_and_rank(live_features):
    """Per-exped: regress E1 blend on the 5 neutral features (median-imputed,
    with intercept), take residuals, rank residuals per exped (pct)."""
    blend = _e1_blend(live_features)
    F = _imp(live_features[NEUT_FEATS].values.astype(np.float32))  # median-imputed, no NaNs
    ones = np.ones((len(live_features), 1))
    D = np.concatenate([ones, F], axis=1)
    idx = live_features.index
    resid = np.empty(len(live_features), dtype=np.float64)
    if "exped" in live_features.columns:
        group_iter = [np.asarray(pos) for pos in
                      live_features.groupby("exped", sort=False).indices.values()]
    else:
        group_iter = [np.arange(len(live_features))]
    for pos in group_iter:
        b, *_ = np.linalg.lstsq(D[pos], blend[pos], rcond=None)
        resid[pos] = blend[pos] - D[pos] @ b
    r = pd.Series(resid, index=idx)
    gr = live_features["exped"] if "exped" in live_features.columns else pd.Series(np.zeros(len(idx), dtype=int), index=idx)
    ranked = r.groupby(gr, sort=False).rank(pct=True).values
    return np.clip(ranked, 0.0, 1.0)

# --- validation CSV ---
val = pd.read_parquet(VAL)
print(f"val rows={len(val)} expeds={val['exped'].nunique()}", flush=True)
preds = _neutralize_and_rank(val)
out = pd.DataFrame({"prediction": preds}, index=val.index)
out.to_csv(OUT_CSV)
print(f"CSV: mean={out['prediction'].mean():.4f} std={out['prediction'].std():.4f}", flush=True)

# --- pickled callable (self-contained, same logic) ---
def predict(live_features):
    """Neutralized E1: 0.5*ridge + 0.5*LGBM, per-exped neutralize vs 5 top
    benchmark-correlated features, per-exped rank residuals, training-direction."""
    X = live_features[_fcols].values.astype(np.float32)
    b = X.astype(np.float64); b[b == -1] = np.nan
    Xi = np.where(np.isnan(b), _med, b)
    p1 = _ridge.predict(Xi).astype(np.float64)
    p2 = _lgbm.predict(pd.DataFrame(X).replace(-1, np.nan)).astype(np.float64)
    blend = 0.5 * p1 + 0.5 * p2
    F = Xi[:, [_fcols.index(c) for c in NEUT_FEATS]]
    D = np.concatenate([np.ones((len(live_features), 1)), F], axis=1)
    resid = np.empty(len(live_features), dtype=np.float64)
    if "exped" in live_features.columns:
        group_iter = [np.asarray(pos) for pos in
                      live_features.groupby("exped", sort=False).indices.values()]
    else:
        group_iter = [np.arange(len(live_features))]
    for pos in group_iter:
        bb, *_ = np.linalg.lstsq(D[pos], blend[pos], rcond=None)
        resid[pos] = blend[pos] - D[pos] @ bb
    r = pd.Series(resid, index=live_features.index)
    gr = live_features["exped"] if "exped" in live_features.columns else pd.Series(np.zeros(len(r), dtype=int), index=r.index)
    ranked = r.groupby(gr, sort=False).rank(pct=True).values
    return pd.DataFrame({"prediction": np.clip(ranked, 0.0, 1.0)}, index=live_features.index)

with open(OUT_PKL, "wb") as f:
    cloudpickle.dump(predict, f)
print(f"wrote {OUT_PKL} ({time.time()-ts:.0f}s)", flush=True)
