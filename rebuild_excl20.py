"""Rebuild Claude Code's excl_top20 (0.1115) recipe on this machine.
- LightGBM (their params, seed 7) on exped_3500-6524, 158 features (20 excluded)
- Predict validation; per-exped neutralize vs own top-20 |corr| among the 158
- Offline CORR/AIMC vs their reported (0.043, 0.015); save model + predictions
"""
import numpy as np, pandas as pd, json, time, pickle
from scipy.stats import norm
import lightgbm as lgb

EXCL = ["feature_Agouti","feature_Tazzarine","feature_Takerkoust","feature_Tagoudiche",
"feature_Imenane","feature_Tafraout","feature_Tagmat","feature_Iferd","feature_Ouarzazate",
"feature_Chefchaouen","feature_Aguelmame","feature_Tidighin","feature_Ounila","feature_Azilal",
"feature_Elksiba","feature_Azaghar","feature_Guigou","feature_Tagant","feature_Amazigh","feature_Tazouta"]
TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
VAL = "/home/hatch/workspace/user/files/eiq_validation.parquet"
MODEL_OUT = "/home/hatch/workspace/everesteer_research/final/excl20_lgbm.pkl"
CSV_OUT = "/home/hatch/workspace/everesteer_research/final/final_excl20_neut20.csv"
K = 20
ts = time.time()

def rgauss(x):
    r = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").values
    return norm.ppf(np.clip((r - 0.5) / len(x), 1e-12, 1 - 1e-12))
def spow(x, p=1.5): return np.sign(x) * np.abs(x) ** p
def per_exped_corr(p, t):
    if np.std(p) == 0 or np.std(t) == 0: return 0.0
    g, tc = rgauss(p), t - t.mean()
    gp, tp = spow(g), spow(tc)
    if np.std(gp) == 0 or np.std(tp) == 0: return 0.0
    return float(np.corrcoef(gp, tp)[0, 1])

print("loading...", flush=True)
tr = pd.read_parquet(TRAIN)
va = pd.read_parquet(VAL)
bench = pd.read_parquet("/home/hatch/workspace/user/files/eiq_train_benchmark_models.parquet")
fcols = [c for c in tr.columns if c.startswith("feature")]
missing = [c for c in EXCL if c not in fcols]
assert not missing, f"excluded features missing from train: {missing}"
keep = [c for c in fcols if c not in EXCL]
print(f"features: {len(fcols)} total, {len(keep)} kept; val expeds: {va['exped'].nunique()} "
      f"({va['exped'].min()}..{va['exped'].max()})", flush=True)
trf = tr[(tr["exped"] >= "exped_3500") & (tr["exped"] <= "exped_6524")].reset_index(drop=True)
print(f"train window rows: {len(trf)} ({trf['exped'].min()}..{trf['exped'].max()})", flush=True)
y = trf["target_everest"].values.astype(np.float32)
Xtr = trf[keep].values.astype(np.float32)
Xva = va[keep].values.astype(np.float32)
Xtr = np.where(Xtr == -1, np.nan, Xtr); Xva = np.where(Xva == -1, np.nan, Xva)

reg = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.01, num_leaves=31,
    colsample_bytree=0.3, subsample=0.7, subsample_freq=1,
    min_child_samples=200, random_state=7, verbose=-1)
reg.fit(Xtr, y)
print(f"trained ({time.time()-ts:.0f}s)", flush=True)
with open(MODEL_OUT, "wb") as f: pickle.dump({"model": reg, "features": keep, "excluded": EXCL}, f)
pred = reg.predict(Xva).astype(np.float64)

# per-exped self-neutralization vs own top-20 |corr| among the 158 (no targets)
Fv = va[keep].values.astype(np.float64)
groups = va.groupby("exped", sort=False).indices
res = np.empty_like(pred)
for pos in groups.values():
    p = pred[pos]
    F = np.where(Fv[pos] == -1, np.nan, Fv[pos])
    ok = ~np.isnan(F)
    pv = p[:, None]; n = ok.sum(0)
    fm = np.where(ok, F, 0).sum(0) / np.maximum(n, 1)
    pm = np.where(ok, pv, 0).sum(0) / np.maximum(n, 1)
    cov = (np.where(ok, (F - fm) * (pv - pm), 0)).sum(0) / np.maximum(n, 1)
    sf = np.sqrt((np.where(ok, (F - fm) ** 2, 0)).sum(0) / np.maximum(n, 1))
    sp_ = np.sqrt((np.where(ok, (pv - pm) ** 2, 0)).sum(0) / np.maximum(n, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.abs(cov / (sf * sp_))
    c = np.where((sf > 0) & (sp_ > 0) & (n > 10), c, -1)
    topk = np.argsort(c)[-K:]
    Fc = F[:, topk]
    colmed = np.nanmedian(Fc, axis=0); colmed = np.where(np.isnan(colmed), 0, colmed)
    Fc = np.where(np.isnan(Fc), colmed, Fc)
    X = np.column_stack([np.ones(len(p)), Fc])
    beta, *_ = np.linalg.lstsq(X, p, rcond=None)
    res[pos] = p - X @ beta
res = np.clip(res, 0, 1)

# offline verification vs their reported CORR 0.043 / AIMC 0.015
yt = va["target_everest"].values.astype(np.float32)
print(f"validation targets: all blank={bool((yt == 0).all())}", flush=True)
# NOTE: validation targets are blank server-side; offline CORR/AIMC cannot be computed.
# Diagnostic: prediction distribution + benchmark correlation on train-window tail.
out = pd.DataFrame({"id": va.index.values, "prediction": res.astype(np.float32)})
assert out["id"].is_unique and np.isfinite(out["prediction"]).all()
out.to_csv(CSV_OUT, index=False)
print(f"wrote {CSV_OUT} rows={len(out)} mean={res.mean():.4f} std={res.std():.4f} "
      f"range=({res.min():.4f},{res.max():.4f}) ({time.time()-ts:.0f}s)", flush=True)
print("REBUILD DONE")
