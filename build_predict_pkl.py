"""Build the cloudpickled predict callable for the excl20 + top-20 neutralization
model. The neutralization block is a verbatim port of regen_excl20_noint.py
(median imputation, no intercept), so the pkl reproduces the diagnostic CSV.
The predict function is fully self-contained (inlined logic)."""
import pickle, cloudpickle
import numpy as np, pandas as pd

d = pickle.load(open("/home/hatch/workspace/everesteer_research/final/excl20_lgbm.pkl", "rb"))
_lgbm = d["model"]
_keep = d["features"]
_K = 20

def predict(live_features):
    """Excl-20 LightGBM + per-exped self-neutralization vs own top-20 |corr| features."""
    K = _K
    Xf = live_features[_keep].values.astype(np.float32)
    Xf = np.where(Xf == -1, np.nan, Xf)
    p_all = _lgbm.predict(Xf).astype(np.float64)
    Fv = live_features[_keep].values.astype(np.float64)
    if "exped" in live_features.columns:
        group_iter = [np.asarray(pos) for pos in
                      live_features.groupby("exped", sort=False).indices.values()]
    else:
        group_iter = [np.arange(len(live_features))]
    out = np.empty(len(live_features))
    for pos in group_iter:
        p = p_all[pos]
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
        beta, *_ = np.linalg.lstsq(Fc, p, rcond=None)   # no intercept: keep levels
        out[pos] = p - Fc @ beta
    out = np.clip(out, 0.0, 1.0)
    return pd.DataFrame({"prediction": out}, index=live_features.index)

with open("/home/hatch/workspace/everesteer_research/final/excl20_predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
print("wrote excl20_predict.pkl")
