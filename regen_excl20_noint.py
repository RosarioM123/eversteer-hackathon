"""Regenerate excl20 validation predictions: neutralize WITHOUT intercept (keep levels),
then clip [0,1] (uploader-accepted format, barely binds). Ranks identical to raw residuals."""
import numpy as np, pandas as pd, pickle, time
ts = time.time()
K = 20
d = pickle.load(open("/home/hatch/workspace/everesteer_research/final/excl20_lgbm.pkl", "rb"))
model, keep = d["model"], d["features"]
va = pd.read_parquet("/home/hatch/workspace/user/files/eiq_validation.parquet")
Xva = va[keep].values.astype(np.float32)
Xva = np.where(Xva == -1, np.nan, Xva)
pred = model.predict(Xva).astype(np.float64)
print(f"raw pred mean={pred.mean():.4f} std={pred.std():.4f}", flush=True)

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
    beta, *_ = np.linalg.lstsq(Fc, p, rcond=None)   # no intercept: keep levels
    res[pos] = p - Fc @ beta
res = np.clip(res, 0, 1)
frac_at_zero = (res == 0).mean()
out = pd.DataFrame({"id": va.index.values, "prediction": res.astype(np.float32)})
assert out["id"].is_unique and np.isfinite(out["prediction"]).all()
out.to_csv("/home/hatch/workspace/everesteer_research/final/final_excl20_neut20.csv", index=False)
print(f"neutralized mean={res.mean():.4f} std={res.std():.4f} range=({res.min():.4f},{res.max():.4f}) "
      f"frac_at_0={frac_at_zero:.4f} ({time.time()-ts:.0f}s)", flush=True)
print("REGEN DONE")
