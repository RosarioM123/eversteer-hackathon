"""Build neutralized variant: E1 trained on 1000-6521, predict validation,
per-exped residualize vs own top-20 |corr| features (k=20 = Claude Code board peak).
Diagnostic pair to final_blend.csv (raw)."""
import numpy as np, pandas as pd, json, time
from sklearn.linear_model import Ridge
import lightgbm as lgb

TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
VAL = "/home/hatch/workspace/user/files/eiq_validation.parquet"
OUT = "/home/hatch/workspace/everesteer_research/final/final_neutralized_k20.csv"
K = 20
ts = time.time()
print("loading...", flush=True)
tr = pd.read_parquet(TRAIN)
va = pd.read_parquet(VAL)
fcols = [c for c in tr.columns if c.startswith("feature")]
expeds = sorted(tr["exped"].unique())
tr["eord"] = tr["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
tr = tr[tr["eord"] >= 1000].reset_index(drop=True)
y = tr["target_everest"].values.astype(np.float32)
Xtr_raw = tr[fcols].values.astype(np.float32)
Xva_raw = va[fcols].values.astype(np.float32)
med = np.nanmedian(np.where(Xtr_raw == -1, np.nan, Xtr_raw), axis=0)
med = np.where(np.isnan(med), 0.0, med)
Xtr = np.where(Xtr_raw == -1, med, Xtr_raw)
Xva = np.where(Xva_raw == -1, med, Xva_raw)
print(f"train {Xtr.shape}, val {Xva.shape}", flush=True)
m1 = Ridge(alpha=1.0); m1.fit(Xtr, y); p1 = m1.predict(Xva).astype(np.float32)
print("ridge done", flush=True)
dtr = lgb.Dataset(pd.DataFrame(np.where(Xtr_raw == -1, np.nan, Xtr_raw)), label=y)
m2 = lgb.train({"objective": "regression", "num_leaves": 31, "learning_rate": 0.05,
                "min_child_samples": 200, "verbosity": -1}, dtr, 200)
p2 = m2.predict(pd.DataFrame(np.where(Xva_raw == -1, np.nan, Xva_raw))).astype(np.float32)
print("lgbm done", flush=True)
pb = 0.5 * p1 + 0.5 * p2

# per-exped self-neutralization, k=20, no targets used
res = np.empty_like(pb)
groups = va.groupby("exped", sort=False).indices
Fv = va[fcols].values.astype(np.float64)
for pos in groups.values():
    p = pb[pos].astype(np.float64)
    F = Fv[pos]; F = np.where(F == -1, np.nan, F)
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
out = pd.DataFrame({"id": va["id"].values, "prediction": res.astype(np.float32)})
assert out["id"].is_unique and out["prediction"].notna().all() and np.isfinite(out["prediction"]).all()
out.to_csv(OUT, index=False)
print(f"wrote {OUT} rows={len(out)} mean={res.mean():.4f} std={res.std():.4f} ({time.time()-ts:.0f}s)")
print("NEUTRALIZED BUILD DONE")
