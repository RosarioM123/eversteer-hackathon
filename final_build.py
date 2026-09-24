"""FINAL: holdout confirmation (untouched 680 expeds) then full-train build.
Stage 1: train B1+B2 on orders 1000-5799, test E1 blend on 5842-6521 (42-exped embargo).
Stage 2 (only if sane): retrain on 1000-6521, predict validation, write 3 CSVs.
"""
import numpy as np, pandas as pd, json, time
from scipy.stats import norm
from sklearn.linear_model import Ridge
import lightgbm as lgb

TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
VAL = "/home/hatch/workspace/user/files/eiq_validation.parquet"
OUTDIR = "/home/hatch/workspace/everesteer_research/final"

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
def corr20(p, t, groups): return float(np.mean([per_exped_corr(p[i], t[i]) for i in groups.values()]))

print("loading...", flush=True)
df = pd.read_parquet(TRAIN)
fcols = [c for c in df.columns if c.startswith("feature")]
expeds = sorted(df["exped"].unique())
df["eord"] = df["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
val = pd.read_parquet(VAL)
val_ids = val.index.values

def impute_fit(Xtr_raw):
    a = Xtr_raw.copy(); a[a == -1] = np.nan
    med = np.nanmedian(a, axis=0); med = np.where(np.isnan(med), 0.0, med)
    return med
def impute_apply(X_raw, med):
    a = X_raw.copy().astype(np.float64); a[a == -1] = np.nan
    return np.where(np.isnan(a), med, a)
def train_predict(tr, teX_raw, med):
    y = tr["target_everest"].values.astype(np.float32)
    Xtr = impute_apply(tr[fcols].values.astype(np.float32), med)
    m1 = Ridge(alpha=1.0); m1.fit(Xtr, y)
    dtr = lgb.Dataset(pd.DataFrame(tr[fcols].values.astype(np.float32)).replace(-1, np.nan), label=y)
    m2 = lgb.train({"objective": "regression", "num_leaves": 31, "learning_rate": 0.05,
                    "min_child_samples": 200, "verbosity": -1}, dtr, 200)
    p1 = m1.predict(impute_apply(teX_raw, med)).astype(np.float32)
    p2 = m2.predict(pd.DataFrame(teX_raw).replace(-1, np.nan)).astype(np.float32)
    return p1, p2, m2

# ---- Stage 1: holdout confirmation ----
tr = df[(df["eord"] >= 1000) & (df["eord"] <= 5799)]
te = df[(df["eord"] >= 5842) & (df["eord"] <= 6521)]
print(f"stage1 train {len(tr)} test {len(te)} ({te['exped'].nunique()} expeds)", flush=True)
med = impute_fit(tr[fcols].values.astype(np.float32))
p1, p2, m2 = train_predict(tr, te[fcols].values.astype(np.float32), med)
pb = np.clip(0.5 * p1 + 0.5 * p2, 0, 1)
groups = te.groupby("exped", sort=False).indices
yt = te["target_everest"].values.astype(np.float32)
c = corr20(pb, yt, groups)
imp = pd.Series(m2.feature_importance(importance_type="gain"), index=fcols).sort_values(ascending=False)
top10 = float(imp.head(10).sum() / imp.sum())
print(f"HOLDOUT E1: CORR={c:.4f} pcorr_b1b2={np.corrcoef(p1,p2)[0,1]:.4f} top10_gain_share={top10:.3f}", flush=True)
print("pred dist: mean=%.4f std=%.4f min=%.4f max=%.4f" % (pb.mean(), pb.std(), pb.min(), pb.max()), flush=True)

# ---- Stage 2: full build ----
import os; os.makedirs(OUTDIR, exist_ok=True)
trf = df[df["eord"] >= 1000]
print(f"stage2 train {len(trf)} -> validation {len(val)}", flush=True)
medf = impute_fit(trf[fcols].values.astype(np.float32))
p1v, p2v, _ = train_predict(trf, val[fcols].values.astype(np.float32), medf)
blend = np.clip(0.5 * p1v + 0.5 * p2v, 0, 1)
for name, arr in [("final_blend", blend), ("final_b1_ridge", np.clip(p1v, 0, 1)), ("final_b2_lgbm", np.clip(p2v, 0, 1))]:
    out = pd.DataFrame({"id": val_ids, "prediction": arr.astype(np.float64)})
    assert out["prediction"].notna().all() and ((out["prediction"] >= 0) & (out["prediction"] <= 1)).all()
    assert out["id"].is_unique and len(out) == len(val)
    out.to_csv(f"{OUTDIR}/{name}.csv", index=False)
    print(f"wrote {name}.csv n={len(out)} mean={arr.mean():.4f} std={arr.std():.4f}", flush=True)
print("FINAL BUILD DONE")
