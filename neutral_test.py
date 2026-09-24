"""Replicate Claude Code's self-neutralization finding on embargoed folds.
Per test exped: rank features by |corr(pred, feature)|, residualize predictions
vs top-k (with intercept), sweep k in {0,10,20,30}. Score E1 blend.
k is chosen from FOLDS (not the practice board).
"""
import numpy as np, pandas as pd, json, time
from scipy.stats import norm
from sklearn.linear_model import Ridge
import lightgbm as lgb

TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
OUT = "/home/hatch/workspace/everesteer_research/neutral_results.json"

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
def aimc20(p, t, b, groups):
    v = []
    for i in groups.values():
        pp, tt, bb = p[i], t[i], b[i]
        if np.std(pp) == 0 or np.std(bb) == 0 or np.std(tt) == 0: v.append(0.0); continue
        g, gb = rgauss(pp), rgauss(bb)
        d = gb @ gb
        r = g - (g @ gb) / d * gb if d > 0 else g
        tc = tt - tt.mean()
        v.append(float(np.mean((r - r.mean()) * tc)))
    return float(np.mean(v))

print("loading...", flush=True)
df = pd.read_parquet(TRAIN)
bench = pd.read_parquet("/home/hatch/workspace/user/files/eiq_train_benchmark_models.parquet")
df["sherpa"] = bench["v1_sherpa"].reindex(df.index).values
fcols = [c for c in df.columns if c.startswith("feature")]
expeds = sorted(df["exped"].unique())
df["eord"] = df["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
FOLDS = [("A_early", 0, 1999, 2042, 2641), ("B_mid", 0, 3599, 3642, 4241), ("C_recent", 0, 5199, 5242, 5841)]
KS = [0, 10, 20, 30]

def impute(Xtr_raw, Xte_raw):
    a, b = Xtr_raw.copy(), Xte_raw.copy()
    a[a == -1] = np.nan; b[b == -1] = np.nan
    med = np.nanmedian(a, axis=0); med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(a), med, a), np.where(np.isnan(b), med, b)

def neutralize_block(pred, Fdf, groups, fcols, k):
    res = np.empty_like(pred)
    for pos in groups.values():
        p = pred[pos].astype(np.float64)
        F = Fdf.iloc[pos][fcols].values.astype(np.float64)
        F[F == -1] = np.nan
        ok = ~np.isnan(F)
        # |corr| per feature, pairwise complete
        pv = p[:, None]
        n = ok.sum(0)
        fm = np.where(ok, F, 0).sum(0) / np.maximum(n, 1)
        pm = np.where(ok, pv, 0).sum(0) / np.maximum(n, 1)
        cov = (np.where(ok, (F - fm) * (pv - pm), 0)).sum(0) / np.maximum(n, 1)
        sf = np.sqrt((np.where(ok, (F - fm) ** 2, 0)).sum(0) / np.maximum(n, 1))
        sp_ = np.sqrt((np.where(ok, (pv - pm) ** 2, 0)).sum(0) / np.maximum(n, 1))
        c = np.where((sf > 0) & (sp_ > 0) & (n > 10), np.abs(cov / (sf * sp_)), -1)
        topk = np.argsort(c)[-k:]
        Fc = F[:, topk]
        colmed = np.nanmedian(Fc, axis=0); colmed = np.where(np.isnan(colmed), 0, colmed)
        Fc = np.where(np.isnan(Fc), colmed, Fc)
        X = np.column_stack([np.ones(len(p)), Fc])
        beta, *_ = np.linalg.lstsq(X, p, rcond=None)
        res[pos] = (p - X @ beta).astype(np.float32)
    return res

results = {}
for name, tr_lo, tr_hi, te_lo, te_hi in FOLDS:
    ts = time.time()
    tr = df[(df["eord"] >= tr_lo) & (df["eord"] <= tr_hi)]
    te = df[(df["eord"] >= te_lo) & (df["eord"] <= te_hi)]
    y = tr["target_everest"].values.astype(np.float32)
    groups = te.groupby("exped", sort=False).indices
    yt = te["target_everest"].values.astype(np.float32); bt = te["sherpa"].values.astype(np.float32)
    Xtr, Xte = impute(tr[fcols].values.astype(np.float32), te[fcols].values.astype(np.float32))
    m1 = Ridge(alpha=1.0); m1.fit(Xtr, y); p1 = m1.predict(Xte).astype(np.float32)
    dtr = lgb.Dataset(pd.DataFrame(tr[fcols].values.astype(np.float32)).replace(-1, np.nan), label=y)
    m2 = lgb.train({"objective": "regression", "num_leaves": 31, "learning_rate": 0.05,
                    "min_child_samples": 200, "verbosity": -1}, dtr, 200)
    p2 = m2.predict(pd.DataFrame(te[fcols].values.astype(np.float32)).replace(-1, np.nan)).astype(np.float32)
    pb = np.clip(0.5 * p1 + 0.5 * p2, 0, 1)
    Fte = te[fcols]
    row = {}
    for k in KS:
        pn = neutralize_block(pb, Fte, groups, fcols, k) if k else pb
        c = corr20(pn, yt, groups); a = aimc20(pn, yt, bt, groups)
        row[f"k{k}"] = dict(corr=round(c, 4), aimc=round(a, 5), blend=round(c + 2 * a, 4))
    results[name] = row
    print(f"{name}: " + " | ".join(f"k{k}={row[f'k{k}']['blend']:.4f}" for k in KS) + f" ({time.time()-ts:.0f}s)", flush=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=1)
print("NEUTRAL DONE")
