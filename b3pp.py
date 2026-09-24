"""B3'': ridge on POOLED benchmark-residualized target.
One global beta from all train rows (stable) vs B3's noisy per-exped betas.
Needs sherpa only at TRAIN time -> deployable without live benchmark values.
Same 3 folds + kernels as ladder.py.
"""
import numpy as np, pandas as pd, json, time
from scipy.stats import norm
from sklearn.linear_model import Ridge

TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
BENCH = "/home/hatch/workspace/user/files/eiq_train_benchmark_models.parquet"
OUT = "/home/hatch/workspace/everesteer_research/b3pp_results.json"

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
def corr20(pred, tgt, groups):
    return float(np.mean([per_exped_corr(pred[pos], tgt[pos]) for pos in groups.values()]))
def aimc20(pred, tgt, benchv, groups):
    vals = []
    for pos in groups.values():
        p, t, b = pred[pos], tgt[pos], benchv[pos]
        if np.std(p) == 0 or np.std(b) == 0 or np.std(t) == 0:
            vals.append(0.0); continue
        g, gb = rgauss(p), rgauss(b)
        den = gb @ gb
        r = g - (g @ gb) / den * gb if den > 0 else g
        tc = t - t.mean()
        vals.append(float(np.mean((r - r.mean()) * tc)))
    return float(np.mean(vals))

print("loading...", flush=True)
df = pd.read_parquet(TRAIN)
bench = pd.read_parquet(BENCH)
df["sherpa"] = bench["v1_sherpa"].reindex(df.index).values
fcols = [c for c in df.columns if c.startswith("feature")]
expeds = sorted(df["exped"].unique())
df["eord"] = df["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
FOLDS = [("A_early", 0, 1999, 2042, 2641), ("B_mid", 0, 3599, 3642, 4241), ("C_recent", 0, 5199, 5242, 5841)]

def impute(Xtr_raw, Xte_raw):
    a, b = Xtr_raw.copy(), Xte_raw.copy()
    a[a == -1] = np.nan; b[b == -1] = np.nan
    med = np.nanmedian(a, axis=0); med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(a), med, a), np.where(np.isnan(b), med, b)

results = {}
for name, tr_lo, tr_hi, te_lo, te_hi in FOLDS:
    ts = time.time()
    tr = df[(df["eord"] >= tr_lo) & (df["eord"] <= tr_hi)]
    te = df[(df["eord"] >= te_lo) & (df["eord"] <= te_hi)]
    y = tr["target_everest"].values.astype(np.float64)
    s = tr["sherpa"].values.astype(np.float64)
    yc, sc = y - y.mean(), s - s.mean()
    beta = (yc @ sc) / (sc @ sc)
    y_resid = (yc - beta * sc).astype(np.float32)
    Xtr, Xte = impute(tr[fcols].values.astype(np.float32), te[fcols].values.astype(np.float32))
    m = Ridge(alpha=1.0); m.fit(Xtr, y_resid)
    pred = m.predict(Xte).astype(np.float32)
    yt = te["target_everest"].values.astype(np.float32)
    bt = te["sherpa"].values.astype(np.float32)
    groups = te.groupby("exped", sort=False).indices
    c = corr20(pred, yt, groups); a = aimc20(pred, yt, bt, groups)
    blend = c + 2 * a
    cs = float(np.corrcoef(pred, bt)[0, 1])
    results[name] = dict(corr=round(c,4), aimc=round(a,5), blend_no_ncorr=round(blend,4),
                         corr_sherpa=round(cs,4), pooled_beta=round(float(beta),4),
                         runtime_s=round(time.time()-ts,1))
    print(f"{name}: CORR={c:.4f} AIMC={a:.5f} BLEND*={blend:.4f} corr_sherpa={cs:.4f} beta={beta:.4f} ({time.time()-ts:.0f}s)", flush=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=1)
print("B3PP DONE")
