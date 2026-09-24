"""B3''': ridge on features screened for target-signal WITHOUT benchmark-signal.
Per fold (train only): |corr(feat,target)| high AND |corr(feat,sherpa)| low.
Raw target, median impute, ridge. No sherpa needed at predict time.
Same 3 folds + kernels as ladder.py.
"""
import numpy as np, pandas as pd, json, time
from scipy.stats import norm
from sklearn.linear_model import Ridge

TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
BENCH = "/home/hatch/workspace/user/files/eiq_train_benchmark_models.parquet"
OUT = "/home/hatch/workspace/everesteer_research/b3ppp_results.json"

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
def pxy(F, mask, y):
    # pooled Pearson per column, pairwise complete
    Fm = np.where(mask, np.nan, F.astype(np.float64))
    ym = y[:, None]
    ok = ~np.isnan(Fm)
    n = ok.sum(0)
    xm = np.where(ok, Fm, 0).sum(0) / np.maximum(n, 1)
    ym_ = np.where(ok, ym, 0).sum(0) / np.maximum(n, 1)
    cov = (np.where(ok, (Fm - xm) * (ym - ym_), 0)).sum(0) / np.maximum(n, 1)
    sx = np.sqrt((np.where(ok, (Fm - xm) ** 2, 0)).sum(0) / np.maximum(n, 1))
    sy = np.sqrt((np.where(ok, (ym - ym_) ** 2, 0)).sum(0) / np.maximum(n, 1))
    return np.where((sx > 0) & (sy > 0) & (n > 1000), cov / (sx * sy), 0.0)

print("loading...", flush=True)
df = pd.read_parquet(TRAIN)
bench = pd.read_parquet(BENCH)
df["sherpa"] = bench["v1_sherpa"].reindex(df.index).values
fcols = [c for c in df.columns if c.startswith("feature")]
expeds = sorted(df["exped"].unique())
df["eord"] = df["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
FOLDS = [("A_early", 0, 1999, 2042, 2641), ("B_mid", 0, 3599, 3642, 4241), ("C_recent", 0, 5199, 5242, 5841)]

def impute_cols(Xtr_raw, Xte_raw):
    a, b = Xtr_raw.copy().astype(np.float64), Xte_raw.copy().astype(np.float64)
    a[a == -1] = np.nan; b[b == -1] = np.nan
    med = np.nanmedian(a, axis=0); med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(a), med, a), np.where(np.isnan(b), med, b)

results = {}
for name, tr_lo, tr_hi, te_lo, te_hi in FOLDS:
    ts = time.time()
    tr = df[(df["eord"] >= tr_lo) & (df["eord"] <= tr_hi)]
    te = df[(df["eord"] >= te_lo) & (df["eord"] <= te_hi)]
    Ftr = tr[fcols].values.astype(np.float32)
    mtr = (Ftr == -1)
    y = tr["target_everest"].values.astype(np.float64)
    s = tr["sherpa"].values.astype(np.float64)
    ct = np.abs(pxy(Ftr, mtr, y)); cs = np.abs(pxy(Ftr, mtr, s))
    keep = (cs <= np.quantile(cs, 0.40)) & (ct >= np.quantile(ct, 0.60))
    sel = [f for f, k in zip(fcols, keep) if k]
    Xtr, Xte = impute_cols(tr[sel].values.astype(np.float32), te[sel].values.astype(np.float32))
    m = Ridge(alpha=1.0); m.fit(Xtr, y.astype(np.float32))
    pred = m.predict(Xte).astype(np.float32)
    yt = te["target_everest"].values.astype(np.float32); bt = te["sherpa"].values.astype(np.float32)
    groups = te.groupby("exped", sort=False).indices
    c = corr20(pred, yt, groups); a = aimc20(pred, yt, bt, groups)
    blend = c + 2 * a
    csh = float(np.corrcoef(pred, bt)[0, 1])
    results[name] = dict(n_sel=len(sel), corr=round(c,4), aimc=round(a,5), blend_no_ncorr=round(blend,4),
                         corr_sherpa=round(csh,4), runtime_s=round(time.time()-ts,1))
    print(f"{name}: n={len(sel)} CORR={c:.4f} AIMC={a:.5f} BLEND*={blend:.4f} corr_sherpa={csh:.4f} ({time.time()-ts:.0f}s)", flush=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=1)
print("B3PPP DONE")
