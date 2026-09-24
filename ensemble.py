"""Two final candidate tests on the 3 ladder folds.
E1: equal-weight blend of B1 (ridge) + B2 (lightgbm), retrained per fold.
E2: lightgbm with sample weights emphasizing benchmark-blind rows
    w = 1 + 3*|target - sherpa|  (sherpa used at TRAIN time only -> deployable).
"""
import numpy as np, pandas as pd, json, time
from scipy.stats import norm
from sklearn.linear_model import Ridge
import lightgbm as lgb

TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
BENCH = "/home/hatch/workspace/user/files/eiq_train_benchmark_models.parquet"
OUT = "/home/hatch/workspace/everesteer_research/ensemble_results.json"

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
def score(pred, yt, bt, groups):
    c = corr20(pred, yt, groups); a = aimc20(pred, yt, bt, groups)
    return c, a, c + 2 * a

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
    y = tr["target_everest"].values.astype(np.float32)
    s = tr["sherpa"].values.astype(np.float32)
    groups = te.groupby("exped", sort=False).indices
    yt = te["target_everest"].values.astype(np.float32); bt = te["sherpa"].values.astype(np.float32)

    # B1 ridge
    Xtr, Xte = impute(tr[fcols].values.astype(np.float32), te[fcols].values.astype(np.float32))
    m1 = Ridge(alpha=1.0); m1.fit(Xtr, y); p1 = m1.predict(Xte).astype(np.float32)
    # B2 lightgbm (native NaN)
    dtr = lgb.Dataset(pd.DataFrame(tr[fcols].values.astype(np.float32)).replace(-1, np.nan), label=y)
    m2 = lgb.train({"objective": "regression", "num_leaves": 31, "learning_rate": 0.05,
                    "min_child_samples": 200, "verbosity": -1}, dtr, 200)
    p2 = m2.predict(pd.DataFrame(te[fcols].values.astype(np.float32)).replace(-1, np.nan)).astype(np.float32)
    # E1 blend
    pb = (0.5 * p1 + 0.5 * p2).astype(np.float32)
    c, a, bl = score(pb, yt, bt, groups)
    pc = float(np.corrcoef(p1, p2)[0, 1])
    # E2 weighted lgbm
    w = (1.0 + 3.0 * np.abs(y - s)).astype(np.float64)
    dtrw = lgb.Dataset(pd.DataFrame(tr[fcols].values.astype(np.float32)).replace(-1, np.nan), label=y, weight=w)
    m3 = lgb.train({"objective": "regression", "num_leaves": 31, "learning_rate": 0.05,
                    "min_child_samples": 200, "verbosity": -1}, dtrw, 200)
    p3 = m3.predict(pd.DataFrame(te[fcols].values.astype(np.float32)).replace(-1, np.nan)).astype(np.float32)
    c3, a3, bl3 = score(p3, yt, bt, groups)
    cs3 = float(np.corrcoef(p3, bt)[0, 1])
    results[name] = dict(E1_blend=dict(corr=round(c,4), aimc=round(a,5), blend=round(bl,4), pcorr_b1b2=round(pc,4)),
                         E2_weighted=dict(corr=round(c3,4), aimc=round(a3,5), blend=round(bl3,4), corr_sherpa=round(cs3,4)))
    print(f"{name}: E1 CORR={c:.4f} AIMC={a:.5f} BLEND={bl:.4f} | E2 CORR={c3:.4f} AIMC={a3:.5f} BLEND={bl3:.4f} cs={cs3:.3f} ({time.time()-ts:.0f}s)", flush=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=1)
print("ENSEMBLE DONE")
