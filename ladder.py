"""Credit-budget ladder: B0/B1/B2/B3 on real train data.
3 chronological folds, 42-exped embargo, -1 masked as missing.
Kernels: corr20 (server: signed 1.5 power on BOTH sides), aimc20, proxy-ncorr20.
BLEND = CORR + 2*AIMC + proxy-NCORR. Final 680 expeds held out untouched.
"""
import numpy as np, pandas as pd, json, time, sys
from scipy.stats import norm

TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
BENCH = "/home/hatch/workspace/user/files/eiq_train_benchmark_models.parquet"
OUT = "/home/hatch/workspace/everesteer_research/ladder_results.json"
LOG = "/home/hatch/workspace/everesteer_research/ladder_log.txt"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f: f.write(msg + "\n")

t0 = time.time()
log("loading train...")
df = pd.read_parquet(TRAIN)
bench = pd.read_parquet(BENCH)
assert (bench.index == df.index).all() or set(bench.index) == set(df.index), "id mismatch"
df["sherpa"] = bench["v1_sherpa"].reindex(df.index).values
assert df["sherpa"].notna().all(), "sherpa join produced NaNs"
fcols = [c for c in df.columns if c.startswith("feature")]
assert len(fcols) == 178, len(fcols)
tcols = [c for c in df.columns if c.startswith("target")]
log(f"train {df.shape}, features {len(fcols)}, sherpa joined. load {time.time()-t0:.0f}s")

expeds = sorted(df["exped"].unique())
assert len(expeds) == 6522
order = {e: i for i, e in enumerate(expeds)}
df["eord"] = df["exped"].map(order).astype(np.int32)

# (name, train_lo, train_hi, test_lo, test_hi) inclusive exped orders; final 680 expeds untouched
FOLDS = [
    ("A_early", 0, 1999, 2042, 2641),
    ("B_mid",   0, 3599, 3642, 4241),
    ("C_recent",0, 5199, 5242, 5841),
]

# ---------- scoring kernels ----------
def rgauss(x):
    r = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").values
    n = len(x)
    u = np.clip((r - 0.5) / n, 1e-12, 1 - 1e-12)
    return norm.ppf(u)

def spow(x, p=1.5):
    return np.sign(x) * np.abs(x) ** p

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

def proxy_ncorr20(pred, tgt, feat_dict, groups):
    # neutralize gaussianized preds vs gaussianized top-20 features + intercept, per exped, then corr20
    vals = []
    for pos in groups.values():
        p = pred[pos]; t = tgt[pos]
        G = np.column_stack([rgauss(feat_dict[k][pos]) for k in feat_dict])
        G = np.column_stack([np.ones(len(p)), G])
        gp = rgauss(p)
        beta, *_ = np.linalg.lstsq(G, gp, rcond=None)
        resid = gp - G @ beta
        vals.append(per_exped_corr(resid, t))
    return float(np.mean(vals))

def topk_features(Xtr_imp, ytr, k=20):
    X = Xtr_imp.astype(np.float64); y = ytr.astype(np.float64)
    Xc = X - X.mean(0); yc = y - y.mean()
    den = np.sqrt((Xc**2).sum(0) * (yc**2).sum())
    corr = np.where(den > 0, (Xc * yc[:, None]).sum(0) / den, 0.0)
    return np.argsort(-np.abs(corr))[:k]

# ---------- data prep per fold ----------
def prep(fold):
    name, tr_lo, tr_hi, te_lo, te_hi = fold
    tr = df[(df["eord"] >= tr_lo) & (df["eord"] <= tr_hi)]
    te = df[(df["eord"] >= te_lo) & (df["eord"] <= te_hi)]
    ytr = tr["target_everest"].values.astype(np.float32)
    yte = te["target_everest"].values.astype(np.float32)
    bte = te["sherpa"].values.astype(np.float32)
    btr = tr["sherpa"].values.astype(np.float32)
    Xtr_raw = tr[fcols].values.astype(np.float32)
    Xte_raw = te[fcols].values.astype(np.float32)
    groups = te.groupby("exped", sort=False).indices  # positional within te
    return dict(name=name, ytr=ytr, yte=yte, btr=btr, bte=bte,
                Xtr_raw=Xtr_raw, Xte_raw=Xte_raw, groups=groups,
                n_tr=len(tr), n_te=len(te))

def impute_median(Xtr_raw, Xte_raw):
    Xn_tr = Xtr_raw.copy(); Xn_te = Xte_raw.copy()
    Xn_tr[Xn_tr == -1] = np.nan; Xn_te[Xn_te == -1] = np.nan
    med = np.nanmedian(Xn_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(Xn_tr), med, Xn_tr), np.where(np.isnan(Xn_te), med, Xn_te)

# ---------- baselines ----------
from sklearn.linear_model import Ridge

def run_B0(P):
    return np.full(P["n_te"], 0.5, dtype=np.float32)

def run_B1(P):
    Xtr, Xte = impute_median(P["Xtr_raw"], P["Xte_raw"])
    m = Ridge(alpha=1.0)
    m.fit(Xtr, P["ytr"])
    return m.predict(Xte).astype(np.float32)

def run_B2(P):
    import lightgbm as lgb
    Xtr = P["Xtr_raw"].copy(); Xte = P["Xte_raw"].copy()
    Xtr[Xtr == -1] = np.nan; Xte[Xte == -1] = np.nan
    dtr = lgb.Dataset(Xtr, label=P["ytr"])
    params = dict(objective="regression", num_leaves=31, learning_rate=0.05,
                  min_child_samples=200, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    m = lgb.train(params, dtr, num_boost_round=200)
    return m.predict(Xte).astype(np.float32)

def run_B3(P):
    # residualize TARGET vs sherpa per exped (train), then ridge on median-imputed features
    tr_exp = df[(df["eord"] >= 0)]  # placeholder, not used
    yresid = np.zeros_like(P["ytr"])
    # need train exped groups: rebuild quickly
    return None  # implemented in driver where train frame is available

def residualize_target(tr_df):
    res = np.zeros(len(tr_df), dtype=np.float64)
    y = tr_df["target_everest"].values.astype(np.float64)
    b = tr_df["sherpa"].values.astype(np.float64)
    for _, pos in tr_df.groupby("exped", sort=False).indices.items():
        tt, bb = y[pos], b[pos]
        tc, bc = tt - tt.mean(), bb - bb.mean()
        den = bc @ bc
        res[pos] = tc - (tc @ bc) / den * bc if den > 0 else tc
    return res.astype(np.float32)

def run_B3_on(tr_df, P):
    yresid = residualize_target(tr_df)
    Xtr, Xte = impute_median(P["Xtr_raw"], P["Xte_raw"])
    m = Ridge(alpha=1.0)
    m.fit(Xtr, yresid)
    return m.predict(Xte).astype(np.float32)

BASELINES = [("B0", run_B0), ("B1", run_B1), ("B2", run_B2)]

results = {"folds": {f[0]: {"train": [f[1], f[2]], "test": [f[3], f[4]]} for f in FOLDS},
           "runs": {}, "pred_corr": {}, "notes": []}
preds = {}  # (bname, fold) -> array

for fold in FOLDS:
    name = fold[0]
    log(f"--- fold {name} ---")
    P = prep(fold)
    log(f"train rows {P['n_tr']}, test rows {P['n_te']}, test expeds {len(P['groups'])}")
    # top-20 features for proxy NCORR (from B1-style imputed train)
    Xtr_imp, _ = impute_median(P["Xtr_raw"], P["Xte_raw"])
    tk = topk_features(Xtr_imp, P["ytr"], 20)
    feat_dict = {fcols[i]: Xtr_imp[:, i] for i in tk}  # placeholder; rebuilt per test below
    Xte_imp = impute_median(P["Xtr_raw"], P["Xte_raw"])[1]
    feat_te = {fcols[i]: Xte_imp[:, i] for i in tk}
    results["runs"][name] = {}
    for bname, fn in BASELINES + [("B3", None)]:
        ts = time.time()
        if bname == "B3":
            tr_df = df[(df["eord"] >= fold[1]) & (df["eord"] <= fold[2])]
            pred = run_B3_on(tr_df, P)
        else:
            pred = fn(P)
        rt = time.time() - ts
        c = corr20(pred, P["yte"], P["groups"])
        a = aimc20(pred, P["yte"], P["bte"], P["groups"])
        pn = proxy_ncorr20(pred, P["yte"], feat_te, P["groups"])
        blend = c + 2 * a + pn
        cs = float(np.corrcoef(pred, P["bte"])[0, 1]) if np.std(pred) > 0 else 0.0
        results["runs"][name][bname] = dict(corr=round(c, 4), aimc=round(a, 5),
            proxy_ncorr=round(pn, 4), blend=round(blend, 4),
            corr_sherpa=round(cs, 4), runtime_s=round(rt, 1), n_test=P["n_te"])
        preds[(bname, name)] = pred
        log(f"{bname}: CORR={c:.4f} AIMC={a:.5f} pNCORR={pn:.4f} BLEND={blend:.4f} corr_sherpa={cs:.4f} ({rt:.0f}s)")
    # pairwise pred correlations on this fold
    bn = ["B0", "B1", "B2", "B3"]
    pc = {}
    for i in range(len(bn)):
        for j in range(i + 1, len(bn)):
            pi, pj = preds[(bn[i], name)], preds[(bn[j], name)]
            v = float(np.corrcoef(pi, pj)[0, 1]) if np.std(pi) > 0 and np.std(pj) > 0 else 0.0
            pc[f"{bn[i]}_vs_{bn[j]}"] = round(v, 4)
    results["pred_corr"][name] = pc
    log(f"pred_corr {name}: {pc}")
    with open(OUT, "w") as f: json.dump(results, f, indent=1)

results["notes"].append("final holdout exped orders 5842-6521 untouched")
with open(OUT, "w") as f: json.dump(results, f, indent=1)
log(f"DONE total {time.time()-t0:.0f}s")
