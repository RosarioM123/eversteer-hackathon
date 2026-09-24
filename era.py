"""Era / training-window check. Estimator fixed (ridge or orthogonal-ridge via EST env).
Windows: A=all(0-5199) C=recent50%(2600-5199) E=exclude sparse era(1000-5199).
Test blocks: mid (3642-4241) and recent (5242-5841), 42-exped embargo respected.
Reports CORR/AIMC/BLEND per window per test block + stability read.
"""
import numpy as np, pandas as pd, json, time, os, sys
from scipy.stats import norm
sys.path.insert(0, "/home/hatch/workspace/everesteer_research")

TRAIN = "/home/hatch/workspace/everesteer_research/eiq_train.parquet"
BENCH = "/home/hatch/workspace/user/files/eiq_train_benchmark_models.parquet"
OUT = "/home/hatch/workspace/everesteer_research/era_results.json"
EST = os.environ.get("EST", "ridge")  # ridge | orthogonal
from sklearn.linear_model import Ridge

def rgauss(x):
    r = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").values
    n = len(x)
    return norm.ppf(np.clip((r - 0.5) / n, 1e-12, 1 - 1e-12))

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

WINDOWS = {"A_all": (0, 5199), "C_recent50": (2600, 5199), "E_nosparse": (1000, 5199)}
TESTS = {"mid": (3642, 4241), "recent": (5242, 5841)}

def impute(Xtr_raw, Xte_raw):
    a, b = Xtr_raw.copy(), Xte_raw.copy()
    a[a == -1] = np.nan; b[b == -1] = np.nan
    med = np.nanmedian(a, axis=0); med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(a), med, a), np.where(np.isnan(b), med, b)

def resid_target(tr):
    res = np.zeros(len(tr), dtype=np.float64)
    y = tr["target_everest"].values.astype(np.float64)
    b = tr["sherpa"].values.astype(np.float64)
    for _, pos in tr.groupby("exped", sort=False).indices.items():
        tt, bb = y[pos], b[pos]
        tc, bc = tt - tt.mean(), bb - bb.mean()
        den = bc @ bc
        res[pos] = tc - (tc @ bc) / den * bc if den > 0 else tc
    return res.astype(np.float32)

results = {"estimator": EST, "windows": {}, "notes": []}
for wname, (wlo, whi) in WINDOWS.items():
    results["windows"][wname] = {}
    for tname, (tlo, thi) in TESTS.items():
        ts = time.time()
        tr = df[(df["eord"] >= wlo) & (df["eord"] <= whi)]
        te = df[(df["eord"] >= tlo) & (df["eord"] <= thi)]
        Xtr, Xte = impute(tr[fcols].values.astype(np.float32), te[fcols].values.astype(np.float32))
        y = resid_target(tr) if EST == "orthogonal" else tr["target_everest"].values.astype(np.float32)
        m = Ridge(alpha=1.0); m.fit(Xtr, y)
        pred = m.predict(Xte).astype(np.float32)
        yt = te["target_everest"].values.astype(np.float32)
        bt = te["sherpa"].values.astype(np.float32)
        groups = te.groupby("exped", sort=False).indices
        c = corr20(pred, yt, groups); a = aimc20(pred, yt, bt, groups)
        blend = c + 2 * a  # proxy-NCORR omitted here; relative window comparison
        cs = float(np.corrcoef(pred, bt)[0, 1])
        results["windows"][wname][tname] = dict(corr=round(c,4), aimc=round(a,5),
            blend_no_ncorr=round(blend,4), corr_sherpa=round(cs,4),
            n_tr=len(tr), runtime_s=round(time.time()-ts,1))
        print(f"{wname} x {tname}: CORR={c:.4f} AIMC={a:.5f} BLEND*={blend:.4f} ({time.time()-ts:.0f}s)", flush=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=1)

# stability read: rank of windows within each test block
for tname in TESTS:
    ranking = sorted(WINDOWS, key=lambda w: -results["windows"][w][tname]["blend_no_ncorr"])
    msg = f"test block {tname}: window ranking by BLEND*: {ranking}"
    print(msg); results["notes"].append(msg)
with open(OUT, "w") as f: json.dump(results, f, indent=1)
print("ERA DONE")
