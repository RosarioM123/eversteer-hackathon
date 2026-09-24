"""Find features with orthogonal alpha: high corr with (target - benchmark)."""
import numpy as np, pandas as pd, time
ts = time.time()
tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
fcols = [c for c in tr.columns if c.startswith("feature")]
expeds = sorted(tr["exped"].unique())
tr["eord"] = tr["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
dtr = tr[tr["eord"] >= 1000]
print(f"rows {len(dtr)}", flush=True)
bm = pd.read_parquet("/home/hatch/workspace/user/files/eiq_train_benchmark_models.parquet")
smap = dict(zip(bm["exped"], bm["v1_sherpa"]))
dtr = dtr.copy()
dtr["sherpa"] = dtr["exped"].map(smap)
# Residual: target - sherpa (what benchmark misses)
# Per-exped: residual = target - (a + b*sherpa) to remove linear fit
resid = np.empty(len(dtr))
for exped, idx in dtr.groupby("exped", sort=False).indices.items():
    s = dtr["sherpa"].values[idx].astype(np.float64)
    y = dtr["target_everest"].values[idx].astype(np.float64)
    A = np.column_stack([np.ones(len(s)), s])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid[idx] = y - (coef[0] + coef[1] * s)
dtr["_resid"] = resid
print(f"resid std: {resid.std():.4f}", flush=True)
# For each feature, corr with resid (per-exped mean, like CORR20)
fcorrs = {}
X = dtr[fcols].values.astype(np.float32)
X[X == -1] = np.nan
for j, fc in enumerate(fcols):
    cs = []
    xj = X[:, j]
    for _, idx in dtr.groupby("exped", sort=False).indices.items():
        x = xj[idx]; r = resid[idx]
        m = ~(np.isnan(x) | np.isnan(r))
        if m.sum() > 10 and np.std(x[m]) > 0 and np.std(r[m]) > 0:
            cs.append(np.corrcoef(x[m], r[m])[0, 1])
    fcorrs[fc] = np.mean(cs) if cs else 0
    if (j+1) % 30 == 0:
        print(f"  {j+1}/{len(fcols)}", flush=True)
# Top features by |corr|
ranked = sorted(fcorrs.items(), key=lambda x: -abs(x[1]))
print("Top 20 orthogonal-alpha features:", flush=True)
for fc, c in ranked[:20]:
    print(f"  {fc}: {c:.4f}", flush=True)
# Save
pd.DataFrame(ranked, columns=["feature", "orth_corr"]).to_csv(
    "/home/hatch/workspace/everesteer_research/feature_orth_corr.csv", index=False)
print(f"done ({time.time()-ts:.0f}s)")
