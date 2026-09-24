"""Pinpoint the regime break: train ridge on windows, test CORR on subsequent
500-exped blocks within train. If CORR flips negative in late train, the break
is INSIDE train (train recent-only). If late train is still positive, the break
is between train end and validation."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
fcols = [c for c in tr.columns if c.startswith("feature")]
expeds = sorted(tr["exped"].unique())
tr["eord"] = tr["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)

def imp_fit(X):
    a = X.copy(); a[a == -1] = np.nan
    m = np.nanmedian(a, axis=0); return np.where(np.isnan(m), 0.0, m)
def imp_apply(X, med):
    a = X.astype(np.float64); a[a == -1] = np.nan
    return np.where(np.isnan(a), med, a)

# test blocks: [5000-5500], [5500-6000], [6000-6521]; train on all before block start
for t0, t1 in [(5000, 5500), (5500, 6000), (6000, 6521)]:
    dtr = tr[tr["eord"] < t0]; dte = tr[(tr["eord"] >= t0) & (tr["eord"] <= t1)]
    med = imp_fit(dtr[fcols].values.astype(np.float32))
    m = Ridge(alpha=1.0)
    m.fit(imp_apply(dtr[fcols].values.astype(np.float32), med),
          dtr["target_everest"].values.astype(np.float32))
    p = m.predict(imp_apply(dte[fcols].values.astype(np.float32), med))
    # per-exped rank corr (approx corr20)
    cs = []
    for _, pos in dte.groupby("exped", sort=False).indices.items():
        pp, tt = p[pos], dte["target_everest"].values[pos].astype(np.float64)
        if np.std(pp) > 0 and np.std(tt) > 0:
            cs.append(np.corrcoef(pd.Series(pp).rank(), pd.Series(tt).rank())[0, 1])
    print(f"train<={t0} test[{t0}-{t1}]: n_expeds={len(cs)} mean_rank_corr={np.mean(cs):.4f} "
          f"frac_pos={np.mean(np.array(cs) > 0):.2f}", flush=True)
