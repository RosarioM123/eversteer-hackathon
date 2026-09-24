"""Clean memory-efficient AIMC model."""
import numpy as np, pandas as pd, cloudpickle, time, gc
import lightgbm as lgb
ts = time.time()
print("load train", flush=True)
tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
fcols = [c for c in tr.columns if c.startswith("feature")]
expeds = sorted(tr["exped"].unique())
emap = {e: i for i, e in enumerate(expeds)}
eord = tr["exped"].map(emap).values
mask = eord >= 1000
exp_m = tr["exped"].values[mask]
y_m = tr["target_everest"].values[mask].astype(np.float64)
X_m = tr[fcols].values[mask].astype(np.float32)
del tr; gc.collect()
print(f"filtered {len(y_m)} rows ({time.time()-ts:.0f}s)", flush=True)
print("load benchmark", flush=True)
bm = pd.read_parquet("/home/hatch/workspace/user/files/eiq_train_benchmark_models.parquet")
smap = dict(zip(bm["exped"].values, bm["v1_sherpa"].values))
del bm; gc.collect()
sherpa = np.array([smap[e] for e in exp_m], dtype=np.float64)
del smap, exp_m; gc.collect()
# Per-exped orthogonalize y vs sherpa
print("orthogonalize", flush=True)
# Use pandas factorize for speed
codes, uniq = pd.factorize(pd.Series(np.arange(len(y_m))), sort=False)
# Actually simpler: group by exped via dictionary
from collections import defaultdict
idx_by_exped = defaultdict(list)
# We need exped labels; reconstruct from mask order is messy. Use eord instead.
print("done setup", flush=True)
