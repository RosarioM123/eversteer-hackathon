"""Winners' pipeline: diverse-target ensemble + neutralize + negate."""
import pandas as pd, numpy as np, pickle, time, lightgbm as lgb
ts = time.time()
targets = ["target_Tiskiouine", "target_Saghro", "target_Gourza", "target_Ayachi", "target_Azurki"]
print(f"Training on {len(targets)} diverse targets...", flush=True)
tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
fcols = [c for c in tr.columns if c.startswith("feature")]
expeds = sorted(tr["exped"].unique())
tr["eord"] = tr["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
dtr = tr[tr["eord"] >= 1000].copy()
X = dtr[fcols].values.astype(np.float32)
X[X == -1] = np.nan
medians = np.nanmedian(X, axis=0)
X = np.where(np.isnan(X), medians, X)
print(f"X: {X.shape}", flush=True)
models = {}
for t in targets:
    y = dtr[t].values.astype(np.float32)
    # Drop NaN targets
    mask = ~np.isnan(y)
    print(f"  {t}: {mask.sum()} rows", flush=True)
    train_data = lgb.Dataset(X[mask], label=y[mask])
    params = {"objective": "regression", "metric": "rmse", "learning_rate": 0.01,
              "num_leaves": 31, "min_child_samples": 200, "verbose": -1, "seed": 7}
    model = lgb.train(params, train_data, num_boost_round=300)
    models[t] = model
    print(f"  {t} done ({time.time()-ts:.0f}s)", flush=True)
# Save models and medians
with open("/home/hatch/workspace/everesteer_research/final/diverse_target_models.pkl", "wb") as f:
    pickle.dump({"models": models, "targets": targets, "features": fcols, "medians": medians}, f)
print(f"Saved ({time.time()-ts:.0f}s)", flush=True)
