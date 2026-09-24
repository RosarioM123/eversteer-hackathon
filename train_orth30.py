"""Train aggressive orth-feature LGBM: top-30 benchmark-orthogonal features, negated."""
import pandas as pd, numpy as np, pickle, time, lightgbm as lgb
ts = time.time()
print("Loading train...", flush=True)
tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
# Top 30 orth features
orth = pd.read_csv("/home/hatch/workspace/everesteer_research/feature_orth_corr.csv")
top30 = orth.head(30)["feature"].tolist()
print(f"Top30: {top30[:5]}...", flush=True)
expeds = sorted(tr["exped"].unique())
tr["eord"] = tr["exped"].map({e: i for i, e in enumerate(expeds)}).astype(np.int32)
dtr = tr[tr["eord"] >= 1000].copy()
print(f"Train rows: {len(dtr)}", flush=True)
# Prep features
X = dtr[top30].values.astype(np.float32)
X[X == -1] = np.nan
medians = np.nanmedian(X, axis=0)
X = np.where(np.isnan(X), medians, X)
y = dtr["target_everest"].values.astype(np.float32)
print("Training LGBM...", flush=True)
train_data = lgb.Dataset(X, label=y)
params = {"objective": "regression", "metric": "rmse", "learning_rate": 0.01,
          "num_leaves": 31, "min_child_samples": 200, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 7}
model = lgb.train(params, train_data, num_boost_round=500)
print(f"Trained ({time.time()-ts:.0f}s)", flush=True)
# Build negated predict fn
def predict(live_df):
    Xl = live_df[top30].values.astype(np.float32)
    Xl[Xl == -1] = np.nan
    Xl = np.where(np.isnan(Xl), medians, Xl)
    raw = model.predict(Xl)
    neg = 1.0 - np.clip(raw, 0, 1)
    return pd.DataFrame({"prediction": np.clip(neg, 0, 1)}, index=live_df.index)
with open("/home/hatch/workspace/everesteer_research/final/orth30_neg_predict.pkl", "wb") as f:
    pickle.dump(predict, f)
print("Saved orth30_neg_predict.pkl", flush=True)
