"""Build pure LGBM with 0.1115 recipe params (different from E1 blend)."""
import pandas as pd, numpy as np, pickle, lightgbm as lgb, cloudpickle
print("Loading data...", flush=True)
train = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
val = pd.read_parquet("/home/hatch/workspace/user/files/eiq_validation.parquet")
fcols = [c for c in train.columns if c.startswith("feature")]
print(f"Features: {len(fcols)}", flush=True)
# Parse exped numbers, use 3500+ per 0.1115 recipe (recent data)
train["exped_num"] = train["exped"].str.extract(r"exped_(\d+)").astype(int)
train_recent = train[train["exped_num"] >= 3500].copy()
print(f"Train rows (exped 3500+): {len(train_recent)}", flush=True)
# Exclude 20 most bench-correlated features (from 0.1115 recipe)
excl = ["feature_Agouti","feature_Tazzarine","feature_Takerkoust","feature_Tagoudiche",
        "feature_Imenane","feature_Tafraout","feature_Tagmat","feature_Iferd",
        "feature_Ouarzazate","feature_Chefchaouen","feature_Aguelmame","feature_Tidighin",
        "feature_Ounila","feature_Azilal","feature_Elksiba","feature_Azaghar",
        "feature_Guigou","feature_Tagant","feature_Amazigh","feature_Tazouta"]
fcols_use = [c for c in fcols if c not in excl]
print(f"Features after exclusion: {len(fcols_use)}", flush=True)
# Median impute
medians = np.array([train_recent[fc].replace(-1, np.nan).median() for fc in fcols_use], dtype=np.float32)
def prep(df):
    X = df[fcols_use].values.astype(np.float32)
    X[X == -1] = np.nan
    return np.where(np.isnan(X), medians, X)
X_train = prep(train_recent)
y_train = train_recent["target_everest"].values
mask = ~np.isnan(y_train)
print("Training pure LGBM (0.1115 recipe)...", flush=True)
d = lgb.Dataset(X_train[mask], label=y_train[mask])
params = {"objective": "regression", "metric": "rmse", "verbosity": -1,
          "num_leaves": 31, "learning_rate": 0.01, "feature_fraction": 0.3,
          "bagging_fraction": 0.7, "bagging_freq": 1, "min_child_samples": 200,
          "seed": 7}
model = lgb.train(params, d, num_boost_round=2000)
print("Training done.", flush=True)
# Save model
with open("/home/hatch/workspace/everesteer_research/final/pure_lgbm.pkl", "wb") as f:
    pickle.dump({"model": model, "features": fcols_use, "medians": medians}, f)
# Validation predictions (training-direction, light neutralize)
print("Generating validation predictions...", flush=True)
X_val = prep(val)
pred = model.predict(X_val)
df = pd.DataFrame({"pred": pred, "exped": val["exped"].values}, index=val.index)
# Light neutralization: per-exped demean
df["dm"] = df.groupby("exped")["pred"].transform(lambda x: x - x.mean())
df["rank"] = df.groupby("exped")["dm"].rank(pct=True)
# Training-direction (NO negation) - for live rounds
final = df["rank"].values.clip(0, 1)
out = pd.DataFrame({"prediction": final}, index=val.index)
out.to_csv("/home/hatch/workspace/everesteer_research/final/pure_lgbm_raw.csv")
print(f"Saved: mean={out['prediction'].mean():.4f}, std={out['prediction'].std():.4f}", flush=True)
# Predict callable
def predict(live_features):
    X = live_features[fcols_use].values.astype(np.float32)
    X[X == -1] = np.nan
    X = np.where(np.isnan(X), medians, X)
    p = model.predict(X)
    d2 = pd.DataFrame({"p": p}, index=live_features.index)
    if "exped" in live_features.columns:
        d2["exped"] = live_features["exped"].values
        d2["dm"] = d2.groupby("exped")["p"].transform(lambda x: x - x.mean())
        d2["r"] = d2.groupby("exped")["dm"].rank(pct=True)
    else:
        d2["r"] = d2["p"].rank(pct=True)
    return pd.DataFrame({"prediction": d2["r"].values.clip(0, 1)}, index=live_features.index)
with open("/home/hatch/workspace/everesteer_research/final/pure_lgbm_predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
print("Saved pure_lgbm_predict.pkl", flush=True)
