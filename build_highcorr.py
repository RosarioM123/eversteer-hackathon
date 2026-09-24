"""Build high-correlation target blend (opposite of failed diverse-target)."""
import pandas as pd, numpy as np, pickle, lightgbm as lgb
print("Loading validation data...", flush=True)
val = pd.read_parquet("/home/hatch/workspace/user/files/eiq_validation.parquet")
train = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
# Use high-corr targets (from earlier scan)
targets = ["target_Tougroute", "target_Ghat", "target_Meltsene"]  # 0.84, 0.60, 0.60 corr
print(f"Targets: {targets}", flush=True)
fcols = [c for c in train.columns if c.startswith("feature")]
print(f"Features: {len(fcols)}", flush=True)
# Train on exped 1000+ (recent, per era check) - parse number from exped_XXXX string
train["exped_num"] = train["exped"].str.extract(r"exped_(\d+)").astype(int)
train_recent = train[train["exped_num"] >= 1000].copy()
print(f"Train rows (exped 1000+): {len(train_recent)}", flush=True)
# Median impute -1
medians = {}
for fc in fcols:
    vals = train_recent[fc].replace(-1, np.nan)
    medians[fc] = vals.median()
medians_arr = np.array([medians[fc] for fc in fcols], dtype=np.float32)
def prep(df):
    X = df[fcols].values.astype(np.float32)
    X[X == -1] = np.nan
    X = np.where(np.isnan(X), medians_arr, X)
    return X
X_train = prep(train_recent)
models = {}
for t in targets:
    print(f"Training {t}...", flush=True)
    y = train_recent[t].values
    # Remove NaN targets
    mask = ~np.isnan(y)
    d = lgb.Dataset(X_train[mask], label=y[mask])
    params = {"objective": "regression", "metric": "rmse", "verbosity": -1,
              "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.8}
    m = lgb.train(params, d, num_boost_round=300)
    models[t] = m
    print(f"  Done {t}", flush=True)
# Save
with open("/home/hatch/workspace/everesteer_research/final/highcorr_models.pkl", "wb") as f:
    pickle.dump({"models": models, "targets": targets, "features": fcols, "medians": medians_arr}, f)
print("Saved highcorr_models.pkl", flush=True)
# Build predictions for validation
print("Generating validation predictions...", flush=True)
X_val = prep(val)
preds = []
for t in targets:
    p = models[t].predict(X_val)
    preds.append(p)
blend = np.mean(preds, axis=0)
# Light neutralization: per-exped demean (not aggressive top-20)
df = pd.DataFrame({"blend": blend, "exped": val["exped"].values}, index=val.index)
df["demeaned"] = df.groupby("exped")["blend"].transform(lambda x: x - x.mean())
# Rank per exped and negate
df["rank"] = df.groupby("exped")["demeaned"].rank(pct=True)
df["final"] = 1.0 - df["rank"]
df["final"] = df["final"].clip(0, 1)
out = pd.DataFrame({"prediction": df["final"].values}, index=val.index)
out.to_csv("/home/hatch/workspace/everesteer_research/final/highcorr_blend_neg.csv")
print(f"Saved predictions: mean={out['prediction'].mean():.4f}, std={out['prediction'].std():.4f}", flush=True)
# Create predict callable
import cloudpickle
def predict(live_features):
    X = live_features[fcols].values.astype(np.float32)
    X[X == -1] = np.nan
    X = np.where(np.isnan(X), medians_arr, X)
    ps = [models[t].predict(X) for t in targets]
    b = np.mean(ps, axis=0)
    d2 = pd.DataFrame({"b": b}, index=live_features.index)
    if "exped" in live_features.columns:
        d2["exped"] = live_features["exped"].values
        d2["dm"] = d2.groupby("exped")["b"].transform(lambda x: x - x.mean())
        d2["r"] = d2.groupby("exped")["dm"].rank(pct=True)
    else:
        d2["r"] = d2["b"].rank(pct=True)
    final = (1.0 - d2["r"].values).clip(0, 1)
    return pd.DataFrame({"prediction": final}, index=live_features.index)
with open("/home/hatch/workspace/everesteer_research/final/highcorr_predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
print("Saved highcorr_predict.pkl", flush=True)
