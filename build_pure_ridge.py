"""Build PURE RIDGE model (training-direction, for live rounds)."""
import pandas as pd, numpy as np, pickle, cloudpickle
from sklearn.linear_model import Ridge

print("Loading data...", flush=True)
train = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
val = pd.read_parquet("/home/hatch/workspace/user/files/eiq_validation.parquet")
print(f"train: {train.shape}, val: {val.shape}", flush=True)

fcols = [c for c in train.columns if c.startswith("feature")]
print(f"Features: {len(fcols)}", flush=True)

# Exped 1000+ (drop sparse early era)
exped_num = train["exped"].astype(str).str.extract(r"exped_(\d+)")[0].astype(int)
train_recent = train[exped_num >= 1000].copy()
print(f"Train rows (exped 1000+): {len(train_recent)}", flush=True)

# Median impute (-1 -> NaN -> median)
medians = np.array(
    [train_recent[fc].replace(-1, np.nan).median() for fc in fcols], dtype=np.float32
)

def prep(df):
    X = df[fcols].values.astype(np.float32)
    X[X == -1] = np.nan
    return np.where(np.isnan(X), medians, X)

X_train = prep(train_recent)
y_train = train_recent["target_everest"].values.astype(np.float64)
mask = np.isfinite(y_train)
print(f"Rows with valid target: {mask.sum()}", flush=True)

print("Training Ridge...", flush=True)
model = Ridge(alpha=1.0)
model.fit(X_train[mask], y_train[mask])
print("Training done.", flush=True)

print("Generating validation predictions (training-direction)...", flush=True)
X_val = prep(val)
pred = model.predict(X_val)
df = pd.DataFrame({"pred": pred, "exped": val["exped"].values}, index=val.index)
# Per-exped rank (pct) - no negation
df["rank"] = df.groupby("exped")["pred"].rank(pct=True)
final = df["rank"].values.clip(0, 1)
out = pd.DataFrame({"prediction": final}, index=val.index)
out.to_csv("/home/hatch/workspace/everesteer_research/final/pure_ridge_raw.csv")
print(f"Saved CSV: mean={out['prediction'].mean():.4f}, std={out['prediction'].std():.4f}", flush=True)

# Predict callable
def predict(live_features):
    X = live_features[fcols].values.astype(np.float32)
    X[X == -1] = np.nan
    X = np.where(np.isnan(X), medians, X)
    p = model.predict(X)
    d2 = pd.DataFrame({"p": p}, index=live_features.index)
    if "exped" in live_features.columns:
        d2["exped"] = live_features["exped"].values
        d2["r"] = d2.groupby("exped")["p"].rank(pct=True)
    else:
        d2["r"] = d2["p"].rank(pct=True)
    return pd.DataFrame({"prediction": d2["r"].values.clip(0, 1)}, index=live_features.index)

with open("/home/hatch/workspace/everesteer_research/final/pure_ridge_predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
print("Saved pure_ridge_predict.pkl", flush=True)

# Validation checks
assert len(out) == 23886, f"row count {len(out)} != 23886"
assert out["prediction"].notna().all(), "NaNs found"
assert ((out["prediction"] >= 0) & (out["prediction"] <= 1)).all(), "out of [0,1]"
print("Validation OK: 23,886 rows, no NaNs, all in [0,1]", flush=True)
