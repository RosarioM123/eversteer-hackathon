"""Build E1 30/70 LGBM-heavy blend (0.3*ridge + 0.7*LightGBM), training-direction (NO negation) for live rounds."""
import pandas as pd, numpy as np, pickle, lightgbm as lgb, cloudpickle
from sklearn.linear_model import Ridge

print("Loading data...", flush=True)
train = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
val = pd.read_parquet("/home/hatch/workspace/user/files/eiq_validation.parquet")
fcols = [c for c in train.columns if c.startswith("feature")]
assert len(fcols) == 178, f"expected 178 features, got {len(fcols)}"
print(f"Features: {len(fcols)}", flush=True)

# Exped 1000+ filter
train["exped_num"] = train["exped"].str.extract(r"exped_(\d+)").astype(int)
tr = train[train["exped_num"] >= 1000].copy()
print(f"Train rows (exped 1000+): {len(tr)}", flush=True)

# Median impute (-1 -> NaN -> median), medians computed on training set
medians = np.array([tr[fc].replace(-1, np.nan).median() for fc in fcols], dtype=np.float32)
def prep(df):
    X = df[fcols].values.astype(np.float32)
    X[X == -1] = np.nan
    return np.where(np.isnan(X), medians, X)

X_train = prep(tr)
y_train = tr["target_everest"].values
mask = ~np.isnan(y_train)
X_train, y_train = X_train[mask], y_train[mask]
print(f"Labeled rows: {len(y_train)}", flush=True)

print("Training Ridge...", flush=True)
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)

print("Training LightGBM (500 rounds, lr 0.05)...", flush=True)
d = lgb.Dataset(X_train, label=y_train)
params = {"objective": "regression", "metric": "rmse", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31, "seed": 42}
lgbm = lgb.train(params, d, num_boost_round=500)
print("Training done.", flush=True)

print("Predicting validation (30/70 blend)...", flush=True)
X_val = prep(val)
ridge_pred = ridge.predict(X_val)
lgbm_pred = lgbm.predict(X_val)
blend = 0.3 * ridge_pred + 0.7 * lgbm_pred

df = pd.DataFrame({"blend": blend, "exped": val["exped"].values}, index=val.index)
# Per-exped rank (pct), NO negation -> training-direction
df["rank"] = df.groupby("exped")["blend"].rank(pct=True)
final = df["rank"].values.clip(0, 1)

out = pd.DataFrame({"prediction": final}, index=val.index)
out.to_csv("/home/hatch/workspace/everesteer_research/final/e1_3070_raw.csv")

# Validation
assert len(out) == 23886, f"row count {len(out)}"
assert out["prediction"].notna().all(), "NaNs found"
assert ((out["prediction"] >= 0) & (out["prediction"] <= 1)).all(), "out of [0,1]"
print(f"Saved: mean={out['prediction'].mean():.4f}, std={out['prediction'].std():.4f}", flush=True)

# Predict callable for live data
def predict(live_features):
    X = live_features[fcols].values.astype(np.float32)
    X[X == -1] = np.nan
    X = np.where(np.isnan(X), medians, X)
    rp = ridge.predict(X)
    lp = lgbm.predict(X)
    b = 0.3 * rp + 0.7 * lp
    d2 = pd.DataFrame({"b": b}, index=live_features.index)
    if "exped" in live_features.columns:
        d2["exped"] = live_features["exped"].values
        d2["r"] = d2.groupby("exped")["b"].rank(pct=True)
    else:
        d2["r"] = d2["b"].rank(pct=True)
    return pd.DataFrame({"prediction": d2["r"].values.clip(0, 1)}, index=live_features.index)

with open("/home/hatch/workspace/everesteer_research/final/e1_3070_predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
print("Saved e1_3070_predict.pkl", flush=True)
