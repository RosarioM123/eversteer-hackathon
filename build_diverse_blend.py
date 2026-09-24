"""Build diverse-target blend, neutralize, negate, validate on practice board."""
import pandas as pd, numpy as np, pickle, time
ts = time.time()
print("Loading diverse-target models...", flush=True)
with open("/home/hatch/workspace/everesteer_research/final/diverse_target_models.pkl", "rb") as f:
    saved = pickle.load(f)
models, targets, fcols, medians = saved["models"], saved["targets"], saved["features"], saved["medians"]
print(f"Loaded {len(models)} models", flush=True)
# Load validation data
val = pd.read_parquet("/home/hatch/workspace/user/files/eiq_validation.parquet")
print(f"Val: {len(val)} rows", flush=True)
Xv = val[fcols].values.astype(np.float32)
Xv[Xv == -1] = np.nan
Xv = np.where(np.isnan(Xv), medians, Xv)
# Predict with each model
preds = {}
for t in targets:
    p = models[t].predict(Xv)
    preds[t] = p
    print(f"  {t}: mean={p.mean():.4f} std={p.std():.4f}", flush=True)
# Blend (equal weight)
blend = np.mean(list(preds.values()), axis=0)
print(f"Blend: mean={blend.mean():.4f} std={blend.std():.4f}", flush=True)
# Feature-neutralize: find top-20 features by |corr| with blend
print("Finding top exposure features...", flush=True)
corrs = []
for i, fc in enumerate(fcols):
    xv_col = Xv[:, i]
    c = np.corrcoef(blend, xv_col)[0, 1]
    if not np.isnan(c):
        corrs.append((abs(c), fc, i))
corrs.sort(reverse=True)
top20 = corrs[:20]
print(f"Top 5: {[(fc, f'{c:.3f}') for c, fc, i in top20[:5]]}", flush=True)
# Per-exped neutralize
val = val.copy()
val["blend"] = blend
def neutralize_group(g):
    idx = [i for c, fc, i in top20]
    F = Xv[g.index.values, :][:, idx]  # careful: need positional index
    return g
# Simpler: per-exped linear residualization
print("Neutralizing per-exped...", flush=True)
val["exped_code"] = pd.Categorical(val["exped"]).codes
neutralized = np.zeros(len(val))
for ecode in np.unique(val["exped_code"]):
    mask = val["exped_code"] == ecode
    # Get positional indices
    pos = np.where(mask)[0]
    y = blend[pos]
    F = Xv[pos][:, [i for c, fc, i in top20]]
    # Add intercept
    F1 = np.column_stack([np.ones(len(pos)), F])
    # Least squares
    try:
        beta, *_ = np.linalg.lstsq(F1, y, rcond=None)
        resid = y - F1 @ beta
    except:
        resid = y - y.mean()
    neutralized[pos] = resid
print(f"Neutralized: mean={neutralized.mean():.4f} std={neutralized.std():.4f}", flush=True)
# Negate and scale to [0,1]
# Rank-normalize per exped then negate
val["neut"] = neutralized
val["rank"] = val.groupby("exped")["neut"].rank(pct=True)
final_pred = 1.0 - val["rank"].values
print(f"Final: mean={final_pred.mean():.4f} std={final_pred.std():.4f} min={final_pred.min():.4f} max={final_pred.max():.4f}", flush=True)
# Save
out = pd.DataFrame({"id": val.index if "id" in val.columns else val["exped"].astype(str) + "_" + val.groupby("exped").cumcount().astype(str), "prediction": final_pred})
# Use the validation index as id
val_reset = pd.read_parquet("/home/hatch/workspace/user/files/eiq_validation.parquet")
out = pd.DataFrame({"id": val_reset.index, "prediction": final_pred})
out.to_csv("/home/hatch/workspace/everesteer_research/final/diverse_blend_neg.csv", index=False)
print(f"Saved diverse_blend_neg.csv ({time.time()-ts:.0f}s)", flush=True)
# Also save the pkl for event lane
with open("/home/hatch/workspace/everesteer_research/final/diverse_blend_neg_predict.pkl", "wb") as f:
    # Save a function that reproduces this
    pickle.dump({"targets": targets, "features": fcols, "medians": medians,
                 "top20_idx": [i for c, fc, i in top20]}, f)
print("Done", flush=True)
