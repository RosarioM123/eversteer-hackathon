"""Regime forensics: why does E1 work on train-holdout (CORR 0.0826) but fail on
validation (official CORR -0.102)? Compare validation vs train feature regimes."""
import numpy as np, pandas as pd
tr = pd.read_parquet("/home/hatch/workspace/everesteer_research/eiq_train.parquet")
va = pd.read_parquet("/home/hatch/workspace/user/files/eiq_validation.parquet")
fcols = [c for c in tr.columns if c.startswith("feature")]

# 1. exped overlap / ordering
tr_exp = set(tr["exped"].unique()); va_exp = set(va["exped"].unique())
print("train expeds:", len(tr_exp), "val expeds:", len(va_exp),
      "overlap:", len(tr_exp & va_exp))
print("train exped range:", min(tr_exp), "-", max(tr_exp))
print("val exped range:", min(va_exp), "-", max(va_exp))

# 2. feature distribution shift: train-recent (last 500 expeds) vs validation
expeds = sorted(tr_exp)
recent = tr[tr["exped"].isin(expeds[-500:])]
def stats(d):
    X = d[fcols].values.astype(np.float64); X[X == -1] = np.nan
    return np.nanmean(X, 0), np.nanstd(X, 0), np.isnan(X).mean(0)
m_tr, s_tr, na_tr = stats(recent)
m_va, s_va, na_va = stats(va)
# standardized mean shift
shift = np.abs(m_va - m_tr) / np.where(s_tr > 0, s_tr, 1)
order = np.argsort(-shift)
print("\nTop 15 shifted features (train-recent vs val), in std units:")
for i in order[:15]:
    print(f"  {fcols[i]}: shift={shift[i]:.2f} mean {m_tr[i]:.3f}->{m_va[i]:.3f} "
          f"std {s_tr[i]:.3f}->{s_va[i]:.3f} na {na_tr[i]:.3f}->{na_va[i]:.3f}")
print(f"\nMedian abs shift: {np.median(shift):.2f} std; max: {shift.max():.2f}")
print(f"Median na-rate train: {np.median(na_tr):.3f} val: {np.median(na_va):.3f}")
# 3. target distribution in train-recent (sanity)
print("\ntarget_everest train-recent: mean=%.4f std=%.4f" % (
    recent["target_everest"].mean(), recent["target_everest"].std()))
print("rows/exped train-recent: %.1f val: %.1f" % (
    len(recent)/recent["exped"].nunique(), len(va)/va["exped"].nunique()))
