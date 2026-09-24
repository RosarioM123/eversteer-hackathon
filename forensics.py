"""
Data forensics for the Everesteer NYC labeled split.
Dataset-agnostic: pass file paths and (once known) column names via CLI args.
With no column names given, it still reports schema, shape, dtypes,
missingness, duplicates, and per-period structure.

Usage:
    python forensics.py --train train.csv [--target TARGET --time EXPED --id ID]
                        [--live live.csv] [--benchmark BENCH]

Output: printed report + forensics_report.json
No modeling. No submissions. Read-only.
"""
import argparse
import json
import sys

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    sys.exit(f"missing dependency: {e} (need pandas, numpy)")


def summarize(df, name):
    out = {"name": name, "rows": int(len(df)), "cols": int(df.shape[1])}
    out["columns"] = list(df.columns)
    out["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    miss = df.isna().sum()
    out["missing_count"] = {c: int(v) for c, v in miss[miss > 0].items()}
    out["missing_fraction"] = {c: round(float(v) / len(df), 4)
                               for c, v in miss[miss > 0].items()}
    # sentinel check: -1 / 0 heavy columns are often missingness sentinels
    num = df.select_dtypes(include=[np.number])
    out["neg1_fraction"] = {c: round(float((num[c] == -1).mean()), 4)
                            for c in num.columns
                            if float((num[c] == -1).mean()) > 0.01}
    out["duplicate_rows"] = int(df.duplicated().sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--live", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--time", default=None, help="exped/period column")
    ap.add_argument("--id", default=None)
    ap.add_argument("--benchmark", default=None)
    args = ap.parse_args()

    report = {}
    train = pd.read_csv(args.train)
    report["train"] = summarize(train, "train")

    # ID uniqueness
    if args.id and args.id in train.columns:
        dup = train[args.id].duplicated().sum()
        report["id_duplicates"] = int(dup)
        report["id_unique"] = int(train[args.id].nunique())

    # Time structure
    if args.time and args.time in train.columns:
        t = train[args.time]
        periods = t.dropna().unique()
        report["n_periods"] = int(len(periods))
        per = train.groupby(args.time).size()
        report["rows_per_period"] = {
            "min": int(per.min()), "max": int(per.max()),
            "mean": round(float(per.mean()), 1)}
        report["period_sorted"] = bool(
            list(t.drop_duplicates()) == sorted(t.drop_duplicates()))
        # leakage heuristic: does row order within file correlate with target?
        if args.target and args.target in train.columns:
            y = train[args.target].values
            order = np.arange(len(y))
            c = np.corrcoef(order, np.nan_to_num(y, nan=np.nanmean(y)))[0, 1]
            report["row_order_target_corr"] = round(float(c), 4)

    # Target forensics
    if args.target and args.target in train.columns:
        y = train[args.target]
        report["target"] = {
            "dtype": str(y.dtype),
            "n_unique": int(y.nunique()),
            "missing": int(y.isna().sum()),
            "mean": round(float(np.nanmean(y)), 6),
            "std": round(float(np.nanstd(y)), 6),
            "min": round(float(np.nanmin(y)), 6),
            "max": round(float(np.nanmax(y)), 6),
        }
        if args.time and args.time in train.columns:
            # target mean/std drift per period -> regime check
            g = train.groupby(args.time)[args.target].agg(["mean", "std", "count"])
            report["target_drift"] = {
                "period_mean_min": round(float(g["mean"].min()), 4),
                "period_mean_max": round(float(g["mean"].max()), 4),
                "period_mean_std": round(float(g["mean"].std()), 4),
            }
        # univariate screen: |corr| of each numeric feature with target
        num = train.select_dtypes(include=[np.number]).columns.tolist()
        feats = [c for c in num if c not in
                 {args.target, args.time, args.id, args.benchmark}]
        corrs = {}
        yv = y.values
        for c in feats:
            x = train[c].values
            m = ~(np.isnan(x) | np.isnan(yv))
            if m.sum() > 10 and np.nanstd(x[m]) > 0:
                corrs[c] = round(float(np.corrcoef(x[m], yv[m])[0, 1]), 4)
        top = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
        report["top_univariate_corr"] = top
        report["n_features_screened"] = len(corrs)

    # Benchmark
    if args.benchmark and args.benchmark in train.columns:
        b = train[args.benchmark].values
        report["benchmark_present"] = True
        if args.target and args.target in train.columns:
            yv = train[args.target].values
            m = ~(np.isnan(b) | np.isnan(yv))
            if m.sum() > 10:
                report["benchmark_target_corr"] = round(
                    float(np.corrcoef(b[m], yv[m])[0, 1]), 4)
    else:
        report["benchmark_present"] = False

    # Live file structure match
    if args.live:
        live = pd.read_csv(args.live)
        report["live"] = summarize(live, "live")
        report["live_has_target"] = bool(args.target and
                                         args.target in live.columns)
        report["columns_match"] = list(train.columns) == list(live.columns)
        if list(train.columns) != list(live.columns):
            report["columns_only_in_train"] = [
                c for c in train.columns if c not in live.columns]
            report["columns_only_in_live"] = [
                c for c in live.columns if c not in train.columns]

    with open("forensics_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print("\nWrote forensics_report.json")


if __name__ == "__main__":
    main()
