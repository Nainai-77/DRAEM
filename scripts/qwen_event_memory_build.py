#!/usr/bin/env python3
"""
qwen_event_memory_build.py

Build event-memory features from qwen_event_daily.csv and merge them into an existing
DRAEM dataset. This solves the sparse same-day text problem by carrying event impact
forward with exponential decay.

Inputs:
  --events results/qwen_event_factor/qwen_event_daily.csv
  --dataset data/dataset_with_qwen_cumulative.csv

Outputs:
  results/qwen_event_factor/qwen_event_memory_features.csv
  data/dataset_with_qwen_event_memory.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_EVENTS = Path(__file__).resolve().parent.parent / 'results' / 'qwen_event_factor' / 'qwen_event_daily.csv'
DEFAULT_DATASET = Path(__file__).resolve().parent.parent / 'data' / 'dataset_with_qwen_cumulative.csv'
DEFAULT_OUT_FEATURES = Path(__file__).resolve().parent.parent / 'results' / 'qwen_event_factor' / 'qwen_event_memory_features.csv'
DEFAULT_OUT_DATASET = Path(__file__).resolve().parent.parent / 'data' / 'dataset_with_qwen_event_memory.csv'

HALF_LIVES = [3, 7, 14, 21]
LAGS = [1, 3, 5, 7]


def add_calendar(events, dataset_dates=None):
    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])
    if dataset_dates is None:
        start, end = events["date"].min(), events["date"].max()
        dates = pd.date_range(start, end, freq="D")
    else:
        dates = pd.to_datetime(pd.Series(dataset_dates)).sort_values().drop_duplicates()
    full = pd.DataFrame({"date": dates})
    out = full.merge(events, on="date", how="left")
    feature_cols = [c for c in out.columns if c != "date"]
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out.sort_values("date").reset_index(drop=True)


def ewm_memory(x, half_life):
    # Recursive carry-forward: mem_t = max(raw_t contribution, decayed previous + raw_t)
    # Additive version is used because several events can accumulate.
    lam = np.exp(np.log(0.5) / half_life)
    mem = np.zeros(len(x), dtype=float)
    prev = 0.0
    for i, v in enumerate(np.asarray(x, dtype=float)):
        prev = prev * lam + v
        mem[i] = prev
    return mem


def build_memory(df):
    df = df.copy().sort_values("date").reset_index(drop=True)
    base_cols = [c for c in df.columns if c != "date"]

    # Limit memory to meaningful semantic intensity columns, not every mechanism count.
    target_cols = [
        c for c in base_cols
        if any(s in c for s in ["_signal", "_abs_signal", "_pos_signal", "_neg_signal", "_uncertainty"])
        or c in ["qef_global_signal_mean", "qef_global_signal_sum", "qef_global_abs_signal", "qef_global_disagreement"]
    ]
    target_cols = sorted(set(target_cols))

    mem = pd.DataFrame({"date": df["date"]})
    # Keep same-day selected features.
    for c in target_cols:
        mem[c] = df[c].astype(float).values

    for c in target_cols:
        vals = df[c].astype(float).values
        for h in HALF_LIVES:
            mem[f"{c}_ewm{h}"] = ewm_memory(vals, h)
        for lag in LAGS:
            mem[f"{c}_lag{lag}"] = pd.Series(vals).shift(lag).fillna(0.0).values

    active_cols = [c for c in base_cols if c.endswith("_active") or c == "qef_global_active_modules"]
    for c in active_cols:
        mem[c] = df[c].astype(float).values
        for w in [3, 7, 14]:
            mem[f"{c}_roll{w}"] = pd.Series(df[c].astype(float)).rolling(w, min_periods=1).sum().values

    # Aggregate active memory indicators.
    signal_mem_cols = [c for c in mem.columns if c.endswith("_ewm7") and "signal" in c and "abs" not in c]
    if signal_mem_cols:
        vals = mem[signal_mem_cols]
        mem["qef_memory_signal_mean_ewm7"] = vals.mean(axis=1)
        mem["qef_memory_signal_sum_ewm7"] = vals.sum(axis=1)
        mem["qef_memory_disagreement_ewm7"] = vals.std(axis=1).fillna(0.0)

    return mem


def train_only_select(dataset, candidate_cols, target_col, train_ratio=0.70, topk=30):
    """Optional transparent train-only correlation screening."""
    n = len(dataset)
    n_train = int(n * train_ratio)
    train = dataset.iloc[:n_train].copy()
    y = train[target_col].astype(float)
    rows = []
    for c in candidate_cols:
        x = train[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if x.std() == 0 or y.std() == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        rows.append((c, corr, abs(corr)))
    sel = pd.DataFrame(rows, columns=["feature", "train_corr", "abs_train_corr"]).sort_values("abs_train_corr", ascending=False)
    return sel.head(topk), sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=DEFAULT_EVENTS)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out-features", default=DEFAULT_OUT_FEATURES)
    ap.add_argument("--out-dataset", default=DEFAULT_OUT_DATASET)
    ap.add_argument("--select-topk", type=int, default=30)
    args = ap.parse_args()

    events = pd.read_csv(args.events)
    dataset = pd.read_csv(args.dataset)
    dataset["date"] = pd.to_datetime(dataset["date"])

    full_events = add_calendar(events, dataset_dates=dataset["date"])
    mem = build_memory(full_events)

    Path(args.out_features).parent.mkdir(parents=True, exist_ok=True)
    mem.to_csv(args.out_features, index=False)

    merged = dataset.merge(mem, on="date", how="left")
    qef_cols = [c for c in merged.columns if c.startswith("qef_")]
    merged[qef_cols] = merged[qef_cols].fillna(0.0)
    Path(args.out_dataset).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dataset, index=False)

    # Produce train-only screening reports for each horizon if targets exist.
    outdir = Path(args.out_features).parent
    for H in [1, 5, 7]:
        target = f"target_H{H}"
        if target in merged.columns:
            sel, all_scores = train_only_select(merged, qef_cols, target, topk=args.select_topk)
            sel.to_csv(outdir / f"qwen_event_memory_selected_H{H}_top{args.select_topk}.csv", index=False)
            all_scores.to_csv(outdir / f"qwen_event_memory_train_corr_H{H}.csv", index=False)

    print(f"Saved memory features: {args.out_features} rows={len(mem)} cols={len(mem.columns)}")
    print(f"Saved merged dataset:  {args.out_dataset} rows={len(merged)} cols={len(merged.columns)}")
    print(f"QEF feature count: {len(qef_cols)}")
    print("Top active/memory columns sample:", qef_cols[:20])


if __name__ == "__main__":
    main()
