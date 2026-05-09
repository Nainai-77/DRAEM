"""
dataset.py — Data loading, splitting, scaling.
"""
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

class CarbonDataset:
    """
    Handles loading, temporal splitting, and scaling of the TAIC dataset.
    Uses per-row NaN filtering (not all-column) to avoid dropping everything.
    """
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, parse_dates=['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.scaler = None

    def split(self, train_ratio=0.70, val_ratio=0.15):
        n = len(self.df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        self.train_idx = list(range(n_train))
        self.val_idx = list(range(n_train, n_train + n_val))
        self.test_idx = list(range(n_train + n_val, n))
        return self

    def prepare(self, struct_cols, text_cols=None, target_col='target_H1', fit=True):
        """Extract X (struct+text), y for train/val/test with per-row NaN handling."""
        use_cols = list(struct_cols)
        if text_cols:
            use_cols += list(text_cols)

        # Only keep columns that exist
        use_cols = [c for c in use_cols if c in self.df.columns]
        X_all = self.df[use_cols].copy()
        y_all = self.df[target_col].copy()

        # Fill NaN with 0 for cross-market and monthly macro columns
        # (they're not missing — they just don't exist yet for early dates)
        fill_zero_cols = [c for c in use_cols if c.startswith('mkt_') or c in ('cpi', 'm2')]
        for c in fill_zero_cols:
            X_all[c] = X_all[c].fillna(0)

        # For remaining NaN (very few), forward-fill then back-fill
        X_all = X_all.ffill().bfill()
        y_all = y_all.ffill().bfill()

        def extract(idx):
            mask = np.array(idx)
            X = X_all.loc[mask].values.astype(np.float32)
            y = y_all.loc[mask].values.astype(np.float32)
            return X, y

        X_tr, y_tr = extract(self.train_idx)
        X_va, y_va = extract(self.val_idx)
        X_te, y_te = extract(self.test_idx)

        if fit:
            self.scaler = StandardScaler()
            X_tr = self.scaler.fit_transform(X_tr)
            X_va = self.scaler.transform(X_va)
            X_te = self.scaler.transform(X_te)
        else:
            X_tr = self.scaler.transform(X_tr)
            X_va = self.scaler.transform(X_va)
            X_te = self.scaler.transform(X_te)

        return X_tr, y_tr, X_va, y_va, X_te, y_te

    def info(self):
        n = len(self.df)
        nt, nv, ne = len(self.train_idx), len(self.val_idx), len(self.test_idx)
        d0, d1 = self.df['date'].min().date(), self.df['date'].max().date()
        return f"Dataset: {n} rows, {d0}~{d1}, train={nt} val={nv} test={ne}"
