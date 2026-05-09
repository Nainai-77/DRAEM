#!/usr/bin/env python3
"""
纯数值特征极限基线（GDEA → CEA）
================================
从 CSV 读取已有 DRAEM 结果，合并对比
基线：Random, Majority, Ridge(Num), XGBoost(Num), LSTM(Num)
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0, Path(__file__).resolve().parent.parent / 'models')
sys.path.insert(0, Path(__file__).resolve().parent.parent / 'config')
from utils import set_seed
from pathlib import Path

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
SEEDS = [42, 123, 456]

DATASET = Path(__file__).resolve().parent.parent / 'data' / 'dataset_with_qwen_cumulative.csv'
OUT_DIR = Path(__file__).resolve().parent.parent / 'results' / 'cross_market'

TARGET_CLOSE = {'SZEA': 'mkt_szea_close', 'CEA': 'mkt_cea_close'}

def get_num_cols():
    return [
        'close', 'open', 'high', 'low', 'volume', 'amount',
        'log_return', 'volatility_20d', 'volatility_60d',
        'range_pct', 'return_lag1', 'return_lag3', 'return_lag5',
        'volume_lag1',
        'mkt_hbea_close', 'mkt_shea_close', 'mkt_cea_close', 'mkt_szea_close',
        'hs300_ret', 'eua_ret', 'blt_oil_ret', 'dq_oil_ret',
        'coal_ret', 'cpi', 'm2',
    ]

def compute_target(close_series, horizon):
    ret = np.log(close_series / close_series.shift(1)).fillna(0).values
    tgt = np.full(len(ret), np.nan, dtype=np.float32)
    for i in range(len(ret) - horizon):
        tgt[i] = ret[i+1:i+1+horizon].sum()
    return tgt

def prepare(df, y, num_cols):
    n = len(y); nt = int(n*TRAIN_RATIO); nv = int(n*VAL_RATIO)
    xs = df[num_cols].fillna(0).ffill().bfill().values.astype(np.float32)
    sc_s = StandardScaler(); sc_s.fit(xs[:nt])
    y = np.nan_to_num(y.astype(np.float32), nan=0.0)
    return (sc_s.transform(xs[:nt]), sc_s.transform(xs[nt:nt+nv]),
            sc_s.transform(xs[nt+nv:]), y[:nt], y[nt:nt+nv], y[nt+nv:], sc_s, nt, nv)

def evaluate(y_true, y_pred):
    y_t = np.array(y_true); y_p = np.array(y_pred)
    da = float(np.mean(np.sign(y_t) == np.sign(y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p)**2)))
    return da, rmse

# ── LSTM ──────────────────────────────────────────────────────
class SimpleLSTM(torch.nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_feat, hidden, batch_first=True, num_layers=2, dropout=0.2)
        self.head = torch.nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

def make_windows(xs, y, w=5):
    X, Y = [], []
    for i in range(w, len(y)):
        X.append(xs[i-w:i]); Y.append(y[i])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def train_lstm(xs_tr, y_tr, seed, w=5):
    set_seed(seed)
    Xw, Yw = make_windows(xs_tr, y_tr, w)
    model = SimpleLSTM(xs_tr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.tensor(Xw, dtype=torch.float32, device=DEV)
    Y = torch.tensor(Yw, dtype=torch.float32, device=DEV)
    best_loss, best_state, wait = float('inf'), None, 0
    for ep in range(100):
        perm = torch.randperm(len(Y))
        for i in range(0, len(Y), 64):
            bi = perm[i:i+64]
            loss = torch.mean((model(X[bi]) - Y[bi])**2)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            loss = float(torch.mean((model(X) - Y)**2).item())
        if loss < best_loss:
            best_loss, best_state, wait = loss, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 15: break
    model.load_state_dict(best_state); model.eval()
    return model

def predict_lstm(model, xs_te, y_te, w=5):
    preds = []
    with torch.no_grad():
        for i in range(w, len(y_te)):
            x = torch.tensor(xs_te[i-w:i][None,:,:], dtype=torch.float32, device=DEV)
            preds.append(float(model(x).item()))
    return np.concatenate([[np.mean(preds)]*w, np.array(preds, dtype=np.float32)])[:len(y_te)]

# ── Main ──────────────────────────────────────────────────────
def run():
    print("=" * 70)
    print("纯数值特征极限基线 vs DRAEM（GDEA → CEA）")
    print("=" * 70)

    df = pd.read_csv(DATASET)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df); nt = int(n*TRAIN_RATIO); nv = int(n*VAL_RATIO)
    num_cols = get_num_cols()

    print(f"数据: {n}天 | 训练:{nt} 验证:{nv} 测试:{n-nt-nv}")
    print(f"数值特征: {len(num_cols)}维")

    tgt_col = TARGET_CLOSE['SZEA']
    tst_col = TARGET_CLOSE['CEA']
    results = []

    for horizon in [1, 5, 7]:
        print(f"\n{'─'*70}\n  H{horizon}\n{'─'*70}")

        y_src = compute_target(df[tgt_col], horizon)
        (xs_tr, xs_va, xs_te, y_tr, y_va, y_te, sc_s, _, _) = prepare(df, y_src, num_cols)

        xs_all = np.vstack([xs_tr, xs_va])
        y_all  = np.concatenate([y_tr, y_va])

        y_tgt = compute_target(df[tst_col], horizon)
        y_tgt = np.nan_to_num(y_tgt.astype(np.float32), nan=0.0)
        y_te  = y_tgt[nt+nv:]

        valid = ~np.isnan(y_te)
        y_v = y_te[valid]; xs_v = xs_te[valid]
        n_v = len(y_v)

        print(f"  有效样本: {n_v}")

        # 1) Random
        np.random.seed(42)
        pred = np.random.choice([-1, 1], n_v)
        da, rmse = evaluate(y_v, pred)
        results.append(('Random', horizon, n_v, da, rmse))
        print(f"  Random:       DA={da*100:5.1f}%  RMSE={rmse:.4f}")

        # 2) Majority
        maj_dir = 1 if (np.sign(y_all)>0).mean() > 0.5 else -1
        da, rmse = evaluate(y_v, np.full(n_v, maj_dir, dtype=float))
        results.append(('Majority', horizon, n_v, da, rmse))
        print(f"  Majority:    DA={da*100:5.1f}%  RMSE={rmse:.4f}")

        # 3) Ridge(Num)
        pred = np.mean([Ridge(alpha=1.0).fit(xs_all, y_all).predict(xs_v) for _ in SEEDS], axis=0)
        da, rmse = evaluate(y_v, pred)
        results.append(('Ridge(Num)', horizon, n_v, da, rmse))
        print(f"  Ridge(Num): DA={da*100:5.1f}%  RMSE={rmse:.4f}")

        # 4) XGBoost(Num)
        try:
            import xgboost as xgb
            pred = np.zeros(n_v)
            for sd in SEEDS:
                m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=sd, verbosity=0)
                m.fit(xs_all, y_all)
                pred += m.predict(xs_v)
            pred /= len(SEEDS)
            da, rmse = evaluate(y_v, pred)
            results.append(('XGBoost(Num)', horizon, n_v, da, rmse))
            print(f"  XGBoost(Num):DA={da*100:5.1f}%  RMSE={rmse:.4f}")
        except ImportError:
            print(f"  XGBoost: [not installed]")

        # 5) LSTM(Num)
        pred = np.zeros(n_v)
        for sd in SEEDS:
            m = train_lstm(xs_all, y_all, sd)
            pred += predict_lstm(m, xs_v, y_v)
            del m; torch.cuda.empty_cache()
        pred /= len(SEEDS)
        da, rmse = evaluate(y_v, pred)
        results.append(('LSTM(Num)', horizon, n_v, da, rmse))
        print(f"  LSTM(Num):   DA={da*100:5.1f}%  RMSE={rmse:.4f}")

    # ── 合并已有 DRAEM 结果 ─────────────────────────────────────
    draem = pd.read_csv(os.path.join(OUT_DIR, 'true_cross_market_transfer.csv'))
    cea = draem[draem['test_market']=='CEA'].copy()
    cea['baseline'] = cea['cfg'].map({'B':'DRAEM B (TEXT)', 'D':'DRAEM D (TEXT+Qwen)'})
    cea['horizon'] = cea['horizon'].str.replace('H','').astype(int)
    cea = cea[['baseline','horizon','da_dir','rmse']].rename(columns={'da_dir':'da_dir','rmse':'rmse'})

    df_res = pd.DataFrame(results, columns=['baseline','horizon','n_test','da_dir','rmse'])
    df_full = pd.concat([df_res, cea], ignore_index=True)
    df_full['horizon'] = 'H' + df_full['horizon'].astype(str)
    df_full = df_full.sort_values(['horizon','da_dir'], ascending=[True, False]).reset_index(drop=True)

    out = os.path.join(OUT_DIR, 'numerical_only_baselines.csv')
    df_full.to_csv(out, index=False)

    # ── 打印汇总表 ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("结果汇总（GDEA → CEA）")
    print(f"{'='*70}")

    ORDER = ['Random','Majority','Ridge(Num)','XGBoost(Num)','LSTM(Num)',
             'DRAEM B (TEXT)','DRAEM D (TEXT+Qwen)']
    for h in ['H1','H5','H7']:
        print(f"\n── {h} ──")
        sub = df_full[df_full['horizon']==h].copy()
        sub['key'] = sub['baseline'].map({v:i for i,v in enumerate(ORDER) if v in sub['baseline'].values})
        sub = sub.sort_values('key').drop(columns='key')
        for _, r in sub.iterrows():
            star = ' ⭐' if 'DRAEM' in r['baseline'] else ''
            print(f"  {r['baseline']:17s}  DA={r['da_dir']*100:5.1f}%  RMSE={r['rmse']:.4f}{star}")

    print(f"\n✅ 保存: {out}")
    return df_full

if __name__ == '__main__':
    run()