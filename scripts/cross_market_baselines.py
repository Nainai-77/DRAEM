#!/usr/bin/env python3
"""
Cross-Market Baselines: GDEA训练 → HBEA/CEA测试
===============================================
基线模型：
  Majority  — 训练集方向先验
  ARIMA     — 时序统计基准
  Ridge     — 数值特征岭回归
  XGBoost   — 数值+文本 GBDT
  Ridge(Text) — 仅文本特征
  LSTM      — 序列神经网络
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import statsmodels.api as sm

sys.path.insert(0, Path(__file__).resolve().parent.parent / 'models')
sys.path.insert(0, Path(__file__).resolve().parent.parent / 'config')
from utils import set_seed
from pathlib import Path

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEEDS = [42, 123, 456]

DATASET  = Path(__file__).resolve().parent.parent / 'data' / 'dataset_with_qwen_cumulative.csv'
OUT_DIR  = Path(__file__).resolve().parent.parent / 'results' / 'cross_market'
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_CLOSE = {
    'HBEA': 'mkt_hbea_close', 'SHEA': 'mkt_shea_close',
    'SZEA': 'mkt_szea_close', 'CEA':  'mkt_cea_close',
}

def get_all_text_cols(df):
    text_mods = ['policy', 'market', 'finance', 'trade', 'compliance']
    text_cols = []
    for m in text_mods:
        text_cols += [c for c in df.columns if c.startswith(f'{m}_')]
    text_cols += ['doc_mask', 'global_cnt', 'global_has', 'global_shock_flag',
                  'log_return', 'range_pct', 'realized_vol_1d']
    return text_cols

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

def prepare(df, y, num_cols, text_cols):
    n = len(y); nt = int(n*TRAIN_RATIO); nv = int(n*VAL_RATIO)
    xs = df[num_cols].fillna(0).ffill().bfill().values.astype(np.float32)
    sc_s = StandardScaler(); sc_s.fit(xs[:nt])
    xt = df[text_cols].fillna(0).ffill().bfill().values.astype(np.float32)
    sc_t = StandardScaler(); sc_t.fit(xt[:nt])
    y = np.nan_to_num(y.astype(np.float32), nan=0.0)
    return (sc_s.transform(xs[:nt]), sc_s.transform(xs[nt:nt+nv]),
            sc_s.transform(xs[nt+nv:]),
            sc_t.transform(xt[:nt]), sc_t.transform(xt[nt:nt+nv]),
            sc_t.transform(xt[nt+nv:]),
            y[:nt], y[nt:nt+nv], y[nt+nv:], sc_s, sc_t, nt, nv)

def evaluate(y_true, y_pred):
    da_dir = float(np.mean(np.sign(np.array(y_true)) == np.sign(np.array(y_pred))))
    rmse   = float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)))
    return da_dir, rmse

# ── LSTM ──────────────────────────────────────────────────────
class SimpleLSTM(torch.nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_feat, hidden, batch_first=True, num_layers=2, dropout=0.2)
        self.head = torch.nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

def make_windows(xs, y, window=5):
    X, Y = [], []
    for i in range(window, len(y)):
        X.append(xs[i-window:i])
        Y.append(y[i])
    return np.array(X), np.array(Y)

def train_lstm(xs_tr, y_tr, seed, window=5):
    set_seed(seed)
    Xw, Yw = make_windows(xs_tr, y_tr, window)
    Xw_t = torch.tensor(Xw, dtype=torch.float32, device=DEV)
    Yw_t = torch.tensor(Yw, dtype=torch.float32, device=DEV)

    model = SimpleLSTM(xs_tr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_state, wait = float('inf'), None, 0

    for ep in range(150):
        model.train()
        perm = torch.randperm(len(Yw_t))
        for i in range(0, len(Yw_t), 64):
            bi = perm[i:i+64]
            pred = model(Xw_t[bi])
            loss = torch.mean((pred - Yw_t[bi])**2)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            all_pred = model(Xw_t)
            loss = float(torch.mean((all_pred - Yw_t)**2).item())
        if loss < best_loss:
            best_loss = loss
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 20: break

    model.load_state_dict(best_state)
    model.eval()
    return model

def predict_lstm(model, xs_te, y_te, window=5):
    preds = []
    with torch.no_grad():
        for i in range(window, len(y_te)):
            x_w = torch.tensor(xs_te[i-window:i][None,:,:], dtype=torch.float32, device=DEV)
            preds.append(float(model(x_w).item()))
    mean_p = float(np.mean(preds)) if preds else 0.0
    full = np.concatenate([[mean_p]*window, np.array(preds, dtype=np.float32)])
    return full[:len(y_te)]

# ── Main ──────────────────────────────────────────────────────
def run_experiment():
    print("=" * 70)
    print("Cross-Market Baselines: GDEA → [HBEA, CEA]")
    print("=" * 70)

    df = pd.read_csv(DATASET)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df); nt = int(n*TRAIN_RATIO); nv = int(n*VAL_RATIO)

    text_cols = get_all_text_cols(df)
    num_cols  = get_num_cols()
    tgt_col   = TARGET_CLOSE['SZEA']   # GDEA = SZEA

    print(f"数据: {n}天 | {df['date'].min()} ~ {df['date'].max()}")
    print(f"训练: {nt}天 | 验证: {nv}天 | 测试: {n-nt-nv}天")
    print(f"数值特征: {len(num_cols)}维  文本特征: {len(text_cols)}维")

    test_markets = [('HBEA','HBEA'), ('CEA','CEA')]
    results = []

    for horizon in [1, 5, 7]:
        print(f"\n{'─'*70}")
        print(f"  Horizon H{horizon}")
        print(f"{'─'*70}")

        y_src = compute_target(df[tgt_col], horizon)
        (xs_tr, xs_va, _,
         xt_tr, xt_va, _,
         y_tr, y_va, _,
         sc_s, sc_t, _, _) = prepare(df, y_src, num_cols, text_cols)

        # 训练集全部数据（train+val）用于基线
        xs_all_tr = np.vstack([xs_tr, xs_va])
        xt_all_tr = np.vstack([xt_tr, xt_va])
        y_all_tr  = np.concatenate([y_tr, y_va])

        for tst_mkt, tst_name in test_markets:
            tst_col = TARGET_CLOSE[tst_mkt]
            y_full = compute_target(df[tst_col], horizon)
            y_full = np.nan_to_num(y_full.astype(np.float32), nan=0.0)
            y_te = y_full[nt+nv:]

            xs_te = sc_s.transform(
                df[num_cols].fillna(0).ffill().bfill().values.astype(np.float32)[nt+nv:])
            xt_te = sc_t.transform(
                df[text_cols].fillna(0).ffill().bfill().values.astype(np.float32)[nt+nv:])
            # 对齐有效长度
            valid = ~np.isnan(y_te)
            y_v = y_te[valid]
            xs_v = xs_te[valid]
            xt_v = xt_te[valid]
            n_v = len(y_v)

            print(f"\n  → {tst_name} (n={n_v})")

            # 1) Majority
            maj_dir = 1 if (np.sign(y_tr)>0).mean() > 0.5 else -1
            pred = np.full(n_v, maj_dir, dtype=float)
            da, rmse = evaluate(y_v, pred)
            results.append(dict(baseline='Majority',  horizon=f'H{horizon}', test_mkt=tst_name, n_test=n_v, da_dir=da, rmse=rmse))
            print(f"    Majority:    DA={da*100:5.1f}%  RMSE={rmse:.4f}")

            # 2) ARIMA: 拟合 source 市场时序，滚动 H-step 预测（每天重估）
            close_src = df[tgt_col].fillna(0).ffill().bfill().values
            close_tst = df[tst_col].fillna(0).ffill().bfill().values
            preds_ar = []
            for i in range(nt, len(df) - horizon):
                try:
                    m = sm.tsa.arima.ARIMA(close_src[:i+1], order=(1,0,1))
                    f = m.fit(solver='lbfgs', maxiter=30, warn_convergence=False)
                    fc = f.forecast(steps=horizon)
                    p_H = fc.iloc[-1]; p_t = close_src[i]
                    preds_ar.append(np.log(p_H/p_t) if p_H>0 and p_t>0 else 0.0)
                except:
                    preds_ar.append(0.0)
            # preds_ar[i] 对应 y[i+horizon]，是从 i 预测 H 步
            # y_te[j] = y[nt+nv+j]，需要 preds_ar 的最后 n_v 个
            n_ar = len(preds_ar)
            pred_ar = np.full(n_v, 0.0)
            take = min(n_ar, n_v)
            pred_ar[-take:] = preds_ar[-take:]
            da, rmse = evaluate(y_v, pred_ar)
            results.append(dict(baseline='ARIMA',     horizon=f'H{horizon}', test_mkt=tst_name, n_test=n_v, da_dir=da, rmse=rmse))
            print(f"    ARIMA:      DA={da*100:5.1f}%  RMSE={rmse:.4f}")

            # 3) Ridge (Numerical)
            pred = np.mean([Ridge(alpha=1.0).fit(xs_all_tr, y_all_tr).predict(xs_v) for _ in SEEDS], axis=0)
            da, rmse = evaluate(y_v, pred)
            results.append(dict(baseline='Ridge(Num)',  horizon=f'H{horizon}', test_mkt=tst_name, n_test=n_v, da_dir=da, rmse=rmse))
            print(f"    Ridge(Num): DA={da*100:5.1f}%  RMSE={rmse:.4f}")

            # 4) Ridge (Text-only)
            pred = np.mean([Ridge(alpha=1.0).fit(xt_all_tr, y_all_tr).predict(xt_v) for _ in SEEDS], axis=0)
            da, rmse = evaluate(y_v, pred)
            results.append(dict(baseline='Ridge(Text)', horizon=f'H{horizon}', test_mkt=tst_name, n_test=n_v, da_dir=da, rmse=rmse))
            print(f"    Ridge(Text):DA={da*100:5.1f}%  RMSE={rmse:.4f}")

            # 5) XGBoost
            try:
                import xgboost as xgb
                pred = np.zeros(n_v)
                for sd in SEEDS:
                    m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8,
                                         random_state=sd, verbosity=0)
                    m.fit(xs_all_tr, y_all_tr)
                    pred += m.predict(xs_v)
                pred /= len(SEEDS)
                da, rmse = evaluate(y_v, pred)
                results.append(dict(baseline='XGBoost',   horizon=f'H{horizon}', test_mkt=tst_name, n_test=n_v, da_dir=da, rmse=rmse))
                print(f"    XGBoost:    DA={da*100:5.1f}%  RMSE={rmse:.4f}")
            except ImportError:
                print(f"    XGBoost:    [not installed, skipped]")

            # 6) LSTM
            pred = np.zeros(n_v)
            for sd in SEEDS:
                model = train_lstm(xs_all_tr, y_all_tr, sd)
                pred += predict_lstm(model, xs_v, y_v)
                del model; torch.cuda.empty_cache()
            pred /= len(SEEDS)
            da, rmse = evaluate(y_v, pred)
            results.append(dict(baseline='LSTM',      horizon=f'H{horizon}', test_mkt=tst_name, n_test=n_v, da_dir=da, rmse=rmse))
            print(f"    LSTM:       DA={da*100:5.1f}%  RMSE={rmse:.4f}")

    df_res = pd.DataFrame(results)
    out = os.path.join(OUT_DIR, 'cross_market_baselines.csv')
    df_res.to_csv(out, index=False)

    print(f"\n{'='*70}")
    print("结果汇总")
    print(f"{'='*70}")
    # pivot table
    for h in ['H1','H5','H7']:
        sub = df_res[df_res['horizon']==h]
        print(f"\n── {h} ──")
        for _, r in sub.sort_values(['test_mkt','baseline']).iterrows():
            print(f"  {r['test_mkt']:5s}  {r['baseline']:15s}  DA={r['da_dir']*100:5.1f}%  RMSE={r['rmse']:.4f}")

    print(f"\n✅ 保存: {out}")
    return df_res

if __name__ == '__main__':
    run_experiment()
