#!/usr/bin/env python3
"""
Stage 1: Numeric-only 分类器 (LSTM)
方案A - Step 1: 确认基线是否能成立
"""
import os, sys, warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from pathlib import Path

DEV = 'cuda'
SEEDS = [42, 123, 777, 2024, 1234]
HORIZON = 7

# 27维数值特征
NUM = ['close','open','high','low','volume','amount','log_return',
       'volatility_20d','volatility_60d','range_pct',
       'return_lag1','return_lag3','return_lag5','volume_lag1',
       'mkt_hbea_close','mkt_bea_close','mkt_shea_close','mkt_szea_close','mkt_cea_close',
       'hs300_ret','eua_ret','blt_oil_ret','dq_oil_ret','coal_ret','cpi','m2']

# 序列窗口长度
SEQ_LEN = 10


class DirectionLSTM(nn.Module):
    def __init__(self, n_feat, hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden,
                           num_layers=n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, (h, c) = self.lstm(x)
        # 取最后一个时间步
        h_last = h[-1]  # (B, hidden)
        return self.head(h_last).squeeze(-1)  # (B,)


def build_sequences(X, seq_len):
    """构建时间序列样本"""
    n = len(X)
    X_seq = []
    for i in range(seq_len, n):
        X_seq.append(X[i-seq_len:i])
    return np.array(X_seq)


def load_data():
    df = pd.read_csv(Path(__file__).resolve().parent.parent / 'features' / 'dataset_with_qwen_cumulative.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    nt = int(n * 0.70)
    nv = int(n * 0.15)
    return df, nt, nv


def train_and_eval(df, nt_train, nt_val, seed, verbose=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    target_col = f'target_H{HORIZON}'

    # 数值特征
    Xs = df[NUM].fillna(0).ffill().bfill().values.astype(np.float32)

    # 时间序列
    X_seq = build_sequences(Xs, SEQ_LEN)
    n_total = len(X_seq)

    # 对齐target: target从SEQ_LEN开始
    y_all = df[target_col].values.astype(np.float32)[SEQ_LEN:]
    date_all = df['date'].values[SEQ_LEN:]

    # 切分
    X_tr = X_seq[:nt_train - SEQ_LEN]
    X_va = X_seq[nt_train - SEQ_LEN:nt_train - SEQ_LEN + nt_val]
    X_te = X_seq[nt_train - SEQ_LEN + nt_val:]

    y_tr = y_all[:nt_train - SEQ_LEN]
    y_va = y_all[nt_train - SEQ_LEN:nt_train - SEQ_LEN + nt_val]
    y_te = y_all[nt_train - SEQ_LEN + nt_val:]

    # 标准化
    sc = StandardScaler()
    n_tr = len(X_tr)
    X_tr_2d = X_tr.reshape(-1, X_tr.shape[-1])
    sc.fit(X_tr_2d)
    X_tr = sc.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    X_va = sc.transform(X_va.reshape(-1, X_va.shape[-1])).reshape(X_va.shape)
    X_te = sc.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)

    # 方向标签
    y_dir_tr = (y_tr > 0).astype(int)
    y_dir_va = (y_va > 0).astype(int)
    y_dir_te = (y_te > 0).astype(int)

    # 有效mask
    m_tr = ~np.isnan(y_tr)
    m_va = ~np.isnan(y_va) & (y_va != 0)
    m_te = ~np.isnan(y_te) & (y_te != 0)

    X_tr = torch.tensor(X_tr[m_tr], dtype=torch.float32, device=DEV)
    y_tr_t = torch.tensor(y_dir_tr[m_tr], dtype=torch.float32, device=DEV)
    X_va = torch.tensor(X_va[m_va], dtype=torch.float32, device=DEV)
    y_va_t = torch.tensor(y_dir_va[m_va], dtype=torch.float32, device=DEV)
    X_te = torch.tensor(X_te[m_te], dtype=torch.float32, device=DEV)
    y_te_np = y_dir_te[m_te]
    y_te_raw = y_te[m_te]

    # pos_weight for BCE
    n_pos = y_dir_tr[m_tr].sum()
    n_neg = (1 - y_dir_tr[m_tr]).sum()
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], device=DEV)

    # 模型
    model = DirectionLSTM(n_feat=len(NUM), hidden=64, n_layers=2, dropout=0.2).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state, best_bacc, wait = None, 0.0, 0
    n_epochs = 200
    batch_size = 64

    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            bi = perm[i:i+batch_size]
            out = model(X_tr[bi])
            loss = criterion(out, y_tr_t[bi])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            out_va = model(X_va)
            pred_va = (torch.sigmoid(out_va) > 0.5).cpu().numpy().astype(int)
            bacc_va = balanced_accuracy_score(y_va_t.cpu().numpy().astype(int), pred_va)

        if bacc_va > best_bacc:
            best_bacc = bacc_va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 30:
                if verbose:
                    print(f"    early stop ep {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        out_te = model(X_te)
        prob_te = torch.sigmoid(out_te).cpu().numpy()

    del model
    torch.cuda.empty_cache()

    # 指标
    pred_dir = (prob_te > 0.5).astype(int)
    da = accuracy_score(y_te_np, pred_dir)
    bacc = balanced_accuracy_score(y_te_np, pred_dir)
    mcc = matthews_corrcoef(y_te_np, pred_dir)
    f1 = f1_score(y_te_np, pred_dir, zero_division=0)

    if verbose:
        print(f"  seed={seed}: DA={da:.1%}, BAcc={bacc:.1%}, MCC={mcc:+.3f}, F1={f1:.3f}")

    return {
        'DA': da, 'BAcc': bacc, 'MCC': mcc, 'F1': f1,
        'prob': prob_te, 'y_true_dir': y_te_np, 'y_true_raw': y_te_raw,
        'pred_dir': pred_dir
    }


def main():
    print("="*70)
    print("Stage 1: Numeric-only LSTM 分类器 (H7, 5-seed)")
    print("="*70)

    df, nt_train, nt_val = load_data()
    print(f"数据: Train={nt_train}, Val={nt_val}, Test={len(df)-nt_train-nt_val}")
    print(f"序列长度: {SEQ_LEN}, 特征: {len(NUM)}")
    print(f"Seeds: {SEEDS}")
    print()

    runs = []
    for seed in SEEDS:
        r = train_and_eval(df, nt_train, nt_val, seed)
        runs.append(r)

    da = np.mean([x['DA'] for x in runs])
    da_s = np.std([x['DA'] for x in runs])
    bacc = np.mean([x['BAcc'] for x in runs])
    bacc_s = np.std([x['BAcc'] for x in runs])
    mcc = np.mean([x['MCC'] for x in runs])
    mcc_s = np.std([x['MCC'] for x in runs])
    f1 = np.mean([x['F1'] for x in runs])

    print(f"\n→ Numeric LSTM (5-seed mean)")
    print(f"  DA={da:.1%}±{da_s*100:.1f}%, BAcc={bacc:.1%}±{bacc_s*100:.1f}%, MCC={mcc:+.3f}±{mcc_s:.3f}, F1={f1:.3f}")

    # 保存
    os.makedirs(Path(__file__).resolve().parent.parent / 'results' / 'formal', exist_ok=True)
    np.save(Path(__file__).resolve().parent.parent / 'results' / 'formal' / 'A1_probs.npy',
            {s: r['prob'] for s, r in zip(SEEDS, runs)})

    # 止损判断
    print("\n" + "="*70)
    print("🚦 Stage 1 基线判断")
    print("="*70)
    c1 = bacc > 0.50
    c2 = mcc > 0
    print(f"  {'✅' if c1 else '❌'} BAcc > 50%: {bacc:.1%}")
    print(f"  {'✅' if c2 else '❌'} MCC > 0: {mcc:+.3f}")
    if c1 and c2:
        print("  → Stage 1 基线成立，可进入 Stage 2")
    else:
        print("  → Stage 1 基线偏弱，但若 A3 能提升仍可继续")


if __name__ == '__main__':
    main()