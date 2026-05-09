#!/usr/bin/env python3
"""
消融实验 - Gate和CrossAttention
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.path.insert(0, Path(__file__).resolve().parent.parent / 'models')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from draem_v3_model import DRAEMv3
from pathlib import Path

DEV = 'cuda'
SEED = 42
HP = dict(d=64, n_enc_layers=3, dropout=0.15, lr=0.0003, epochs=1500, batch=64, patience=30)

NUM = ['close','open','high','low','volume','amount','log_return',
       'volatility_20d','volatility_60d','range_pct',
       'return_lag1','return_lag3','return_lag5','volume_lag1',
       'mkt_hbea_close','mkt_bea_close','mkt_shea_close','mkt_szea_close','mkt_cea_close',
       'hs300_ret','eua_ret','blt_oil_ret','dq_oil_ret','coal_ret','cpi','m2']

M = {'policy':['policy_cnt','policy_has','policy_alen','policy_roll3','policy_roll7','policy_roll14','policy_lag1','policy_lag3','policy_lag5','policy_gap','policy_sent'],
     'market':['market_cnt','market_has','market_alen','market_roll3','market_roll7','market_roll14','market_lag1','market_lag3','market_lag5','market_gap','market_sent'],
     'finance':['finance_cnt','finance_has','finance_alen','finance_roll3','finance_roll7','finance_roll14','finance_lag1','finance_lag3','finance_lag5','finance_gap','finance_sent'],
     'trade':['trade_cnt','trade_has','trade_alen','trade_roll3','trade_roll7','trade_roll14','trade_lag1','trade_lag3','trade_lag5','trade_gap','trade_sent'],
     'compliance':['compliance_cnt','compliance_has','compliance_alen','compliance_roll3','compliance_roll7','compliance_roll14','compliance_lag1','compliance_lag3','compliance_lag5','compliance_gap','compliance_sent'],
}
GLOBAL = ['global_cnt','global_has','quality_score','importance_score','global_shock_flag','compliance_window','doc_mask']
TEXT62 = [c for m in M.values() for c in m] + GLOBAL
QWEN = ['qwen_dir_mean','qwen_conf_mean','qwen_signal_strength','qwen_disagreement']


def make_module_dict(Xt, text_cols):
    xd = {}
    for mname, cols in M.items():
        idx = [text_cols.index(c) for c in cols if c in text_cols]
        if idx:
            xd[mname] = torch.tensor(Xt[:, idx], dtype=torch.float32, device=DEV)
        else:
            xd[mname] = torch.zeros((Xt.shape[0], 1), dtype=torch.float32, device=DEV)
    return xd


def train_and_eval(df, target_col, text_cols, seed=42, use_cross_attn=True, use_gate=True, use_calibration=True):
    np.random.seed(seed); torch.manual_seed(seed)
    n = len(df); nt = int(n*0.7); nv = int(n*0.15)
    
    Xs = df[NUM].fillna(0).ffill().bfill().values.astype(np.float32)
    Xt = df[text_cols].fillna(0).ffill().bfill().values.astype(np.float32) if text_cols else np.zeros((n, 1), dtype=np.float32)
    
    sc_s = StandardScaler(); sc_t = StandardScaler()
    Xs_tr, Xs_va, Xs_te = sc_s.fit_transform(Xs[:nt]), sc_s.transform(Xs[nt:nt+nv]), sc_s.transform(Xs[nt+nv:])
    Xt_tr, Xt_va, Xt_te = sc_t.fit_transform(Xt[:nt]), sc_t.transform(Xt[nt:nt+nv]), sc_t.transform(Xt[nt+nv:])
    
    y_all = df[target_col].values.astype(np.float32)
    y_tr_r, y_va_r, y_te_r = y_all[:nt].copy(), y_all[nt:nt+nv].copy(), y_all[nt+nv:].copy()
    mask = ~np.isnan(y_tr_r) & (y_tr_r != 0)
    mu, sd = np.nanmean(y_tr_r[mask]), np.nanstd(y_tr_r[mask]) + 1e-8
    y_tr_z = np.where(mask, (y_tr_r-mu)/sd, 0.0)
    
    ms = {m:sum(1 for c in cols if c in text_cols) for m,cols in M.items()}
    ms = {k:v for k,v in ms.items() if v>0}
    
    model = DRAEMv3(
        n_struct=len(NUM), text_module_sizes=ms, total_text_feat=len(text_cols),
        d=HP['d'], n_enc_layers=HP['n_enc_layers'], dropout=HP['dropout'],
        use_uncertainty=False, horizon_lambda={'reg':1.0,'dir':1.5,'regime':0.15}
    ).to(DEV)
    
    opt = torch.optim.AdamW(model.parameters(), lr=HP['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    best_state, best_loss, wait = None, float('inf'), 0
    
    for ep in range(HP['epochs']):
        model.train()
        perm = np.random.permutation(nt)
        for i in range(0, nt, HP['batch']):
            bi = perm[i:i+HP['batch']]
            xs = torch.tensor(Xs_tr[bi], dtype=torch.float32, device=DEV)
            xt = torch.tensor(Xt_tr[bi], dtype=torch.float32, device=DEV)
            xd = make_module_dict(Xt_tr[bi], text_cols) if text_cols else {m: torch.zeros((len(bi),1), device=DEV) for m in ms}
            yt = torch.tensor(y_tr_z[bi], dtype=torch.float32, device=DEV)
            yr, yd, pr, *_ = model(xs, xd, xt)
            dl = F.binary_cross_entropy_with_logits(yd.squeeze(), (yt>0).float())
            rl = F.mse_loss(yr.squeeze(), yt)
            loss = rl + 1.5*dl
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(Xs_va, dtype=torch.float32, device=DEV)
            tv = torch.tensor(Xt_va, dtype=torch.float32, device=DEV)
            xd_v = make_module_dict(Xt_va, text_cols) if text_cols else {m: torch.zeros((nv,1), device=DEV) for m in ms}
            yr_v, yd_v, *_ = model(xv, xd_v, tv)[:2]
            val_loss = F.mse_loss(yr_v, torch.tensor((y_va_r-mu)/sd, device=DEV)).item()
        
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss; best_state = {k:v.clone() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= HP['patience']: break
    
    if best_state: model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        xs_t = torch.tensor(Xs_te, dtype=torch.float32, device=DEV)
        xt_t = torch.tensor(Xt_te, dtype=torch.float32, device=DEV)
        xd_t = make_module_dict(Xt_te, text_cols) if text_cols else {m: torch.zeros((n-nt-nv,1), device=DEV) for m in ms}
        yr_t, yd_t, *_ = model(xs_t, xd_t, xt_t)[:2]
        dp_t = torch.sigmoid(yd_t).cpu().numpy()
        
        y_te = y_te_r
        mt = ~np.isnan(y_te) & (y_te != 0)
        yt = y_te[mt]; dp = dp_t[mt]
        
        # 校准
        if use_calibration:
            # val-driven flip
            y_va = y_all[nt:nt+nv]
            mv = ~np.isnan(y_va) & (y_va != 0)
            dp_va = torch.sigmoid(yd_v).cpu().numpy()[mv]
            yv = y_va[mv]
            da_raw = (np.sign(yv)==np.sign(dp_va-0.5)).mean()
            da_flip = (np.sign(yv)==np.sign(1-dp_va-0.5)).mean()
            flip = da_flip > da_raw
            dp_vu = 1-dp_va if flip else dp_va
            best_thr, best_da_va = 0.5, da_raw if not flip else da_flip
            for thr in np.arange(0.30, 0.71, 0.05):
                da = (np.sign(yv)==np.sign(dp_vu-thr)).mean()
                if da > best_da_va: best_da_va = da; best_thr = thr
            dp_tu = 1-dp if flip else dp
            da = (np.sign(yt)==np.sign(dp_tu-best_thr)).mean()
        else:
            da = (np.sign(yt)==np.sign(dp-0.5)).mean()
        
        acc = balanced_accuracy_score((np.sign(yt)>0).astype(int), (dp>0.5).astype(int))
        mcc = matthews_corrcoef((np.sign(yt)>0).astype(int), (dp>0.5).astype(int))
    
    del model; torch.cuda.empty_cache()
    return dict(da=da, acc=acc, mcc=mcc)


def main():
    print("="*70)
    print("消融实验 - Gate & CrossAttention & Calibration")
    print("="*70)
    
    df = pd.read_csv(Path(__file__).resolve().parent.parent / 'features' / 'dataset_with_qwen_cumulative.csv')
    df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date').reset_index(drop=True)
    print(f"数据: {len(df)}天, H5")
    
    TARGET = 'target_H5'
    text_cols = TEXT62 + QWEN
    
    EXPS = [
        ('Full (DRAEM-D)', True, True, True),
        ('-Gate', False, True, True),
        ('-CrossAttn', True, False, True),
        ('-Calibration', True, True, False),
        ('-Text (A)', True, True, True),  # Numeric only
    ]
    
    RESULTS = []
    for name, use_gate, use_cross, use_cal in EXPS:
        print(f"\n--- {name} ---")
        cols = [] if 'Text' in name else text_cols
        r = train_and_eval(df, TARGET, cols, SEED, use_cross, use_gate, use_cal)
        print(f"  DA={r['da']*100:.1f}% MCC={r['mcc']:.3f}")
        RESULTS.append(dict(Exp=name, DA=r['da'], MCC=r['mcc'], Acc=r['acc']))
    
    pd.DataFrame(RESULTS).to_csv(Path(__file__).resolve().parent.parent / 'results' / 'ablation_gate_cross_cal.csv', index=False)
    
    print("\n" + "="*70)
    print("结果汇总")
    print("="*70)
    for r in RESULTS:
        print(f"{r['Exp']:20s}: DA={r['DA']*100:.1f}% MCC={r['mcc']:.3f}")


if __name__=='__main__':
    main()