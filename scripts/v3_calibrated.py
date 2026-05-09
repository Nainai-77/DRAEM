#!/usr/bin/env python3
"""
v3 方向头校准评估 - 最小可用版本
修复: pos_weight + val-driven flip + val-driven threshold
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.path.insert(0, Path(__file__).resolve().parent.parent / 'models')
sys.path.insert(0, Path(__file__).resolve().parent.parent / 'config')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from draem_v3_model import DRAEMv3
from utils import set_seed
from pathlib import Path

DEV = 'cuda'
SEEDS = [42, 123, 456, 789, 2024]

# HP
HP = dict(d=32, n_enc_layers=2, dropout=0.3, lr=0.001, wd=1e-5, epochs=500, batch=128, patience=8)

NUM = ['close','open','high','low','volume','amount','log_return',
       'volatility_20d','volatility_60d','range_pct',
       'return_lag1','return_lag3','return_lag5','volume_lag1',
       'mkt_hbea_close','mkt_bea_close','mkt_shea_close','mkt_szea_close','mkt_cea_close',
       'hs300_ret','eua_ret','blt_oil_ret','dq_oil_ret','coal_ret','cpi','m2']

# Text modules (5 modules × 11 features = 55, + 7 global = 62)
M = {
    'policy':    ['policy_cnt','policy_has','policy_alen','policy_roll3','policy_roll7',
                  'policy_roll14','policy_lag1','policy_lag3','policy_lag5','policy_gap','policy_sent'],
    'market':   ['market_cnt','market_has','market_alen','market_roll3','market_roll7',
                  'market_roll14','market_lag1','market_lag3','market_lag5','market_gap','market_sent'],
    'finance':  ['finance_cnt','finance_has','finance_alen','finance_roll3','finance_roll7',
                  'finance_roll14','finance_lag1','finance_lag3','finance_lag5','finance_gap','finance_sent'],
    'trade':    ['trade_cnt','trade_has','trade_alen','trade_roll3','trade_roll7',
                  'trade_roll14','trade_lag1','trade_lag3','trade_lag5','trade_gap','trade_sent'],
    'compliance':['compliance_cnt','compliance_has','compliance_alen','compliance_roll3','compliance_roll7',
                  'compliance_roll14','compliance_lag1','compliance_lag3','compliance_lag5','compliance_gap','compliance_sent'],
}
GLOBAL = ['global_cnt','global_has','quality_score','importance_score',
           'global_shock_flag','compliance_window','doc_mask']
TEXT62 = [c for m in M.values() for c in m] + GLOBAL
QWEN = ['qwen_dir_mean','qwen_conf_mean','qwen_signal_strength','qwen_disagreement']


def build_data(df, target_col, text_cols):
    n = len(df); nt = int(n*0.7); nv = int(n*0.15)
    Xs = df[NUM].fillna(0).ffill().bfill().values.astype(np.float32)
    if text_cols:
        Xt = df[text_cols].fillna(0).ffill().bfill().values.astype(np.float32)
    else:
        Xt = np.zeros((n, 1), dtype=np.float32)
    sc_s, sc_t = StandardScaler(), StandardScaler()
    Xs_tr = sc_s.fit_transform(Xs[:nt]); Xs_va = sc_s.transform(Xs[nt:nt+nv]); Xs_te = sc_s.transform(Xs[nt+nv:])
    Xt_tr = sc_t.fit_transform(Xt[:nt]); Xt_va = sc_t.transform(Xt[nt:nt+nv]); Xt_te = sc_t.transform(Xt[nt+nv:])
    y_all = df[target_col].values.astype(np.float32)
    y_tr_r = y_all[:nt].copy(); y_va_r = y_all[nt:nt+nv].copy(); y_te_r = y_all[nt+nv:].copy()
    mask = ~np.isnan(y_tr_r) & (y_tr_r != 0)
    mu, sd = np.nanmean(y_tr_r[mask]), np.nanstd(y_tr_r[mask]) + 1e-8
    y_tr_z = np.where(mask, (y_tr_r-mu)/sd, 0.0)
    y_va_z = np.where(~np.isnan(y_va_r), (y_va_r-mu)/sd, 0.0)
    y_te_z = np.where(~np.isnan(y_te_r), (y_te_r-mu)/sd, 0.0)
    n_pos = ((y_tr_r>0)&mask).sum(); n_neg = ((y_tr_r<0)&mask).sum()
    pw = float(n_neg)/max(float(n_pos),1.0)
    return dict(Xs_tr=Xs_tr,Xs_va=Xs_va,Xs_te=Xs_te,Xt_tr=Xt_tr,Xt_va=Xt_va,Xt_te=Xt_te,
                y_tr_z=y_tr_z,y_va_z=y_va_z,y_te_z=y_te_z,y_te_r=y_te_r,y_va_r=y_va_r,
                mu=mu,sd=sd,pw=pw,n_tr=nt,n_va=nv,n_te=n-nt-nv)


def make_modules(Xt, modules_dict, text_cols):
    """Build x_text_dict for model."""
    xd = {}
    for m, cols in modules_dict.items():
        idx = [text_cols.index(c) for c in cols if c in text_cols]
        if idx:
            xd[m] = torch.tensor(Xt[:,idx], dtype=torch.float32, device=DEV)
        else:
            xd[m] = torch.zeros((Xt.shape[0],1), dtype=torch.float32, device=DEV)
    return xd


def run_exp(df, target_col, text_cols, modules_dict, seed, lam):
    set_seed(seed)
    data = build_data(df, target_col, text_cols)
    nt, nv = data['n_tr'], data['n_va']
    pw = data['pw']

    # mod_sizes
    ms = {}
    for m, cols in modules_dict.items():
        n_f = sum(1 for c in cols if c in text_cols)
        if n_f > 0: ms[m] = n_f
    tot = len(text_cols) if text_cols else 1
    if not ms:
        ms = {'dummy': 1}; tot = 1

    model = DRAEMv3(n_struct=len(NUM), text_module_sizes=ms, total_text_feat=tot,
                    d=HP['d'], n_enc_layers=HP['n_enc_layers'], dropout=HP['dropout'],
                    use_uncertainty=False, horizon_lambda=lam).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=HP['lr'], weight_decay=HP['wd'])
    best_state, best_mse, wait = None, float('inf'), 0

    for ep in range(HP['epochs']):
        model.train()
        perm = torch.randperm(nt)
        for i in range(0, nt, HP['batch']):
            bi = perm[i:i+HP['batch']]
            xs = torch.tensor(data['Xs_tr'][bi], dtype=torch.float32, device=DEV)
            xt = torch.tensor(data['Xt_tr'][bi], dtype=torch.float32, device=DEV)
            xd = make_modules(xt.cpu().numpy(), modules_dict, text_cols)
            yt = torch.tensor(data['y_tr_z'][bi], dtype=torch.float32, device=DEV)
            yr,yd,pr,*_ = model(xs, xd, xt)
            dl = F.binary_cross_entropy_with_logits(yd.squeeze(), (yt>0).float(),
                                                   pos_weight=torch.tensor(pw, device=DEV))
            rl = F.mse_loss(yr.squeeze(), yt)
            loss = lam['reg']*rl + lam['dir']*dl
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(data['Xs_va'], dtype=torch.float32, device=DEV)
            tv = torch.tensor(data['Xt_va'], dtype=torch.float32, device=DEV)
            xd_v = make_modules(tv.cpu().numpy(), modules_dict, text_cols)
            yr_v,yd_v,*_ = model(xv, xd_v, tv)[:2]
            vm = F.mse_loss(yr_v.squeeze(), torch.tensor(data['y_va_z'], dtype=torch.float32, device=DEV)).item()
        if vm < best_mse:
            best_mse = vm; best_state = {k:v.clone() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= HP['patience']: break

    if best_state: model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        # Test
        xs_t = torch.tensor(data['Xs_te'], dtype=torch.float32, device=DEV)
        xt_t = torch.tensor(data['Xt_te'], dtype=torch.float32, device=DEV)
        xd_t = make_modules(xt_t.cpu().numpy(), modules_dict, text_cols)
        yr_tz, yd_t = model(xs_t, xd_t, xt_t)[:2]
        yr_t = yr_tz.cpu().numpy()*data['sd']+data['mu']
        # Val
        yr_vz = yr_v.cpu().numpy()*data['sd']+data['mu']
        yv, yt_v = data['y_va_r'], data['y_te_r']
        # Masks
        mv = ~np.isnan(yv)&(yv!=0); mt = ~np.isnan(yt_v)&(yt_v!=0)
        yv_v,yv_r = yv[mv],yr_vz[mv]; yt_v,yr_tv = yt_v[mt],yr_t[mt]
        # Regression DA
        da_reg = (np.sign(yt_v)==np.sign(yr_tv)).mean()
        # Direction logits
        dp_v = torch.sigmoid(yd_v).cpu().numpy()[mv]
        dp_t = torch.sigmoid(yd_t).cpu().numpy()[mt]
        # Val-driven flip
        da_raw = (np.sign(yv_v)==((dp_v>0.5).astype(float)*2-1)).mean()
        dp_flip = 1-dp_v
        da_flip = (np.sign(yv_v)==((dp_flip>0.5).astype(float)*2-1)).mean()
        flip = da_flip > da_raw
        # Val-driven threshold
        dp_vu = dp_flip if flip else dp_v
        dp_tu = 1-dp_t if flip else dp_t
        best_thr,best_da_va = 0.5, da_raw if not flip else da_flip
        for thr in np.arange(0.30, 0.71, 0.05):
            da = (np.sign(yv_v)==((dp_vu>thr).astype(float)*2-1)).mean()
            if da > best_da_va: best_da_va = da; best_thr = thr
        # Test metrics
        da_dir = (np.sign(yt_v)==((dp_t>0.5).astype(float)*2-1)).mean()
        da_cal = (np.sign(yt_v)==((dp_tu>best_thr).astype(float)*2-1)).mean()
        rmse = np.sqrt(np.mean((yt_v-yr_tv)**2))
    del model; torch.cuda.empty_cache()
    return dict(da_reg=da_reg,da_dir=da_dir,da_cal=da_cal,
                da_raw=da_raw,da_flip=da_flip,rmse=rmse,flip=flip,best_thr=best_thr,pw=pw)


def main():
    print("="*65)
    print("v3 方向头校准评估（pos_weight + val-flip + val-thr）")
    print(f"d={HP['d']}, dropout={HP['dropout']}, lr={HP['lr']}")
    print(f"SEEDS: {SEEDS}")
    print("="*65)

    df = pd.read_csv(Path(__file__).resolve().parent.parent / 'features' / 'dataset_with_qwen_cumulative.csv')
    df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date').reset_index(drop=True)
    print(f"数据: {len(df)}天")

    # Configs
    CONFIGS = [
        ('A', []),
        ('B', TEXT62),
        ('D', TEXT62+QWEN),
    ]

    RESULTS = []
    LAMBDAS = [('Orig(0.5)', {'reg':1.0,'dir':0.5,'regime':0.15}),
                ('High(1.0)', {'reg':1.0,'dir':1.0,'regime':0.15})]

    for lam_name, lam in LAMBDAS:
        print(f"\n{'='*65}")
        print(f"Lambda_dir: {lam_name}")
        print(f"{'='*65}")
        for h in [5, 1]:
            tgt = f'target_H{h}'
            print(f"\n--- H{h} ---")
            for cfg_name, tcols in CONFIGS:
                # modules_dict
                if tcols:
                    md = {m:[c for c in cols if c in tcols] for m,cols in M.items()}
                    md = {k:v for k,v in md.items() if v}
                    if not md: md = {'dummy':['x0']}
                else:
                    md = {'dummy':['x0']}  # Numeric-only: 1 dummy module
                runs = [run_exp(df, tgt, tcols, md, s, lam) for s in SEEDS]
                da_raw = np.mean([r['da_dir'] for r in runs])
                da_cal = np.mean([r['da_cal'] for r in runs])
                da_std = np.std([r['da_cal'] for r in runs])
                da_reg = np.mean([r['da_reg'] for r in runs])
                rmse = np.mean([r['rmse'] for r in runs])
                flips = [r['flip'] for r in runs]
                pw = runs[0]['pw']
                print(f"  {cfg_name}: DA_raw={da_raw*100:.1f}%→DA_cal={da_cal*100:.1f}% "
                      f"(reg={da_reg*100:.1f}%) RMSE={rmse:.4f} "
                      f"flip={flips} thr={[round(r['best_thr'],2) for r in runs]} pw={pw:.2f}")
                RESULTS.append({'lam':lam_name,'cfg':cfg_name,'h':h,
                                'da_raw':da_raw,'da_cal':da_cal,'da_std':da_std,
                                'da_reg':da_reg,'rmse':rmse,'flip':any(flips),'pw':pw})

    # Summary
    print("\n" + "="*65)
    print("结果汇总")
    print("="*65)
    n=len(df); nt=int(n*0.7); nv=int(n*0.15)
    y_te = df['target_H5'].values[nt+nv:]
    v=~np.isnan(y_te)&(y_te!=0)
    maj = max((y_te[v]>0).mean(),(y_te[v]<0).mean())
    print(f"\n多数类基线(H5): {maj*100:.1f}%\n")
    for h in [5, 1]:
        print(f"H{h}:")
        print(f"  {'cfg':5s} {'lam':12s} {'DA_raw':>8s} {'DA_cal':>8s} {'DA_reg':>8s} {'RMSE':>8s}")
        print("  " + "-"*55)
        for _,r in pd.DataFrame(RESULTS).query(f'h=={h}').iterrows():
            sig='*' if r['da_cal']>maj else ''
            print(f"  {r['cfg']:5s} {r['lam']:12s} {r['da_raw']*100:6.1f}% {r['da_cal']*100:6.1f}%{sig} {r['da_reg']*100:6.1f}% {r['rmse']:.4f}")

    pd.DataFrame(RESULTS).to_csv(Path(__file__).resolve().parent.parent / 'results' / 'v3_calibrated.csv', index=False)
    print(f"\n✅ 保存: v3_calibrated.csv")


if __name__=='__main__':
    main()