#!/usr/bin/env python3
"""
v3 5-seed 正式实验（B vs D）
配置：d=32, n_enc_layers=2（与主实验一致）
"""
import os, warnings; warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import sys
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
SEEDS = [42, 123, 777, 2024, 1234]
HP = dict(d=32, dropout=0.20, lr=2e-4, batch_size=64, epochs=300, patience=50)
LAM = {'reg': 1.0, 'dir': 0.5, 'regime': 0.1}

NUM = ['close','open','high','low','volume','amount','log_return',
       'volatility_20d','volatility_60d','range_pct',
       'return_lag1','return_lag3','return_lag5','volume_lag1',
       'mkt_hbea_close','mkt_bea_close','mkt_shea_close','mkt_szea_close','mkt_cea_close',
       'hs300_ret','eua_ret','blt_oil_ret','dq_oil_ret','coal_ret','cpi','m2']

TEXT62 = ['policy_cnt','policy_has','policy_alen','policy_roll3','policy_roll7','policy_roll14',
         'policy_lag1','policy_lag3','policy_lag5','policy_gap','policy_sent',
         'market_cnt','market_has','market_alen','market_roll3','market_roll7','market_roll14',
         'market_lag1','market_lag3','market_lag5','market_gap','market_sent',
         'finance_cnt','finance_has','finance_alen','finance_roll3','finance_roll7','finance_roll14',
         'finance_lag1','finance_lag3','finance_lag5','finance_gap','finance_sent',
         'trade_cnt','trade_has','trade_alen','trade_roll3','trade_roll7','trade_roll14',
         'trade_lag1','trade_lag3','trade_lag5','trade_gap','trade_sent',
         'compliance_cnt','compliance_has','compliance_alen','compliance_roll3','compliance_roll7',
         'compliance_roll14','compliance_lag1','compliance_lag3','compliance_lag5','compliance_gap',
         'compliance_sent','global_cnt','global_has','quality_score','importance_score',
         'global_shock_flag','compliance_window','doc_mask']

QWEN = ['qwen_dir_mean','qwen_conf_mean','qwen_signal_strength','qwen_disagreement']


def prepare(df_sub, add_qwen):
    txt_cols = [c for c in (TEXT62 + (QWEN if add_qwen else [])) if c in df_sub.columns]
    xt = df_sub[txt_cols].fillna(0).values.astype(np.float32)
    sc = StandardScaler()
    xt = sc.fit_transform(xt)
    return xt, txt_cols


def run_exp(xt, y, seed):
    set_seed(seed)
    n = len(y)
    n_text = xt.shape[1]

    model = DRAEMv3(
        n_struct=len(NUM), text_module_sizes={}, total_text_feat=n_text,
        d=HP['d'], n_enc_layers=2, dropout=HP['dropout'],
        use_uncertainty=False, horizon_lambda=LAM,
    ).to(DEV)

    opt = torch.optim.Adam(model.parameters(), lr=HP['lr'])
    best_state, best_mse, wait = None, float('inf'), 0

    for ep in range(HP['epochs']):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, HP['batch_size']):
            bi = perm[i:i+HP['batch_size']]
            xs_b = torch.randn(len(bi), len(NUM), device=DEV)
            xt_b = torch.tensor(xt[bi], dtype=torch.float32, device=DEV)
            yt_b = torch.tensor(y[bi], dtype=torch.float32, device=DEV)
            rt_b = torch.zeros(len(bi), dtype=torch.float32, device=DEV)
            yr, yd, pr, *_ = model(xs_b, {}, xt_b)
            loss, _ = model.compute_loss(yr, yd, pr, yt_b, rt_b)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            xs_all = torch.randn(n, len(NUM), device=DEV)
            vr = model(xs_all, {}, torch.tensor(xt, dtype=torch.float32, device=DEV))[0]
            vmse = F.mse_loss(vr, torch.tensor(y, dtype=torch.float32, device=DEV)).item()

        if vmse < best_mse:
            best_mse = vmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= HP['patience']:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        xs_all = torch.randn(n, len(NUM), device=DEV)
        yr, yd, pr, *_ = model(xs_all, {}, torch.tensor(xt, dtype=torch.float32, device=DEV))
        da = float(np.mean(np.sign(y) == np.sign(yr.cpu().numpy())))
        dp = torch.sigmoid(yd).cpu().numpy()
        da_dir = float(np.mean(np.sign(y) == ((dp > 0.5).astype(float)*2-1)))
        rmse = float(np.sqrt(np.mean((y - yr.cpu().numpy())**2)))

    del model; torch.cuda.empty_cache()
    return dict(da=round(da,4), da_dir=round(da_dir,4), rmse=round(rmse,4))


def main():
    print("=" * 60)
    print("v3 5-seed 正式实验")
    print(f"配置: d={HP['d']}, layers=2, epochs={HP['epochs']}, patience={HP['patience']}")
    print(f"SEEDS: {SEEDS}")
    print("=" * 60)

    df = pd.read_csv(Path(__file__).resolve().parent.parent / 'features' / 'dataset_with_qwen_full.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 子样本定义
    q_avail = df['qwen_available'] > 0
    q_conf = df['qwen_conf_mean'] >= 1.5

    subs = [
        ('全样本', df),
        ('活跃子样本', df[q_avail]),
        ('高置信度', df[q_avail & q_conf]),
    ]

    results = []
    for sub_name, df_sub in subs:
        n = len(df_sub)
        print(f"\n{'='*50}\n{sub_name}: n={n}")

        for h in [1, 5, 7]:
            tgt = f'target_H{h}'
            y = df_sub[tgt].fillna(0).values.astype(np.float32)

            # B
            xt_b, tc_b = prepare(df_sub, add_qwen=False)
            b_runs = [run_exp(xt_b, y, s) for s in SEEDS]
            b_dir = np.mean([r['da_dir'] for r in b_runs])
            b_std = np.std([r['da_dir'] for r in b_runs])
            b_rmse = np.mean([r['rmse'] for r in b_runs])
            b_dirs = [r['da_dir']*100 for r in b_runs]
            print(f"  B H{h}: {[f'{v:.1f}' for v in b_dirs]}")
            print(f"     mean={b_dir*100:.1f}±{b_std*100:.1f}%, RMSE={b_rmse:.4f}")

            # D
            xt_d, tc_d = prepare(df_sub, add_qwen=True)
            d_runs = [run_exp(xt_d, y, s) for s in SEEDS]
            d_dir = np.mean([r['da_dir'] for r in d_runs])
            d_std = np.std([r['da_dir'] for r in d_runs])
            d_rmse = np.mean([r['rmse'] for r in d_runs])
            d_dirs = [r['da_dir']*100 for r in d_runs]
            print(f"  D H{h}: {[f'{v:.1f}' for v in d_dirs]}")
            print(f"     mean={d_dir*100:.1f}±{d_std*100:.1f}%, RMSE={d_rmse:.4f}")

            delta = (d_dir - b_dir) * 100
            results.append({
                'subsample': sub_name, 'horizon': h, 'n': n,
                'B_da_dir': round(b_dir,4), 'B_std': round(b_std,4),
                'D_da_dir': round(d_dir,4), 'D_std': round(d_std,4),
                'B_rmse': round(b_rmse,4), 'D_rmse': round(d_rmse,4),
                'Δ_da_dir': round(delta,1),
                'Δ_rmse': round((d_rmse-b_rmse)*100, 1),  # RMSE delta in %
                'sig': '✅' if delta > 0 else '⚠️'
            })

    # 汇总
    print("\n" + "=" * 70)
    print("结果汇总 (5-seed mean ± std)")
    print("=" * 70)
    print(f"\n{'子样本':12s} {'H':2s} {'n':>5s}  {'B DA_dir':>10s}  {'D DA_dir':>10s}  {'Δ':>7s}  {'B RMSE':>8s}  {'D RMSE':>8s}  {'提升?':>4s}")
    print("-" * 80)
    for r in results:
        print(f"{r['subsample']:12s} H{r['horizon']}  {r['n']:>5d}  "
              f"{r['B_da_dir']*100:7.1f}±{r['B_std']*100:3.1f}%  "
              f"{r['D_da_dir']*100:7.1f}±{r['D_std']*100:3.1f}%  "
              f"{r['Δ_da_dir']:+.1f}pp  "
              f"{r['B_rmse']:.4f}  {r['D_rmse']:.4f}  {r['sig']}")

    # 基准对比
    print("\n" + "=" * 70)
    print("基准对比")
    print("=" * 70)
    print("主实验 No-Text (基准): H1=44.3%, H5=57.8%")
    print("主实验 Full DRAEM:     H1=44.1%, H5=57.8%")
    print()

    out = pd.DataFrame(results)
    out.to_csv(Path(__file__).resolve().parent.parent / 'results' / 'v3_5seed_formal.csv', index=False)
    print(f"\n✅ 保存: v3_5seed_formal.csv")


if __name__ == '__main__':
    main()