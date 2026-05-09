#!/usr/bin/env python3
"""
memory_beta_sensitivity.py — EMA β sensitivity experiment for DRAEM

Tests different EMA memory decay settings to produce a proper beta sensitivity table.
Uses the same model, seeds, folds, and evaluation protocol as the main experiment.

Memory variants:
  - No memory (β=0):    raw day-level QEF signals only, no EMA features
  - Short memory (β≈0.5): raw signals + ewm3 features
  - Default memory (β≈0.7): raw signals + ewm7 features (paper default)
  - Long memory (β≈0.87):  raw signals + ewm14 features

All variants use the same model architecture, seeds {42, 123, 777}, and rolling-origin folds.
Results are reported as 3-seed × 5-fold means for H7.
"""
import random, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score

DATA = Path(__file__).resolve().parent.parent / 'data' / 'dataset_with_qwen_event_memory.csv'
OUT = Path(__file__).resolve().parent.parent / 'results' / 'qwen_event_factor' / 'memory_sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 777]
HORIZON = 7

HP = dict(d=32, dropout=.30, lr=8e-4, wd=1e-4, epochs=45, patience=7, batch=256, reg_w=.08)

# ── Numeric features (same as main experiment) ──
NUM = ['close','open','high','low','volume','amount','log_return',
       'volatility_20d','volatility_60d','range_pct',
       'return_lag1','return_lag3','return_lag5','volume_lag1',
       'mkt_hbea_close','mkt_bea_close','mkt_shea_close','mkt_szea_close','mkt_cea_close',
       'hs300_ret','eua_ret','blt_oil_ret','dq_oil_ret','coal_ret','cpi','m2']

MODULES = ['policy', 'market', 'finance', 'trade', 'compliance']

# ── Availability features (same as main experiment) ──
AVAIL = ['global_has','global_cnt','doc_mask','quality_score','importance_score',
         'policy_has','market_has','finance_has','trade_has','compliance_has',
         'qef_global_active_modules','qef_global_abs_signal',
         'qef_global_signal_sum','qef_memory_signal_sum_ewm7','qef_memory_disagreement_ewm7']

# ── Per-module raw signal features (day-level, no EMA) ──
RAW_SIGNALS = ['_signal', '_pos_signal', '_neg_signal', '_abs_signal', '_uncertainty',
               '_active']

# ── EMA feature suffixes for each beta variant ──
EMA_VARIANTS = {
    'no_memory': {
        'suffixes': [],  # raw signals only
        'global_suffixes': [],
        'beta_label': 'β=0 (no memory)',
    },
    'short': {
        'suffixes': ['_signal_ewm3', '_pos_signal_ewm3', '_neg_signal_ewm3',
                     '_abs_signal_ewm3', '_active_roll3'],
        'global_suffixes': ['qef_global_abs_signal_ewm3', 'qef_global_signal_sum_ewm3',
                            'qef_global_disagreement_ewm3'],
        'beta_label': 'β≈0.5 (short)',
    },
    'default': {
        'suffixes': ['_signal_ewm7', '_pos_signal_ewm7', '_neg_signal_ewm7',
                     '_abs_signal_ewm7', '_active_roll7'],
        'global_suffixes': ['qef_global_abs_signal_ewm7', 'qef_global_signal_sum_ewm7',
                            'qef_global_disagreement_ewm7',
                            'qef_memory_signal_mean_ewm7', 'qef_memory_signal_sum_ewm7',
                            'qef_memory_disagreement_ewm7'],
        'beta_label': 'β=0.7 (default)',
    },
    'long': {
        'suffixes': ['_signal_ewm14', '_pos_signal_ewm14', '_neg_signal_ewm14',
                     '_abs_signal_ewm14', '_active_roll14'],
        'global_suffixes': ['qef_global_abs_signal_ewm14', 'qef_global_signal_sum_ewm14',
                            'qef_global_disagreement_ewm14'],
        'beta_label': 'β≈0.87 (long)',
    },
}

def seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def ex(df, cols):
    return [c for c in cols if c in df.columns]

def build_qef_features(df, variant_name):
    """Build per-module QEF feature dict and global features for a given EMA variant."""
    cfg = EMA_VARIANTS[variant_name]
    grp = {}
    for m in MODULES:
        cols = [f'qef_{m}{s}' for s in RAW_SIGNALS + cfg['suffixes']]
        cols = [c for c in cols if c in df.columns]
        if cols:
            grp[m] = cols

    # Global QEF features: base global + variant-specific
    global_base = ['qef_global_signal_mean', 'qef_global_signal_sum',
                   'qef_global_abs_signal', 'qef_global_disagreement',
                   'qef_global_active_modules']
    global_cols = global_base + cfg['global_suffixes']
    global_cols = [c for c in global_cols if c in df.columns]
    grp['global'] = global_cols

    return grp

def folds(df):
    yrs = sorted(df.date.dt.year.unique()); fs = []
    for ty in yrs:
        vy = ty - 1; tr = [y for y in yrs if y < vy]
        if ty < 2021 or vy not in yrs or len(tr) < 3: continue
        fs.append(dict(fold=f'Y{ty}', train=tr, val=vy, test=ty))
    return fs

def split(df, f):
    y = df.date.dt.year
    return (df.index[y.isin(f['train'])].values,
            df.index[y == f['val']].values,
            df.index[y == f['test']].values)

def build(df, idx, num, grp, avail, target):
    cols = num + [c for v in grp.values() for c in v] + avail + [target]
    sub = df.loc[idx, cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    y = sub[target].astype(float).values.astype('float32'); keep = y != 0
    sub = sub.iloc[np.where(keep)[0]]; y = y[keep]
    return (sub[num].fillna(0).values.astype('float32'),
            {k: sub[v].fillna(0).values.astype('float32') for k, v in grp.items()},
            sub[avail].fillna(0).values.astype('float32') if avail else np.zeros((len(sub), 1), 'float32'),
            y, sub.index.values)

def scalefit(xs, xg, xa):
    return (StandardScaler().fit(xs),
            {k: StandardScaler().fit(v) for k, v in xg.items()},
            StandardScaler().fit(xa))

def scale(scs, scg, sca, xs, xg, xa):
    return (scs.transform(xs).astype('float32'),
            {k: scg[k].transform(v).astype('float32') for k, v in xg.items()},
            sca.transform(xa).astype('float32'))

def tt(xs, xg, xa, idx=None):
    if idx is None:
        return (torch.tensor(xs, dtype=torch.float32, device=DEV),
                {k: torch.tensor(v, dtype=torch.float32, device=DEV) for k, v in xg.items()},
                torch.tensor(xa, dtype=torch.float32, device=DEV))
    return (torch.tensor(xs[idx], dtype=torch.float32, device=DEV),
            {k: torch.tensor(v[idx], dtype=torch.float32, device=DEV) for k, v in xg.items()},
            torch.tensor(xa[idx], dtype=torch.float32, device=DEV))

# ── Model (same as main experiment) ──
class Model(nn.Module):
    def __init__(self, nnum, gdim, navail, mode):
        super().__init__(); self.names = list(gdim); self.mode = mode; d = HP['d']; drop = HP['dropout']; self.d = d
        self.num = nn.Sequential(nn.Linear(nnum, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
        self.enc = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(v, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop),
                             nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
            for k, v in gdim.items()
        })
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.av = nn.Sequential(nn.Linear(navail, d // 2), nn.LayerNorm(d // 2), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(d + d + d // 2, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))
        self.reg = nn.Sequential(nn.Linear(2 * d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, xs, xg, xa):
        hs = self.num(xs)
        toks = torch.stack([self.enc[k](xg[k]) for k in self.names], 1)
        if 'noattn' in self.mode:
            w = torch.ones(toks.size(0), toks.size(1), device=toks.device) / toks.size(1)
            ev = toks.mean(1)
        else:
            score = (self.q(hs).unsqueeze(1) * self.k(toks)).sum(-1) / (self.d ** .5)
            w = torch.softmax(score, 1)
            ev = (w.unsqueeze(-1) * self.v(toks)).sum(1)
        ev = self.norm(ev); av = self.av(xa)
        if 'gate1' in self.mode:
            g = torch.ones(xs.size(0), device=xs.device)
        else:
            g = torch.sigmoid(self.gate(torch.cat([hs, ev, av], 1))).squeeze(1)
        fu = torch.cat([hs, g[:, None] * ev], 1)
        return self.reg(fu).squeeze(1), self.head(fu).squeeze(1), g, w

def train_model(xs, xg, xa, y, xv, gv, av, yv, mode, seed_value):
    seed(seed_value)
    m = Model(xs.shape[1], {k: v.shape[1] for k, v in xg.items()}, xa.shape[1], mode).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=HP['lr'], weight_decay=HP['wd'])
    pos = max((y > 0).sum(), 1); neg = max((y < 0).sum(), 1)
    pw = torch.tensor([neg / pos], dtype=torch.float32, device=DEV)
    best = None; bb = -9; wait = 0; n = len(y)
    XV, GV, AV = tt(xv, gv, av); yvb = (yv > 0).astype(int)

    for ep in range(HP['epochs']):
        m.train(); perm = np.random.permutation(n)
        for i in range(0, n, HP['batch']):
            bi = perm[i:i + HP['batch']]
            X, G, A = tt(xs, xg, xa, bi)
            yt = torch.tensor(y[bi], dtype=torch.float32, device=DEV)
            yb = (yt > 0).float()
            yr, yd, g, w = m(X, G, A)
            loss = F.binary_cross_entropy_with_logits(yd, yb, pos_weight=pw) + HP['reg_w'] * F.smooth_l1_loss(yr, yt)
            if 'learnedgate' in mode:
                loss = loss + 0.001 * ((g - .55) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2); opt.step()

        m.eval()
        with torch.no_grad():
            _, yd, g, w = m(XV, GV, AV)
            pred = (torch.sigmoid(yd).cpu().numpy() > .5).astype(int)
            b = balanced_accuracy_score(yvb, pred) if len(np.unique(yvb)) > 1 else 0
        if b > bb:
            bb = b; wait = 0
            best = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= HP['patience']: break

    if best: m.load_state_dict(best)
    return m, bb

def evaluate(m, xt, gt, at, yt):
    m.eval()
    with torch.no_grad():
        _, yd, g, w = m(xt, gt, at)
        prob = torch.sigmoid(yd).cpu().numpy()
        pred = (prob > .5).astype(int)
        gg = g.cpu().numpy()

    y_true = (yt > 0).astype(int)
    return {
        'n': len(y_true),
        'DA': accuracy_score(y_true, pred),
        'BAcc': balanced_accuracy_score(y_true, pred),
        'MCC': matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0,
        'F1': f1_score(y_true, pred, zero_division=0),
        'gate_mean': float(gg.mean()),
    }

def main():
    print("=" * 70)
    print("EMA β Sensitivity Experiment — DRAEM H7")
    print("=" * 70)
    print(f"Device: {DEV}")
    print(f"Seeds: {SEEDS}")
    print(f"Horizon: H{HORIZON}")
    print(f"Variants: {list(EMA_VARIANTS.keys())}")
    print()

    df = pd.read_csv(DATA)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    num = ex(df, NUM); avail = ex(df, AVAIL)
    fs = folds(df)
    target = f'target_H{HORIZON}'

    all_rows = []

    for variant_name, cfg in EMA_VARIANTS.items():
        print(f"\n{'=' * 60}")
        print(f"  Variant: {variant_name} ({cfg['beta_label']})")
        print(f"{'=' * 60}")

        grp = build_qef_features(df, variant_name)
        n_qef = sum(len(v) for v in grp.values())
        print(f"  QEF features: {n_qef} (modules: {list(grp.keys())})")

        for f in fs:
            tr, va, te = split(df, f)
            for sd in SEEDS:
                xs, xg, xa, y, itr = build(df, tr, num, grp, avail, target)
                xv, gv, av, yv, iva = build(df, va, num, grp, avail, target)
                xt, gt, at, yt, ite = build(df, te, num, grp, avail, target)

                scs, scg, sca = scalefit(xs, xg, xa)
                xs, xg, xa = scale(scs, scg, sca, xs, xg, xa)
                xv, gv, av = scale(scs, scg, sca, xv, gv, av)
                xt, gt, at = scale(scs, scg, sca, xt, gt, at)

                m, vb = train_model(xs, xg, xa, y, xv, gv, av, yv, 'attn_learnedgate', sd)
                m.eval()

                XT, GT, AT = tt(xt, gt, at)
                r = evaluate(m, XT, GT, AT, yt)
                r.update({
                    'variant': variant_name,
                    'beta_label': cfg['beta_label'],
                    'fold': f['fold'],
                    'test_year': f['test'],
                    'seed': sd,
                    'val_BAcc': float(vb),
                    'n_qef': n_qef,
                })
                all_rows.append(r)

                print(f"  {f['fold']} seed={sd}: MCC={r['MCC']:+.3f} BAcc={r['BAcc']:.3f} DA={r['DA']:.3f}", flush=True)
                del m
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Save raw results ──
    runs = pd.DataFrame(all_rows)
    runs.to_csv(OUT / 'memory_sensitivity_runs.csv', index=False)

    # ── Aggregate by variant ──
    print("\n" + "=" * 70)
    print("📊 EMA β Sensitivity Results — H7 (3-seed × 5-fold mean)")
    print("=" * 70)

    summary = runs.groupby(['variant', 'beta_label']).agg(
        MCC_mean=('MCC', 'mean'), MCC_std=('MCC', 'std'),
        BAcc_mean=('BAcc', 'mean'), BAcc_std=('BAcc', 'std'),
        DA_mean=('DA', 'mean'), DA_std=('DA', 'std'),
        F1_mean=('F1', 'mean'),
        gate_mean=('gate_mean', 'mean'),
        n=('n', 'mean'),
    ).reset_index().sort_values('MCC_mean', ascending=False)

    print(f"\n  {'Variant':<18s} {'Beta':>16s}  {'MCC':>10s}  {'BAcc':>10s}  {'DA':>10s}  {'Gate':>6s}  {'N':>5s}")
    print("  " + "-" * 80)
    for _, row in summary.iterrows():
        print(f"  {row['variant']:<18s} {row['beta_label']:>16s}  "
              f"{row['MCC_mean']:+.3f}±{row['MCC_std']:.3f}  "
              f"{row['BAcc_mean']:.3f}±{row['BAcc_std']:.3f}  "
              f"{row['DA_mean']:.3f}±{row['DA_std']:.3f}  "
              f"{row['gate_mean']:.3f}  {row['n']:.0f}")

    summary.to_csv(OUT / 'memory_sensitivity_summary.csv', index=False)

    # ── Also run per-seed best-seed selection ──
    print("\n" + "=" * 70)
    print("📊 Best-seed results (for reference)")
    print("=" * 70)

    for variant_name in EMA_VARIANTS:
        vruns = runs[runs['variant'] == variant_name]
        # Group by fold, pick best seed by val_BAcc
        best_rows = []
        for fold in vruns['fold'].unique():
            fold_runs = vruns[vruns['fold'] == fold]
            best_idx = fold_runs['val_BAcc'].idxmax()
            best_rows.append(fold_runs.loc[best_idx])
        best_df = pd.DataFrame(best_rows)
        mcc_m = best_df['MCC'].mean()
        mcc_s = best_df['MCC'].std()
        bacc_m = best_df['BAcc'].mean()
        da_m = best_df['DA'].mean()
        beta_label = EMA_VARIANTS[variant_name]['beta_label']
        print(f"  {variant_name:<18s} {beta_label:>16s}  MCC={mcc_m:+.3f}±{mcc_s:.3f}  BAcc={bacc_m:.3f}  DA={da_m:.3f}")

    print(f"\n✅ Saved to {OUT}")
    print("Files: memory_sensitivity_runs.csv, memory_sensitivity_summary.csv")

if __name__ == '__main__':
    main()
