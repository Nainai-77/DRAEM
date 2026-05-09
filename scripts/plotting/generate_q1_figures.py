#!/usr/bin/env python3
"""Generate polished Q1-style figures without overwriting legacy figures.
Outputs are written to figures_q1_redraw/ and referenced by the manuscript.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent / 'results'
PROJECT = Path('/root/autodl-tmp/1_DRAEM_LLM')
OUT = Path(__file__).resolve().parent.parent.parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='white', context='paper')
# ── Global style: LaTeX-native rendering (mathptmx = Times Roman) ──
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Times'],
    'text.latex.preamble': r'\usepackage{mathptmx}\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}',
    'axes.labelsize': 11,
    'font.size': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 180,
    'savefig.dpi': 600,
})

COL = {
    'blue': '#2F5F8F',
    'light_blue': '#8FB7D9',
    'red': '#B94A48',
    'green': '#3F8F62',
    'orange': '#D8903A',
    'purple': '#6D5A9E',
    'gray': '#6F747A',
    'light_gray': '#E8ECEF',
    'dark': '#27313A',
}

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ------------------------------------------------------------------
# Figure 1: QEF construction and temporal coverage
# ------------------------------------------------------------------
qef_path = ROOT / 'qwen_event_daily.csv'
if qef_path.exists():
    qef = pd.read_csv(qef_path)
else:
    qef = pd.read_csv(Path(__file__).resolve().parent.parent.parent / 'data' / 'dataset_with_qwen_event_memory.csv')
qef['date'] = pd.to_datetime(qef['date'])
mods = ['policy','market','finance','trade','compliance']
active_cols = [f'qef_{m}_active' for m in mods if f'qef_{m}_active' in qef.columns]
if active_cols:
    cov = qef.set_index('date')[active_cols].resample('ME').mean().rename(columns=lambda c: c.replace('qef_','').replace('_active','').title())
else:
    has_cols = [f'{m}_has' for m in mods if f'{m}_has' in qef.columns]
    cov = qef.set_index('date')[has_cols].resample('ME').mean().rename(columns=lambda c: c.replace('_has','').title())
if 'qef_global_abs_signal' in qef.columns:
    sig = qef.set_index('date')['qef_global_abs_signal'].resample('ME').mean()
elif 'qef_memory_signal_sum_ewm7' in qef.columns:
    sig = qef.set_index('date')['qef_memory_signal_sum_ewm7'].resample('ME').mean()
else:
    sig = cov.mean(axis=1)

fig = plt.figure(figsize=(12.2, 6.1))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)
ax0 = fig.add_subplot(gs[0, :])
ax0.axis('off')
boxes = [
    ('Raw event text', 'policy news\nmarket reports\ncompliance signals'),
    ('LLM extraction', 'event module\ndirection / confidence\nuncertainty'),
    ('QEF memory', 'decayed semantic factors\nmodule activity\ndisagreement'),
    ('Forecast alignment', 'rolling-origin split\nH1 / H5 / H7 labels\nmodel input tensor'),
]
x = np.linspace(0.08, 0.86, len(boxes))
for i, ((title, body), xi) in enumerate(zip(boxes, x)):
    rect = plt.Rectangle((xi, 0.26), 0.16, 0.46, transform=ax0.transAxes,
                         facecolor='#F7FAFC', edgecolor=COL['blue'], lw=1.15)
    ax0.add_patch(rect)
    ax0.text(xi+0.08, 0.61, title, transform=ax0.transAxes, ha='center', va='center',
             fontsize=9.8, fontweight='bold', color=COL['dark'])
    ax0.text(xi+0.08, 0.43, body, transform=ax0.transAxes, ha='center', va='center',
             fontsize=8.1, color=COL['gray'], linespacing=1.3)
    if i < len(boxes)-1:
        ax0.annotate('', xy=(x[i+1]-0.015, 0.49), xytext=(xi+0.175, 0.49), xycoords=ax0.transAxes,
                     arrowprops=dict(arrowstyle='-|>', lw=1.2, color=COL['blue']))
ax0.text(0.0, 0.96, 'A', transform=ax0.transAxes, fontsize=13, fontweight='bold', color=COL['dark'])
ax0.set_title('QEF construction and chronological alignment', loc='left', pad=2, fontweight='bold')

ax1 = fig.add_subplot(gs[1, 0])
if len(cov) > 0:
    ax1.stackplot(cov.index, [cov[c].values for c in cov.columns], labels=cov.columns,
                  colors=['#2F5F8F','#8FB7D9','#D8903A','#6D5A9E','#3F8F62'][:len(cov.columns)], alpha=0.88)
ax1.set_ylabel('Monthly active ratio')
ax1.set_xlabel('Calendar time')
ax1.set_title('B  Event-module coverage over time', loc='left', fontweight='bold')
ax1.legend(frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.02))
ax1.spines[['top','right']].set_visible(False)
ax1.grid(axis='y', color=COL['light_gray'], lw=0.8)

ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(sig.index, sig.values, color=COL['red'], lw=1.8)
ax2.fill_between(sig.index, sig.values, color=COL['red'], alpha=0.18)
ax2.set_ylabel('Mean absolute QEF signal')
ax2.set_xlabel('Calendar time')
ax2.set_title('C  Semantic signal intensity', loc='left', fontweight='bold')
ax2.spines[['top','right']].set_visible(False)
ax2.grid(axis='y', color=COL['light_gray'], lw=0.8)
save(fig, 'fig01_qef_construction_coverage')

# ------------------------------------------------------------------
# Figure 2: Stage-I model-to-model heatmap + horizon profile
# ------------------------------------------------------------------
base = pd.read_csv(ROOT / 'final_baseline_result_stage_1.csv')
method_order = ['Naive','SVM','XGBoost','MLP','LSTM','TCN','Transformer','PatchTST','DeepMLP','Residual-QEF-DRAEM']
base['Method'] = pd.Categorical(base['Method'], categories=method_order, ordered=True)
base = base.sort_values('Method')
heat = pd.DataFrame({f'H{h}': base[f'H{h}_MCC'].values for h in [1,5,7]}, index=base['Method'].astype(str))
heat_bacc = pd.DataFrame({f'H{h}': base[f'H{h}_BAcc'].values*100 for h in [1,5,7]}, index=base['Method'].astype(str))
fig = plt.figure(figsize=(9.6, 6.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 0.92], wspace=0.35)
ax = fig.add_subplot(gs[0, 0])
cmap = LinearSegmentedColormap.from_list('mcc_div', ['#B94A48','#F7F7F7','#2F5F8F'])
norm = TwoSlopeNorm(vmin=min(-0.06, float(heat.min().min())), vcenter=0, vmax=max(0.16, float(heat.max().max())))
annot = heat.map(lambda x: f'{x:.3f}')
sns.heatmap(heat, ax=ax, cmap=cmap, norm=norm, annot=annot, fmt='', linewidths=0.8, linecolor='white',
            cbar_kws={'label':'MCC', 'shrink':0.72})
ax.set_title('A  Repeated-run mean MCC', loc='left', fontweight='bold')
ax.set_xlabel('Forecast horizon')
ax.set_ylabel('Compared model')

axp = fig.add_subplot(gs[0, 1])
focus = ['XGBoost','LSTM','PatchTST','Residual-QEF-DRAEM']
markers = {'XGBoost':'o','LSTM':'s','PatchTST':'^','Residual-QEF-DRAEM':'D'}
colors = {'XGBoost':COL['blue'],'LSTM':COL['orange'],'PatchTST':COL['purple'],'Residual-QEF-DRAEM':COL['red']}
for m in focus:
    vals = [float(base.loc[base['Method'].astype(str)==m, f'H{h}_MCC'].iloc[0]) for h in [1,5,7]]
    axp.plot(['H1','H5','H7'], vals, marker=markers[m], color=colors[m], lw=2.0, ms=5.8, label=m)
    for xlab, val in zip(['H1','H5','H7'], vals):
        if m == 'Residual-QEF-DRAEM':
            axp.text(xlab, val+0.008, f'{val:.3f}', ha='center', fontsize=7.8, color=colors[m])
axp.axhline(0, color=COL['dark'], lw=0.8, ls='--', alpha=0.55)
axp.set_title('B  Horizon profile of key models', loc='left', fontweight='bold')
axp.set_ylabel('MCC')
axp.set_xlabel('Forecast horizon')
axp.legend(frameon=False, loc='upper left')
axp.spines[['top','right']].set_visible(False)
axp.grid(axis='y', color=COL['light_gray'])
fig.suptitle('Stage-I model-to-model comparison under rolling-origin evaluation', y=1.02, fontsize=12, fontweight='bold')
save(fig, 'fig02_stage1_model_comparison')

# ------------------------------------------------------------------
# Figure 3: Stage-II conditional gain landscape
# ------------------------------------------------------------------
win = pd.read_csv(ROOT / 'final_window_result_stage_2.csv')
win = win.copy()
win['Horizon'] = 'H' + win['H'].astype(str)
win['Window'] = win['window'].str.replace('_', '\n', regex=False)
win['gain'] = win['gap_ours_minus_reference']
fig = plt.figure(figsize=(11.7, 6.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)
for i, h in enumerate([5,7]):
    ax = fig.add_subplot(gs[0, i])
    sub = win[win['H']==h].sort_values('gain')
    y = np.arange(len(sub))
    colors = np.where(sub['gain'] >= 0, COL['green'], COL['red'])
    size = 50 + 240*(sub['ours_n']-sub['ours_n'].min())/(sub['ours_n'].max()-sub['ours_n'].min()+1e-9)
    ax.hlines(y, 0, sub['gain'], color=colors, lw=2.2, alpha=0.86)
    ax.scatter(sub['gain'], y, s=size, c=colors, edgecolor='white', linewidth=0.8, zorder=3)
    ax.axvline(0, color=COL['dark'], lw=0.9, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels(sub['window'])
    ax.set_xlabel('MCC gain relative to reference baseline')
    if i == 0:
        ax.set_ylabel('Event-relevant window')
    ax.set_title(f"{'A' if h==5 else 'B'}  H{h} conditional gain landscape", loc='left', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x', color=COL['light_gray'])
    for yy, (_, r) in zip(y, sub.iterrows()):
        if abs(r['gain']) >= 0.075:
            ax.text(r['gain'] + (0.008 if r['gain']>=0 else -0.008), yy, f"{r['gain']:+.3f}",
                    va='center', ha='left' if r['gain']>=0 else 'right', fontsize=7.7, color=COL['dark'])
fig.suptitle('Stage-II event-window analysis: semantic residual gains are regime-dependent', y=1.02, fontsize=12, fontweight='bold')
save(fig, 'fig03_stage2_conditional_gain')

# ------------------------------------------------------------------
# Figure 4: Ablation effect plot
# ------------------------------------------------------------------
abl = pd.read_csv(ROOT / 'final_ablation_result.csv')
order = ['full','no_text','no_gate','no_crossattention']
abl['variant'] = pd.Categorical(abl['variant'], categories=order, ordered=True)
abl = abl.sort_values('variant')
full_mcc = float(abl.loc[abl['variant'].astype(str)=='full','MCC'].iloc[0])
abl['delta_mcc'] = abl['MCC'] - full_mcc
fig = plt.figure(figsize=(9.8, 4.8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.32)
ax1 = fig.add_subplot(gs[0,0])
bar_colors = [COL['blue'] if v=='full' else COL['gray'] for v in abl['variant'].astype(str)]
ax1.bar(abl['variant'].astype(str), abl['MCC'], color=bar_colors, width=0.62)
ax1.axhline(0, color=COL['dark'], lw=0.8, ls='--')
for i, (_, r) in enumerate(abl.iterrows()):
    ax1.text(i, r['MCC'] + (0.012 if r['MCC']>=0 else -0.018), f"{r['MCC']:.3f}", ha='center', fontsize=8)
ax1.set_ylabel('MCC')
ax1.set_xlabel('Model variant')
ax1.set_title('A  Ablation performance', loc='left', fontweight='bold')
ax1.tick_params(axis='x', rotation=18)
ax1.spines[['top','right']].set_visible(False)
ax1.grid(axis='y', color=COL['light_gray'])

ax2 = fig.add_subplot(gs[0,1])
sub = abl[abl['variant'].astype(str)!='full'].copy()
ax2.barh(sub['variant'].astype(str), sub['delta_mcc'], color=COL['red'], height=0.55)
ax2.axvline(0, color=COL['dark'], lw=0.8)
for i, (_, r) in enumerate(sub.iterrows()):
    ax2.text(r['delta_mcc']-0.01, i, f"{r['delta_mcc']:+.3f}", va='center', ha='right', fontsize=8)
ax2.set_xlabel(r'$Delta$MCC relative to full model')
ax2.set_title('B  Component removal effect', loc='left', fontweight='bold')
ax2.spines[['top','right']].set_visible(False)
ax2.grid(axis='x', color=COL['light_gray'])
fig.suptitle('Ablation under the H7 extreme-disagreement event-conflict window', y=1.04, fontsize=12, fontweight='bold')
save(fig, 'fig04_ablation_effect')

# ------------------------------------------------------------------
# Figure 5: Decision path + attention mechanism
# ------------------------------------------------------------------
paths_path = PROJECT / 'final_paper_archive/case_study/interpretability_true_attention/case_attention_paths.csv'
attn_path = PROJECT / 'final_paper_archive/case_study/interpretability_true_attention/case_attention_weights.csv'
if paths_path.exists() and attn_path.exists():
    paths = pd.read_csv(paths_path).sort_values('date')
    attn = pd.read_csv(attn_path)
    fig = plt.figure(figsize=(11.8, 6.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.35, 1.0], hspace=0.36)
    ax = fig.add_subplot(gs[0,0])
    x = np.arange(len(paths))
    width = 0.48
    for i, (_, r) in enumerate(paths.iterrows()):
        base_logit = r['base_logit']; delta = r['residual_delta']; final = r['final_logit']
        ax.bar(i, base_logit, width=width, color='#A7ADB4', label='Base logit' if i==0 else None)
        bottom = base_logit if delta >= 0 else base_logit + delta
        ax.bar(i, abs(delta), width=width, bottom=bottom, color=COL['green'] if delta>=0 else COL['red'],
               alpha=0.90, label='Semantic residual correction' if i==0 else None)
        ax.scatter(i, final, marker='D', s=70, color=COL['blue'], zorder=4, label='Final logit' if i==0 else None)
        ax.text(i, final + (0.08 if final >= 0 else -0.13), f"p={r['ours_prob']:.2f}",
                ha='center', fontsize=8, color=COL['blue'])
    ax.axhline(0, color=COL['dark'], lw=0.9, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(paths['date'])
    ax.set_ylabel('Logit value')
    ax.set_title('A  Gate-controlled correction path', loc='left', fontweight='bold')
    ax.legend(frameon=False, ncol=3, loc='upper right')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', color=COL['light_gray'])

    axh = fig.add_subplot(gs[1,0])
    pivot = attn.pivot(index='date', columns='module', values='attention_weight').fillna(0).loc[paths['date']]
    sns.heatmap(pivot, ax=axh, cmap='YlGnBu', annot=annot_df(pivot, 2), fmt='', linewidths=0.7, linecolor='white',
                cbar_kws={'label':'attention weight', 'shrink':0.85})
    axh.set_title('B  Module-level attention allocation', loc='left', fontweight='bold')
    axh.set_xlabel('Semantic module')
    axh.set_ylabel('Case date')
    fig.suptitle('Decision-level mechanism visualization for H7 event-conflict cases', y=1.02, fontsize=12, fontweight='bold')
    save(fig, 'fig05_decision_mechanism')

print('Generated Q1 redraw figures in:', OUT)
for p in sorted(OUT.glob('*.png')):
    print(p.name)
