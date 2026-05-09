#!/usr/bin/env python3
"""
Generate Q1-quality figures for the final manuscript.
Design patterns learned from ConEm (KBS 2025):
- Clean, modular block diagrams with color-coded pathways
- Grouped bar charts for ablation/improvement studies
- Overlaid density plots for model comparison
- Qualitative prediction overlays
- Consistent typography and spacing
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

ROOT = Path('/root/autodl-tmp/1_DRAEM_LLM')
PR = Path('.')
OUT = PR / 'figures_q1_redraw'
OUT.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Global style (following ConEm KBS style)
# =====================================================================
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

# Color palette (professional, KBS-quality)
COL = {
    'blue': '#2B5B84',     # Primary: numeric branch
    'lb': '#B8D4E8',       # Light blue
    'green': '#2E8B57',    # Semantic branch
    'lg': '#C8E6C9',       # Light green
    'red': '#C0392B',      # Correction/residual
    'orange': '#E67E22',   # Gate/gating
    'purple': '#7B68AE',   # Fusion
    'gray': '#7F8C8D',     # Neutral
    'dark': '#2C3E50',     # Text
    'light': '#ECF0F1',    # Background
    'cream': '#FDF6E3',    # Highlight
    'gold': '#F39C12',     # Accent
}

# Model group colors
GROUP_COLORS = {
    'Traditional ML': COL['blue'],
    'RNN/CNN': COL['purple'],
    'Transformer': COL['green'],
    'DeepMLP': COL['gray'],
    'Our Method': COL['orange'],
}

def save(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved {name}')


# =====================================================================
# Load data
# =====================================================================
df_w = pd.read_csv(PR / 'final_window_result_stage_2.csv')
df_b = pd.read_csv(PR / 'final_baseline_result_stage_1.csv')
df_a = pd.read_csv(PR / 'final_ablation_result.csv')
lef = pd.read_csv(ROOT / 'results/qwen_event_factor/qwen_event_daily.csv')
ds = pd.read_csv(ROOT / 'data/dataset_with_qwen_event_memory.csv')
ds['date'] = pd.to_datetime(ds['date'])
lef['date'] = pd.to_datetime(lef['date'])
lef_cols = ['date'] + [c for c in lef.columns if c not in ds.columns]
merged = ds.merge(lef[lef_cols], on='date', how='left')

# Rename method label for display
df_b['Method'] = df_b['Method'].replace({'Residual-QEF-DRAEM': 'DRAEM'})
df_w['ours_variant'] = df_w['ours_variant'].replace({'residual_logistic': 'DRAEM', 'residual_ridge': 'DRAEM'})

# Load daily predictions for qualitative plots
daily_fn = ROOT / 'results/window_residual_daily/residual_daily_predictions.csv'
daily = pd.read_csv(daily_fn) if daily_fn.exists() else None
if daily is not None:
    daily['date'] = pd.to_datetime(daily['date'])

mods = ['policy', 'market', 'finance', 'trade', 'compliance']
mnames = [m.title() for m in mods]
mcolors = [COL['blue'], '#5B9BD5', COL['orange'], COL['purple'], COL['green']]

print("Data loaded. Generating figures...")


# =====================================================================
# FIGURE 1: Architecture Diagram (ConEm Fig 3 style)
# =====================================================================
def generate_fig00_architecture():
    """Architecture diagram inspired by ConEm Fig 3-5.
    Clean, modular, color-coded blocks with orthogonal arrows."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_aspect('equal')

    def draw_box(ax, x, y, w, h, label, sublabel='', fc='#F0F4F8', ec=COL['blue'],
                 lw=1.2, fontsize=8.5, subfontsize=7.0, alpha=1.0):
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                              fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
                ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color=COL['dark'], zorder=3)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.18, sublabel,
                    ha='center', va='center', fontsize=subfontsize,
                    color=COL['gray'], zorder=3)

    def draw_arrow(ax, x1, y1, x2, y2, color=COL['blue'], lw=1.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    connectionstyle='arc3,rad=0'),
                    zorder=1)

    def draw_curved_arrow(ax, x1, y1, x2, y2, color=COL['blue'], lw=1.0, rad=0.3):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    connectionstyle=f'arc3,rad={rad}'),
                    zorder=1)

    # === Row 1: Input Layer ===
    draw_box(ax, 0.3, 6.2, 2.5, 1.2, 'Structured Features',
             r'$\mathbf{x}_t$: OHLCV, returns', COL['lb'], COL['blue'])
    draw_box(ax, 0.3, 4.5, 2.5, 1.2, 'Daily Event Text',
             'policy / market / finance', COL['lg'], COL['green'])

    # === Row 2: Processing Layer ===
    # Base predictor
    draw_box(ax, 3.5, 6.2, 2.8, 1.2, 'Structured Base Predictor',
             r'$z_t^{\mathrm{base}} = f_{\mathrm{base}}(\mathbf{x}_t)$', '#F8F9FA', COL['blue'])

    # LEF extraction
    draw_box(ax, 3.5, 4.5, 2.8, 0.6, 'LEF Extraction', '', COL['lg'], COL['green'], fontsize=8)

    # Semantic encoder
    draw_box(ax, 3.5, 3.5, 2.8, 0.6, 'Semantic Encoder',
             r'$h_t^{\mathrm{lef}}$', '#E8F5E9', COL['green'], fontsize=8)

    # Event memory
    draw_box(ax, 3.5, 2.5, 2.8, 0.6, 'Rolling Event Memory',
             r'$M_t$: 7d/14d/30d', '#E8F5E9', COL['green'], fontsize=8)

    # === Row 3: Fusion Layer ===
    # Cross-module conditioning
    draw_box(ax, 7.2, 5.0, 2.8, 1.0, 'Cross-Module Conditioning',
             'align structured + semantic', '#F3E5F5', COL['purple'])

    # Residual head
    draw_box(ax, 7.2, 3.5, 2.8, 1.0, 'Residual Head',
             r'$\Delta z_t = f_{\mathrm{res}}(x_t, h_t)$', '#FFF3E0', COL['red'])

    # Gate
    draw_box(ax, 7.2, 2.0, 2.8, 1.0, 'Learned Gate',
             r'$g_t = \sigma(f_{\mathrm{gate}}(x_t, h_t))$', '#FFF8E1', COL['orange'])

    # === Row 4: Output Layer ===
    draw_box(ax, 10.8, 4.0, 2.8, 1.5, 'Final Prediction',
             r'$z_t^{\mathrm{final}} = z_t^{\mathrm{base}} + g_t \cdot \Delta z_t$' + '\n' +
             r'$\hat{y}_{t+H} = \mathbb{1}[\sigma(z_t^{\mathrm{final}}) > \tau]$',
             '#FAFAFA', COL['dark'], fontsize=9, subfontsize=7.5)

    # === Arrows ===
    # Input -> Processing
    draw_arrow(ax, 2.8, 6.8, 3.5, 6.8, COL['blue'])
    draw_arrow(ax, 2.8, 5.1, 3.5, 4.8, COL['green'])
    draw_arrow(ax, 2.8, 5.1, 3.5, 3.8, COL['green'])

    # Processing -> Fusion
    draw_arrow(ax, 6.3, 6.8, 7.2, 5.8, COL['blue'])
    draw_arrow(ax, 6.3, 6.8, 7.2, 4.0, COL['blue'])
    draw_arrow(ax, 6.3, 4.1, 7.2, 5.5, COL['green'])
    draw_arrow(ax, 6.3, 3.8, 7.2, 4.2, COL['green'])
    draw_arrow(ax, 6.3, 2.8, 7.2, 2.5, COL['green'])

    # Fusion -> Output
    draw_arrow(ax, 10.0, 5.5, 10.8, 5.2, COL['purple'])
    draw_arrow(ax, 10.0, 4.0, 10.8, 4.5, COL['red'])
    draw_arrow(ax, 10.0, 2.5, 10.8, 4.2, COL['orange'])

    # === Legend ===
    legend_y = 0.8
    for i, (label, color) in enumerate([('Numeric branch', COL['blue']),
                                         ('Semantic branch', COL['green']),
                                         ('Fusion', COL['purple']),
                                         ('Gating', COL['orange']),
                                         ('Residual', COL['red'])]):
        ax.plot([1 + i*2.5, 1.4 + i*2.5], [legend_y, legend_y], color=color, lw=2.5)
        ax.text(1.5 + i*2.5, legend_y, label, fontsize=7.5, va='center', color=COL['dark'])

    # Title
    ax.text(7, 7.6, 'DRAEM Architecture',
            ha='center', fontsize=13, fontweight='bold', color=COL['dark'])

    save(fig, 'fig00_architecture_v2')


def generate_fig01_qef_construction():
    """Minimal LEF overview with cleaner typography.
    Kept as a backup asset; the manuscript can use fig08 when a more formal pipeline view is preferred."""
    fig = plt.figure(figsize=(12.2, 6.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.12], wspace=0.22)

    # Panel A: simplified pipeline overview
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    def box(x, y, w, h, title, body='', fc='#F8FAFC', ec=COL['blue'], tsize=9.8, bsize=8.0):
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                              fc=fc, ec=ec, lw=1.0)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.66, title, ha='center', va='center', fontsize=tsize,
                fontweight='bold', color=COL['dark'])
        if body:
            ax.text(x + w/2, y + h*0.33, body, ha='center', va='center', fontsize=bsize,
                    color=COL['gray'], linespacing=1.25)

    def arrow(x1,y1,x2,y2,color=COL['blue']):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='-|>', lw=1.1, color=color,
                                    shrinkA=4, shrinkB=4, connectionstyle='arc3,rad=0'))

    box(0.06,0.62,0.22,0.18,'Event documents','News / notices /\nmarket reports',fc=COL['lg'],ec=COL['green'])
    box(0.38,0.62,0.22,0.18,'LEF extraction','Direction $\cdot$ strength\nconfidence $\cdot$ relevance',fc='#EAF5EC',ec=COL['green'])
    box(0.70,0.62,0.22,0.18,'Day-level LEF','Channel-wise semantic\nvector $q_t$',fc='#FFF7E8',ec=COL['orange'])
    box(0.38,0.26,0.22,0.18,'Rolling memory','Short-term event\ncarryover $M_t$',fc='#F9F1E8',ec=COL['orange'])
    box(0.70,0.26,0.22,0.18,'Forecast input','Semantic state used by\nDRAEM',fc='#EEF4FB',ec=COL['blue'])
    arrow(0.28,0.71,0.38,0.71,COL['green'])
    arrow(0.60,0.71,0.70,0.71,COL['orange'])
    arrow(0.49,0.62,0.49,0.44,COL['orange'])
    arrow(0.60,0.35,0.70,0.35,COL['blue'])
    arrow(0.81,0.62,0.81,0.44,COL['blue'])
    ax.text(0.06,0.90,'A. LEF construction workflow',fontsize=10.5,fontweight='bold',color=COL['dark'])
    ax.text(0.06,0.10,'Clean overview only; detailed pipeline is provided separately in fig08.',fontsize=7.8,color=COL['gray'])

    # Panel B: temporal coverage / module activity
    ax = fig.add_subplot(gs[0, 1])
    merged2 = merged.copy().sort_values('date')
    merged2['ym'] = merged2['date'].dt.to_period('M').astype(str)
    abs_avail = [f'qef_{ch}_abs_signal' for ch in mods if f'qef_{ch}_abs_signal' in merged2.columns]
    monthly = merged2.groupby('ym')[abs_avail].mean()
    monthly.columns = [m.title() for m in mods if f'qef_{m}_abs_signal' in merged2.columns]
    monthly.plot.area(ax=ax, color=mcolors[:len(monthly.columns)], alpha=0.86, lw=0)
    ax.set_ylabel('Monthly mean absolute signal', fontsize=9)
    ax.set_xlabel('Calendar time', fontsize=9)
    ax.set_title('B. Temporal semantic coverage by channel', fontsize=10.5,
                 fontweight='bold', loc='left', color=COL['dark'])
    ax.legend(frameon=True, fontsize=7.2, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.02))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=COL['light'], alpha=0.55)
    for label in ax.get_xticklabels():
        label.set_rotation(40)
        label.set_fontsize(6.5)

    fig.suptitle('LEF construction overview and temporal coverage',
                 fontsize=12.4, fontweight='bold', color=COL['dark'], y=0.98)
    save(fig, 'fig01_qef_construction_coverage')


# =====================================================================
# FIGURE 2: Stage-I Improvement Bar Chart (ConEm Fig 7 style)
# =====================================================================
def generate_fig02_improvement():
    """Stage-I percentage improvement over strongest baseline.
    Follows ConEm Fig 7: grouped bar chart by horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    # Get our method and best baseline
    ours = df_b[df_b['Method'] == 'DRAEM'].iloc[0]
    # Find best baseline per horizon
    baselines = df_b[df_b['Method'] != 'DRAEM']

    metrics = ['MCC', 'BAcc', 'DA']
    horizons = ['H1', 'H5', 'H7']
    horizon_labels = ['H1 (1-day)', 'H5 (5-day)', 'H7 (7-day)']

    for ax_idx, (metric, ax) in enumerate(zip(metrics, axes)):
        x = np.arange(len(horizons))
        width = 0.35

        ours_vals = []
        best_base_vals = []
        improvements = []

        for h in horizons:
            our_val = ours[f'{h}_{metric}']
            best_base = baselines[f'{h}_{metric}'].max()
            ours_vals.append(our_val)
            best_base_vals.append(best_base)
            if metric == 'MCC':
                # For MCC, improvement = absolute difference
                imp = (our_val - best_base)
            else:
                # For BAcc/DA, improvement = relative percentage
                imp = ((our_val - best_base) / best_base) * 100 if best_base > 0 else 0
            improvements.append(imp)

        bars1 = ax.bar(x - width/2, best_base_vals, width, label='Best Baseline',
                       color=COL['lb'], edgecolor=COL['blue'], linewidth=0.8)
        bars2 = ax.bar(x + width/2, ours_vals, width, label='DRAEM',
                       color=COL['orange'], edgecolor=COL['dark'], linewidth=0.8, alpha=0.85)

        # Add improvement annotations
        for i, (bv, ov, imp) in enumerate(zip(best_base_vals, ours_vals, improvements)):
            if metric == 'MCC':
                label = f'+{imp:.3f}' if imp > 0 else f'{imp:.3f}'
            else:
                label = f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%'
            color = COL['green'] if imp > 0 else COL['red']
            ax.annotate(label, xy=(x[i] + width/2, ov), xytext=(0, 8),
                       textcoords='offset points', ha='center', fontsize=7.5,
                       fontweight='bold', color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(horizon_labels, fontsize=9)
        ax.set_ylabel(metric, fontsize=10, fontweight='bold')
        ax.set_title(f'{metric}', fontsize=11, fontweight='bold', color=COL['dark'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color=COL['light'], alpha=0.7)
        if ax_idx == 0:
            ax.legend(frameon=True, fontsize=8, loc='upper left')

    fig.suptitle('Stage-I Model-to-Model Comparison: Improvement over Best Baseline',
                 fontsize=12, fontweight='bold', color=COL['dark'], y=1.02)
    plt.tight_layout()
    save(fig, 'fig02_stage1_improvement_v2')


# =====================================================================
# FIGURE 3: Density Comparison (ConEm Fig 8 style)
# =====================================================================
def generate_fig03_density():
    """Density comparison of MCC distributions across horizons.
    Follows ConEm Fig 8: overlaid KDE plots."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    horizons = ['H1', 'H5', 'H7']
    for ax, h in zip(axes, horizons):
        mcc_col = f'{h}_MCC'

        # Get MCC values for each group
        groups = df_b.groupby('Group')
        for group_name, group_df in groups:
            if group_name == 'Our Method':
                continue
            vals = group_df[mcc_col].values
            color = GROUP_COLORS.get(group_name, COL['gray'])
            # Plot individual points as rug plot
            for v in vals:
                ax.axvline(v, color=color, alpha=0.3, linewidth=0.8, ymin=0, ymax=0.1)

        # Our method - single point with marker
        ours_mcc = df_b[df_b['Method'] == 'DRAEM'][mcc_col].values[0]
        ax.axvline(ours_mcc, color=COL['orange'], linewidth=2.5, linestyle='--',
                   label=f'DRAEM ({ours_mcc:.3f})')

        # Best baseline
        baseline_mcc = df_b[df_b['Method'] != 'DRAEM'][mcc_col].max()
        best_method = df_b[df_b[mcc_col] == baseline_mcc]['Method'].values[0]
        ax.axvline(baseline_mcc, color=COL['blue'], linewidth=2, linestyle='-',
                   label=f'Best Baseline: {best_method} ({baseline_mcc:.3f})')

        ax.set_xlabel('MCC', fontsize=9)
        ax.set_title(h, fontsize=11, fontweight='bold', color=COL['dark'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=True, fontsize=7, loc='upper right')
        ax.grid(axis='x', color=COL['light'], alpha=0.7)

    axes[0].set_ylabel('Density', fontsize=9)
    fig.suptitle('MCC Distribution Across Horizons',
                 fontsize=12, fontweight='bold', color=COL['dark'], y=1.02)
    plt.tight_layout()
    save(fig, 'fig03_density_v2')


# =====================================================================
# FIGURE 4: Ablation Study (ConEm Fig 11 style)
# =====================================================================
def generate_fig04_ablation():
    """Ablation study visualization.
    Follows ConEm Fig 11: grouped bar chart with per-metric comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mean across seeds
    abl_mean = df_a.groupby('variant').agg({
        'MCC': 'mean', 'DA': 'mean', 'BAcc': 'mean', 'F1': 'mean'
    }).reset_index()

    # Sort by MCC descending
    abl_mean = abl_mean.sort_values('MCC', ascending=True)

    variants = abl_mean['variant'].tolist()
    x = np.arange(len(variants))

    # Panel A: MCC comparison
    ax = axes[0]
    colors = [COL['orange'] if v == 'Full' else COL['lb'] for v in variants]
    bars = ax.barh(x, abl_mean['MCC'], color=colors, edgecolor=COL['blue'],
                   linewidth=0.8, height=0.6)
    for i, (v, m) in enumerate(zip(variants, abl_mean['MCC'])):
        ax.text(m + 0.003, i, f'{m:.3f}', va='center', fontsize=8, color=COL['dark'])
    ax.set_yticks(x)
    ax.set_yticklabels([v.replace('_', ' ') for v in variants], fontsize=8.5)
    ax.set_xlabel('MCC', fontsize=10, fontweight='bold')
    ax.set_title('MCC', fontsize=11, fontweight='bold', color=COL['dark'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color=COL['light'], alpha=0.7)

    # Panel B: DA comparison
    ax = axes[1]
    colors = [COL['orange'] if v == 'Full' else COL['lg'] for v in variants]
    bars = ax.barh(x, abl_mean['DA'], color=colors, edgecolor=COL['green'],
                   linewidth=0.8, height=0.6)
    for i, (v, d) in enumerate(zip(variants, abl_mean['DA'])):
        ax.text(d + 0.3, i, f'{d:.1f}%', va='center', fontsize=8, color=COL['dark'])
    ax.set_yticks(x)
    ax.set_yticklabels([v.replace('_', ' ') for v in variants], fontsize=8.5)
    ax.set_xlabel('DA (%)', fontsize=10, fontweight='bold')
    ax.set_title('DA', fontsize=11, fontweight='bold', color=COL['dark'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color=COL['light'], alpha=0.7)

    fig.suptitle('Ablation Study: Component Contribution Analysis',
                 fontsize=12, fontweight='bold', color=COL['dark'], y=1.02)
    plt.tight_layout()
    save(fig, 'fig04_ablation_v2')


# =====================================================================
# FIGURE 5: Window Gain Landscape (ConEm Fig 7 style)
# =====================================================================
def generate_fig05_window_gain():
    """Stage-II window analysis with actual populated panels.
    Uses numeric horizons in the source table."""
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), sharex=False)

    horizon_specs = [(5, 'H5'), (7, 'H7')]
    for ax, (h_num, h_label) in zip(axes, horizon_specs):
        hw = df_w[df_w['H'] == h_num].copy()
        if hw.empty:
            ax.text(0.5, 0.5, f'No {h_label} windows available', ha='center', va='center', fontsize=11, color=COL['gray'])
            ax.axis('off')
            continue

        # Focus on strongest-magnitude windows for readability
        summary = hw.groupby('window').agg({
            'gap_ours_minus_reference': 'mean',
            'ours_n': 'mean'
        }).reset_index()
        summary = summary.sort_values('gap_ours_minus_reference')
        if len(summary) > 8:
            summary = pd.concat([summary.head(4), summary.tail(4)], axis=0)

        y = np.arange(len(summary))
        vals = summary['gap_ours_minus_reference'].to_numpy()
        ns = summary['ours_n'].fillna(summary['ours_n'].median()).to_numpy()
        widths = 0.42 + 0.28 * (ns / ns.max())
        colors = [COL['green'] if g > 0 else COL['red'] for g in vals]
        ax.barh(y, vals, color=colors, alpha=0.82, edgecolor='white', linewidth=0.9, height=widths)
        for i, v in enumerate(vals):
            ax.text(v + (0.004 if v >= 0 else -0.004), i, f'{v:+.3f}',
                    va='center', ha='left' if v >= 0 else 'right', fontsize=8.1, color=COL['dark'])
        ax.set_yticks(y)
        ax.set_yticklabels(summary['window'].tolist(), fontsize=7.8)
        ax.axvline(0, color='k', linewidth=0.6, linestyle=':')
        ax.set_xlabel('MCC gain versus reference baseline', fontsize=9)
        ax.set_title(f'{h_label} event windows', fontsize=10.8, fontweight='bold', color=COL['dark'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color=COL['light'], alpha=0.65)
        ax.text(0.02, 0.04, 'Bar thickness $propto$ window sample size', transform=ax.transAxes,
                fontsize=7.7, color=COL['gray'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COL['light']))

    fig.suptitle('Stage-II event-window analysis: where DRAEM gains are concentrated',
                 fontsize=12.2, fontweight='bold', color=COL['dark'], y=0.98)
    plt.tight_layout()
    save(fig, 'fig05_window_gain_v2')


# =====================================================================
# FIGURE 6: Qualitative Prediction Case (ConEm Fig 9-10 style)
# =====================================================================
def generate_fig06_qualitative():
    """Qualitative case study showing base vs final prediction.
    Follows ConEm Fig 9-10: time-series overlay plots."""
    if daily is None:
        print("  SKIP fig06: no daily predictions")
        return

    # Get H7 predictions
    h7 = daily[daily['H'] == 7].copy() if 'H' in daily.columns else daily.copy()
    if len(h7) < 30:
        print("  SKIP fig06: insufficient H7 data")
        return

    # Take a recent window of 60 days
    h7 = h7.sort_values('date').tail(60).reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    dates = pd.to_datetime(h7['date'])
    x = np.arange(len(dates))

    # Panel A: Base vs Final probability
    ax = axes[0]
    base_prob = h7['base_prob'].astype(float)
    final_prob = base_prob + h7.get('residual_delta', pd.Series(0, index=h7.index)).astype(float)
    final_prob = final_prob.clip(0, 1)

    ax.fill_between(x, base_prob, final_prob,
                    where=final_prob > base_prob,
                    alpha=0.3, color=COL['green'], label='Semantic boost')
    ax.fill_between(x, base_prob, final_prob,
                    where=final_prob <= base_prob,
                    alpha=0.3, color=COL['red'], label='Semantic suppression')
    ax.plot(x, base_prob, color=COL['blue'], linewidth=1.5,
            label=r'$p_{\mathrm{base}}$', marker='', linestyle='-')
    ax.plot(x, final_prob, color=COL['orange'], linewidth=1.5,
            label=r'$p_{\mathrm{final}}$', marker='', linestyle='-')
    ax.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.set_ylabel('Probability', fontsize=9)
    ax.set_title('A. Base vs Final Prediction Probability', fontsize=10,
                 fontweight='bold', loc='left', color=COL['dark'])
    ax.legend(frameon=True, fontsize=8, ncol=4, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=COL['light'], alpha=0.5)

    # Panel B: Gate activation
    ax = axes[1]
    gate = h7['gate'].astype(float)
    colors_gate = [COL['orange'] if g > 0.5 else COL['lb'] for g in gate]
    ax.bar(x, gate, color=colors_gate, width=0.8, alpha=0.7)
    ax.axhline(0.5, color=COL['red'], linewidth=1, linestyle='--', alpha=0.7)
    ax.set_ylabel(r'Gate $g_t$', fontsize=9)
    ax.set_title(r'B. Gate Activation ($g_t > 0.5$: semantic injection active)',
                 fontsize=10, fontweight='bold', loc='left', color=COL['dark'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', color=COL['light'], alpha=0.5)

    # Panel C: Actual direction vs prediction
    ax = axes[2]
    if 'y_true' in h7.columns:
        y_true = h7['y_true'].astype(int)
        correct = (h7['pred'].astype(int) == y_true)
        colors_pred = [COL['green'] if c else COL['red'] for c in correct]
        ax.bar(x, y_true * 0.5 + 0.25, color=colors_pred, width=0.8, alpha=0.6)
        ax.set_yticks([0.25, 0.75])
        ax.set_yticklabels(['Down', 'Up'], fontsize=8)
    ax.set_ylabel('Direction', fontsize=9)
    ax.set_xlabel('Time (recent 60 trading days)', fontsize=9)
    ax.set_title('C. Actual Direction (green=correct, red=incorrect)',
                 fontsize=10, fontweight='bold', loc='left', color=COL['dark'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=COL['light'], alpha=0.5)

    # X-axis labels
    tick_idx = np.linspace(0, len(dates)-1, min(10, len(dates)), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([dates.iloc[i].strftime('%Y-%m') for i in tick_idx],
                        rotation=30, fontsize=8)

    fig.suptitle('Qualitative Case: Semantic Residual Correction Mechanism (H7)',
                 fontsize=12, fontweight='bold', color=COL['dark'], y=1.01)
    plt.tight_layout()
    save(fig, 'fig06_qualitative_v2')


# =====================================================================
# FIGURE 7: LEF Module Contribution Dashboard (enhanced)
# =====================================================================
def generate_fig07_qef_dashboard():
    """Compact LEF dashboard for manuscript use.
    Reduced from five crowded panels to four clearer views."""
    fig = plt.figure(figsize=(12.6, 7.8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.05],
                           hspace=0.36, wspace=0.28)

    abs_cols = [f'qef_{ch}_abs_signal' for ch in mods if f'qef_{ch}_abs_signal' in merged.columns]
    sig_cols = [f'qef_{ch}_signal' for ch in mods if f'qef_{ch}_signal' in merged.columns]
    conf_cols = [f'qef_{ch}_confidence' for ch in mods if f'qef_{ch}_confidence' in merged.columns]
    act_cols = [f'qef_{ch}_active' for ch in mods if f'qef_{ch}_active' in merged.columns]

    means = [merged[c].mean() for c in abs_cols]
    confs = [merged[c].mean() for c in conf_cols]
    act_means = [(merged[c] > 0).mean() * 100 for c in act_cols]

    # Panel A: channel intensity ranking
    ax = fig.add_subplot(gs[0, 0])
    order = np.argsort(means)[::-1]
    ax.bar(np.arange(len(order)), np.array(means)[order],
           color=np.array(mcolors)[order], edgecolor='white', linewidth=0.9, width=0.62)
    for j, idx in enumerate(order):
        ax.text(j, means[idx] + max(means) * 0.03, f'{means[idx]:.3f}',
                ha='center', va='bottom', fontsize=8.2, fontweight='bold', color=COL['dark'])
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(np.array(mnames)[order], fontsize=8.5)
    ax.set_ylabel('Mean absolute signal', fontsize=9)
    ax.set_title('A. Which channels carry stronger semantic signals?', loc='left',
                 fontsize=10, fontweight='bold', color=COL['dark'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=COL['light'], alpha=0.7)

    # Panel B: activity-confidence map with clearer semantics
    ax = fig.add_subplot(gs[0, 1])
    sizes = np.array(means) / max(means) * 420 + 100
    ax.scatter(act_means, confs, s=sizes, c=mcolors[:len(means)],
               edgecolor='white', linewidth=0.9, alpha=0.95)
    for i, name in enumerate(mnames[:len(means)]):
        ax.text(act_means[i] + 1.2, confs[i], name, fontsize=8.2, va='center', color=COL['dark'])
    ax.set_xlabel('Active trading days (%)', fontsize=9)
    ax.set_ylabel('Mean confidence', fontsize=9)
    ax.set_title('B. Which channels are frequent and reliable?', loc='left',
                 fontsize=10, fontweight='bold', color=COL['dark'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color=COL['light'], alpha=0.7)
    ax.text(0.02, 0.04, 'Bubble size = mean absolute signal', transform=ax.transAxes,
            fontsize=7.8, color=COL['gray'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COL['light']))

    # Panel C: activity ratio as direct ranking
    ax = fig.add_subplot(gs[1, 0])
    order_act = np.argsort(act_means)
    ax.barh(np.arange(len(order_act)), np.array(act_means)[order_act],
            color=np.array(mcolors)[order_act], edgecolor='white', linewidth=0.9, height=0.55)
    for j, idx in enumerate(order_act):
        ax.text(act_means[idx] + 1.0, j, f'{act_means[idx]:.1f}%', va='center', fontsize=8.2, color=COL['dark'])
    ax.set_yticks(np.arange(len(order_act)))
    ax.set_yticklabels(np.array(mnames)[order_act], fontsize=8.5)
    ax.set_xlabel('Share of active days (%)', fontsize=9)
    ax.set_title('C. How often is each channel activated?', loc='left',
                 fontsize=10, fontweight='bold', color=COL['dark'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color=COL['light'], alpha=0.7)

    # Panel D: temporal mixture kept as one overview panel
    ax = fig.add_subplot(gs[1, 1])
    merged2 = merged.copy().sort_values('date')
    merged2['ym'] = merged2['date'].dt.to_period('M').astype(str)
    if abs_cols:
        monthly = merged2.groupby('ym')[abs_cols].mean()
        monthly.columns = [m.title() for m in mods if f'qef_{m}_abs_signal' in merged2.columns]
        monthly.plot.area(ax=ax, color=mcolors[:len(monthly.columns)], alpha=0.86, lw=0)
        ax.set_ylabel('Monthly mean absolute signal', fontsize=9)
        ax.set_xlabel('')
        ax.set_title('D. How does the module mix shift over time?', loc='left',
                     fontsize=10, fontweight='bold', color=COL['dark'])
        ax.legend(frameon=True, fontsize=7.1, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.02))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color=COL['light'], alpha=0.55)
        for label in ax.get_xticklabels():
            label.set_rotation(40)
            label.set_fontsize(6.5)

    fig.suptitle('LEF module dashboard: intensity, activity, confidence, and temporal mix',
                 fontsize=12.6, fontweight='bold', color=COL['dark'], y=0.99)
    save(fig, 'fig07_qef_dashboard_v2')


# =====================================================================
# FIGURE 8: LEF pipeline detail schematic
# =====================================================================
def generate_fig08_qef_pipeline_detail():
    """Detailed LEF pipeline schematic with non-overlapping text layout."""
    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    def box(x, y, w, h, title, body='', fc='#F8FAFC', ec=COL['blue'], title_size=10, body_size=8.2):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle='round,pad=0.012,rounding_size=0.02',
                              fc=fc, ec=ec, lw=1.1)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.67, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=COL['dark'])
        if body:
            ax.text(x + w / 2, y + h * 0.34, body, ha='center', va='center',
                    fontsize=body_size, color=COL['gray'], linespacing=1.28)
        return rect

    def arrow(x1, y1, x2, y2, color=COL['blue']):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.15,
                                    shrinkA=4, shrinkB=4, connectionstyle='arc3,rad=0'))

    box(0.04, 0.60, 0.17, 0.20,
        'Daily event corpus',
        'News, policy notices,\nindustry reports',
        fc=COL['lg'], ec=COL['green'])
    ax.text(0.125, 0.55, 'Corpus coverage: 850 events, 19.3% trading days',
            ha='center', va='center', fontsize=8.0, color=COL['gray'])

    box(0.28, 0.56, 0.21, 0.26,
        'LEF Tokenizer',
        'Dir $\cdot$ Str $\cdot$ Conf $\cdot$ Unc\nDecay $\cdot$ Rel $\cdot$ Active',
        fc='#EAF5EC', ec=COL['green'], title_size=10.2, body_size=8.6)

    box(0.56, 0.58, 0.18, 0.22,
        'Structural sub-signals',
        'Policy $\cdot$ Market $\cdot$ Finance\nTrade $\cdot$ Compliance $\cdot$ Global',
        fc='#EEF4FB', ec=COL['blue'], title_size=9.6, body_size=8.1)

    box(0.79, 0.60, 0.16, 0.20,
        'Daily LEF vector',
        '$q_t$ with channel-wise\ndirection and intensity',
        fc='#FFF7E8', ec=COL['orange'])

    box(0.79, 0.24, 0.16, 0.18,
        'Rolling event memory',
        '$M_t$: short-term carryover\nand disagreement cues',
        fc='#F9F1E8', ec=COL['orange'])

    box(0.28, 0.18, 0.20, 0.18,
        'Day-level aggregation',
        'Aggregate documents into one\ntrading-day semantic summary',
        fc='#F8FAFC', ec=COL['purple'])

    box(0.56, 0.16, 0.18, 0.22,
        'Auxiliary statistics',
        'Signal strength\nDisagreement\nAvailability / recency',
        fc='#F8FAFC', ec=COL['purple'])

    arrow(0.21, 0.69, 0.28, 0.69, COL['green'])
    arrow(0.49, 0.69, 0.56, 0.69, COL['green'])
    arrow(0.74, 0.69, 0.79, 0.69, COL['blue'])

    # Lowered arrows to avoid text overlap
    arrow(0.385, 0.56, 0.385, 0.36, COL['green'])
    arrow(0.65, 0.58, 0.65, 0.39, COL['blue'])
    arrow(0.48, 0.27, 0.56, 0.27, COL['purple'])
    arrow(0.74, 0.27, 0.79, 0.31, COL['orange'])
    arrow(0.87, 0.60, 0.87, 0.42, COL['orange'])

    ax.text(0.04, 0.92, 'Detailed LEF construction pipeline', fontsize=13,
            fontweight='bold', color=COL['dark'])
    ax.text(0.04, 0.88,
            'The schematic expands the document-to-factor conversion step and shows how day-level LEF vectors and rolling event memory are formed.',
            fontsize=8.6, color=COL['gray'])

    save(fig, 'fig08_qef_pipeline_detail')


# =====================================================================
# Generate all figures
# =====================================================================
if __name__ == '__main__':
    print("Generating fig00: Architecture...")
    generate_fig00_architecture()

    print("Generating fig01: LEF Construction Overview...")
    generate_fig01_qef_construction()

    print("Generating fig02: Stage-I Improvement...")
    generate_fig02_improvement()

    print("Generating fig03: Density Comparison...")
    generate_fig03_density()

    print("Generating fig04: Ablation Study...")
    generate_fig04_ablation()

    print("Generating fig05: Window Gain...")
    generate_fig05_window_gain()

    print("Generating fig06: Qualitative Case...")
    generate_fig06_qualitative()

    print("Generating fig07: LEF Dashboard...")
    generate_fig07_qef_dashboard()

    print("Generating fig08: LEF Pipeline Detail...")
    generate_fig08_qef_pipeline_detail()

    print("\nAll figures generated!")
    print(f"Output directory: {OUT}")
