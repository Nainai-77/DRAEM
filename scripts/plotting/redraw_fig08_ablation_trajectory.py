#!/usr/bin/env python3
"""Redraw Figure 8: Ablation trajectory with darker, more visible line colors.
Two-panel figure: (A) MCC degradation, (B) DA misleading stability.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parent.parent.parent / 'results'
OUT = ROOT / 'figures_q1_redraw'

# ── Style ──
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

# Dark, saturated colors — high contrast on white background
COL = {
    'mcc_line':   '#1A3A5C',   # Very dark navy for MCC trajectory
    'mcc_fill':   '#1A3A5C',   # Same for fill
    'da_line':    '#8B1A1A',   # Very dark red for DA trajectory
    'da_fill':    '#8B1A1A',
    'full_dot':   '#0D47A1',   # Deep blue for Full model marker
    'warn':       '#BF360C',   # Deep orange-red for warning markers
    'accent':     '#2E7D32',   # Deep green for correct-direction
    'dark':       '#1A1A1A',   # Near-black for text
    'gray':       '#555555',   # Dark gray for annotations
    'light_gray': '#E0E0E0',   # Grid lines
}

# ── Load ablation data ──
abl = pd.read_csv(ROOT / 'final_ablation_result.csv')

# Define ordering: Full → w/o text → w/o gate → w/o cross-attn
variant_order = ['full', 'no_text', 'no_gate', 'no_crossattention']
display_labels = ['Full\nDRAEM', 'w/o\ntext', 'w/o\ngate', 'w/o\ncross-attn']

# Aggregate across seeds (mean)
abl_mean = abl.groupby('variant').agg({'MCC': 'mean', 'DA': 'mean'}).reset_index()
abl_mean['variant'] = pd.Categorical(abl_mean['variant'], categories=variant_order, ordered=True)
abl_mean = abl_mean.sort_values('variant')

x = np.arange(len(variant_order))
mcc_vals = abl_mean['MCC'].values
da_vals = abl_mean['DA'].values * 100  # Convert to percentage

# ── Create figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

# === Panel A: MCC degradation trajectory ===
# Fill area under the trajectory to emphasize the drop
ax1.fill_between(x, mcc_vals, alpha=0.15, color=COL['mcc_fill'], zorder=1)
# Main trajectory line — thick and dark
ax1.plot(x, mcc_vals, color=COL['mcc_line'], linewidth=2.8, marker='o',
         markersize=9, markerfacecolor='white', markeredgecolor=COL['mcc_line'],
         markeredgewidth=2.2, zorder=3, label='MCC')

# Highlight the Full model point
ax1.plot(0, mcc_vals[0], marker='o', markersize=12, markerfacecolor=COL['full_dot'],
         markeredgecolor='white', markeredgewidth=2.0, zorder=5)
ax1.annotate(f'{mcc_vals[0]:.3f}', xy=(0, mcc_vals[0]),
             xytext=(0, 0.035), textcoords='offset points',
             ha='center', fontsize=9, fontweight='bold', color=COL['full_dot'])

# Annotate each point with its value
for i, (v, m) in enumerate(zip(variant_order, mcc_vals)):
    if i == 0:
        continue
    offset_y = 0.025 if m >= 0 else -0.035
    color = COL['warn'] if m < 0 else COL['gray']
    ax1.annotate(f'{m:.3f}', xy=(i, m), xytext=(0, offset_y * 100),
                 textcoords='offset points', ha='center', fontsize=9,
                 fontweight='bold', color=color)

# Arrow annotation for the cliff drop
ax1.annotate('Cliff drop\n(gate removal)',
             xy=(2, mcc_vals[2]), xytext=(2.4, mcc_vals[1] + 0.02),
             fontsize=8.5, color=COL['warn'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COL['warn'], lw=1.5),
             ha='center')

ax1.axhline(0, color=COL['dark'], linewidth=0.7, linestyle='--', alpha=0.6, zorder=1)
ax1.set_xticks(x)
ax1.set_xticklabels(display_labels, fontsize=9)
ax1.set_ylabel('MCC', fontsize=11, fontweight='bold')
ax1.set_xlabel('Ablation variant', fontsize=10)
ax1.set_title('A  MCC degradation (primary metric)', loc='left', fontsize=11,
              fontweight='bold', color=COL['dark'])
ax1.spines[['top', 'right']].set_visible(False)
ax1.grid(axis='y', color=COL['light_gray'], linewidth=0.7, alpha=0.8)
ax1.set_ylim(min(mcc_vals) - 0.06, max(mcc_vals) + 0.06)

# === Panel B: DA misleading stability ===
ax2.fill_between(x, da_vals, alpha=0.15, color=COL['da_fill'], zorder=1)
ax2.plot(x, da_vals, color=COL['da_line'], linewidth=2.8, marker='s',
         markersize=9, markerfacecolor='white', markeredgecolor=COL['da_line'],
         markeredgewidth=2.2, zorder=3, label='DA')

# Highlight Full model
ax2.plot(0, da_vals[0], marker='s', markersize=12, markerfacecolor=COL['full_dot'],
         markeredgecolor='white', markeredgewidth=2.0, zorder=5)
ax2.annotate(f'{da_vals[0]:.1f}\\%', xy=(0, da_vals[0]),
             xytext=(0, 8), textcoords='offset points',
             ha='center', fontsize=9, fontweight='bold', color=COL['full_dot'])

# Annotate each point
for i, (v, d) in enumerate(zip(variant_order, da_vals)):
    if i == 0:
        continue
    offset_y = 8 if d >= da_vals[0] else -14
    color = COL['warn'] if v == 'no_gate' else COL['gray']
    ax2.annotate(f'{d:.1f}\\%', xy=(i, d), xytext=(0, offset_y),
                 textcoords='offset points', ha='center', fontsize=9,
                 fontweight='bold', color=color)

# Warning annotation for misleading DA
ax2.annotate('Superficially high DA\nmasks MCC collapse',
             xy=(2, da_vals[2]), xytext=(1.2, da_vals[2] + 1.8),
             fontsize=8.5, color=COL['warn'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COL['warn'], lw=1.5),
             ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                       edgecolor=COL['warn'], alpha=0.9))

ax2.axhline(50, color=COL['dark'], linewidth=0.7, linestyle='--', alpha=0.6, zorder=1)
ax2.set_xticks(x)
ax2.set_xticklabels(display_labels, fontsize=9)
ax2.set_ylabel('DA (\\%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Ablation variant', fontsize=10)
ax2.set_title('B  Directional accuracy (auxiliary metric)', loc='left', fontsize=11,
              fontweight='bold', color=COL['dark'])
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(axis='y', color=COL['light_gray'], linewidth=0.7, alpha=0.8)
ax2.set_ylim(min(da_vals) - 3, max(da_vals) + 3)

fig.suptitle('Ablation trajectory under the H7 extreme-disagreement event-conflict window',
             y=1.02, fontsize=12.5, fontweight='bold', color=COL['dark'])
plt.tight_layout()

# Save
fig.savefig(OUT / 'fig05_ablation_trajectory.png', bbox_inches='tight', facecolor='white')
fig.savefig(OUT / 'fig05_ablation_trajectory.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved: fig05_ablation_trajectory.{pdf,png}')

# Also save to v3 directory
import shutil
v3_dir = ROOT / 'figures_q1_redraw_v3'
v3_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(OUT / 'fig05_ablation_trajectory.pdf', v3_dir / 'fig05_ablation_trajectory.pdf')
shutil.copy2(OUT / 'fig05_ablation_trajectory.png', v3_dir / 'fig05_ablation_trajectory.png')
print('Copied to figures_q1_redraw_v3/')
