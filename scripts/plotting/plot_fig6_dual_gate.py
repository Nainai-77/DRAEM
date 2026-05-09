"""
Fig 6: Dual-Gate Mechanism Analysis
Reads actual data from final_v2_all.csv
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / 'figures'

# Load final v2 results
df = pd.read_csv(OUT + 'final_v2_all.csv')
print("Final v2 data:")
print(df.to_string())

# Load ablation results (No-CrossAttn)
nc_data = {
    'H1': {'alpha_reg': [0.505, 0.553], 'alpha_dir': [0.600, 0.592]},
    'H5': {'alpha_reg': [0.539, 0.573], 'alpha_dir': [0.601, 0.631]},
    'H7': {'alpha_reg': [0.566, 0.493], 'alpha_dir': [0.532, 0.514]},
}

colors = {'H1': '#2196F3', 'H5': '#FF9800', 'H7': '#E91E63'}
markers = {'H1': 'o', 'H5': 's', 'H7': '^'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Left: Full model scatter ──────────────────────────────────────
ax = axes[0]
for h in ['H1', 'H5', 'H7']:
    sub = df[df['horizon'] == h]
    ar = sub['alpha_reg'].values
    ad = sub['alpha_dir'].values
    da = sub['da_dir'].values
    ax.scatter(ar, ad, c=colors[h], marker=markers[h], s=130,
               edgecolors='white', linewidths=1.5, zorder=5, label=h)
    # Mean point (larger)
    ax.scatter(ar.mean(), ad.mean(), c=colors[h], marker=markers[h],
               s=280, edgecolors='black', linewidths=2, zorder=6, alpha=0.7)
    # Annotate seeds
    for _, row in sub.iterrows():
        ax.annotate(f"s={int(row['seed'])}", (row['alpha_reg'], row['alpha_dir']),
                    textcoords='offset points', xytext=(5, 5), fontsize=7, color='gray')
    delta = ar.mean() - ad.mean()
    print(f"{h}: α_reg={ar.mean():.3f}, α_dir={ad.mean():.3f}, Δ={delta:+.3f}")

# Diagonal
lims = [0.25, 0.80]
ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1.5, label=r'$\alpha_{\rm reg}=\alpha_{\rm dir}$')
ax.fill_between(lims, lims, [lims[1]]*2, alpha=0.04, color='red', label='α_reg > α_dir zone')
ax.fill_between(lims, [lims[0]]*2, lims, alpha=0.04, color='blue', label='α_dir > α_reg zone')

ax.set_xlabel(r'$\alpha_{\rm reg}$ (Magnitude Gate)', fontsize=13)
ax.set_ylabel(r'$\alpha_{\rm dir}$ (Direction Gate)', fontsize=13)
ax.set_title('Full DRAEM: Learned Gate Weights\n(Dual-Gate Asymmetric Decoupling)', fontsize=12, fontweight='bold')
ax.set_xlim(0.40, 0.75)
ax.set_ylim(0.25, 0.65)
ax.set_aspect('1.3')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)

ax.text(0.68, 0.32, 'Text dominates\nmagnitude path', fontsize=9, color='#1565C0', style='italic')

# ── Right: Ablation comparison ────────────────────────────────────
ax2 = axes[1]

# Full model means
full_means = {}
for h in ['H1', 'H5', 'H7']:
    sub = df[df['horizon'] == h]
    full_means[h] = (sub['alpha_reg'].mean(), sub['alpha_dir'].mean())
    ax2.scatter(sub['alpha_reg'], sub['alpha_dir'], c=colors[h], marker=markers[h],
                s=100, edgecolors='white', linewidths=1.5, alpha=0.6, label=f'{h} Full')
    ax2.scatter(full_means[h][0], full_means[h][1], c=colors[h], marker=markers[h],
                s=250, edgecolors='black', linewidths=2, zorder=6, alpha=0.5)

# No-CrossAttn
for h in ['H1', 'H5', 'H7']:
    nc_reg = nc_data[h]['alpha_reg']
    nc_dir = nc_data[h]['alpha_dir']
    ax2.scatter(nc_reg, nc_dir, c=colors[h], marker=markers[h],
                s=100, edgecolors='black', linewidths=1.5, alpha=0.9,
                facecolors='none', label=f'{h} No-CrossAttn')
    # Arrows from Full mean to No-CrossAttn mean
    nx = np.mean(nc_reg)
    ny = np.mean(nc_dir)
    fx, fy = full_means[h]
    ax2.annotate('', xy=(nx, ny), xytext=(fx, fy),
                 arrowprops=dict(arrowstyle='->', color=colors[h], lw=2.0, alpha=0.7))

ax2.plot(lims, lims, 'k--', alpha=0.4, linewidth=1.5)
ax2.set_xlabel(r'$\alpha_{\rm reg}$', fontsize=13)
ax2.set_ylabel(r'$\alpha_{\rm dir}$', fontsize=13)
ax2.set_title('Gate Shift: Full → w/o Cross-Attn\n(arrows show direction of change)', fontsize=12, fontweight='bold')
ax2.set_xlim(0.40, 0.75)
ax2.set_ylim(0.25, 0.65)
ax2.set_aspect('1.3')
ax2.grid(True, alpha=0.3)

# Manual legend
p1 = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Full (solid)', linestyle='None')
p2 = plt.Line2D([0], [0], marker='o', color='gray', markerfacecolor='white', markeredgewidth=2, markersize=10, label='No-CrossAttn (outline)', linestyle='None')
ax2.legend(handles=[p1, p2], loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(OUT + 'fig6_dual_gate.pdf', dpi=150, bbox_inches='tight')
plt.savefig(OUT + 'fig6_dual_gate.png', dpi=150, bbox_inches='tight')
print("\nFig 6 saved!")
