"""
Fig 8: Cross-Market Generalization - DRAEM v3
HBEA (Hubei), SHEA (Shanghai), CEA (Chongqing)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / 'figures'

cm = pd.read_csv(Path(__file__).resolve().parent.parent.parent / 'results' / 'cross_market' / 'cross_market_cleaned.csv')
cm['Model'] = cm['Model'].str.strip()

df_v3 = pd.read_csv(OUT + 'cross_market_v3_results.csv')

markets = ['HBEA', 'SHEA', 'CEA']
horizons = ['H1', 'H5', 'H7']

# v1 (DRAEM) results from cross_market_cleaned.csv
v1_data = {
    'HBEA': {'H1': 43.8, 'H5': 47.4, 'H7': 48.8},
    'SHEA': {'H1': 31.8, 'H5': 51.8, 'H7': 52.7},
    'CEA':  {'H1': 35.8, 'H5': 46.9, 'H7': 47.8},
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors_v1 = {'HBEA': '#FF8F00', 'SHEA': '#FFB300', 'CEA': '#FFA000'}
colors_v3 = {'HBEA': '#1565C0', 'SHEA': '#1976D2', 'CEA': '#1E88E5'}
x = np.arange(len(horizons))
width = 0.35

for j, market in enumerate(markets):
    ax = axes[j]
    v1_vals = [v1_data[market][h] for h in horizons]
    v3_row = df_v3[df_v3['market'] == market]
    v3_vals = [float(v3_row[v3_row['horizon'] == h]['da_dir'].values[0]) for h in horizons]

    bars1 = ax.bar(x - width/2, v1_vals, width, label='v1 (DRAEM)', color=colors_v1[market], alpha=0.7, edgecolor='white')
    bars2 = ax.bar(x + width/2, v3_vals, width, label='v3 (Asymmetric)', color=colors_v3[market], alpha=0.9, edgecolor='white')

    for bar, v in zip(bars1, v1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=9, color='#795548')
    for bar, v in zip(bars2, v3_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=9, color=colors_v3[market], fontweight='bold')

    # Improvement annotations
    for i, h in enumerate(horizons):
        delta = v3_vals[i] - v1_vals[i]
        if abs(delta) > 1.0:
            color = 'green' if delta > 0 else 'red'
            ax.annotate(f'{'+' if delta > 0 else ''}{delta:.1f}', xy=(x[i] + width/2, max(v1_vals[i], v3_vals[i]) + 1.5),
                       ha='center', fontsize=8.5, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=12)
    ax.set_ylabel('Direction Accuracy (%)', fontsize=11)
    ax.set_title(f'{market}', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 65)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.2)
    if j == 0:
        ax.legend(loc='lower right', fontsize=9)

fig.suptitle('Fig 8: Cross-Market Generalization — Direction Accuracy (DA_dir)\n'
             'DRAEM v1 vs Asymmetric DRAEM (ours)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'fig8_cross_market.pdf', dpi=150, bbox_inches='tight')
plt.savefig(OUT + 'fig8_cross_market.png', dpi=150, bbox_inches='tight')
print("Fig 8 saved!")

# Print summary table
print("\nCross-market summary:")
print(f"{'Market':<8} {'Horizon':<8} {'v1':<8} {'v3':<8} {'Δ':<8}")
for market in markets:
    for h in horizons:
        v1v = v1_data[market][h]
        v3v = float(df_v3[(df_v3['market']==market) & (df_v3['horizon']==h)]['da_dir'].values[0])
        delta = v3v - v1v
        sig = '*' if abs(delta) > 3 else ''
        print(f"{market:<8} {h:<8} {v1v:<8.1f} {v3v:<8.1f} {delta:+.1f}{sig}")
