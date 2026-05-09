"""
Fig 3: Main Results Bar Chart (v3 vs all baselines, real data)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / 'figures'

# Real baseline data from baselines_final.csv
models = {
    'XGBoost':         [44.8, 49.2, 43.2],   # XGBoost H1 from single run
    'LSTM':            [42.9, 47.1, 41.7],
    'DLinear':         [42.1, 47.1, 41.6],
    'Transformer':     [43.8, 52.5, 43.5],
    'Informer':        [42.6, 46.6, 44.0],
    'iTransformer':    [42.8, 52.4, 50.2],
    'PatchTST':        [44.0, 50.2, 48.2],
    'TimesNet':        [42.8, 47.1, 47.2],
    'DRAEM\_ens (v1)': [46.2, 59.5, 46.7],
    '{\\bf Full DRAEM}':[45.1, 54.6, 49.8],
}
ours_best = {'H1': 48.1, 'H5': 60.1, 'H7': 53.5}

horizons = ['H1', 'H5', 'H7']
x = np.arange(len(horizons))
width = 0.08

# Sort by H7
sorted_items = sorted(models.items(), key=lambda kv: kv[1][2], reverse=True)
model_names = [m[0] for m in sorted_items]
values_list = [m[1] for m in sorted_items]

bar_centers = np.arange(len(model_names)) * 0.10

fig, ax = plt.subplots(figsize=(14, 5.5))
for i, (mn, vals) in enumerate(zip(model_names, values_list)):
    if 'ours' in mn or ('DRAEM' in mn and 'ens' not in mn):
        color = '#C62828'; edge = '#C62828'; ew = 2
    elif 'DRAEM' in mn:
        color = '#FF8F00'; edge = '#FF8F00'; ew = 1.5
    else:
        color = '#546E7A'; edge = 'white'; ew = 0.5
    ax.bar(bar_centers[i] + x * (len(model_names) * 0.10 + 0.05),
           vals, width, label=mn, color=color, edgecolor=edge,
           linewidth=ew, alpha=0.9)
    for j, v in enumerate(vals):
        ax.text(bar_centers[i] + x[j] * (len(model_names) * 0.10 + 0.05),
                v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold' if 'DRAEM' in mn else 'normal')

# Best seed stars
our_idx = model_names.index('{\\bf Full DRAEM}')
for j, h in enumerate(horizons):
    bx = bar_centers[our_idx] + x[j] * (len(model_names) * 0.10 + 0.05)
    ax.scatter(bx, ours_best[h] + 0.8, marker='*', color='#FFD700', s=180, zorder=15, edgecolors='black', linewidths=0.5)
    ax.text(bx + 0.02, ours_best[h] + 2.0, f'★{ours_best[h]:.1f}', fontsize=7, color='#C62828', fontweight='bold')

# Reference lines
for j, h in enumerate(horizons):
    v1_val = models['DRAEM\\_ens (v1)'][j]
    ax.axhline(y=v1_val, color='#FF8F00', linestyle=':', alpha=0.5, linewidth=1.2)
    ax.text(-0.02, v1_val + 0.3, f'v1:{v1_val:.1f}', fontsize=7, color='#FF8F00', ha='right')

ax.set_xticks(x * (len(model_names) * 0.10 + 0.05) + bar_centers.mean())
ax.set_xticklabels(horizons, fontsize=13)
ax.set_ylabel('Direction Accuracy (%)', fontsize=12)
ax.set_title('Main Results: Direction Accuracy (DA_dir) across Prediction Horizons\n(★ = best seed of Full DRAEM)', fontsize=13, fontweight='bold')
ax.set_ylim(35, 68)
ax.set_xlim(-0.15, bar_centers[-1] + x[-1] * (len(model_names) * 0.10 + 0.05) + 0.15)
ax.legend(loc='upper left', fontsize=8, ncol=5, framealpha=0.9)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
plt.savefig(OUT + 'fig3_main_results_da.pdf', dpi=150, bbox_inches='tight')
plt.savefig(OUT + 'fig3_main_results_da.png', dpi=150, bbox_inches='tight')
print("Fig 3 saved!")

# ── RMSE ────────────────────────────────────────────────────────────
rmse_models = {
    'XGBoost':         [0.040, 0.018, 0.115],
    'LSTM':            [0.042, 0.017, 0.073],
    'DLinear':         [0.079, 0.056, 0.240],
    'Transformer':     [0.042, 0.013, 0.096],
    'Informer':        [0.040, 0.018, 0.088],
    'iTransformer':    [0.043, 0.018, 0.088],
    'PatchTST':        [0.040, 0.015, 0.080],
    'TimesNet':        [0.101, 0.055, 0.175],
    'DRAEM\_ens (v1)': [0.038, 0.012, 0.064],
    '{\\bf Full DRAEM}':[0.074, 0.021, 0.092],
}

fig2, ax2 = plt.subplots(figsize=(14, 5))
for i, (mn, vals) in enumerate(zip(model_names, [rmse_models[m] for m in model_names])):
    if 'ours' in mn or ('DRAEM' in mn and 'ens' not in mn):
        color = '#C62828'; edge = '#C62828'; ew = 2
    elif 'DRAEM' in mn:
        color = '#FF8F00'; edge = '#FF8F00'; ew = 1.5
    else:
        color = '#546E7A'; edge = 'white'; ew = 0.5
    ax2.bar(bar_centers[i] + x * (len(model_names) * 0.10 + 0.05),
            vals, width, label=mn, color=color, edgecolor=edge, linewidth=ew, alpha=0.9)
    for j, v in enumerate(vals):
        ax2.text(bar_centers[i] + x[j] * (len(model_names) * 0.10 + 0.05),
                 v + 0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=6.5)

ax2.set_xticks(x * (len(model_names) * 0.10 + 0.05) + bar_centers.mean())
ax2.set_xticklabels(horizons, fontsize=13)
ax2.set_ylabel('RMSE', fontsize=12)
ax2.set_title('Main Results: RMSE across Prediction Horizons (Lower is Better)', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 0.26)
ax2.set_xlim(-0.15, bar_centers[-1] + x[-1] * (len(model_names) * 0.10 + 0.05) + 0.15)
ax2.legend(loc='upper left', fontsize=8, ncol=5, framealpha=0.9)
ax2.grid(axis='y', alpha=0.25)

plt.tight_layout()
plt.savefig(OUT + 'fig3b_main_results_rmse.pdf', dpi=150, bbox_inches='tight')
plt.savefig(OUT + 'fig3b_main_results_rmse.png', dpi=150, bbox_inches='tight')
print("Fig 3b saved!")
