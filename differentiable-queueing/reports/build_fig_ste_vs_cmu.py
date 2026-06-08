# -*- coding: utf-8 -*-
"""§7 Tables 1-5 figure — STE-trained policy vs cμ baseline across 11 networks."""
import json, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

with open('results/reproduction/ste_vs_cmu_benchmark.json') as f:
    d = json.load(f)

envs = list(d.keys())
labels = [e.replace('criss_cross_bh', 'criss-cross').replace('reentrant_', 'r_') for e in envs]
cmu_m = np.array([d[e]['cmu_mean'] for e in envs])
ste_m = np.array([d[e]['ste_mean'] for e in envs])
cmu_s = np.array([d[e]['cmu_std'] for e in envs])
ste_s = np.array([d[e]['ste_std'] for e in envs])
improvement = np.array([d[e]['improvement_pct'] for e in envs])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: bar chart of cμ vs STE
ax = axes[0]
x = np.arange(len(envs)); w = 0.4
ax.bar(x - w/2, cmu_m, w, yerr=cmu_s, capsize=3, color='C1', label='cμ baseline', alpha=0.85)
ax.bar(x + w/2, ste_m, w, yerr=ste_s, capsize=3, color='C0', label='PATHWISE (STE)', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel('avg cost (T=50K, B=100 episodes)')
ax.set_title('§7 Tables 1-5: STE vs cμ baseline on 11 networks')
ax.legend(); ax.grid(alpha=0.3, axis='y')

# Right: improvement %
ax = axes[1]
colors = ['C2' if i > 0 else 'C3' for i in improvement]
ax.bar(x, improvement, color=colors)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel('(cμ − STE) / cμ  [%]')
ax.set_title('STE improvement over cμ (positive = STE better)')
ax.axhline(0, color='k', lw=0.5)
for i, v in enumerate(improvement):
    ax.text(i, v + (1 if v >= 0 else -1), f'{v:+.1f}%', ha='center',
            va='bottom' if v >= 0 else 'top', fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Summary stat
n_better = sum(1 for v in improvement if v > 0)
mean_improve = np.mean(improvement)
fig.suptitle(f'§7 STE beats cμ on {n_better}/{len(envs)} networks; mean improvement {mean_improve:+.1f}%', y=1.02)
fig.tight_layout()
fig.savefig('reports/figures/fig_section7_ste_vs_cmu.png')
fig.savefig('reports/figures/fig_section7_ste_vs_cmu.pdf')
plt.close(fig)
print('saved reports/figures/fig_section7_ste_vs_cmu.{png,pdf}')
