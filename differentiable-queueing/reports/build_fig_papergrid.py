# -*- coding: utf-8 -*-
"""§5.2 paper-grid figure — 5 gaps × 4 alphas heatmap PATHWISE vs REINFORCE."""
import json, statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

pw = json.load(open('results/reproduction/cmu_papergrid_pathwise.json'))
rf = json.load(open('results/reproduction/cmu_papergrid_reinforce.json'))

alphas = sorted(pw.keys(), key=float)
gaps   = sorted(pw[alphas[0]].keys(), key=lambda x: -float(x))

def grid(data):
    g = np.zeros((len(alphas), len(gaps)))
    for i, a in enumerate(alphas):
        for j, gp in enumerate(gaps):
            costs = [r['avg_cost'] for r in data[a][gp]]
            g[i, j] = statistics.mean(costs)
    return g

pw_g = grid(pw); rf_g = grid(rf)
diff = (rf_g - pw_g) / pw_g * 100

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
vmin = min(pw_g.min(), rf_g.min()); vmax = max(pw_g.max(), rf_g.max())

for ax, mat, title in zip(axes[:2], [pw_g, rf_g], ['PATHWISE', 'REINFORCE']):
    im = ax.imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(gaps))); ax.set_xticklabels(gaps)
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
    ax.set_xlabel('gap'); ax.set_ylabel('alpha (step size)')
    ax.set_title(f'{title} avg cost')
    for i in range(len(alphas)):
        for j in range(len(gaps)):
            ax.text(j, i, f'{mat[i,j]:.1f}', ha='center', va='center',
                    color='white' if mat[i,j] > (vmin+vmax)/2 else 'black', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[2]
amx = abs(diff).max()
im = ax.imshow(diff, cmap='RdBu_r', vmin=-amx, vmax=amx, aspect='auto')
ax.set_xticks(range(len(gaps))); ax.set_xticklabels(gaps)
ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
ax.set_xlabel('gap'); ax.set_ylabel('alpha')
ax.set_title('(RF − PW) / PW  [%]')
for i in range(len(alphas)):
    for j in range(len(gaps)):
        ax.text(j, i, f'{diff[i,j]:.1f}', ha='center', va='center',
                color='black' if abs(diff[i,j]) < amx*0.5 else 'white', fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('§5.2 CμRule 10-class, ρ=0.95 — PAPER GRID (5 gaps × 4 alphas, 50 trials)', y=1.02)
fig.savefig('reports/figures/fig_section52_cmu_papergrid.png')
fig.savefig('reports/figures/fig_section52_cmu_papergrid.pdf')
plt.close(fig)
print('saved reports/figures/fig_section52_cmu_papergrid.{png,pdf}')
