# -*- coding: utf-8 -*-
"""§5.1 Figure 4 — full reproduction across 4 networks + 4 rhos + 3 policies.

Aggregates the new GPU-canonical results:
- criss_cross_bh (4 rhos × 3 policies) — gradient_gpu_canonical_v2.json
- reentrant_2/3/4 (1 rho × 3 policies)  — gradient_gpu_reentrant_*.json
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Load
cc = json.load(open('results/reproduction/gradient_gpu_canonical_v2.json'))
r2 = json.load(open('results/reproduction/gradient_gpu_reentrant_2.json'))
r3 = json.load(open('results/reproduction/gradient_gpu_reentrant_3.json'))
r4 = json.load(open('results/reproduction/gradient_gpu_reentrant_4.json'))

# ─── Panel A: criss-cross heatmap (rho × policy) PATHWISE vs REINFORCE ───
policies = ['sPR', 'sMW', 'sMP']
rhos     = ['0.8', '0.9', '0.95', '0.99']

pw_grid = np.zeros((len(rhos), len(policies)))
rf_grid = np.zeros((len(rhos), len(policies)))
for i, rho in enumerate(rhos):
    for j, pol in enumerate(policies):
        d = cc[rho][pol]
        pw_grid[i, j] = d['pathwise_mean']
        rf_grid[i, j] = d['reinforce_mean']

fig = plt.figure(figsize=(15, 5))

# Left: PATHWISE heatmap
ax = fig.add_subplot(1, 4, 1)
im = ax.imshow(pw_grid, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(policies))); ax.set_xticklabels(policies)
ax.set_yticks(range(len(rhos))); ax.set_yticklabels([f'ρ={r}' for r in rhos])
ax.set_title('PATHWISE cossim\n(criss-cross)')
for i in range(len(rhos)):
    for j in range(len(policies)):
        ax.text(j, i, f'{pw_grid[i,j]:+.2f}', ha='center', va='center', fontsize=9,
                color='white' if abs(pw_grid[i,j])>0.5 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(1, 4, 2)
im = ax.imshow(rf_grid, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(policies))); ax.set_xticklabels(policies)
ax.set_yticks(range(len(rhos))); ax.set_yticklabels([f'ρ={r}' for r in rhos])
ax.set_title('REINFORCE cossim\n(criss-cross)')
for i in range(len(rhos)):
    for j in range(len(policies)):
        ax.text(j, i, f'{rf_grid[i,j]:+.2f}', ha='center', va='center', fontsize=9,
                color='white' if abs(rf_grid[i,j])>0.5 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(1, 4, 3)
diff = pw_grid - rf_grid
amx = abs(diff).max()
im = ax.imshow(diff, cmap='RdBu_r', vmin=-amx, vmax=amx, aspect='auto')
ax.set_xticks(range(len(policies))); ax.set_xticklabels(policies)
ax.set_yticks(range(len(rhos))); ax.set_yticklabels([f'ρ={r}' for r in rhos])
ax.set_title('PATHWISE − REINFORCE\n(criss-cross)')
for i in range(len(rhos)):
    for j in range(len(policies)):
        ax.text(j, i, f'{diff[i,j]:+.2f}', ha='center', va='center', fontsize=9,
                color='black' if abs(diff[i,j])<amx*0.5 else 'white')
plt.colorbar(im, ax=ax, fraction=0.046)

# Right: PATHWISE vs REINFORCE on reentrant_2/3/4 at rho=0.95
ax = fig.add_subplot(1, 4, 4)
nets = [('r_2', r2['0.95']), ('r_3', r3['0.95']), ('r_4', r4['0.95'])]
x = np.arange(len(nets) * len(policies))
labels = []
pw_vals, rf_vals = [], []
for net_name, net_d in nets:
    for pol in policies:
        d = net_d[pol]
        pw_vals.append(d['pathwise_mean'])
        rf_vals.append(d['reinforce_mean'])
        labels.append(f'{net_name}/{pol}')
w = 0.4
ax.bar(x - w/2, pw_vals, w, label='PATHWISE', color='C0')
ax.bar(x + w/2, rf_vals, w, label='REINFORCE', color='C3')
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('cosine similarity')
ax.set_title('Reentrant networks\n(ρ=0.95)')
ax.axhline(0, color='k', lw=0.5)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

fig.suptitle(
    '§5.1 Figure 4 reproduction — Gradient cosine similarity vs ground truth\n'
    'PATHWISE > REINFORCE on sPR uniformly; advantage on sMW/sMP grows in heavy traffic',
    y=1.02, fontsize=11
)
fig.tight_layout()
fig.savefig('reports/figures/fig_section51_gradient_full.png')
fig.savefig('reports/figures/fig_section51_gradient_full.pdf')
plt.close(fig)
print('saved reports/figures/fig_section51_gradient_full.{png,pdf}')

# Export companion CSV
import csv
with open('reports/figures/fig_section51_gradient_full.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['network', 'rho', 'policy', 'pathwise_mean', 'pathwise_std',
                'reinforce_mean', 'reinforce_std', 'n_pw', 'n_rf'])
    for net_label, data in [('criss_cross_bh', cc), ('reentrant_2', r2),
                            ('reentrant_3', r3), ('reentrant_4', r4)]:
        for rho in data.keys():
            for pol in policies:
                d = data[rho][pol]
                w.writerow([net_label, rho, pol,
                            d['pathwise_mean'], d['pathwise_std'],
                            d['reinforce_mean'], d['reinforce_std'],
                            d.get('n_pw_kept', '-'), d.get('n_rf_kept', '-')])
print('saved reports/figures/fig_section51_gradient_full.csv')
