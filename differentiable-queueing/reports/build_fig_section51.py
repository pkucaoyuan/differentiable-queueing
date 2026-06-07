# -*- coding: utf-8 -*-
"""§5.1 gradient cossim figure — heatmap across (ρ, policy) for PATHWISE vs REINFORCE."""
import json, glob, os, statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Find all gradient_comparison JSONs
files = sorted(glob.glob('results/reproduction/gradient_comparison_*.json'))
if not files:
    print("No gradient_comparison_*.json found"); raise SystemExit

# Parse rho from filename
def parse_rho(path):
    base = os.path.basename(path).replace('.json', '')
    return float(base.split('rho')[-1])

rhos = []
data = {}
for f in files:
    rho = parse_rho(f)
    rhos.append(rho)
    with open(f) as fh:
        d = json.load(fh)
    data[rho] = d

rhos = sorted(set(rhos))
policies = sorted({p for d in data.values() for p in d.keys() if p in ('sPR', 'sMW', 'sMP')})
if not policies:
    # Maybe schema differs — print one and exit
    print("Schema not (rho,policy) keyed — first file keys:", list(data[rhos[0]].keys())[:5])
    raise SystemExit

# Build matrices: rho × policy → mean cossim
def get_means(method):
    g = np.full((len(rhos), len(policies)), np.nan)
    for i, rho in enumerate(rhos):
        for j, p in enumerate(policies):
            v = data[rho].get(p, {})
            # Schema: each policy has list of cossims for that method
            if method in v:
                vals = v[method]
                if isinstance(vals, list) and vals:
                    g[i, j] = statistics.mean(vals)
            elif 'pathwise_cosines' in v and method == 'PATHWISE':
                vals = v['pathwise_cosines']
                if vals: g[i, j] = statistics.mean(vals)
            elif 'reinforce_cosines' in v and method == 'REINFORCE':
                vals = v['reinforce_cosines']
                if vals: g[i, j] = statistics.mean(vals)
    return g

pw = get_means('PATHWISE')
rf = get_means('REINFORCE')

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
vmin = min(np.nanmin(pw), np.nanmin(rf), -1)
vmax = max(np.nanmax(pw), np.nanmax(rf), 1)

for ax, mat, title in zip(axes[:2], [pw, rf], ['PATHWISE', 'REINFORCE']):
    im = ax.imshow(mat, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(policies))); ax.set_xticklabels(policies)
    ax.set_yticks(range(len(rhos))); ax.set_yticklabels([f'ρ={r}' for r in rhos])
    ax.set_title(f'{title} cosine similarity')
    for i in range(len(rhos)):
        for j in range(len(policies)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(v) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[2]
diff = pw - rf
im = ax.imshow(diff, cmap='RdBu', vmin=-1.0, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(policies))); ax.set_xticklabels(policies)
ax.set_yticks(range(len(rhos))); ax.set_yticklabels([f'ρ={r}' for r in rhos])
ax.set_title('PATHWISE − REINFORCE')
for i in range(len(rhos)):
    for j in range(len(policies)):
        v = diff[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('§5.1 Gradient cosine similarity vs ground truth (mid-canonical: 30 samples × 30 estimators)', y=1.02)
fig.savefig('reports/figures/fig_section51_gradient_cossim.png')
fig.savefig('reports/figures/fig_section51_gradient_cossim.pdf')
plt.close(fig)
print('saved reports/figures/fig_section51_gradient_cossim.{png,pdf}')
