# -*- coding: utf-8 -*-
"""§7 Polyak avg-iterate vs last-iterate comparison figure."""
import json, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

d = json.load(open('results/reproduction/polyak_eval.json'))
envs = list(d.keys())
last = [d[e]['last_iterate_cost'] for e in envs]
avg  = [d[e]['polyak_avg_cost'] for e in envs]
diff = [d[e]['diff_pct'] for e in envs]

labels = [e.replace('criss_cross_bh', 'criss-cross').replace('reentrant_', 'r_') for e in envs]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: absolute costs
ax = axes[0]
x = np.arange(len(envs)); w = 0.4
ax.bar(x - w/2, last, w, label='Last iterate', color='C1')
ax.bar(x + w/2, avg,  w, label='Polyak avg (last 50 epochs)', color='C0')
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_yscale('log')
ax.set_ylabel('avg cost (T=50K, B=100 episodes, log scale)')
ax.set_title('§7 Last-iterate vs Polyak-averaged policy cost')
ax.legend()
ax.grid(alpha=0.3, axis='y', which='both')

# Right: relative diff (Polyak − last) / last %
ax = axes[1]
colors = ['C2' if d_pct < 0 else 'C3' for d_pct in diff]
ax.bar(x, diff, color=colors)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('(Polyak − Last) / Last  [%]')
ax.set_title('Polyak averaging effect (negative = better)')
for i, d_pct in enumerate(diff):
    y = d_pct + (3 if d_pct >= 0 else -3)
    va = 'bottom' if d_pct >= 0 else 'top'
    ax.text(i, y, f'{d_pct:+.0f}%', ha='center', va=va, fontsize=8)
ax.grid(alpha=0.3, axis='y')

fig.suptitle('§7 Reproduction protocol: Polyak averaging changes cost in 5/10 envs by >30%', y=1.02)
fig.tight_layout()
fig.savefig('reports/figures/fig_section7_polyak_vs_last.png')
fig.savefig('reports/figures/fig_section7_polyak_vs_last.pdf')
plt.close(fig)
print('saved reports/figures/fig_section7_polyak_vs_last.{png,pdf}')
