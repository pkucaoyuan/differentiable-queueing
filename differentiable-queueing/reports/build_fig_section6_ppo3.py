# -*- coding: utf-8 -*-
"""§6 Figure 12: PPO 3 variants comparison (vanilla / +BC / +WC) on criss-cross.

Uses the existing PPO/*.json files from the April 2026 PPO runs.
Reproduces paper Figure 12 narrative: vanilla unstable, +BC stable but weak,
+WC stable AND strong (beats cμ).
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

# Load the 3 PPO variant results + cμ baseline
ppo_wc  = json.load(open('PPO/WC_results.json'))
ppo_bc  = json.load(open('PPO/vanilla_bc_results.json'))
ppo_van = json.load(open('PPO/vanilla_results.json'))
# cμ baseline JSON has separate entries; use the criss-cross matching one (first)
cmu     = json.load(open('PPO/cmu_results.json'))
cmu_cost = list(cmu.values())[0]['avg_cost']  # 17.44 — criss-cross cμ baseline

# Build training-iteration curves
iters = list(range(len(ppo_wc['test_cost'])))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# ── Left panel: full range (log scale to fit vanilla's 2000+) ──
ax = axes[0]
ax.semilogy(iters, ppo_van['test_cost'], 'o-', color='C3', label='vanilla PPO', lw=1.8, ms=3)
ax.semilogy(iters, ppo_bc['test_cost'],  's-', color='C2', label='PPO + BC init', lw=1.8, ms=3)
ax.semilogy(iters, ppo_wc['test_cost'],  '^-', color='C0', label='PPO-WC (work-conserving)', lw=1.8, ms=3)
ax.axhline(cmu_cost, color='k', ls='--', lw=1.5, label=f'cμ baseline ({cmu_cost:.2f})', alpha=0.7)
ax.set_xlabel('iteration')
ax.set_ylabel('test cost (log scale)')
ax.set_title('§6 Figure 12 (full) — PPO 3 variants on criss-cross')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3, which='both')

# ── Right panel: zoom to converged region (linear, range 10-60) ──
ax = axes[1]
ax.plot(iters, ppo_bc['test_cost'],  's-', color='C2', label='PPO + BC init', lw=1.8, ms=3)
ax.plot(iters, ppo_wc['test_cost'],  '^-', color='C0', label='PPO-WC', lw=1.8, ms=3)
ax.axhline(cmu_cost, color='k', ls='--', lw=1.5, label=f'cμ baseline ({cmu_cost:.2f})', alpha=0.7)
ax.set_xlabel('iteration')
ax.set_ylabel('test cost')
ax.set_ylim(10, 60)
ax.set_title('Zoom: PPO+WC < cμ; PPO+BC degrades over training')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
# Annotate end values
ax.annotate(f"WC final {ppo_wc['test_cost'][-1]:.2f}", xy=(iters[-1], ppo_wc['test_cost'][-1]),
            xytext=(70, 14), fontsize=9, color='C0',
            arrowprops=dict(arrowstyle='->', color='C0', lw=0.5))
ax.annotate(f"BC final {ppo_bc['test_cost'][-1]:.2f}", xy=(iters[-1], ppo_bc['test_cost'][-1]),
            xytext=(70, 52), fontsize=9, color='C2',
            arrowprops=dict(arrowstyle='->', color='C2', lw=0.5))

fig.suptitle(
    '§6 Figure 12 — vanilla PPO collapses, +BC degrades, +WC stable < cμ\n'
    'NOTE: WC curve = our PPO run (job 8556856, 67h CPU); '
    'vanilla & +BC curves = upstream-provided data (not independently re-run)',
    y=1.04, fontsize=10
)
fig.tight_layout()
fig.savefig('reports/figures/fig_section6_ppo3_variants.png')
fig.savefig('reports/figures/fig_section6_ppo3_variants.pdf')
plt.close(fig)
print('saved reports/figures/fig_section6_ppo3_variants.{png,pdf}')

# Export CSV companion
import csv
with open('reports/figures/fig_section6_ppo3_variants.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['iteration', 'vanilla_PPO', 'PPO_BC', 'PPO_WC', 'cmu_baseline'])
    for i, (v, b, wc) in enumerate(zip(ppo_van['test_cost'], ppo_bc['test_cost'], ppo_wc['test_cost'])):
        w.writerow([i, v, b, wc, cmu_cost])
print('saved reports/figures/fig_section6_ppo3_variants.csv')
