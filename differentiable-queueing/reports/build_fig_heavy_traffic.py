# -*- coding: utf-8 -*-
"""E5 heavy-traffic curve — PATHWISE cost as ρ → 1, multiple gap values."""
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Parse the log file (E5 stdout) since the JSON write got killed mid-run
log_path = 'logs/direct/E5_heavy_traffic.out'
with open(log_path) as f:
    text = f.read()

# Pattern: "  PATHWISE rho=0.80: cost=3.01±0.03 (722.2s)" under "--- Gap = 0.01 ---"
sections = re.split(r'--- (Gap = [\d.]+) ---', text)
# sections is [pre, label1, body1, label2, body2, ...]
data = {}  # gap -> [(rho, cost, std)]
for i in range(1, len(sections), 2):
    label = sections[i]
    body = sections[i+1]
    if 'PATHWISE' not in body:
        continue
    gap = float(label.replace('Gap = ', ''))
    pw_rows = []
    for m in re.finditer(r'PATHWISE rho=([\d.]+): cost=([\d.]+)±([\d.]+)', body):
        pw_rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    if pw_rows:
        data[gap] = pw_rows

if not data:
    print("No PATHWISE data found in log"); raise SystemExit

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
colors = {0.01: 'C3', 0.5: 'C1', 1.0: 'C0'}
for gap in sorted(data.keys()):
    rhos = [r[0] for r in data[gap]]
    costs = [r[1] for r in data[gap]]
    stds  = [r[2] for r in data[gap]]
    ax.errorbar(rhos, costs, yerr=stds, marker='o', lw=2, ms=6, capsize=3,
                color=colors.get(gap, 'gray'), label=f'gap = {gap}')

ax.set_xlabel('traffic intensity ρ')
ax.set_ylabel('avg cost')
ax.set_title('E5 (revision) — Heavy-traffic curve: PATHWISE cost as ρ → 1')
ax.legend(title='holding-cost gap', loc='upper left')
ax.grid(alpha=0.3)
fig.text(0.5, 0.01,
         'Cost grows as ρ → 1, sharper for larger gap. Reviewer-1\'s heavy-traffic ask.',
         ha='center', fontsize=9, style='italic')
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig('reports/figures/fig_E5_heavy_traffic.png')
fig.savefig('reports/figures/fig_E5_heavy_traffic.pdf')
plt.close(fig)
print('saved reports/figures/fig_E5_heavy_traffic.{png,pdf}')
print(f'  gaps included: {sorted(data.keys())}')
print(f'  rhos per gap: {len(data[list(data.keys())[0]])}')
