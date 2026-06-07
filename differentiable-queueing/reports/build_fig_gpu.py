# -*- coding: utf-8 -*-
"""Build §4.3.1 GPU benchmark figure from gpu_benchmark.json + gpu_benchmark_large.json"""
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

d_small = json.load(open('results/reproduction/gpu_benchmark.json'))
d_large = json.load(open('results/reproduction/gpu_benchmark_large.json'))

# CPU results
cpu = [(r['batch_size'], r['throughput_events_per_s'], r['median_time_s'])
       for r in d_small['results'] if r['device'] == 'cpu']
# GPU results (combine small + large)
gpu_small = [(r['batch_size'], r['throughput_events_per_s'], r['median_time_s'])
             for r in d_small['results'] if r['device'] == 'cuda']
gpu_large = [(r['batch_size'], r['throughput_events_per_s'], r['median_time_s'])
             for r in d_large['results']]
gpu = sorted(gpu_small + gpu_large)

cpu_B  = [r[0] for r in cpu]; cpu_T = [r[1] for r in cpu]; cpu_t = [r[2] for r in cpu]
gpu_B  = [r[0] for r in gpu]; gpu_T = [r[1] for r in gpu]; gpu_t = [r[2] for r in gpu]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: throughput (events per second)
ax = axes[0]
ax.loglog(cpu_B, cpu_T, 'o-', color='C1', lw=2, markersize=8, label='CPU (16 cores)')
ax.loglog(gpu_B, gpu_T, 's-', color='C0', lw=2, markersize=8, label='GPU (A100-80GB)')
ax.set_xlabel('batch size B')
ax.set_ylabel('throughput (events / second)')
ax.set_title('§4.3.1 PATHWISE simulation throughput — CPU vs GPU')
ax.legend(loc='lower right'); ax.grid(alpha=0.3, which='both')
for B, T in zip(gpu_B, gpu_T):
    if B in (1024, 65536):
        ax.annotate(f'{T/1e6:.1f}M ev/s', xy=(B, T), xytext=(B*0.5, T*1.3),
                    fontsize=8, color='C0',
                    arrowprops=dict(arrowstyle='-', color='C0', lw=0.5))

# Right: speedup (GPU / CPU at matching B)
ax = axes[1]
cpu_by_B = dict(zip(cpu_B, cpu_t))
gpu_by_B = dict(zip(gpu_B, gpu_t))
common = sorted(set(cpu_B) & set(gpu_B))
speedup = [cpu_by_B[b] / gpu_by_B[b] for b in common]
ax.semilogx(common, speedup, 'D-', color='C2', lw=2, markersize=9)
for b, s in zip(common, speedup):
    ax.annotate(f'{s:.2f}×', (b, s), xytext=(2, 5), textcoords='offset points', fontsize=8)
ax.axhline(1.0, ls=':', color='gray')
ax.set_xlabel('batch size B')
ax.set_ylabel('speedup (CPU time / GPU time)')
ax.set_title('GPU/CPU speedup vs batch size (crossover at B~1024)')
ax.grid(alpha=0.3, which='both')
# Show extrapolated GPU-only point at largest batch
B_max = gpu_B[-1]
ax.annotate(f'@ B={B_max}:  GPU throughput\n{gpu_T[-1]/1e6:.1f}M ev/s\n(CPU would take {gpu_T[-1]/cpu_T[-1]*gpu_t[-1]/3600:.1f}h)',
            xy=(0.95, 0.05), xycoords='axes fraction', ha='right', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

fig.suptitle('§4.3.1  GPU unlocks ≥84× higher throughput on criss-cross simulator (T=1000)', y=1.02)
fig.tight_layout()
fig.savefig('reports/figures/fig_section4_3_1_gpu_benchmark.png')
fig.savefig('reports/figures/fig_section4_3_1_gpu_benchmark.pdf')
plt.close(fig)
print('saved reports/figures/fig_section4_3_1_gpu_benchmark.{png,pdf}')
