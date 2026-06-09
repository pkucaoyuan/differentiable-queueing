# -*- coding: utf-8 -*-
"""Figure 14 benchmark summary — combines §5.2/§5.3/§6/§7 into one publication-quality figure."""
import json, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

# ─── Panel A: §5.2 PATHWISE vs REINFORCE — 10-class with paper grid ───
ax = fig.add_subplot(gs[0, 0])
pw = json.load(open('results/reproduction/cmu_papergrid_pathwise.json'))
rf = json.load(open('results/reproduction/cmu_papergrid_reinforce.json'))
alphas = sorted(pw.keys(), key=float)
gaps = sorted(pw[alphas[0]].keys(), key=lambda x: -float(x))
def get_means(d):
    g = np.zeros((len(alphas), len(gaps)))
    for i, a in enumerate(alphas):
        for j, gp in enumerate(gaps):
            g[i, j] = np.mean([r['avg_cost'] for r in d[a][gp]])
    return g
diff = (get_means(rf) - get_means(pw)) / get_means(pw) * 100
amx = abs(diff).max()
im = ax.imshow(diff, cmap='RdBu_r', vmin=-amx, vmax=amx, aspect='auto')
ax.set_xticks(range(len(gaps))); ax.set_xticklabels(gaps)
ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
ax.set_xlabel('gap'); ax.set_ylabel('α (step size)')
ax.set_title('§5.2 CμRule 10-class: |RF − PW| / PW [%]\n(40 cells; all match within ≤3.1%)')
plt.colorbar(im, ax=ax, fraction=0.046)

# ─── Panel B: §5.3 admission — PATHWISE vs SPSA on 12 networks ───
ax = fig.add_subplot(gs[0, 1])
summ = json.load(open('results/admission_control_summary.json'))
families = {'reentrant_1': ['reentrant_2.yaml', 'reentrant_3.yaml', 'reentrant_4.yaml',
                            'reentrant_5.yaml', 'reentrant_6.yaml', 'reentrant_7.yaml'],
            'reentrant_2': ['re-reentrant_2.yaml', 're-reentrant_3.yaml', 're-reentrant_4.yaml',
                            're-reentrant_5.yaml', 're-reentrant_6.yaml', 're-reentrant_7.yaml']}
Ks = [6, 9, 12, 15, 18, 21]
for fam, files in families.items():
    pw = [summ[f]['PATHWISE_B1']['mean'] for f in files]
    sp = [summ[f]['SPSA_B1000']['mean'] for f in files]
    lstyle = '-' if fam == 'reentrant_1' else '--'
    ax.plot(Ks, pw, 'o' + lstyle, color='C0', lw=2,
            label=f'PATHWISE_B1 ({fam})' if fam == 'reentrant_1' else None)
    ax.plot(Ks, sp, 's' + lstyle, color='C3', lw=2,
            label=f'SPSA_B1000 ({fam})' if fam == 'reentrant_1' else None)
ax.set_xlabel('# queue classes K'); ax.set_ylabel('final avg cost')
ax.set_title('§5.3 Admission: PATHWISE scales,\nSPSA collapses at K ≥ 15')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.axvline(15, ls=':', color='gray', alpha=0.5)

# ─── Panel C: §6 WC vs Vanilla criss-cross training curves ───
ax = fig.add_subplot(gs[0, 2])
wc = json.load(open('loss/criss_cross_bh_ppg_softmax.json'))
van = json.load(open('loss/criss_cross_bh_ppg_vanilla.json'))
wc_e = [r['epoch'] for r in wc]; wc_t = [r['test_loss'] for r in wc]
van_e = [r['epoch'] for r in van]; van_t = [r['test_loss'] for r in van]
ax.plot(wc_e, wc_t, color='C0', label=f'WC softmax (min {min(wc_t):.2f})', lw=1.8)
ax.plot(van_e[1:], van_t[1:], color='C3', label=f'Vanilla softmax (min {min(van_t):.2f})', lw=1.8)
ax.set_xlabel('epoch'); ax.set_ylabel('test cost')
ax.set_ylim(10, 26)
ax.set_title('§6 WC > Vanilla on criss-cross (+13.2%)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ─── Panel D: §4.3.1 GPU benchmark throughput ───
ax = fig.add_subplot(gs[1, 0])
d_small = json.load(open('results/reproduction/gpu_benchmark.json'))
d_large = json.load(open('results/reproduction/gpu_benchmark_large.json'))
cpu_B = [r['batch_size'] for r in d_small['results'] if r['device']=='cpu']
cpu_T = [r['throughput_events_per_s'] for r in d_small['results'] if r['device']=='cpu']
gpu_small = [(r['batch_size'], r['throughput_events_per_s']) for r in d_small['results'] if r['device']=='cuda']
gpu_large = [(r['batch_size'], r['throughput_events_per_s']) for r in d_large['results']]
gpu = sorted(gpu_small + gpu_large)
ax.loglog(cpu_B, cpu_T, 'o-', color='C1', label='CPU (16 cores)', lw=2)
ax.loglog([b for b,_ in gpu], [t for _,t in gpu], 's-', color='C0', label='GPU (A100-80GB)', lw=2)
ax.set_xlabel('batch size B'); ax.set_ylabel('throughput (events/s)')
ax.set_title('§4.3.1 GPU 84× speedup @ B=65K')
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

# ─── Panel E: §7 STE vs cμ benchmark on 10 networks ───
ax = fig.add_subplot(gs[1, 1])
d = json.load(open('results/reproduction/ste_vs_cmu_benchmark.json'))
envs = list(d.keys())
labels = [e.replace('criss_cross_bh', 'criss-cross').replace('reentrant_', 'r_') for e in envs]
improvement = [d[e]['improvement_pct'] for e in envs]
colors = ['C2' if v > 0 else 'C3' for v in improvement]
x = np.arange(len(envs))
ax.bar(x, improvement, color=colors)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel('(cμ − STE) / cμ [%]')
ax.set_title('§7 STE beats cμ on 7/10 (tie 3/10)\nmean improvement +3.4%')
ax.axhline(0, color='k', lw=0.5)
ax.grid(alpha=0.3, axis='y')

# ─── Panel F: §8 Theorem 2 (documented mismatch) ───
ax = fig.add_subplot(gs[1, 2])
d = json.load(open('results/reproduction/theorem2_validation_v2.json'))
rhos = sorted([float(k) for k in d.keys() if k != 'fit'])
gaps_v = [1 - r for r in rhos]
pw_vars = [d[str(r)]['pw_var'] for r in rhos]
rf_vars = [d[str(r)]['rf_var'] for r in rhos]
ax.loglog(gaps_v, pw_vars, 'o-', label=f'PW (fit slope {d["fit"]["pw_slope_all"]:.2f})', color='C0')
ax.loglog(gaps_v, rf_vars, 's-', label=f'RF (fit slope {d["fit"]["rf_slope_all"]:.2f})', color='C3')
ax.set_xlabel('1 − ρ'); ax.set_ylabel('estimator variance')
ax.set_title('§8 Theorem 2 (methodology mismatch:\nour RF is SPSA-style, not LR-style)')
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

fig.suptitle(
    'Reproduction Benchmark Summary — OPRE-2025-02-1714 '
    '(Differentiable Discrete Event Simulation for Queuing Network Control)',
    y=0.995, fontsize=12, fontweight='bold')
fig.savefig('reports/figures/fig_benchmark_summary.png')
fig.savefig('reports/figures/fig_benchmark_summary.pdf')
plt.close(fig)
print('saved reports/figures/fig_benchmark_summary.{png,pdf}')
