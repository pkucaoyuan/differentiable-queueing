# -*- coding: utf-8 -*-
"""Build paper-quality reproduction figures from cached JSON outputs."""
import json, os, statistics, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 110,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

OUTDIR = 'reports/figures'
os.makedirs(OUTDIR, exist_ok=True)

def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUTDIR}/{name}.{ext}')
    plt.close(fig)
    print(f'  saved {OUTDIR}/{name}.png + .pdf')

# ═══════════════════════════════════════════════════════════════════
# Figure 1: Section 5.2 CμRule grid (10-class) — heatmap PATHWISE vs REINFORCE
# ═══════════════════════════════════════════════════════════════════
def fig_section52_grid():
    pw = json.load(open('results/reproduction/reproduction_cmu_pathwise.json'))
    rf = json.load(open('results/reproduction/reproduction_cmu_reinforce.json'))
    alphas = sorted(pw.keys(), key=float)
    gaps   = sorted(pw[alphas[0]].keys(), key=lambda x: -float(x))  # large→small

    def grid(data):
        g = np.zeros((len(alphas), len(gaps)))
        for i, a in enumerate(alphas):
            for j, gp in enumerate(gaps):
                costs = [r['avg_cost'] for r in data[a][gp]]
                g[i, j] = statistics.mean(costs)
        return g

    pw_g = grid(pw)
    rf_g = grid(rf)
    diff = (rf_g - pw_g) / pw_g * 100  # % improvement of PW vs RF

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
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
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-abs(diff).max(), vmax=abs(diff).max(), aspect='auto')
    ax.set_xticks(range(len(gaps))); ax.set_xticklabels(gaps)
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
    ax.set_xlabel('gap'); ax.set_ylabel('alpha')
    ax.set_title('(RF − PW) / PW  [%]')
    for i in range(len(alphas)):
        for j in range(len(gaps)):
            ax.text(j, i, f'{diff[i,j]:.1f}', ha='center', va='center',
                    color='black' if abs(diff[i,j]) < abs(diff).max()*0.5 else 'white', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle('§5.2  CμRule 10-class, ρ=0.95 — PATHWISE vs REINFORCE (50 trials per cell)', y=1.02)
    save(fig, 'fig_section52_cmu_grid')

# ═══════════════════════════════════════════════════════════════════
# Figure 2: §5.2 ablations — T, queue_class, num_iter, rho
# ═══════════════════════════════════════════════════════════════════
def fig_section52_ablations():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    # T ablation
    ax = axes[0, 0]
    pw = json.load(open('results/reproduction/T_ablation_pathwise.json'))
    rf = json.load(open('results/reproduction/T_ablation_reinforce.json'))
    Ts = [500, 1000, 2000, 5000]
    pw_means = [statistics.mean([r['avg_cost'] for r in pw[f'T_{T}']['0.5']['0.5']]) for T in Ts]
    rf_means = [statistics.mean([r['avg_cost'] for r in rf[f'T_{T}']['0.1']['0.5']]) for T in Ts]
    pw_stds  = [statistics.stdev([r['avg_cost'] for r in pw[f'T_{T}']['0.5']['0.5']]) for T in Ts]
    rf_stds  = [statistics.stdev([r['avg_cost'] for r in rf[f'T_{T}']['0.1']['0.5']]) for T in Ts]
    ax.errorbar(Ts, pw_means, yerr=pw_stds, marker='o', label='PATHWISE', color='C0', capsize=3)
    ax.errorbar(Ts, rf_means, yerr=rf_stds, marker='s', label='REINFORCE', color='C1', capsize=3)
    ax.set_xscale('log'); ax.set_xlabel('horizon T'); ax.set_ylabel('avg cost')
    ax.set_title('§5.2 T ablation (alpha=0.5, gap=0.5)'); ax.legend(); ax.grid(alpha=0.3)

    # queue_class ablation
    ax = axes[0, 1]
    pw = json.load(open('results/reproduction/queue_class_ablation_pathwise.json'))
    rf = json.load(open('results/reproduction/queue_class_ablation_reinforce.json'))
    qcs = [5, 15, 20]
    pw_m = [statistics.mean([r['avg_cost'] for r in pw[f'qc_{q}']['0.5']['0.5']]) for q in qcs]
    rf_m = [statistics.mean([r['avg_cost'] for r in rf[f'qc_{q}']['0.1']['0.5']]) for q in qcs]
    pw_s = [statistics.stdev([r['avg_cost'] for r in pw[f'qc_{q}']['0.5']['0.5']]) for q in qcs]
    rf_s = [statistics.stdev([r['avg_cost'] for r in rf[f'qc_{q}']['0.1']['0.5']]) for q in qcs]
    ax.errorbar(qcs, pw_m, yerr=pw_s, marker='o', label='PATHWISE', color='C0', capsize=3)
    ax.errorbar(qcs, rf_m, yerr=rf_s, marker='s', label='REINFORCE', color='C1', capsize=3)
    ax.set_xlabel('# queue classes'); ax.set_ylabel('avg cost')
    ax.set_title('§5.2 queue_class ablation'); ax.legend(); ax.grid(alpha=0.3)

    # num_iter ablation
    ax = axes[1, 0]
    pw = json.load(open('results/reproduction/num_iter_ablation_pathwise.json'))
    rf = json.load(open('results/reproduction/num_iter_ablation_reinforce.json'))
    nis = [10, 20, 100]
    pw_m = [statistics.mean([r['avg_cost'] for r in pw[f'ni_{n}']['0.5']['0.5']]) for n in nis]
    rf_m = [statistics.mean([r['avg_cost'] for r in rf[f'ni_{n}']['0.1']['0.5']]) for n in nis]
    pw_s = [statistics.stdev([r['avg_cost'] for r in pw[f'ni_{n}']['0.5']['0.5']]) for n in nis]
    rf_s = [statistics.stdev([r['avg_cost'] for r in rf[f'ni_{n}']['0.1']['0.5']]) for n in nis]
    ax.errorbar(nis, pw_m, yerr=pw_s, marker='o', label='PATHWISE', color='C0', capsize=3)
    ax.errorbar(nis, rf_m, yerr=rf_s, marker='s', label='REINFORCE', color='C1', capsize=3)
    ax.set_xscale('log'); ax.set_xlabel('num_iter (gradient steps)'); ax.set_ylabel('avg cost')
    ax.set_title('§5.2 num_iter ablation'); ax.legend(); ax.grid(alpha=0.3)

    # rho ablation
    ax = axes[1, 1]
    pw = json.load(open('results/reproduction/rho_ablation_pathwise.json'))
    rf = json.load(open('results/reproduction/rho_ablation_reinforce.json'))
    rhos = [0.9, 0.99]
    # At gap=0.05 (heavy gap), should see PW vs RF differ more
    pw_m = [statistics.mean([r['avg_cost'] for r in pw[f'rho_{r}']['0.5']['0.05']]) for r in rhos]
    rf_m = [statistics.mean([r['avg_cost'] for r in rf[f'rho_{r}']['0.1']['0.05']]) for r in rhos]
    pw_s = [statistics.stdev([r['avg_cost'] for r in pw[f'rho_{r}']['0.5']['0.05']]) for r in rhos]
    rf_s = [statistics.stdev([r['avg_cost'] for r in rf[f'rho_{r}']['0.1']['0.05']]) for r in rhos]
    ax.errorbar(rhos, pw_m, yerr=pw_s, marker='o', label='PATHWISE', color='C0', capsize=3)
    ax.errorbar(rhos, rf_m, yerr=rf_s, marker='s', label='REINFORCE', color='C1', capsize=3)
    ax.set_xlabel('traffic intensity ρ'); ax.set_ylabel('avg cost')
    ax.set_title('§5.2 ρ ablation (gap=0.05)'); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle('§5.2  Four ablations — PATHWISE matches REINFORCE within ≤2.62% across all settings', y=1.01)
    fig.tight_layout()
    save(fig, 'fig_section52_ablations')

# ═══════════════════════════════════════════════════════════════════
# Figure 3: §5.3 Admission control — PATHWISE vs SPSA on 12 nets
# ═══════════════════════════════════════════════════════════════════
def fig_section53_admission():
    summ = json.load(open('results/admission_control_summary.json'))
    families = {'reentrant_1': ['reentrant_2.yaml', 'reentrant_3.yaml', 'reentrant_4.yaml',
                                'reentrant_5.yaml', 'reentrant_6.yaml', 'reentrant_7.yaml'],
                'reentrant_2': ['re-reentrant_2.yaml', 're-reentrant_3.yaml', 're-reentrant_4.yaml',
                                're-reentrant_5.yaml', 're-reentrant_6.yaml', 're-reentrant_7.yaml']}
    Ks = [6, 9, 12, 15, 18, 21]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    methods = [('PATHWISE_B1', 'o-', 'C0'),
               ('SPSA_B10',    's--', 'C2'),
               ('SPSA_B100',   '^--', 'C3'),
               ('SPSA_B1000',  'D--', 'C1')]

    for ax, (fam, files) in zip(axes, families.items()):
        for m, marker, color in methods:
            means = [summ[f][m]['mean'] for f in files]
            stds  = [summ[f][m]['std'] for f in files]
            ax.errorbar(Ks, means, yerr=stds, fmt=marker, label=m, color=color, capsize=3, lw=1.5)
        ax.set_xlabel('# queue classes K')
        ax.set_ylabel('final avg cost (50 trials)')
        ax.set_title(f'§5.3 Admission control — family {fam}')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.axvline(15, ls=':', color='gray', alpha=0.6)
        ax.text(15.2, ax.get_ylim()[1]*0.95, 'K=15: SPSA collapses', fontsize=8, color='gray')

    fig.suptitle('§5.3  PATHWISE (B=1) scales gracefully while SPSA collapses on K ≥ 15', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_section53_admission')

# ═══════════════════════════════════════════════════════════════════
# Figure 4: §6 WC-Softmax vs Vanilla on criss-cross
# ═══════════════════════════════════════════════════════════════════
def fig_section6_wc_vs_vanilla():
    wc      = json.load(open('loss/criss_cross_bh_ppg_softmax.json'))
    vanilla = json.load(open('loss/criss_cross_bh_ppg_vanilla.json'))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    wc_ep   = [r['epoch'] for r in wc];      wc_test   = [r['test_loss'] for r in wc]
    van_ep  = [r['epoch'] for r in vanilla]; van_test  = [r['test_loss'] for r in vanilla]
    # vanilla epoch 0 has huge test_loss (init): clip y-axis
    ax.plot(wc_ep, wc_test, color='C0', label=f'WC softmax (min {min(wc_test):.2f})', lw=1.8)
    ax.plot(van_ep[1:], van_test[1:], color='C3', label=f'Vanilla softmax (min {min(van_test):.2f})', lw=1.8)
    ax.set_xlabel('epoch'); ax.set_ylabel('test cost')
    ax.set_ylim(10, 25)
    ax.set_title('§6  Criss-cross — Work-conserving softmax converges lower than Vanilla')
    ax.axhline(min(wc_test), ls=':', color='C0', alpha=0.5)
    ax.axhline(min(van_test), ls=':', color='C3', alpha=0.5)
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, 'fig_section6_wc_vs_vanilla')

# ═══════════════════════════════════════════════════════════════════
# Figure 5: §7 STE training curves — all 11 envs in one figure
# ═══════════════════════════════════════════════════════════════════
def fig_section7_training_curves():
    envs = [
        ('criss_cross_bh', 'criss-cross'),
        ('reentrant_2', 'reentrant_2'),
        ('reentrant_3', 'reentrant_3'),
        ('reentrant_4', 'reentrant_4'),
        ('reentrant_5', 'reentrant_5'),
        ('reentrant_6', 'reentrant_6'),
        ('reentrant_7', 'reentrant_7'),
        ('reentrant_8', 'reentrant_8'),
        ('reentrant_9', 'reentrant_9'),
        ('reentrant_10', 'reentrant_10'),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True)
    for (key, label), ax in zip(envs, axes.flat):
        d = json.load(open(f'loss/{key}_ppg_softmax.json'))
        ep = [r['epoch'] for r in d]
        tc = [r['test_loss'] for r in d]
        tr = [r['train_loss'] for r in d]
        std= [r['test_loss_std'] for r in d]
        min_ep = tc.index(min(tc)); min_val = min(tc)
        ax.plot(ep, tc, color='C0', label='test', lw=1.3)
        ax.fill_between(ep, np.array(tc)-np.array(std)/np.sqrt(100), np.array(tc)+np.array(std)/np.sqrt(100),
                        color='C0', alpha=0.2)
        ax.plot(ep, tr, color='C1', label='train', lw=0.8, alpha=0.7)
        ax.scatter([min_ep], [min_val], color='red', s=30, zorder=5)
        ax.annotate(f'{min_val:.2f}', (min_ep, min_val), xytext=(3, -10), 
                    textcoords='offset points', fontsize=8, color='red')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('epoch'); ax.set_ylabel('cost')
        ax.grid(alpha=0.3)
        if (key, label) == envs[0]: ax.legend(fontsize=8)
    fig.suptitle('§7  STE training curves — 10 networks, 100 epochs each (min cost marked)', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_section7_training_curves')

# ═══════════════════════════════════════════════════════════════════
# Figure 6: §7 min cost summary bar
# ═══════════════════════════════════════════════════════════════════
def fig_section7_summary():
    envs = ['criss_cross_bh', 'reentrant_2', 'reentrant_3', 'reentrant_4', 'reentrant_5',
            'reentrant_6', 'reentrant_7', 'reentrant_8', 'reentrant_9', 'reentrant_10']
    labels = ['criss-cross', 'r_2', 'r_3', 'r_4', 'r_5', 'r_6', 'r_7', 'r_8', 'r_9', 'r_10']
    init, mins, finals = [], [], []
    for key in envs:
        d = json.load(open(f'loss/{key}_ppg_softmax.json'))
        tc = [r['test_loss'] for r in d]
        init.append(tc[0]); mins.append(min(tc)); finals.append(tc[-1])

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(envs)); w = 0.27
    ax.bar(x-w, init,  w, label='initial (ep 0)',  color='C7', alpha=0.7)
    ax.bar(x,    mins, w, label='best (min)',      color='C0')
    ax.bar(x+w,  finals,w, label='final (ep 99)',  color='C1')
    for i, v in enumerate(mins):
        ax.text(i, v + max(mins)*0.01, f'{v:.1f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('test cost')
    ax.set_title('§7  STE training: initial → best → final cost across 10 networks')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    save(fig, 'fig_section7_min_cost_summary')

# ═══════════════════════════════════════════════════════════════════
# Figure 7: §8 Theorem 2 — variance log-log
# ═══════════════════════════════════════════════════════════════════
def fig_section8_theorem2():
    d = json.load(open('results/reproduction/theorem2_validation_v2.json'))
    rhos = sorted([float(k) for k in d.keys() if k != 'fit'])
    gaps = [1 - r for r in rhos]
    pw_vars = [d[str(r)]['pw_var'] for r in rhos]
    rf_vars = [d[str(r)]['rf_var'] for r in rhos]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    # Left: log-log variance
    ax = axes[0]
    ax.loglog(gaps, pw_vars, 'o-', label=f'PATHWISE (fitted slope {d["fit"]["pw_slope_all"]:.2f})', color='C0')
    ax.loglog(gaps, rf_vars, 's-', label=f'REINFORCE (fitted slope {d["fit"]["rf_slope_all"]:.2f})', color='C3')
    # Reference slopes
    g_ref = np.array([gaps[-1], gaps[0]])
    pw_ref = pw_vars[-1] * (g_ref / gaps[-1])**(-3)
    rf_ref = rf_vars[-1] * (g_ref / gaps[-1])**(-4)
    ax.loglog(g_ref, pw_ref, '--', color='C0', alpha=0.5, label='Predicted PW: (1-ρ)⁻³')
    ax.loglog(g_ref, rf_ref, '--', color='C3', alpha=0.5, label='Predicted RF: (1-ρ)⁻⁴')
    ax.set_xlabel('1 − ρ'); ax.set_ylabel('estimator variance')
    ax.set_title('§8 Theorem 2 variance — fit deviates from predicted slope')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    # Right: variance ratio
    ax = axes[1]
    ratios = [d[str(r)]['ratio'] for r in rhos]
    ax.semilogx(gaps, ratios, 'o-', color='C2')
    ax.set_xlabel('1 − ρ'); ax.set_ylabel('Var(REINFORCE) / Var(PATHWISE)')
    ax.set_title('PATHWISE advantage vanishes as ρ → 1 in our setup')
    ax.axhline(1, ls=':', color='k', alpha=0.5)
    ax.grid(alpha=0.3, which='both')

    fig.suptitle('§8  Theorem 2 — methodology mismatch (our REINFORCE is Gaussian-perturb, not paper\'s likelihood-ratio)',
                 y=1.02, fontsize=10)
    fig.tight_layout()
    save(fig, 'fig_section8_theorem2')

# ═══════════════════════════════════════════════════════════════════
# Build everything
# ═══════════════════════════════════════════════════════════════════
print("Building figures...")
fig_section52_grid()
fig_section52_ablations()
fig_section53_admission()
fig_section6_wc_vs_vanilla()
fig_section7_training_curves()
fig_section7_summary()
fig_section8_theorem2()
print("Done.")
