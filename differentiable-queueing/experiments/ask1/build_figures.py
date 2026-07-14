"""Ask-1 deliverable figures.

fig1: paper-Figure-8-style heatmap grid (PW/RF x sMP/sMW/sPR, networks x rho)
fig2: GT reliability (LOO vs no-baseline split-half cosine), criss-cross stage1
fig3: per-theta cossim distributions (the bimodal story)

Data: results/ask1/stage1 (criss, 4 rhos, 100x100, dual GT)
      results/ask1/stage2quick (8 reentrant nets, rho 0.9/0.99, 25x25, LOO GT)
Output: reports/ask1_figs/
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', '..', 'results', 'ask1')
OUT = os.path.join(HERE, '..', '..', 'reports', 'ask1_figs')
os.makedirs(OUT, exist_ok=True)

ROWS = [  # paper Figure 8 row order (top to bottom)
    ('reentrant_2', 'Reentrant (6 cls)'),
    ('re-reentrant_2', 'Reentrant-2 (6 cls)'),
    ('reentrant_3', 'Reentrant (9 cls)'),
    ('re-reentrant_3', 'Reentrant-2 (9 cls)'),
    ('reentrant_4', 'Reentrant (12 cls)'),
    ('re-reentrant_4', 'Reentrant-2 (12 cls)'),
    ('reentrant_5', 'Reentrant (15 cls)'),
    ('re-reentrant_5', 'Reentrant-2 (15 cls)'),
    ('criss_cross_bh', 'Criss Cross'),
]
RHOS = [0.8, 0.9, 0.95, 0.99]
POLS = [('sMP', 'MaxPressure (sMP)*'), ('sMW', 'MaxWeight (sMW)'), ('sPR', 'Random Priority (sPR)')]


def load_all():
    cells = {}
    # later stages overwrite earlier ones: stage2 (full spec) > stage2quick
    for stage in ['stage1', 'stage2quick', 'stage2']:
        for f in glob.glob(os.path.join(RES, stage, '*.npz')):
            d = np.load(f, allow_pickle=True)
            meta = json.loads(str(d['meta']))
            if meta['scaling'] != 'paper':
                continue
            key = (meta['env'], meta['rho'], meta['policy'])
            cells[key] = {
                'pw': np.nanmean(d['pw_cos']),
                'rf': np.nanmean(d['rf_cos']),
                'pw_theta': np.nanmean(d['pw_cos'], axis=1),
                'rf_theta': np.nanmean(d['rf_cos'], axis=1),
                'meta': meta,
            }
    return cells


def fig1_heatmaps(cells):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharey=True)
    cmap = plt.get_cmap('RdYlBu_r').copy()
    cmap.set_bad('#dddddd')
    for r, (est, est_lab) in enumerate([('pw', 'PATHWISE (B=1)'), ('rf', 'REINFORCE (B=1000)')]):
        for c, (pol, pol_lab) in enumerate(POLS):
            M = np.full((len(ROWS), len(RHOS)), np.nan)
            for i, (env, _) in enumerate(ROWS):
                for j, rho in enumerate(RHOS):
                    k = (env, rho, pol)
                    if k in cells:
                        M[i, j] = cells[k][est]
            ax = axes[r, c]
            im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect='auto')
            for i in range(len(ROWS)):
                for j in range(len(RHOS)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f'{M[i,j]:+.2f}', ha='center', va='center', fontsize=7,
                                color='black')
            ax.set_xticks(range(len(RHOS)), [str(r_) for r_ in RHOS], fontsize=8)
            ax.set_yticks(range(len(ROWS)), [lab for _, lab in ROWS], fontsize=8)
            if r == 0:
                ax.set_title(pol_lab, fontsize=11)
            if r == 1:
                ax.set_xlabel('traffic intensity ρ', fontsize=9)
            if c == 0:
                ax.set_ylabel(est_lab, fontsize=11)
    fig.suptitle('Cosine similarity to ground-truth gradient — released-code policies, LOO-baselined GT\n'
                 '(paper Fig. 8 reports PATHWISE ≈ 1.0; upstream repo notes admit the figure code is lost. '
                 '*sMP is code-identical to sMW in the released code)', fontsize=9.5)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label('mean cossim (negatives clipped to 0 on color scale)', fontsize=8)
    fig.savefig(os.path.join(OUT, 'fig1_cossim_grid.png'), dpi=180, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'fig1_cossim_grid.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('fig1 done')


def fig2_gt_reliability():
    files = glob.glob(os.path.join(RES, 'stage1', '*.npz'))
    data = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d['meta']))
        if meta['scaling'] != 'paper':
            continue
        data[(meta['policy'], meta['rho'])] = (np.nanmedian(d['gt_split_cos']),
                                               np.nanmedian(d['gt_nb_split_cos']))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, pol in zip(axes, ['sMP', 'sMW', 'sPR']):
        loo = [data[(pol, r)][0] for r in RHOS]
        nb = [data[(pol, r)][1] for r in RHOS]
        x = np.arange(len(RHOS))
        ax.bar(x - 0.18, loo, 0.34, label='GT with LOO baseline (ours)', color='#2c7fb8')
        ax.bar(x + 0.18, nb, 0.34, label='GT no baseline (released code)', color='#d95f0e')
        ax.axhline(1.0, color='gray', lw=0.6, ls='--')
        ax.set_xticks(x, [str(r) for r in RHOS])
        ax.set_title(pol)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('ρ')
    axes[0].set_ylabel('GT split-half cosine\n(median over 100 θ)')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Ground-truth reliability (criss-cross, GT = 1e5 REINFORCE trajectories per θ): '
                 'the released code has no value baseline → its GT is noise-dominated', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_gt_reliability.png'), dpi=180, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'fig2_gt_reliability.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('fig2 done')


def fig3_theta_dist(cells):
    picks = [('criss_cross_bh', 0.9, 'sPR'), ('criss_cross_bh', 0.9, 'sMW'),
             ('criss_cross_bh', 0.99, 'sPR'), ('criss_cross_bh', 0.99, 'sMW')]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    bins = np.linspace(-1, 1, 21)
    for ax, key in zip(axes, picks):
        c = cells.get(key)
        if c is None:
            ax.axis('off')
            continue
        ax.hist(c['pw_theta'], bins=bins, alpha=0.65, label='PATHWISE (B=1)', color='#2c7fb8')
        ax.hist(c['rf_theta'], bins=bins, alpha=0.65, label='REINFORCE (B=1000)', color='#d95f0e')
        ax.axvline(0, color='gray', lw=0.6)
        ax.set_title(f'{key[2]}  ρ={key[1]}', fontsize=10)
        ax.set_xlabel('per-θ mean cossim')
    axes[0].set_ylabel('# of θ (out of 100)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Per-θ distributions are bimodal: a sub-population reaches the paper-level cossim ≈ 1, '
                 'another has ≈ 0 or negative (STE directional bias)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_theta_dist.png'), dpi=180, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'fig3_theta_dist.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('fig3 done')


def fig4_batch_sweep():
    files = glob.glob(os.path.join(RES, 'sweep', '*.npz'))
    data = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d['meta']))
        data[(meta['env'], meta['rho'], meta['policy'])] = d
    nets = [('criss_cross_bh', 'Criss Cross'), ('reentrant_3', 'Reentrant (9 cls)'),
            ('re-reentrant_3', 'Reentrant-2 (9 cls)')]
    rhos = [0.9, 0.99]
    bs = [1, 2, 5, 10, 50, 100, 1000, 10000]
    colors = {'sMP': '#1b9e77', 'sMW': '#d95f02', 'sPR': '#7570b3'}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
    for r, rho in enumerate(rhos):
        for c, (env, lab) in enumerate(nets):
            ax = axes[r, c]
            for pol in ['sMP', 'sMW', 'sPR']:
                d = data.get((env, rho, pol))
                if d is None:
                    continue
                pw = [np.nanmean(d[f'pw_B{b}']) for b in bs]
                rf = [np.nanmean(d[f'rf_B{b}']) for b in bs]
                ax.plot(bs, pw, 'o-', color=colors[pol], label=f'{pol} PW', ms=4)
                ax.plot(bs, rf, 's--', color=colors[pol], label=f'{pol} RF', ms=4, alpha=0.6)
            ax.set_xscale('log')
            ax.axhline(0, color='gray', lw=0.5)
            if r == 0:
                ax.set_title(lab, fontsize=11)
            if r == 1:
                ax.set_xlabel('batch size B (trajectories per estimate)')
            if c == 0:
                ax.set_ylabel(f'ρ = {rho}\nmean cossim to GT')
    axes[0, 0].legend(fontsize=7, ncol=2, loc='upper left')
    fig.suptitle('Cosine similarity vs batch size — PATHWISE (solid) vs REINFORCE (dashed), '
                 'released-code policies, LOO-baselined GT', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_batch_sweep.png'), dpi=180, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'fig4_batch_sweep.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('fig4 done')


if __name__ == '__main__':
    cells = load_all()
    print(f'{len(cells)} paper-scaling cells loaded')
    fig1_heatmaps(cells)
    fig2_gt_reliability()
    fig3_theta_dist(cells)
    fig4_batch_sweep()


def fig5_paper_apparatus():
    import matplotlib.patches as mpatches
    res = {}
    for f in glob.glob(os.path.join(RES, 'paper_impl', '*.npz')):
        d = np.load(f, allow_pickle=True)
        m = json.loads(str(d['meta']))
        variant = 'nomask' if f.endswith('_nomask.npz') else 'masked'
        res[(m['rho'], m['policy'].replace('paper_', ''), variant)] = m['pw_mean']
    released = {}
    for f in glob.glob(os.path.join(RES, 'stage1', '*.npz')):
        d = np.load(f, allow_pickle=True)
        m = json.loads(str(d['meta']))
        if m['scaling'] == 'paper':
            released[(m['rho'], m['policy'])] = float(np.nanmean(d['pw_cos']))
    pols = ['sMP', 'sMW', 'sPR']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, rho in zip(axes, [0.9, 0.99]):
        x = np.arange(len(pols))
        bars = [
            ('released code', [released.get((rho, p), np.nan) for p in pols], '#bdbdbd'),
            ('paper policies + V-baseline (masked)', [res.get((rho, p, 'masked'), np.nan) for p in pols], '#fdae6b'),
            ('paper-literal (no mask) + V-baseline', [res.get((rho, p, 'nomask'), np.nan) for p in pols], '#2c7fb8'),
        ]
        for k, (lab, vals, col) in enumerate(bars):
            ax.bar(x + (k - 1) * 0.27, vals, 0.25, label=lab, color=col)
        ax.axhline(1.0, color='red', lw=0.8, ls='--')
        ax.text(2.45, 1.02, 'paper ≈1', color='red', fontsize=8)
        ax.set_xticks(x, pols)
        ax.set_title(f'ρ = {rho}')
        ax.set_ylim(0, 1.12)
    axes[0].set_ylabel('PATHWISE (B=1)\nmean cossim to GT')
    axes[0].legend(fontsize=7.5, loc='upper left')
    fig.suptitle('Recovering the paper apparatus (criss-cross): value baseline fixes the GT; removing the\n'
                 "released-code masking recovers the paper's Fig.8 magnitudes (10θ x 20 draws pilot)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_paper_apparatus.png'), dpi=180, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'fig5_paper_apparatus.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('fig5 done')
