"""Ask-2 deliverable: Fig 12-style multi-seed PPO curves (reentrant_2, 6 classes).

Aggregates ppo_runs/<variant>_s<k>/{WC,vanilla}_results.json (+ *_bc_results.json)
into cost-vs-iteration curves with across-seed bands, against the cmu baseline.
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RUNS = os.path.join(REPO, 'ppo_runs')
OUT = os.path.join(REPO, 'reports', 'ask1_figs')

LABELS = {'WC': ('PPO-WC (work-conserving)', '#2c7fb8'),
          'vanilla': ('PPO vanilla', '#d95f02'),
          'bc': ('PPO-BC (behavior cloned)', '#7570b3')}


def load_runs():
    """Prefer final *_results.json; fall back to parsing per-iteration test
    costs from run.log so in-progress runs can be plotted at any time."""
    import re
    curves = {}
    for d in sorted(glob.glob(os.path.join(RUNS, '*_s*'))):
        rid = os.path.basename(d)
        var = rid.rsplit('_s', 1)[0]
        js = glob.glob(os.path.join(d, '*_results.json'))
        if js:
            res = json.load(open(js[0]))
            arr = np.array(res['test_cost'], dtype=float)
        else:
            log = os.path.join(d, 'run.log')
            if not os.path.exists(log):
                continue
            txt = open(log, errors='ignore').read().replace('\r', '\n')
            vals = re.findall(r'test cost:?\s*([\d.]+)', txt)
            if len(vals) < 3:
                continue
            arr = np.array(vals, dtype=float)
        curves.setdefault(var, []).append(arr)
    return curves


def main():
    curves = load_runs()
    matched = os.path.join(RUNS, 'cmu_matched.json')
    if os.path.exists(matched):  # protocol-matched baseline (25 envs x 2500 steps)
        cmu_cost = float(json.load(open(matched))['avg_cost'])
    else:
        cmu = json.load(open(os.path.join(REPO, 'PPO', 'cmu_results.json')))
        cmu_cost = float(cmu['reentrant_2']['avg_cost'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, ylim, tag in [(axes[0], None, 'log'), (axes[1], (10, 40), 'zoom')]:
        for var, runs in curves.items():
            L = min(len(r) for r in runs)
            M = np.stack([r[:L] for r in runs])
            lab, col = LABELS.get(var, (var, 'gray'))
            x = np.arange(L)
            ax.plot(x, np.median(M, 0), color=col, label=f'{lab} (n={len(runs)})')
            ax.fill_between(x, M.min(0), M.max(0), color=col, alpha=0.18)
        ax.axhline(cmu_cost, color='k', ls='--', lw=1, label=f'cμ rule ({cmu_cost:.1f})')
        ax.set_xlabel('policy iteration')
        if tag == 'log':
            ax.set_yscale('log')
            ax.set_ylabel('avg holding cost (test, 25 envs)')
            ax.legend(fontsize=8)
        else:
            ax.set_ylim(*ylim)
            ax.set_title('zoom', fontsize=10)
    fig.suptitle('Fig 12 reproduction, multi-seed (reentrant_2 / 6 classes; SB3 PPO; '
                 'episode_steps=15000 x 50 iters, reduced from paper 50000x100)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig8_ppo_wc_multiseed.png'), dpi=180, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'fig8_ppo_wc_multiseed.pdf'), bbox_inches='tight')
    # console summary
    print(f'cmu = {cmu_cost:.2f}')
    for var, runs in curves.items():
        finals = [np.mean(r[-5:]) for r in runs]
        print(f'{var}: n={len(runs)} final(last5 mean) = '
              + ', '.join(f'{v:.1f}' for v in finals))
    print('fig8 done')


if __name__ == '__main__':
    main()
