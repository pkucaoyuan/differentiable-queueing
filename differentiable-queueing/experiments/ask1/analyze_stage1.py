"""Stage-1 analysis: rho-scaling A/B comparison on criss-cross + GT reliability."""
import glob
import json
import os
import sys

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'ask1', 'stage1')


def load_cells():
    cells = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, '*.npz'))):
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d['meta']))
        cells[(meta['rho'], meta['policy'], meta['scaling'])] = (d, meta)
    return cells


def per_theta_stats(cos_mat):
    """cos_mat (n_theta, n_draws) -> per-theta mean; distribution summary."""
    m = np.nanmean(cos_mat, axis=1)
    return m


def main():
    cells = load_cells()
    print(f'loaded {len(cells)} cells\n')

    print('=== A/B: rho scaling (cell mean cossim; per-theta median in brackets) ===')
    print(f'{"rho":>5} {"policy":>4} | {"PW paper":>16} {"PW author":>16} | {"RF paper":>9} {"RF author":>9}')
    for rho in [0.8, 0.9, 0.95, 0.99]:
        for pol in ['sMP', 'sMW', 'sPR']:
            row = []
            for est in ['pw_cos', 'rf_cos']:
                for sc in ['paper', 'author']:
                    if (rho, pol, sc) in cells:
                        d, _ = cells[(rho, pol, sc)]
                        m = per_theta_stats(d[est])
                        row.append(f'{np.nanmean(m):+.3f}[{np.nanmedian(m):+.2f}]')
                    else:
                        row.append('  --  ')
            print(f'{rho:>5} {pol:>4} | {row[0]:>16} {row[1]:>16} | {row[2]:>9} {row[3]:>9}')

    print('\n=== GT reliability: split-half cosine median (LOO / no-baseline control) ===')
    for rho in [0.8, 0.9, 0.95, 0.99]:
        line = f'rho={rho}: '
        for pol in ['sMP', 'sMW', 'sPR']:
            k = (rho, pol, 'paper')
            if k in cells:
                _, meta = cells[k]
                line += (f'{pol} {meta["gt_split_cos_median"]:+.2f}/'
                         f'{meta["gt_nb_split_cos_median"]:+.2f}   ')
        print(line)

    print('\n=== per-theta distribution of PW cossim (paper scaling) ===')
    for rho in [0.9, 0.99]:
        for pol in ['sMP', 'sMW', 'sPR']:
            k = (rho, pol, 'paper')
            if k not in cells:
                continue
            d, _ = cells[k]
            m = per_theta_stats(d['pw_cos'])
            qs = np.nanpercentile(m, [10, 25, 50, 75, 90])
            frac_high = np.nanmean(m > 0.8)
            print(f'rho={rho} {pol}: q10..q90 = {[f"{x:+.2f}" for x in qs]}  '
                  f'share(theta with PW>0.8) = {frac_high:.0%}')

    print('\n=== PW vs RF win-rate per theta (paper scaling, all cells) ===')
    wins, total = 0, 0
    for (rho, pol, sc), (d, _) in cells.items():
        if sc != 'paper':
            continue
        pw_m = per_theta_stats(d['pw_cos'])
        rf_m = per_theta_stats(d['rf_cos'])
        ok = ~np.isnan(pw_m) & ~np.isnan(rf_m)
        wins += int((pw_m[ok] > rf_m[ok]).sum())
        total += int(ok.sum())
    if total:
        print(f'PW > RF for {wins}/{total} thetas = {wins/total:.1%} (paper reports 94.5%)')


if __name__ == '__main__':
    main()
