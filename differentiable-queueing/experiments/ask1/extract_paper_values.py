"""Numerically extract the paper's Figure 8 cell values by inverting the jet
colormap on the original high-res jpg (6480x2946). Panels are auto-detected by
projecting the colored-pixel mask; each panel is split into 9 rows x 4 cols and
the median color of each cell's central region is mapped to a value via
nearest-neighbor lookup in the jet LUT."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, '..', '..', 'literature', 'arxiv_sources',
                   'che2024differentiable', 'plot', 'gradient_eval',
                   'gradient_comparsion_policy.jpg')
OUT = os.path.join(HERE, '..', '..', 'results', 'ask1', 'paper_fig8_values.json')

ROW_LABELS = ['Reentrant (6)', 'Reentrant-2 (6)', 'Reentrant (9)', 'Reentrant-2 (9)',
              'Reentrant (12)', 'Reentrant-2 (12)', 'Reentrant (15)', 'Reentrant-2 (15)',
              'Criss Cross']
RHOS = ['0.8', '0.9', '0.95', '0.99']
PANELS = [('pw', 'sMP'), ('pw', 'sMW'), ('pw', 'sPR'),
          ('rf', 'sMP'), ('rf', 'sMW'), ('rf', 'sPR')]


def bands(profile, min_gap, thresh_frac=0.15):
    on = profile > profile.max() * thresh_frac
    edges = np.flatnonzero(np.diff(on.astype(int)))
    if on[0]:
        edges = np.r_[0, edges]
    if on[-1]:
        edges = np.r_[edges, len(on) - 1]
    spans = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    # merge close spans
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def main():
    img = plt.imread(IMG).astype(float)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    colored = (mx - mn > 60) & (mx > 60)   # saturated & not near-black
    # exclude colorbar: rightmost colored column band
    col_prof = colored.sum(0)
    xb = bands(col_prof, min_gap=80)
    xb = [s for s in xb if s[1] - s[0] > 300]           # drop thin bands (colorbar)
    row_prof = colored[:, xb[0][0]:xb[-1][1]].sum(1)
    yb = bands(row_prof, min_gap=120)
    yb = [s for s in yb if s[1] - s[0] > 300]
    assert len(xb) == 3 and len(yb) == 2, (xb, yb)

    jet = plt.get_cmap('jet')(np.linspace(0, 1, 256))[:, :3] * 255

    def invert(rgb):
        return float(np.argmin(((jet - rgb) ** 2).sum(1)) / 255.0)

    values = {}
    for pi, (est_pol, (x0, x1), (y0, y1)) in enumerate(
            [(PANELS[j * 3 + i], xb[i], yb[j]) for j in range(2) for i in range(3)]):
        est, pol = est_pol
        H, W = y1 - y0, x1 - x0
        for ri, rlab in enumerate(ROW_LABELS):
            for ci, rho in enumerate(RHOS):
                cy = y0 + int((ri + 0.5) / 9 * H)
                cx = x0 + int((ci + 0.5) / 4 * W)
                patch = img[cy - H // 40: cy + H // 40, cx - W // 30: cx + W // 30]
                pr, pg, pb = patch[..., 0], patch[..., 1], patch[..., 2]
                pmx = np.maximum(np.maximum(pr, pg), pb)
                pmn = np.minimum(np.minimum(pr, pg), pb)
                sel = (pmx - pmn > 60) & (pmx > 60)     # drop text/grid pixels
                if sel.sum() < 10:
                    values[f'{est}|{pol}|{rlab}|{rho}'] = None
                    continue
                med = np.median(patch[sel].reshape(-1, 3), axis=0)
                values[f'{est}|{pol}|{rlab}|{rho}'] = round(invert(med), 3)

    with open(OUT, 'w') as f:
        json.dump(values, f, indent=1)
    print(f'saved {OUT}')
    print('\nCriss Cross row (paper values, jet-inverted):')
    print(f'{"":>4} {"sMP":>22} {"sMW":>22} {"sPR":>22}')
    for est in ['pw', 'rf']:
        for rho in RHOS:
            row = [values.get(f'{est}|{p}|Criss Cross|{rho}') for p in ['sMP', 'sMW', 'sPR']]
            print(f'{est} ρ={rho:<5} ' + '  '.join(f'{v:.2f}' if v is not None else ' -- ' for v in row))


if __name__ == '__main__':
    main()
