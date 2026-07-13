"""Ask-1 runner: upstream-code cossim grid + batch-size sweep, multi-GPU via file claims.

Stages:
  stage1: criss_cross_bh x 4 rho x 3 policies x 2 lam-scalings (A/B test)   = 24 cells
  stage2: 8 reentrant nets x 4 rho x 3 policies, chosen scaling            = 96 cells
  sweep : 3 nets x 2 rho x 3 policies, B in {1..10000}, PW & RF            = 18 cells

Per main-grid cell (quick spec, GT=1e5; paper uses 1e6):
  100 policy inits (upstream default init); GT = REINFORCE mean over 1e5 trajs each;
  100 draws each of PATHWISE(B=1) and REINFORCE(B=1000); T=1000, gamma=0.999.
  GT reliability: split-half cosine between even/odd chunk halves, per init.

Usage (one worker per GPU):
  CUDA_VISIBLE_DEVICES=i python run_cossim.py --stage stage1
Results: results/ask1/<stage>/<cell_id>.npz ; claims in results/ask1/claims/.
"""
import argparse
import json
import os
import time

# limit CPU threads BEFORE importing torch: 8 workers x default 96 threads
# oversubscribe the 96-core host and slow env-step CPU ops ~10x
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
import torch

torch.set_num_threads(4)

from common import (CAPS, HORIZON, cosine_rows, env_dims, init_params,
                    load_scaled_cfg, pathwise_grads, param_dim, reinforce_grads,
                    repeat_params, slice_params, stable_seed)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'ask1')

N_THETA = 100
N_DRAWS = 100
GT_TRAJS = 100_000          # quick version of the paper's 1e6 (user decision 2026-07-13)
RF_B = 1000                 # REINFORCE estimator batch (paper/upstream)
WITH_CONTROL_GT = True      # also compute the no-baseline control GT (option C)

RHOS = [0.8, 0.9, 0.95, 0.99]
POLICIES = ['sMP', 'sMW', 'sPR']
NETS_STAGE2 = ['reentrant_2', 'reentrant_3', 'reentrant_4', 'reentrant_5',
               're-reentrant_2', 're-reentrant_3', 're-reentrant_4', 're-reentrant_5']

SWEEP_NETS = ['criss_cross_bh', 'reentrant_3', 're-reentrant_3']
SWEEP_RHOS = [0.9, 0.99]
SWEEP_BS = [1, 2, 5, 10, 50, 100, 1000, 10000]
SWEEP_DRAWS = {1: 50, 2: 50, 5: 50, 10: 50, 50: 30, 100: 30, 1000: 15, 10000: 8}
SWEEP_THETAS = 25           # first 25 inits of the matching main-grid cell


def cell_id(env, rho, policy, scaling):
    return f'{env}__rho{rho}__{policy}__{scaling}'


def main_cells(stage, scaling=None):
    cells = []
    if stage == 'stage1':
        for sc in ['paper', 'author']:
            for rho in RHOS:
                for pol in POLICIES:
                    cells.append(('criss_cross_bh', rho, pol, sc))
    elif stage == 'stage2':
        assert scaling in ('paper', 'author')
        order = sorted(NETS_STAGE2, key=lambda n: -int(n.split('_')[-1]))  # big first
        for env in order:
            for rho in RHOS:
                for pol in POLICIES:
                    cells.append((env, rho, pol, scaling))
    elif stage == 'stage2quick':
        # reduced spec for the fast reportable pass: topology axis at 2 rhos
        order = sorted(NETS_STAGE2, key=lambda n: -int(n.split('_')[-1]))
        for env in order:
            for rho in [0.9, 0.99]:
                for pol in POLICIES:
                    cells.append((env, rho, pol, scaling))
    return cells


def claim(name):
    os.makedirs(os.path.join(RESULTS, 'claims'), exist_ok=True)
    try:
        os.mkdir(os.path.join(RESULTS, 'claims', name))
        return True
    except FileExistsError:
        return False


def chunked_grads(fn, cfg, policy, pair_params, B_per_group, cap, seed_tag, device):
    """Run fn over G pair-params in chunks bounded by cap total trajectories."""
    P_total = pair_params[0].shape[0]
    per = max(1, cap // B_per_group)
    out = None
    i, c = 0, 0
    while i < P_total:
        j = min(P_total, i + per)
        g = fn(cfg, policy, slice_params(pair_params, slice(i, j)), B_per_group,
               seed=stable_seed(seed_tag, c), device=device)
        if out is None:
            out = torch.empty(P_total, g.shape[1], dtype=g.dtype)
        out[i:j] = g
        i, c = j, c + 1
    return out


def run_main_cell(env, rho, policy, scaling, device):
    cid = cell_id(env, rho, policy, scaling)
    t0 = time.time()
    cfg = load_scaled_cfg(env, rho, scaling)
    s, q = env_dims(cfg)
    cap = CAPS[q]
    P = param_dim(policy, s, q)
    params = init_params(policy, N_THETA, s, q, seed=stable_seed(env, rho, policy, 'init'))

    # --- GT: REINFORCE averaged over GT_TRAJS trajectories per init ---
    # primary GT uses the unbiased LOO baseline; a no-baseline control GT
    # (released-code verbatim) is computed alongside to document GT reliability.
    # nan chunks dropped per-init; split-half cosine as the reliability metric.
    B_c = max(1, cap['rf'] // N_THETA)
    n_chunks = max(2, GT_TRAJS // B_c)

    def compute_gt(tag, use_loo):
        halves = [torch.zeros(N_THETA, P, dtype=torch.float64) for _ in range(2)]
        counts = [torch.zeros(N_THETA) for _ in range(2)]
        for c in range(n_chunks):
            g = reinforce_grads(cfg, policy, params, B_c, loo_baseline=use_loo,
                                seed=stable_seed(cid, tag, c), device=device)
            good = ~torch.isnan(g).any(dim=1)
            halves[c % 2][good] += g[good]
            counts[c % 2][good] += 1
        gt_cnt = counts[0] + counts[1]
        gt = (halves[0] + halves[1]) / gt_cnt.clamp_min(1).unsqueeze(1)
        gt[gt_cnt == 0] = float('nan')
        split = np.array([
            float(cosine_rows((halves[0][i] / max(counts[0][i].item(), 1)).unsqueeze(0),
                              halves[1][i] / max(counts[1][i].item(), 1))[0])
            if counts[0][i] > 0 and counts[1][i] > 0 else float('nan')
            for i in range(N_THETA)], dtype=np.float32)
        drops = int((n_chunks - gt_cnt).sum().item())
        return gt, split, drops

    gt, gt_split_cos, n_gt_chunk_drops = compute_gt('gt', True)
    if WITH_CONTROL_GT:
        gt_nb, gt_nb_split_cos, n_gt_nb_drops = compute_gt('gtnb', False)
    else:
        gt_nb = torch.full_like(gt, float('nan'))
        gt_nb_split_cos = np.full(N_THETA, np.nan, dtype=np.float32)
        n_gt_nb_drops = 0
    t_gt = time.time() - t0
    print(f'[{cid}] GT done ({n_chunks} chunks x2, drops {n_gt_chunk_drops}/{n_gt_nb_drops}, '
          f'split-half LOO={np.nanmedian(gt_split_cos):+.3f} '
          f'nobl={np.nanmedian(gt_nb_split_cos):+.3f}, {t_gt/60:.1f} min)', flush=True)

    # --- estimator draws: pairs (init_i, draw_j) ---
    pair_params = repeat_params(params, N_DRAWS)
    rf = chunked_grads(reinforce_grads, cfg, policy, pair_params, RF_B,
                       cap['rf'], (cid, 'rf'), device)
    t_rf = time.time() - t0 - t_gt
    pw = chunked_grads(pathwise_grads, cfg, policy, pair_params, 1,
                       cap['pw'], (cid, 'pw'), device)

    def score(gt_mat):
        pc = np.full((N_THETA, N_DRAWS), np.nan, dtype=np.float32)
        rc = np.full((N_THETA, N_DRAWS), np.nan, dtype=np.float32)
        bad = 0
        for i in range(N_THETA):
            g = gt_mat[i]
            if torch.isnan(g).any() or g.norm() < 1e-12:
                bad += 1
                continue
            sl = slice(i * N_DRAWS, (i + 1) * N_DRAWS)
            pc[i] = cosine_rows(pw[sl], g).numpy()
            rc[i] = cosine_rows(rf[sl], g).numpy()
        return pc, rc, bad

    pw_cos, rf_cos, n_bad_gt = score(gt)
    pw_cos_nb, rf_cos_nb, n_bad_gt_nb = score(gt_nb)

    meta = dict(env=env, rho=rho, policy=policy, scaling=scaling, s=s, q=q, param_dim=P,
                n_theta=N_THETA, n_draws=N_DRAWS, gt_trajs=GT_TRAJS, rf_b=RF_B,
                horizon=HORIZON, n_bad_gt=n_bad_gt, n_gt_chunk_drops=n_gt_chunk_drops,
                n_gt_nb_drops=n_gt_nb_drops,
                gt_split_cos_median=float(np.nanmedian(gt_split_cos)),
                gt_nb_split_cos_median=float(np.nanmedian(gt_nb_split_cos)),
                pw_mean=float(np.nanmean(pw_cos)), rf_mean=float(np.nanmean(rf_cos)),
                pw_mean_nb=float(np.nanmean(pw_cos_nb)), rf_mean_nb=float(np.nanmean(rf_cos_nb)),
                t_gt_s=round(t_gt, 1), t_rf_s=round(t_rf, 1),
                t_total_s=round(time.time() - t0, 1))
    payload = dict(gt=gt.numpy(), gt_nb=gt_nb.numpy(), pw_cos=pw_cos, rf_cos=rf_cos,
                   pw_cos_nb=pw_cos_nb, rf_cos_nb=rf_cos_nb,
                   gt_split_cos=gt_split_cos, gt_nb_split_cos=gt_nb_split_cos,
                   meta=json.dumps(meta))
    for k, p in enumerate(params):
        payload[f'param{k}'] = p.numpy()
    return payload, meta


def run_sweep_cell(env, rho, policy, scaling, device):
    cid = cell_id(env, rho, policy, scaling)
    src = os.path.join(RESULTS, 'stage1' if env == 'criss_cross_bh' else 'stage2', cid + '.npz')
    d = np.load(src)
    gt = torch.tensor(d['gt'][:SWEEP_THETAS])
    params = []
    for k in range(2):
        if f'param{k}' in d:
            params.append(torch.tensor(d[f'param{k}'][:SWEEP_THETAS]))
    cfg = load_scaled_cfg(env, rho, scaling)
    s, q = env_dims(cfg)
    cap = CAPS[q]

    out = {}
    for B in SWEEP_BS:
        nd = SWEEP_DRAWS[B]
        pair_params = repeat_params(params, nd)
        for est, fn, cap_key in [('pw', pathwise_grads, 'pw'), ('rf', reinforce_grads, 'rf')]:
            t0 = time.time()
            g = chunked_grads(fn, cfg, policy, pair_params, B, cap[cap_key],
                              (cid, 'sweep', est, B), device)
            cos = np.full((SWEEP_THETAS, nd), np.nan, dtype=np.float32)
            for i in range(SWEEP_THETAS):
                gi = gt[i]
                if torch.isnan(gi).any() or gi.norm() < 1e-12:
                    continue
                cos[i] = cosine_rows(g[i * nd:(i + 1) * nd], gi).numpy()
            out[f'{est}_B{B}'] = cos
            print(f'[{cid}] sweep {est} B={B}: mean={np.nanmean(cos):+.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    out['meta'] = json.dumps(dict(env=env, rho=rho, policy=policy, scaling=scaling,
                                  n_theta=SWEEP_THETAS, draws=SWEEP_DRAWS, bs=SWEEP_BS))
    return out


def worker(stage, scaling, device):
    outdir = os.path.join(RESULTS, stage)
    os.makedirs(outdir, exist_ok=True)
    if stage == 'sweep':
        cells = [(e, r, p, scaling) for e in SWEEP_NETS for r in SWEEP_RHOS for p in POLICIES]
    elif stage == 'stage2quick':
        cells = main_cells(stage, scaling)
    else:
        cells = main_cells(stage, scaling)
    for env, rho, pol, sc in cells:
        cid = cell_id(env, rho, pol, sc)
        out_path = os.path.join(outdir, cid + '.npz')
        if os.path.exists(out_path) or not claim(stage + '__' + cid):
            continue
        print(f'>>> {stage} {cid}', flush=True)
        try:
            if stage == 'sweep':
                payload = run_sweep_cell(env, rho, pol, sc, device)
                np.savez_compressed(out_path, **payload)
            else:
                payload, meta = run_main_cell(env, rho, pol, sc, device)
                np.savez_compressed(out_path, **payload)
                print(f'<<< {cid} PW={meta["pw_mean"]:+.3f} RF={meta["rf_mean"]:+.3f} '
                      f'gt_split={meta["gt_split_cos_median"]:+.3f} '
                      f'total={meta["t_total_s"]/60:.1f}min', flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f'!!! OOM on {cid}, releasing claim', flush=True)
            os.rmdir(os.path.join(RESULTS, 'claims', stage + '__' + cid))
            raise
    print('worker done: no unclaimed cells left', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['stage1', 'stage2', 'stage2quick', 'sweep'])
    ap.add_argument('--scaling', default='paper', choices=['paper', 'author'],
                    help='lam scaling for stage2/sweep (stage1 runs both)')
    ap.add_argument('--n-theta', type=int, default=None)
    ap.add_argument('--n-draws', type=int, default=None)
    ap.add_argument('--gt-trajs', type=int, default=None)
    ap.add_argument('--no-control-gt', action='store_true')
    args = ap.parse_args()
    if args.n_theta:
        N_THETA = args.n_theta
    if args.n_draws:
        N_DRAWS = args.n_draws
    if args.gt_trajs:
        GT_TRAJS = args.gt_trajs
    if args.no_control_gt:
        WITH_CONTROL_GT = False
    print(f'GPU: {torch.cuda.get_device_name(0)}  spec: theta={N_THETA} draws={N_DRAWS} '
          f'gt={GT_TRAJS} control={WITH_CONTROL_GT}', flush=True)
    worker(args.stage, args.scaling, 'cuda')
