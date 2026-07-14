"""External validation of the paper-apparatus GT via finite differences.

The paper_impl GT is baselined REINFORCE — same estimator family as the RF-BL
draws, so its correctness needs an independent check. Here: central finite
differences of J_rf(theta) = E[avg discounted-free total cost] under *sampled*
actions (the objective REINFORCE differentiates), with common random numbers
(same torch seed for +h and -h) and B=2e5 trajectories per evaluation.

cos(FD, GT_nomask) ~ 1 validates the GT; cos(FD, PW_mean) then measures the
pathwise estimator against an estimator-family-free reference.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PAPERIMPL_NOMASK', '1')
from common import REPO, load_scaled_cfg, probs_from_params, routing_matrix, stable_seed
from paper_impl import maybe_mask, NOMASK
from queuetorch.env import load_env

DEV = 'cuda'
T = 1000


def J_hat(cfg, policy, theta_q, B, seed):
    """Monte-Carlo estimate of J(theta) = E[sum_t c_t * tau_t] (undiscounted,
    matching J_N in the paper) under sampled actions; CRN via fixed seed."""
    torch.manual_seed(seed)
    dq = load_env(cfg, temp=1.0, batch=B, seed=seed, device=DEV)
    s, q = dq.s, dq.q
    ctx = {'mu': dq.mu.double(), 'R': routing_matrix(dq).to(DEV)}
    net64 = dq.network.double()
    th = theta_q.double().to(DEV).unsqueeze(0)
    obs, state = dq.reset()
    total = torch.zeros(B, device=DEV)
    with torch.no_grad():
        for _ in range(T):
            queues, _ = obs
            q64 = queues.double()
            probs = maybe_mask(
                probs_from_params('paper_' + policy if not policy.startswith('paper_') else policy,
                                  [th.expand(B, q)], q64, B, 1, s, q, ctx), net64, q64)
            action = torch.distributions.OneHotCategorical(probs=probs).sample()
            obs, state, cost, _ = dq.step(state, action.float())
            total += cost.squeeze(1)
    return total.mean().item()


def fd_gradient(cfg, policy, theta_q, B, h_rel, seed):
    q = theta_q.shape[0]
    g = np.zeros(q)
    for j in range(q):
        h = h_rel * float(theta_q[j])
        tp, tm = theta_q.clone(), theta_q.clone()
        tp[j] += h
        tm[j] -= h
        jp = J_hat(cfg, policy, tp, B, seed)   # CRN: same seed both sides
        jm = J_hat(cfg, policy, tm, B, seed)
        g[j] = (jp - jm) / (2 * h)
    return g


def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


if __name__ == '__main__':
    assert NOMASK, 'run with PAPERIMPL_NOMASK=1'
    rho = 0.9
    cfg = load_scaled_cfg('criss_cross_bh', rho, 'paper')
    for pol in ['paper_sMP', 'paper_sMW', 'paper_sPR']:
        cid = f'criss_cross_bh__rho{rho}__{pol}__paperimpl_nomask'
        d = np.load(os.path.join(REPO, 'results', 'ask1', 'paper_impl', cid + '.npz'),
                    allow_pickle=True)
        gt, thetas, pw = d['gt'], d['theta'], d['pw_cos']
        # pick 3 thetas with reliable GT (split ~1) and defined pw
        order = np.argsort(-d['gt_split'])[:3]
        for i in order:
            th = torch.tensor(thetas[i])
            fds = [fd_gradient(cfg, pol, th, B=200_000, h_rel=hr,
                               seed=stable_seed('fd', pol, i, hr))
                   for hr in (0.05, 0.1)]
            c_gt = [cos(f, gt[i]) for f in fds]
            c_fd = cos(fds[0], fds[1])
            print(f'{pol} theta{i} (GTsplit={d["gt_split"][i]:+.2f}, '
                  f'PWmean={np.nanmean(pw[i]):+.2f}): '
                  f'cos(FD_h.05,FD_h.10)={c_fd:+.3f}  '
                  f'cos(FD,GT)={c_gt[0]:+.3f}/{c_gt[1]:+.3f}', flush=True)
