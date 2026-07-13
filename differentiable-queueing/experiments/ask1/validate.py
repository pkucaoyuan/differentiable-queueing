"""Validation for the grouped upstream-faithful implementation in common.py.

1. Exact forward equivalence with queuetorch/policies.py modules.
2. No cross-group leakage: per-group loss grads only touch their own param rows.
3. Smoke test (criss-cross rho=0.9): GT split-half consistency; PW/RF cossim.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (REPO, cosine_rows, init_params, load_scaled_cfg, mask_probs,
                    pathwise_grads, probs_from_params, reinforce_grads,
                    repeat_params, slice_params, stable_seed, _to_device_params)

sys.path.insert(0, os.path.join(REPO, 'experiments'))
from gradient_comparison import get_policy
from queuetorch.env import load_env

DEV = 'cuda'


def test_forward_equivalence():
    """Functional grouped policies must match upstream nn.Module forward exactly."""
    s, q, G, B = 2, 3, 4, 7
    queues = torch.randint(0, 20, (G * B, q)).double()
    for kind in ['sPR', 'sMW', 'sMP']:
        params = init_params(kind, G, s, q, seed=11)
        mine = probs_from_params(kind, params, queues, G, B, s, q)
        worst = 0.
        for g in range(G):
            net = get_policy(kind, s, q).double()
            with torch.no_grad():
                if kind == 'sPR':
                    net.theta.copy_(params[0][g])
                else:
                    net.fc.weight.copy_(params[0][g])
                    net.fc.bias.copy_(params[1][g])
            ref = net(queues[g * B:(g + 1) * B])
            worst = max(worst, (ref - mine[g * B:(g + 1) * B]).abs().max().item())
        print(f'forward equivalence {kind}: max diff={worst:.2e}')
        assert worst < 1e-12


def rollout_group_losses(cfg, policy, params_cpu, B_per_group, seed, T, mode):
    G = params_cpu[0].shape[0]
    B = G * B_per_group
    torch.manual_seed(seed)
    dq = load_env(cfg, temp=1.0, batch=B, seed=seed, device=DEV)
    s, q = dq.s, dq.q
    net64 = dq.network.double()
    params = _to_device_params(params_cpu, DEV)
    obs, state = dq.reset()
    if mode == 'rf':
        log_probs, rewards = [], []
        for _ in range(T):
            queues, _ = obs
            q64 = queues.double()
            probs = mask_probs(probs_from_params(policy, params, q64, G, B_per_group, s, q), net64, q64)
            dist = torch.distributions.OneHotCategorical(probs=probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action).sum(1))
            obs, state, cost, _ = dq.step(state, action.float())
            rewards.append(-cost.squeeze(1).double())
        ret = torch.zeros(B, dtype=torch.float64, device=DEV)
        loss = torch.zeros(B, dtype=torch.float64, device=DEV)
        for t in reversed(range(T)):
            ret = rewards[t] + 0.999 * ret
            loss = loss - log_probs[t] * ret
    else:
        loss = torch.zeros(B, device=DEV)
        for _ in range(T):
            queues, _ = obs
            q64 = queues.double()
            probs = mask_probs(probs_from_params(policy, params, q64, G, B_per_group, s, q), net64, q64)
            obs, state, cost, _ = dq.step(state, probs.float())
            loss = loss + cost.squeeze(1)
        loss = (loss / T).double()
    return params, loss.reshape(G, B_per_group).mean(1)


def test_no_leakage(mode):
    cfg = load_scaled_cfg('criss_cross_bh', 0.9, 'author')
    G, Bpg, T = 3, 2, 50
    params_cpu = init_params('sMW', G, 2, 3, seed=5)
    params, losses = rollout_group_losses(cfg, 'sMW', params_cpu, Bpg, seed=7, T=T, mode=mode)
    grouped = torch.autograd.grad(losses.sum(), params, retain_graph=True)
    max_leak, max_err = 0., 0.
    for g in range(G):
        gi = torch.autograd.grad(losses[g], params, retain_graph=True, allow_unused=True)
        for p_i, p_g in zip(gi, grouped):
            own_err = (p_i[g] - p_g[g]).abs().max().item()
            leak = p_i[torch.arange(G) != g].abs().max().item()
            max_err, max_leak = max(max_err, own_err), max(max_leak, leak)
    print(f'[{mode}] own-row err={max_err:.2e}  cross-group leak={max_leak:.2e}')
    assert max_leak == 0.0 and max_err < 1e-9, 'group leakage detected'


def test_smoke_cossim():
    cfg = load_scaled_cfg('criss_cross_bh', 0.9, 'author')
    n_th, n_dr = 4, 10
    for pol in ['sPR', 'sMW']:
        params = init_params(pol, n_th, 2, 3, seed=stable_seed('smoke', pol))
        acc = [None, None]
        for c in range(4):  # 4 chunks x 6250 = 2.5e4 GT trajs per init
            g = reinforce_grads(cfg, pol, params, 25_000 // n_th, seed=stable_seed('smokegt', pol, c), device=DEV)
            h = c % 2
            acc[h] = g if acc[h] is None else acc[h] + g
        gt1, gt2 = acc[0] / 2, acc[1] / 2
        gt = (gt1 + gt2) / 2
        split = torch.stack([cosine_rows(gt1[i].unsqueeze(0), gt2[i])[0] for i in range(n_th)])
        pair = repeat_params(params, n_dr)
        pw = pathwise_grads(cfg, pol, pair, 1, seed=1, device=DEV)
        rf = reinforce_grads(cfg, pol, pair, 1000, seed=2, device=DEV)
        pw_m = torch.stack([cosine_rows(pw[i*n_dr:(i+1)*n_dr], gt[i]).nanmean() for i in range(n_th)]).mean()
        rf_m = torch.stack([cosine_rows(rf[i*n_dr:(i+1)*n_dr], gt[i]).nanmean() for i in range(n_th)]).mean()
        print(f'smoke {pol}: GT split-half cos={split.mean():+.3f}  PW={pw_m:+.3f}  RF={rf_m:+.3f}')


if __name__ == '__main__':
    test_forward_equivalence()
    test_no_leakage('pw')
    test_no_leakage('rf')
    test_smoke_cossim()
    print('ALL VALIDATION PASSED')
