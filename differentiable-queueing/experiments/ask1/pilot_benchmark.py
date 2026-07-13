"""Pilot benchmark for Ask-1 (§5.1 gradient cossim, paper-canonical rerun).

Measures wall-clock and memory for the atomic units of the experiment so we can
budget the full 9-networks x 4-rho x 3-policies grid on 8x A800.

Units measured (per network):
  1. REINFORCE rollout+backward, B=1000, T=1000   (one RF estimator draw)
  2. REINFORCE rollout+backward, B=B_big, T=1000  (GT chunk)
  3. PATHWISE rollout+backward through env graph, B=100 (grouped PW draws)
  4. NaN check at rho=0.99
"""
import os, sys, time
import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'experiments'))
os.chdir(REPO)

from queuetorch.env import load_env

DEV = 'cuda'


def load_cfg(env_name, rho):
    cfg = yaml.safe_load(open(f'configs/env/{env_name}.yaml'))
    if cfg['lam_params']['val'] is None:
        lam = np.load(f'env_data/{env_name}/{env_name}_lam.npy')
    else:
        lam = np.array(cfg['lam_params']['val'], dtype=float)
    # baseline configs are all at rho=0.9; scale to target rho
    cfg['lam_params']['val'] = (lam * rho / 0.9).tolist()
    return cfg


def make_policy_probs(theta, mu, network, queues):
    """Paper eq (soft_policies) sPR: logits_ij = theta_j * mu_ij, masked."""
    logits = theta.unsqueeze(1) * mu[0]  # theta (B,1,q) -> broadcast over servers
    logits = logits.masked_fill(network[0] == 0, -1e9)
    return torch.softmax(logits, dim=-1)


def mask_probs(probs, network, queues):
    """Author's feasibility wrapper from gradient_comparison.py."""
    B, s, q = probs.shape
    probs = probs * network[:B]
    probs = torch.min(torch.stack((probs, queues.unsqueeze(1).expand(-1, s, -1)), dim=3), dim=3).values
    all_zero = torch.all(probs == 0., dim=2, keepdim=True)
    probs = probs + all_zero * network[:B]
    ssum = probs.sum(-1, keepdim=True)
    probs = probs / torch.where(ssum == 0, torch.ones_like(ssum), ssum)
    return probs


def bench_reinforce(cfg, B, T, gamma=0.999):
    torch.cuda.reset_peak_memory_stats()
    dq = load_env(cfg, temp=1.0, batch=B, seed=None, device=DEV)
    q = dq.q
    theta = torch.distributions.LogNormal(0., 1.).sample((q,)).to(DEV).requires_grad_(True)
    obs, state = dq.reset()
    t0 = time.time()
    log_probs, rewards = [], []
    for _ in range(T):
        queues, _ = obs
        probs = make_policy_probs(theta.unsqueeze(0).expand(B, -1), dq.mu, dq.network, queues)
        probs = mask_probs(probs, dq.network, queues)
        dist = torch.distributions.OneHotCategorical(probs=probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action).sum(1))
        obs, state, cost, _ = dq.step(state, action)
        rewards.append(-cost.squeeze(1))
    loss = torch.zeros(1, device=DEV)
    ret = torch.zeros(B, device=DEV)
    for t in reversed(range(T)):
        ret = rewards[t] + gamma * ret
        loss = loss - (log_probs[t] * ret).mean()
    loss.backward()
    torch.cuda.synchronize()
    g = theta.grad.detach()
    return time.time() - t0, torch.cuda.max_memory_allocated() / 1e9, g


def bench_pathwise(cfg, B, T):
    torch.cuda.reset_peak_memory_stats()
    dq = load_env(cfg, temp=1.0, batch=B, seed=None, device=DEV)
    q = dq.q
    theta = torch.distributions.LogNormal(0., 1.).sample((B, q)).to(DEV).requires_grad_(True)
    obs, state = dq.reset()
    t0 = time.time()
    total = torch.zeros(1, device=DEV)
    for _ in range(T):
        queues, _ = obs
        probs = make_policy_probs(theta, dq.mu, dq.network, queues)
        probs = mask_probs(probs, dq.network, queues)
        obs, state, cost, _ = dq.step(state, probs)
        total = total + cost.sum()
    (total / T).backward()
    torch.cuda.synchronize()
    g = theta.grad.detach()
    return time.time() - t0, torch.cuda.max_memory_allocated() / 1e9, g


if __name__ == '__main__':
    for env_name, B_big in [('criss_cross_bh', 100_000), ('reentrant_5', 20_000)]:
        for rho in [0.9, 0.99]:
            cfg = load_cfg(env_name if env_name != 'criss_cross_bh' else 'criss_cross_bh', rho)
            print(f'\n== {env_name} rho={rho} ==', flush=True)
            dt, mem, g = bench_reinforce(cfg, 1000, 1000)
            print(f'  RF  B=1000   T=1000: {dt:6.1f}s  peak {mem:5.1f}GB  nan={torch.isnan(g).any().item()}  |g|={g.norm():.3e}', flush=True)
            dt, mem, g = bench_reinforce(cfg, B_big, 1000)
            print(f'  RF  B={B_big} T=1000: {dt:6.1f}s  peak {mem:5.1f}GB  nan={torch.isnan(g).any().item()}  |g|={g.norm():.3e}', flush=True)
            dt, mem, g = bench_pathwise(cfg, 100, 1000)
            print(f'  PW  B=100    T=1000: {dt:6.1f}s  peak {mem:5.1f}GB  nan={torch.isnan(g).any().item()}  |g|={g.norm():.3e}', flush=True)
