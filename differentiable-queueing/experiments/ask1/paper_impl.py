"""Full paper-faithful apparatus for §5.1 (arXiv:2409.03740) — the missing pieces.

Implements exactly what the paper describes but the released code omits:
  1. Policies: eq.(soft_policies) diagonal parameterization, theta in R^q_+,
     theta ~ Lognormal(0,1)  (common.py paper_* kinds).
  2. REINFORCE with a *value-function baseline* (eq. BASELINE): for each theta,
     V(x) is fitted on 1e6 state transitions (= 1000 trajectories x N=1000, as
     stated in the paper's intro figure caption), then the estimator uses the
     advantage G_t - V(x_t).
  3. GT = baselined REINFORCE averaged over many trajectories; PATHWISE beta=1.

Everything else identical to the paper spec: gamma=0.999, N=1000, PW B=1,
RF B=1000, cossim over estimator draws. Pilot: criss-cross, 3 policies x 2 rho.

sMP sign: standard MaxPressure pressure -(theta*x) @ R (the paper's printed
mu (.) R(theta x) has an inverted sign vs. the MaxPressure definition).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (GAMMA, HORIZON, REPO, cosine_rows, init_params, load_scaled_cfg,
                    mask_probs, pathwise_grads, probs_from_params, routing_matrix,
                    stable_seed, _to_device_params)
from queuetorch.env import load_env

DEV = 'cuda'
OUT = os.path.join(REPO, 'results', 'ask1', 'paper_impl')
os.makedirs(OUT, exist_ok=True)

# PAPERIMPL_NOMASK=1: use the raw eq.(soft_policies) softmax as the policy
# (paper-literal; no released-code min-with-queues masking). The env still
# enforces feasibility internally, but log-probs / pathwise actions are of the
# unmasked distribution.
NOMASK = os.environ.get('PAPERIMPL_NOMASK', '0') == '1'
SUFFIX = '_nomask' if NOMASK else ''


def maybe_mask(probs, network, queues):
    return probs if NOMASK else mask_probs(probs, network, queues)


class ValueNet(nn.Module):
    """V(x, t/N): time feature is essential — with gamma=0.999 and N=1000 the
    return-to-go depends strongly on the remaining horizon, and a state-only
    V(x) explains almost none of the variance (measured R2 ~ 0.05). Any
    function of (state, t) is a valid unbiased baseline."""

    def __init__(self, q):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(q + 1, 64), nn.ReLU(),
                               nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.mu = 0.0
        self.sd = 1.0

    def features(self, x, t_frac):
        return torch.cat([torch.log1p(x), t_frac.unsqueeze(-1)], dim=-1)

    def forward(self, x, t_frac):
        return self.f(self.features(x, t_frac)).squeeze(-1) * self.sd + self.mu


def rollout_states_returns(cfg, policy, theta_q, B, seed, T=HORIZON, gamma=GAMMA):
    """no-grad rollout under sampled actions; returns states (B,T,q) and
    discounted returns-to-go (B,T)."""
    torch.manual_seed(seed)
    dq = load_env(cfg, temp=1.0, batch=B, seed=seed, device=DEV)
    s, q = dq.s, dq.q
    ctx = {'mu': dq.mu.double(), 'R': routing_matrix(dq).to(DEV)}
    net64 = dq.network.double()
    theta_b = theta_q.double().to(DEV).unsqueeze(0).expand(B, q)
    obs, state = dq.reset()
    xs = torch.empty(B, T, q, device=DEV)
    rws = torch.empty(B, T, device=DEV)
    with torch.no_grad():
        for t in range(T):
            queues, _ = obs
            xs[:, t] = queues
            q64 = queues.double()
            probs = maybe_mask(probs_from_params(policy, [theta_b[:1].expand(B, q)],
                                                 q64, B, 1, s, q, ctx), net64, q64)
            action = torch.distributions.OneHotCategorical(probs=probs).sample()
            obs, state, cost, _ = dq.step(state, action.float())
            rws[:, t] = -cost.squeeze(1)
    ret = torch.empty_like(rws)
    acc = torch.zeros(B, device=DEV)
    for t in reversed(range(T)):
        acc = rws[:, t] + gamma * acc
        ret[:, t] = acc
    return xs, ret


def fit_value(cfg, policy, theta_q, seed, n_traj=1000, epochs=3, bs=8192):
    """Fit V(x, t/N) on n_traj x T = 1e6 (x, t, G) tuples (paper caption: 1e6
    state transitions)."""
    xs, ret = rollout_states_returns(cfg, policy, theta_q, n_traj, seed)
    B, T, q = xs.shape
    X = xs.reshape(-1, q)
    TF = (torch.arange(T, device=DEV).float() / T).unsqueeze(0).expand(B, T).reshape(-1)
    Y = ret.reshape(-1)
    v = ValueNet(q).to(DEV)
    v.mu = Y.mean().item()
    v.sd = Y.std().clamp_min(1e-6).item()
    Yn = (Y - v.mu) / v.sd
    feats = v.features(X, TF)
    opt = torch.optim.Adam(v.f.parameters(), lr=1e-3)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = v.f(feats[idx]).squeeze(-1)
            loss = ((pred - Yn[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        r2 = 1 - tot / n
    v.eval()
    for p in v.parameters():
        p.requires_grad_(False)
    return v, r2


def rf_grads_baseline(cfg, policy, theta_q, vnet, G, B_per_group, seed,
                      T=HORIZON, gamma=GAMMA):
    """Baselined REINFORCE (eq. BASELINE): per-group estimates, all groups share
    the same theta value (groups = independent draws). Returns (G, q)."""
    B = G * B_per_group
    torch.manual_seed(seed)
    dq = load_env(cfg, temp=1.0, batch=B, seed=seed, device=DEV)
    s, q = dq.s, dq.q
    ctx = {'mu': dq.mu.double(), 'R': routing_matrix(dq).to(DEV)}
    net64 = dq.network.double()
    params = _to_device_params([theta_q.double().unsqueeze(0).expand(G, q)], DEV)
    obs, state = dq.reset()
    log_probs, rewards, vpreds = [], [], []
    for t in range(T):
        queues, _ = obs
        q64 = queues.double()
        probs = probs_from_params(policy, params, q64, G, B_per_group, s, q, ctx)
        probs = maybe_mask(probs, net64, q64)
        dist = torch.distributions.OneHotCategorical(probs=probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action).sum(dim=1))
        with torch.no_grad():
            tf = torch.full((B,), t / T, device=DEV)
            vpreds.append(vnet(queues, tf).double())
        obs, state, cost, _ = dq.step(state, action.float())
        rewards.append(-cost.squeeze(1).double())
    ret = torch.zeros(B, dtype=torch.float64, device=DEV)
    loss = torch.zeros(B, dtype=torch.float64, device=DEV)
    for t in reversed(range(T)):
        ret = rewards[t] + gamma * ret
        loss = loss - log_probs[t] * (ret - vpreds[t])
    loss.reshape(G, B_per_group).mean(dim=1).sum().backward()
    g = params[0].grad.reshape(G, q).detach().cpu()
    del dq
    return g


def pw_grads_local(cfg, policy, theta_groups, seed, T=HORIZON):
    """PATHWISE B=1 per group, honoring NOMASK (common.pathwise_grads always masks)."""
    G, q = theta_groups.shape
    torch.manual_seed(seed)
    dq = load_env(cfg, temp=1.0, batch=G, seed=seed, device=DEV)
    s = dq.s
    ctx = {'mu': dq.mu.double(), 'R': routing_matrix(dq).to(DEV)}
    net64 = dq.network.double()
    params = _to_device_params([theta_groups], DEV)
    obs, state = dq.reset()
    total = torch.zeros(G, device=DEV)
    for _ in range(T):
        queues, _ = obs
        q64 = queues.double()
        probs = maybe_mask(probs_from_params(policy, params, q64, G, 1, s, q, ctx),
                           net64, q64)
        obs, state, cost, _ = dq.step(state, probs.float())
        total = total + cost.squeeze(1)
    (total / T).double().sum().backward()
    return params[0].grad.reshape(G, q).detach().cpu()


def run_cell(env, rho, policy, n_theta=10, n_draws=20, gt_trajs=100_000,
             rf_b=1000, gt_chunk=10_000):
    base_cid = f'{env}__rho{rho}__{policy}__paperimpl'
    cid = base_cid + SUFFIX
    if os.path.exists(os.path.join(OUT, cid + '.npz')):
        print(f'== {cid}: already done, skipping', flush=True)
        return
    t0 = time.time()
    cfg = load_scaled_cfg(env, rho, 'paper')
    thetas = init_params(policy, n_theta, 2, 3, seed=stable_seed(base_cid, 'theta'))[0]
    q = thetas.shape[1]
    gt = torch.zeros(n_theta, q, dtype=torch.float64)
    gt_split = np.zeros(n_theta, dtype=np.float32)
    pw_cos = np.full((n_theta, n_draws), np.nan, dtype=np.float32)
    rfb_cos = np.full((n_theta, n_draws), np.nan, dtype=np.float32)
    r2s = []
    for i in range(n_theta):
        th = thetas[i]
        vnet, r2 = fit_value(cfg, policy, th, seed=stable_seed(cid, 'vfit', i))
        r2s.append(r2)
        # GT: baselined REINFORCE, gt_trajs total, split-half
        acc = [torch.zeros(q, dtype=torch.float64), torch.zeros(q, dtype=torch.float64)]
        cnt = [0, 0]
        n_chunks = gt_trajs // gt_chunk
        for c in range(n_chunks):
            g = rf_grads_baseline(cfg, policy, th, vnet, 1, gt_chunk,
                                  seed=stable_seed(cid, 'gt', i, c))[0]
            if not torch.isnan(g).any():
                acc[c % 2] += g
                cnt[c % 2] += 1
        g1 = acc[0] / max(cnt[0], 1)
        g2 = acc[1] / max(cnt[1], 1)
        gt[i] = (acc[0] + acc[1]) / max(cnt[0] + cnt[1], 1)
        gt_split[i] = float(cosine_rows(g1.unsqueeze(0), g2)[0])
        # estimator draws: RF-BL (B=1000) x n_draws grouped; PW B=1 x n_draws
        rfb = rf_grads_baseline(cfg, policy, th, vnet, n_draws, rf_b,
                                seed=stable_seed(cid, 'rfb', i))
        pw = pw_grads_local(cfg, policy, th.unsqueeze(0).expand(n_draws, q).contiguous(),
                            seed=stable_seed(cid, 'pw', i))
        rfb_cos[i] = cosine_rows(rfb, gt[i]).numpy()
        pw_cos[i] = cosine_rows(pw, gt[i]).numpy()
        print(f'  [{cid}] theta {i}: V-R2={r2:.3f} GTsplit={gt_split[i]:+.2f} '
              f'PW={np.nanmean(pw_cos[i]):+.3f} RF-BL={np.nanmean(rfb_cos[i]):+.3f}',
              flush=True)
    meta = dict(env=env, rho=rho, policy=policy, n_theta=n_theta, n_draws=n_draws,
                gt_trajs=gt_trajs, rf_b=rf_b, v_r2_mean=float(np.mean(r2s)),
                gt_split_median=float(np.nanmedian(gt_split)),
                pw_mean=float(np.nanmean(pw_cos)), rfb_mean=float(np.nanmean(rfb_cos)),
                t_total_s=round(time.time() - t0, 1))
    np.savez_compressed(os.path.join(OUT, cid + '.npz'), gt=gt.numpy(),
                        theta=thetas.numpy(), pw_cos=pw_cos, rfb_cos=rfb_cos,
                        gt_split=gt_split, meta=json.dumps(meta))
    print(f'== {cid}: PW={meta["pw_mean"]:+.3f} RF-BL={meta["rfb_mean"]:+.3f} '
          f'GTsplit={meta["gt_split_median"]:+.2f} V-R2={meta["v_r2_mean"]:.3f} '
          f'({meta["t_total_s"]/60:.1f} min)', flush=True)


if __name__ == '__main__':
    for rho in [0.9, 0.99]:
        for pol in ['paper_sMP', 'paper_sMW', 'paper_sPR']:
            run_cell('criss_cross_bh', rho, pol)
    print('PAPER-IMPL PILOT DONE')
