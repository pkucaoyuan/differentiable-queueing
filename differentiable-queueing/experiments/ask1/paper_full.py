"""Full paper-spec run of §5.1 on the criss-cross row (12 cells).

Paper spec, implemented literally:
  - policies eq.(soft_policies), theta in R^q_+, theta ~ Lognormal(0,1), NO masking
  - 100 thetas per cell; estimators drawn 100 times each
  - GT: baselined REINFORCE averaged over 1e6 trajectories per theta
    (baseline: per-theta V(x, t/N) MLP fitted on rollout transitions, per the
    intro caption's "value function baseline fitted using 1e6 state transitions";
    we fit on 2e5 subsampled transitions per theta — R2 plateaus well before that)
  - PATHWISE B=1 (env STE beta=1), REINFORCE B=1000 (with the same baseline)
  - gamma=0.999, N=1000; sMP uses standard MaxPressure sign

Grouped implementation: all 100 thetas share env batches; per-theta V nets are a
single grouped MLP (batched einsum layers). Claims dir allows 8-GPU parallelism.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (GAMMA, HORIZON, REPO, cosine_rows, init_params, load_scaled_cfg,
                    probs_from_params, routing_matrix, stable_seed, _to_device_params)
from queuetorch.env import load_env

DEV = 'cuda'
OUT = os.path.join(REPO, 'results', 'ask1', 'paper_full')
CLAIMS = os.path.join(OUT, 'claims')
os.makedirs(CLAIMS, exist_ok=True)

N_THETA = 100
N_DRAWS = 100
GT_TRAJS = 1_000_000
RF_B = 1000
VFIT_TRAJS = 1000          # per theta -> 1e6 transitions generated; subsampled x5
VFIT_SUBSTEP = 5
RF_CAP = 80_000            # total trajectories per rollout (fp64 log-prob graph)
PW_CAP = 14_000


class GVal(torch.nn.Module):
    """G independent V(x, t/N) MLPs as batched einsum layers."""

    def __init__(self, G, q, h=64):
        super().__init__()
        def lin(i, o):
            b = 1.0 / (i ** 0.5)
            return (torch.nn.Parameter((torch.rand(G, i, o) * 2 - 1) * b),
                    torch.nn.Parameter((torch.rand(G, o) * 2 - 1) * b))
        self.W1, self.b1 = lin(q + 1, h)
        self.W2, self.b2 = lin(h, h)
        self.W3, self.b3 = lin(h, 1)
        self.register_buffer('mu', torch.zeros(G))
        self.register_buffer('sd', torch.ones(G))

    def core(self, f):
        """f (G, Bp, q+1) -> (G, Bp) normalized prediction."""
        z = torch.relu(torch.einsum('gbi,gio->gbo', f, self.W1) + self.b1.unsqueeze(1))
        z = torch.relu(torch.einsum('gbi,gio->gbo', z, self.W2) + self.b2.unsqueeze(1))
        return (torch.einsum('gbi,gio->gbo', z, self.W3) + self.b3.unsqueeze(1)).squeeze(-1)

    def value(self, x, t_frac, gidx=None):
        """x (B,q), t_frac scalar, gidx (B,) group index or None for arange-repeat."""
        f = torch.cat([torch.log1p(x), torch.full_like(x[:, :1], t_frac)], dim=-1)
        G = self.W1.shape[0]
        Bp = x.shape[0] // G
        f = f.reshape(G, Bp, -1)
        out = self.core(f) * self.sd.unsqueeze(1) + self.mu.unsqueeze(1)
        return out.reshape(-1)


def rollout_collect(cfg, thetas, policy, Bpg, seed, T=HORIZON, gamma=GAMMA):
    """Grouped no-grad rollout under sampled actions. Returns subsampled
    (xs (G,Bpg,Ts,q), tf (Ts,), rets (G,Bpg,Ts))."""
    G, q = thetas.shape
    B = G * Bpg
    torch.manual_seed(seed)
    dq = load_env(cfg, temp=1.0, batch=B, seed=seed, device=DEV)
    s = dq.s
    ctx = {'mu': dq.mu.double(), 'R': routing_matrix(dq).to(DEV)}
    th = thetas.double().to(DEV)
    obs, state = dq.reset()
    keep = list(range(0, T, VFIT_SUBSTEP))
    xs = torch.empty(B, len(keep), q, device=DEV)
    rws = torch.empty(B, T, device=DEV)
    ki = 0
    with torch.no_grad():
        for t in range(T):
            queues, _ = obs
            if ki < len(keep) and t == keep[ki]:
                xs[:, ki] = queues
                ki += 1
            probs = probs_from_params(policy, [th], queues.double(), G, Bpg, s, q, ctx)
            action = torch.distributions.OneHotCategorical(probs=probs).sample()
            obs, state, cost, _ = dq.step(state, action.float())
            rws[:, t] = -cost.squeeze(1)
    ret = torch.empty_like(rws)
    acc = torch.zeros(B, device=DEV)
    for t in reversed(range(T)):
        acc = rws[:, t] + gamma * acc
        ret[:, t] = acc
    rets = ret[:, keep]
    tf = torch.tensor(keep, device=DEV).float() / T
    return (xs.reshape(G, Bpg, len(keep), q), tf, rets.reshape(G, Bpg, len(keep)))


def fit_gval(xs, tf, rets, epochs=4, bs=4096):
    """xs (G,Bp,Ts,q), rets (G,Bp,Ts) -> trained GVal + per-group R2."""
    G, Bp, Ts, q = xs.shape
    N = Bp * Ts
    X = xs.reshape(G, N, q)
    Y = rets.reshape(G, N)
    v = GVal(G, q).to(DEV)
    with torch.no_grad():
        v.mu.copy_(Y.mean(dim=1))
        v.sd.copy_(Y.std(dim=1).clamp_min(1e-6))
    Yn = (Y - v.mu.unsqueeze(1)) / v.sd.unsqueeze(1)
    TF = tf.repeat(Bp)  # (N,) matching reshape order (Bp, Ts) -> Bp*Ts
    feats = torch.cat([torch.log1p(X), TF.expand(G, N).unsqueeze(-1)], dim=-1)
    opt = torch.optim.Adam(v.parameters(), lr=1e-3)
    last = None
    for ep in range(epochs):
        perm = torch.randperm(N, device=DEV)
        tot = torch.zeros(G, device=DEV)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            pred = v.core(feats[:, idx])
            loss_g = ((pred - Yn[:, idx]) ** 2).mean(dim=1)
            opt.zero_grad()
            loss_g.sum().backward()
            opt.step()
            tot += loss_g.detach() * len(idx)
        last = 1 - tot / N
    for p in v.parameters():
        p.requires_grad_(False)
    return v, last.cpu().numpy()


def rf_bl_grads(cfg, policy, thetas_pergroup, vnet, B_per_group, seed,
                T=HORIZON, gamma=GAMMA):
    """Grouped baselined REINFORCE, no masking. thetas_pergroup (G,q) — the V
    net must have G groups aligned with these thetas. Returns (G,q)."""
    G, q = thetas_pergroup.shape
    B = G * B_per_group
    torch.manual_seed(seed)
    dq = load_env(cfg, temp=1.0, batch=B, seed=seed, device=DEV)
    s = dq.s
    ctx = {'mu': dq.mu.double(), 'R': routing_matrix(dq).to(DEV)}
    params = _to_device_params([thetas_pergroup.double()], DEV)
    obs, state = dq.reset()
    log_probs, rewards, vpreds = [], [], []
    for t in range(T):
        queues, _ = obs
        probs = probs_from_params(policy, params, queues.double(), G, B_per_group, s, q, ctx)
        dist = torch.distributions.OneHotCategorical(probs=probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action).sum(dim=1))
        with torch.no_grad():
            vpreds.append(vnet.value(queues, t / T).double())
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


def pw_grads(cfg, policy, thetas_pergroup, seed, T=HORIZON):
    G, q = thetas_pergroup.shape
    torch.manual_seed(seed)
    dq = load_env(cfg, temp=1.0, batch=G, seed=seed, device=DEV)
    s = dq.s
    ctx = {'mu': dq.mu.double(), 'R': routing_matrix(dq).to(DEV)}
    params = _to_device_params([thetas_pergroup.double()], DEV)
    obs, state = dq.reset()
    total = torch.zeros(G, device=DEV)
    for _ in range(T):
        queues, _ = obs
        probs = probs_from_params(policy, params, queues.double(), G, 1, s, q, ctx)
        obs, state, cost, _ = dq.step(state, probs.float())
        total = total + cost.squeeze(1)
    (total / T).double().sum().backward()
    return params[0].grad.reshape(G, q).detach().cpu()


class RepeatVal:
    """View of a GVal where group g of the pair-layout maps to theta i = g // n_rep."""

    def __init__(self, base, n_rep):
        self.base = base
        self.n_rep = n_rep

    def value(self, x, t_frac):
        G0 = self.base.W1.shape[0]
        B = x.shape[0]
        Bp = B // (G0 * self.n_rep)
        f = torch.cat([torch.log1p(x), torch.full_like(x[:, :1], t_frac)], dim=-1)
        f = f.reshape(G0, self.n_rep * Bp, -1)
        out = self.base.core(f) * self.base.sd.unsqueeze(1) + self.base.mu.unsqueeze(1)
        return out.reshape(-1)


def run_cell(env, rho, policy):
    cid = f'{env}__rho{rho}__{policy}__paperfull'
    out_path = os.path.join(OUT, cid + '.npz')
    if os.path.exists(out_path):
        return
    try:
        os.mkdir(os.path.join(CLAIMS, cid))
    except FileExistsError:
        return
    print(f'>>> {cid}', flush=True)
    t0 = time.time()
    cfg = load_scaled_cfg(env, rho, 'paper')
    thetas = init_params(policy, N_THETA, 2, 3, seed=stable_seed(cid, 'theta'))[0]
    q = thetas.shape[1]

    # 1. V baseline: grouped rollout (100 theta x 1000 trajs) + grouped fit
    xs, tf, rets = rollout_collect(cfg, thetas, policy, VFIT_TRAJS,
                                   seed=stable_seed(cid, 'vfit'))
    vnet, r2 = fit_gval(xs, tf, rets)
    del xs, rets
    print(f'[{cid}] V fitted: R2 median={np.median(r2):.3f} '
          f'({(time.time()-t0)/60:.1f} min)', flush=True)

    # 2. GT: 1e6 baselined-REINFORCE trajectories per theta, split-half diag
    B_c = max(1, RF_CAP // N_THETA)
    n_chunks = GT_TRAJS // B_c
    halves = [torch.zeros(N_THETA, q, dtype=torch.float64) for _ in range(2)]
    counts = [torch.zeros(N_THETA) for _ in range(2)]
    for c in range(n_chunks):
        g = rf_bl_grads(cfg, policy, thetas, vnet, B_c, seed=stable_seed(cid, 'gt', c))
        good = ~torch.isnan(g).any(dim=1)
        halves[c % 2][good] += g[good]
        counts[c % 2][good] += 1
        if c % 200 == 0:
            print(f'[{cid}] GT chunk {c}/{n_chunks} ({(time.time()-t0)/60:.1f} min)',
                  flush=True)
    cnt = counts[0] + counts[1]
    gt = (halves[0] + halves[1]) / cnt.clamp_min(1).unsqueeze(1)
    gt[cnt == 0] = float('nan')
    gt_split = np.array([float(cosine_rows(
        (halves[0][i] / max(counts[0][i].item(), 1)).unsqueeze(0),
        halves[1][i] / max(counts[1][i].item(), 1))[0]) for i in range(N_THETA)],
        dtype=np.float32)
    t_gt = time.time() - t0
    print(f'[{cid}] GT done: split median={np.nanmedian(gt_split):+.3f} '
          f'avg drops/theta={n_chunks - cnt.float().mean().item():.1f} '
          f'({t_gt/60:.1f} min)', flush=True)

    # 3. estimator draws: pairs (theta_i, draw_j), theta-major order
    pair_thetas = thetas.repeat_interleave(N_DRAWS, dim=0)
    rvnet = RepeatVal(vnet, N_DRAWS)
    per = max(1, RF_CAP // RF_B)
    rf = torch.empty(N_THETA * N_DRAWS, q, dtype=torch.float64)
    i = 0
    c = 0
    while i < pair_thetas.shape[0]:
        j = min(pair_thetas.shape[0], i + per)
        # RepeatVal requires aligned blocks: process per-theta blocks of N_DRAWS
        gsl = rf_bl_grads(cfg, policy, pair_thetas[i:j],
                          RepeatValSlice(vnet, i, j, N_DRAWS), RF_B,
                          seed=stable_seed(cid, 'rf', c))
        rf[i:j] = gsl
        i, c = j, c + 1
    pw = torch.empty(N_THETA * N_DRAWS, q, dtype=torch.float64)
    i, c = 0, 0
    while i < pair_thetas.shape[0]:
        j = min(pair_thetas.shape[0], i + PW_CAP)
        pw[i:j] = pw_grads(cfg, policy, pair_thetas[i:j], seed=stable_seed(cid, 'pw', c))
        i, c = j, c + 1

    pw_cos = np.full((N_THETA, N_DRAWS), np.nan, dtype=np.float32)
    rf_cos = np.full((N_THETA, N_DRAWS), np.nan, dtype=np.float32)
    for i in range(N_THETA):
        g = gt[i]
        if torch.isnan(g).any() or g.norm() < 1e-12:
            continue
        sl = slice(i * N_DRAWS, (i + 1) * N_DRAWS)
        pw_cos[i] = cosine_rows(pw[sl], g).numpy()
        rf_cos[i] = cosine_rows(rf[sl], g).numpy()

    meta = dict(env=env, rho=rho, policy=policy, n_theta=N_THETA, n_draws=N_DRAWS,
                gt_trajs=GT_TRAJS, rf_b=RF_B, gamma=GAMMA, horizon=HORIZON,
                nomask=True, v_r2_median=float(np.median(r2)),
                gt_split_median=float(np.nanmedian(gt_split)),
                pw_mean=float(np.nanmean(pw_cos)), rf_mean=float(np.nanmean(rf_cos)),
                t_total_s=round(time.time() - t0, 1))
    np.savez_compressed(out_path, gt=gt.numpy(), theta=thetas.numpy(),
                        pw_cos=pw_cos, rf_cos=rf_cos, gt_split=gt_split,
                        v_r2=r2, meta=json.dumps(meta))
    print(f'== {cid}: PW={meta["pw_mean"]:+.3f} RF-BL={meta["rf_mean"]:+.3f} '
          f'GTsplit={meta["gt_split_median"]:+.2f} ({meta["t_total_s"]/60:.1f} min)',
          flush=True)


class RepeatValSlice:
    """V lookup for a contiguous pair slice [i, j) in theta-major layout."""

    def __init__(self, base, i, j, n_draws):
        self.base = base
        self.i, self.j, self.nd = i, j, n_draws

    def value(self, x, t_frac):
        B = x.shape[0]
        n_pairs = self.j - self.i
        Bp = B // n_pairs
        theta_idx = (torch.arange(self.i, self.j, device=x.device) // self.nd)
        f = torch.cat([torch.log1p(x), torch.full_like(x[:, :1], t_frac)], dim=-1)
        f = f.reshape(n_pairs, Bp, -1)
        W1 = self.base.W1[theta_idx]
        b1 = self.base.b1[theta_idx]
        W2 = self.base.W2[theta_idx]
        b2 = self.base.b2[theta_idx]
        W3 = self.base.W3[theta_idx]
        b3 = self.base.b3[theta_idx]
        z = torch.relu(torch.einsum('gbi,gio->gbo', f, W1) + b1.unsqueeze(1))
        z = torch.relu(torch.einsum('gbi,gio->gbo', z, W2) + b2.unsqueeze(1))
        out = (torch.einsum('gbi,gio->gbo', z, W3) + b3.unsqueeze(1)).squeeze(-1)
        out = out * self.base.sd[theta_idx].unsqueeze(1) + self.base.mu[theta_idx].unsqueeze(1)
        return out.reshape(-1)


if __name__ == '__main__':
    cells = [('criss_cross_bh', rho, pol)
             for rho in [0.9, 0.99, 0.95, 0.8]
             for pol in ['paper_sMP', 'paper_sMW', 'paper_sPR']]
    for env, rho, pol in cells:
        run_cell(env, rho, pol)
    print('PAPER-FULL DONE', flush=True)
