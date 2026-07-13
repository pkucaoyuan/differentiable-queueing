"""Ask-1 (§5.1 Figure 8 rerun): upstream-faithful policies, grouped gradient estimators.

Authority: the authors' released code (namkoong-lab/differentiable-queueing@0c21ed7,
byte-identical to this repo) — experiments/gradient_comparison.py + queuetorch/policies.py.
Per upstream's own experiments/reports/report_subtleties.md, the code that generated the
paper's Figure 8 is lost and the released code differs from the paper's description
(nn.Linear policies instead of the diagonal theta parameterization; no value-function
baseline for REINFORCE). We reproduce the *released* implementation faithfully
(user decision 2026-07-13), with engineering hardening only:

  - float64 for the policy/masking forward-backward: the fp32 renormalization
    (probs/ssum) overflows in backward when the softmax saturates (root cause of the
    historical rho>=0.95 nans); fp64 preserves semantics and removes most nans.
  - grouped batching: G independent (theta, draw) groups share one env batch; each
    group has its own parameter copy so param.grad rows are per-group estimates
    (no cross-group leakage; verified in validate.py).
  - per-theta GT split-half cosine diagnostic to quantify GT reliability.

Upstream semantics kept verbatim: policy classes (sPR: free logits randn(s,q);
sMW/sMP: Linear(q -> s*q) with default kaiming-uniform init — note sMP is
structurally identical to sMW in the released code), feasibility masking, REINFORCE
gamma=0.999 without baseline, PATHWISE action=probs with env temp beta=1, T=1000.
"""
import math
import os
import sys
import zlib

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

from queuetorch.env import load_env  # noqa: E402

GAMMA = 0.999
HORIZON = 1000
BASELINE_RHO = 0.9  # actual traffic intensity of all shipped configs

# per-env total-trajectory caps for one 80GB A800; the policy/log-prob graph is
# float64 (2x the fp32 pilot numbers), hence roughly half the fp32 capacity
CAPS = {
    3:  {'rf': 100_000, 'pw': 14_000},
    6:  {'rf': 50_000,  'pw': 8_000},
    9:  {'rf': 22_000,  'pw': 5_000},
    12: {'rf': 12_000,  'pw': 3_500},
    15: {'rf': 8_000,   'pw': 2_500},
}

NET_LABELS = {
    'criss_cross_bh': 'Criss Cross',
    'reentrant_2': 'Reentrant (6 class)',
    'reentrant_3': 'Reentrant (9 class)',
    'reentrant_4': 'Reentrant (12 class)',
    'reentrant_5': 'Reentrant (15 class)',
    're-reentrant_2': 'Reentrant-2 (6 class)',
    're-reentrant_3': 'Reentrant-2 (9 class)',
    're-reentrant_4': 'Reentrant-2 (12 class)',
    're-reentrant_5': 'Reentrant-2 (15 class)',
}


def stable_seed(*parts):
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def load_scaled_cfg(env_name, rho, scaling):
    """scaling='paper': lam * rho/0.9 (true intensity == label).
    scaling='author': lam * rho, the literal gradient_comparison.py behavior
    (note: upstream applied no scaling at all when val is null, i.e. reentrant
    configs; we extend the author rule to the npy-loaded lam)."""
    cfg = yaml.safe_load(open(os.path.join(REPO, 'configs', 'env', f'{env_name}.yaml')))
    if cfg['lam_params']['val'] is None:
        lam = np.load(os.path.join(REPO, 'env_data', env_name, f'{env_name}_lam.npy'))
    else:
        lam = np.array(cfg['lam_params']['val'], dtype=float)
    factor = rho / BASELINE_RHO if scaling == 'paper' else rho
    cfg['lam_params']['val'] = (lam * factor).tolist()
    return cfg


# ---------------------------------------------------------------------------
# Grouped functional equivalents of queuetorch/policies.py (float64 params).
# Param layout matches module.parameters() flatten order for cossim in the
# same space as upstream: sPR -> [theta(s,q)]; sMW/sMP -> [W(s*q,q), b(s*q)].
# ---------------------------------------------------------------------------

def init_params(kind, G, s, q, seed):
    gen = torch.Generator().manual_seed(seed)
    if kind == 'sPR':
        return [torch.randn(G, s, q, generator=gen, dtype=torch.float64)]
    if kind in ('sMW', 'sMP'):
        bound = 1.0 / math.sqrt(q)  # nn.Linear default (kaiming_uniform a=sqrt(5))
        W = (torch.rand(G, s * q, q, generator=gen, dtype=torch.float64) * 2 - 1) * bound
        b = (torch.rand(G, s * q, generator=gen, dtype=torch.float64) * 2 - 1) * bound
        return [W, b]
    if kind.startswith('paper_'):
        # paper eq.(soft_policies): theta in R^q_+, theta ~ Lognormal(0,1)
        return [torch.randn(G, q, generator=gen, dtype=torch.float64).exp()]
    raise ValueError(kind)


def param_dim(kind, s, q):
    if kind == 'sPR':
        return s * q
    if kind in ('sMW', 'sMP'):
        return s * q * q + s * q
    return q  # paper_* policies


def routing_matrix(dq):
    """R (q x q), column j = queue-length delta when a class-j job completes."""
    qeo = dq.queue_event_options[dq.q:2 * dq.q]
    assert qeo.shape == (dq.q, dq.q)
    return qeo.T.contiguous().double()


def flatten_grads(params):
    return torch.cat([p.grad.reshape(p.shape[0], -1) for p in params], dim=1).cpu()


def probs_from_params(kind, params, queues, G, Bpg, s, q, ctx=None):
    """queues (G*Bpg, q) float64 -> probs (G*Bpg, s, q) float64.
    Upstream kinds (sPR/sMW/sMP) match queuetorch/policies.py forward exactly.
    paper_* kinds implement the paper's eq.(soft_policies); ctx supplies
    mu (B,s,q) and R (q,q) in float64. sMP sign: standard MaxPressure
    pressure -(theta*x) @ R (the printed formula's sign is inverted)."""
    if kind == 'sPR':
        logits = params[0].unsqueeze(1).expand(G, Bpg, s, q).reshape(G * Bpg, s, q)
    elif kind in ('sMW', 'sMP'):
        W, b = params
        x = queues.reshape(G, Bpg, q)
        logits = torch.einsum('gbq,gpq->gbp', x, W) + b.unsqueeze(1)
        logits = logits.reshape(G * Bpg, s, q)
    else:
        theta_b = params[0].unsqueeze(1).expand(G, Bpg, q).reshape(G * Bpg, q)
        if kind == 'paper_sPR':
            v = theta_b
        elif kind == 'paper_sMW':
            v = theta_b * queues
        elif kind == 'paper_sMP':
            v = -torch.matmul(theta_b * queues, ctx['R'])
        else:
            raise ValueError(kind)
        logits = ctx['mu'] * v.unsqueeze(1)
    return F.softmax(logits, dim=2)


def mask_probs(probs, network, queues):
    """Feasibility wrapper, verbatim logic from authors' gradient_comparison.py."""
    B, s, q = probs.shape
    probs = probs * network
    probs = torch.min(torch.stack((probs, queues.unsqueeze(1).expand(-1, s, -1)), dim=3), dim=3).values
    all_zero = torch.all(probs == 0., dim=2, keepdim=True)
    probs = probs + all_zero * network
    ssum = probs.sum(-1, keepdim=True)
    probs = probs / torch.where(ssum == 0, torch.ones_like(ssum), ssum)
    return probs


def _to_device_params(param_list, device):
    return [p.detach().clone().to(device).requires_grad_(True) for p in param_list]


def reinforce_grads(cfg, policy, params_cpu, B_per_group, seed, device,
                    T=HORIZON, gamma=GAMMA, loo_baseline=False):
    """params_cpu: list of (G, ...) tensors. Returns (G, P) flattened REINFORCE
    estimates, each averaged over B_per_group trajectories.

    loo_baseline: subtract the leave-one-out within-group mean of the
    return-to-go (a state-independent, time-dependent baseline; strictly
    unbiased). Used only for the GT: the released code has no baseline and its
    GT is noise-dominated at feasible sample sizes (split-half cos << 1),
    while the paper states its GT/estimators used a value-function baseline."""
    G = params_cpu[0].shape[0]
    B = G * B_per_group
    torch.manual_seed(seed)
    dq = load_env(cfg, temp=1.0, batch=B, seed=seed, device=device)
    s, q = dq.s, dq.q
    net64 = dq.network.double()
    ctx = {'mu': dq.mu.double(), 'R': routing_matrix(dq).to(device)}
    params = _to_device_params(params_cpu, device)
    obs, state = dq.reset()
    log_probs, rewards = [], []
    for _ in range(T):
        queues, _ = obs
        q64 = queues.double()
        probs = probs_from_params(policy, params, q64, G, B_per_group, s, q, ctx)
        probs = mask_probs(probs, net64, q64)
        dist = torch.distributions.OneHotCategorical(probs=probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action).sum(dim=1))
        obs, state, cost, _ = dq.step(state, action.float())
        rewards.append(-cost.squeeze(1).double())
    ret = torch.zeros(B, dtype=torch.float64, device=device)
    loss = torch.zeros(B, dtype=torch.float64, device=device)
    for t in reversed(range(T)):
        ret = rewards[t] + gamma * ret
        adv = ret
        if loo_baseline and B_per_group > 1:
            grp = ret.reshape(G, B_per_group)
            loo = (grp.sum(dim=1, keepdim=True) - grp) / (B_per_group - 1)
            adv = (grp - loo).reshape(B)
        loss = loss - log_probs[t] * adv
    loss.reshape(G, B_per_group).mean(dim=1).sum().backward()
    g = flatten_grads(params)
    del dq, params, log_probs, rewards, loss, ret
    return g


def pathwise_grads(cfg, policy, params_cpu, B_per_group, seed, device, T=HORIZON):
    """Returns (G, P) flattened PATHWISE gradients of mean_b J_T / T."""
    G = params_cpu[0].shape[0]
    B = G * B_per_group
    torch.manual_seed(seed)
    dq = load_env(cfg, temp=1.0, batch=B, seed=seed, device=device)
    s, q = dq.s, dq.q
    net64 = dq.network.double()
    ctx = {'mu': dq.mu.double(), 'R': routing_matrix(dq).to(device)}
    params = _to_device_params(params_cpu, device)
    obs, state = dq.reset()
    total = torch.zeros(B, device=device)
    for _ in range(T):
        queues, _ = obs
        q64 = queues.double()
        probs = probs_from_params(policy, params, q64, G, B_per_group, s, q, ctx)
        probs = mask_probs(probs, net64, q64)
        obs, state, cost, _ = dq.step(state, probs.float())
        total = total + cost.squeeze(1)
    (total / T).double().reshape(G, B_per_group).mean(dim=1).sum().backward()
    g = flatten_grads(params)
    del dq, params, total
    return g


def slice_params(params, idx):
    return [p[idx] for p in params]


def repeat_params(params, n):
    return [p.repeat_interleave(n, dim=0) for p in params]


def env_dims(cfg):
    dq = load_env(cfg, temp=1.0, batch=1, seed=0, device='cpu')
    return dq.s, dq.q


def cosine_rows(est, gt):
    """est (D, P) draws vs gt (P,) -> (D,) cosines; nan/zero rows -> nan."""
    est, gt = est.double(), gt.double()
    gtn = gt / gt.norm().clamp_min(1e-12)
    norms = est.norm(dim=1)
    cos = (est @ gtn) / norms.clamp_min(1e-12)
    bad = (norms < 1e-12) | torch.isnan(est).any(dim=1)
    return torch.where(bad, torch.full_like(cos, float('nan')), cos)
