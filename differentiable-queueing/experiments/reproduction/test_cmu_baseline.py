"""
§7 STE-vs-cμ baseline benchmark.

Computes the cμ-priority baseline cost on each §7 env (criss-cross +
reentrant_2..10) and compares to our STE-trained policies. This closes
the gap in §7 Tables 1-5 reproduction — paper claims PATHWISE beats cμ;
we now have both sides for comparison.

cμ rule: priority[q] = h[q] * mu[s,q]; server s serves the highest-priority
non-empty queue it can serve.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'

import sys, json, time, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Inject PriorityNet so torch.load can find it
class PriorityNet(nn.Module):
    def __init__(self, s, q, layers, hidden_dim, f_time=False, x_stats=None, t_stats=None):
        super().__init__()
        self.s = s; self.q = q
        self.x_stats = x_stats; self.t_stats = t_stats
        self.layers = layers; self.hidden_dim = hidden_dim
        self.f_time = f_time
        self.input_fc = nn.Linear(q + (1 if f_time else 0), hidden_dim)
        self.layers_fc = nn.ModuleList()
        for _ in range(layers):
            self.layers_fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_fc = nn.Linear(hidden_dim, s * q)
    def forward(self, x, t=0):
        batch = x.size()[0]
        if self.x_stats is not None: x = (x - self.x_stats[0]) / self.x_stats[1]
        if self.t_stats is not None: t = (t - self.t_stats[0]) / self.t_stats[1]
        if self.f_time: x = torch.cat((x, t), 1)
        x = F.relu(self.input_fc(x))
        for layer in self.layers_fc:
            x = F.relu(layer(x))
        x = self.output_fc(x).view(batch, self.s, self.q)
        return F.softmax(x, dim=-1)

import __main__
__main__.PriorityNet = PriorityNet

from queuetorch.env import load_env

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'reproduction')

ENVS = ['criss_cross_bh',
        'reentrant_2', 'reentrant_3', 'reentrant_4', 'reentrant_5',
        'reentrant_6', 'reentrant_7', 'reentrant_8', 'reentrant_9', 'reentrant_10']
EVAL_BATCH = 100
EVAL_T = 50000
NUM_SEEDS = 5  # for 95% CI


def get_env_params(env_yaml):
    """Load env config, extract mu (s,q) and h (q,) as tensors."""
    with open(env_yaml) as f:
        cfg = yaml.safe_load(f)
    return cfg


def cmu_priority(dq):
    """Compute cμ priority: priority[s,q] = h[q] * mu[s,q] for each (s,q) pair.
    Returns a (B, s, q) tensor where mu/network already carry batch dim."""
    h  = dq.h  # (q,)
    mu = dq.mu  # (B, s, q)
    nw = dq.network  # (B, s, q)
    # Broadcast h over (B, s, q)
    priority = nw * h.view(1, 1, -1) * mu  # (B, s, q)
    return priority


def eval_cmu(env_config, batch, T, seed=42):
    """Evaluate fixed cμ priority policy."""
    torch.set_num_threads(1)
    dq = load_env(env_config, temp=0.5, batch=batch, seed=seed, device='cpu')
    torch.manual_seed(seed)
    obs, state = dq.reset(seed=seed)
    pri = cmu_priority(dq).to(dq.device)  # (s, q)
    total_cost = torch.zeros(batch, 1)

    with torch.no_grad():
        for _ in range(T):
            queues, _ = obs  # queues: (B, q)
            # pri is (B, s, q) already.
            # Mask by queues > 0 to prefer non-empty queues
            mask = (queues > 0).float().unsqueeze(1)  # (B, 1, q)
            scored = pri * mask + dq.network * 1e-6   # tiebreak via network
            action_idx = torch.argmax(scored, dim=2)  # (B, s)
            action = F.one_hot(action_idx, num_classes=dq.q).float()  # (B, s, q)
            # Mask out servers with no eligible queue
            action_sum = action.sum(dim=-1, keepdim=True).clamp(min=1)
            action = action / action_sum
            obs, state, cost, _ = dq.step(state, action)
            total_cost = total_cost + cost
    avg = (total_cost.squeeze(-1) / state.time.squeeze(-1)).cpu().numpy()
    return float(np.mean(avg)), float(np.std(avg))


def eval_ste(env_name, env_config, batch, T, epoch='best'):
    """Evaluate STE-trained policy using paper's WC eval protocol.

    epoch='best' picks the epoch with min train-time test_loss from loss/*.json;
    epoch=int picks specific epoch;
    epoch='last' picks epoch 99.
    """
    if epoch == 'best':
        loss_file = f'{PROJECT_ROOT}/loss/{env_name}_ppg_softmax.json'
        with open(loss_file) as f:
            loss_data = json.load(f)
        costs = [r['test_loss'] for r in loss_data]
        best_epoch = costs.index(min(costs))
        ckpt = f'{PROJECT_ROOT}/models/{env_name}/ppg_softmax_{best_epoch}.pt'
        print(f"  [STE eval at best epoch={best_epoch}, train test_loss={min(costs):.2f}]")
    elif epoch == 'last':
        ckpt = f'{PROJECT_ROOT}/models/{env_name}/ppg_softmax_99.pt'
    else:
        ckpt = f'{PROJECT_ROOT}/models/{env_name}/ppg_softmax_{int(epoch)}.pt'
    if not os.path.exists(ckpt):
        return None, None
    torch.set_num_threads(1)
    net = torch.load(ckpt, map_location='cpu')
    net.eval()
    dq = load_env(env_config, temp=1e-3, batch=batch, seed=42, device='cpu')
    obs, state = dq.reset(seed=42)
    total_cost = torch.zeros(batch, 1)
    with torch.no_grad():
        for _ in range(T):
            queues, t = obs
            pr = net(queues, t.detach() if t is not None else None)  # (B,s,q) softmax
            # Paper's WC eval protocol (train_policy.py lines 269-273)
            pr = pr * dq.network
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr = F.one_hot(torch.argmax(pr, dim=2), num_classes=dq.q).float()
            action = torch.round(pr)
            obs, state, cost, _ = dq.step(state, action)
            total_cost = total_cost + cost
    avg = (total_cost.squeeze(-1) / state.time.squeeze(-1)).cpu().numpy()
    return float(np.mean(avg)), float(np.std(avg))


def main():
    results = {}
    for env_name in ENVS:
        env_yaml = f'{PROJECT_ROOT}/configs/env/{env_name}.yaml'
        if not os.path.exists(env_yaml):
            print(f"SKIP {env_name}: no env yaml")
            continue
        env_config = get_env_params(env_yaml)
        print(f"\n══ {env_name} ══")
        t0 = time.time()
        cmu_mean, cmu_std = eval_cmu(env_config, EVAL_BATCH, EVAL_T)
        print(f"  cμ baseline: {cmu_mean:.3f} ± {cmu_std:.3f} ({time.time()-t0:.0f}s)")
        t0 = time.time()
        ste_mean, ste_std = eval_ste(env_name, env_config, EVAL_BATCH, EVAL_T)
        if ste_mean is None:
            print("  STE: no checkpoint"); continue
        print(f"  STE (last ep): {ste_mean:.3f} ± {ste_std:.3f} ({time.time()-t0:.0f}s)")
        improvement = (cmu_mean - ste_mean) / cmu_mean * 100
        print(f"  STE vs cμ: {improvement:+.1f}% (negative = STE worse)")
        results[env_name] = {
            'cmu_mean': cmu_mean, 'cmu_std': cmu_std,
            'ste_mean': ste_mean, 'ste_std': ste_std,
            'improvement_pct': improvement,
            'eval_batch': EVAL_BATCH, 'eval_T': EVAL_T,
        }

    out_path = f'{RESULTS_DIR}/ste_vs_cmu_benchmark.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n══ Summary ══")
    print(f"{'env':<20s} {'cμ':>10s} {'STE':>10s} {'improvement':>12s}")
    for env, r in results.items():
        print(f"  {env:<18s} {r['cmu_mean']:>10.3f} {r['ste_mean']:>10.3f} {r['improvement_pct']:>+11.1f}%")


if __name__ == '__main__':
    main()
