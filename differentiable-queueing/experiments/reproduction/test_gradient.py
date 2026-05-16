"""
Experiment 2: Section 5.1 Gradient Cosine Similarity

Verifies PATHWISE (B=1) vs REINFORCE (B=1000) cosine similarity against
ground truth on criss-cross_bh network. Ground truth = 100K-batch REINFORCE
(accumulated as 10 batches of 10K to avoid OOM).
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import numpy as np
import json
import time
import sys
import copy
import yaml
import pathos.multiprocessing as mp
import torch.nn.functional as F
import torch.distributions.one_hot_categorical as one_hot_sample

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from queuetorch.env import load_env
from queuetorch.policies import SoftPriorityPolicy, SoftMaxWeightPolicy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
NUM_CORES = int(os.environ.get('NSLOTS', min(4, os.cpu_count() or 1)))
NUM_SAMPLES = 50
T = 1000
GT_BATCH = 10000
GT_REPEATS = 10  # 10 × 10000 = 100K total ground truth samples


def make_net(policy_type):
    if policy_type == 'sPR':
        return SoftPriorityPolicy(2, 3)
    else:
        return SoftMaxWeightPolicy(2, 3)


def pathwise_worker(args):
    torch.set_num_threads(1)
    policy_type, env_config, state_dict, gt_grad, seed = args
    torch.manual_seed(seed)
    net = make_net(policy_type)
    net.load_state_dict(copy.deepcopy(state_dict))
    net.zero_grad()
    dq = load_env(env_config, temp=1.0, batch=1, seed=None, device='cpu')
    obs, state = dq.reset()
    total_cost = torch.zeros(1)
    for _ in range(T):
        queues, _ = obs
        probs = net(queues) * dq.network
        probs = torch.minimum(probs, queues.unsqueeze(1).repeat(1, dq.s, 1))
        mask = torch.all(probs == 0., dim=2).reshape(1, dq.s, 1)
        probs = probs + mask.repeat(1, 1, dq.q) * dq.network
        sm = torch.sum(probs, dim=-1, keepdim=True)
        sm = torch.where(sm == 0, torch.ones_like(sm), sm)
        probs = probs / sm
        obs, state, cost, _ = dq.step(state, probs)
        total_cost = total_cost + cost.mean()
    loss = total_cost / T
    loss.backward()
    grad = torch.cat([p.grad.view(-1).detach() for p in net.parameters()
                     if p.grad is not None])
    return F.cosine_similarity(grad.unsqueeze(0), gt_grad.unsqueeze(0)).item()


def reinforce_worker(args):
    torch.set_num_threads(1)
    policy_type, env_config, state_dict, gt_grad, batch, gamma, seed = args
    torch.manual_seed(seed)
    net = make_net(policy_type)
    net.load_state_dict(copy.deepcopy(state_dict))
    net.zero_grad()
    dq = load_env(env_config, temp=1.0, batch=batch, seed=None, device='cpu')
    obs, state = dq.reset()
    log_probs, rewards = [], []
    for _ in range(T):
        queues, _ = obs
        probs = net(queues) * dq.network
        probs = torch.minimum(probs, queues.unsqueeze(1).repeat(1, dq.s, 1))
        mask = torch.all(probs == 0., dim=2).reshape(batch, dq.s, 1)
        probs = probs + mask.repeat(1, 1, dq.q) * dq.network
        sm = torch.sum(probs, dim=-1, keepdim=True)
        sm = torch.where(sm == 0, torch.ones_like(sm), sm)
        probs = probs / sm
        dist = one_hot_sample.OneHotCategorical(probs=probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action).sum(dim=1))
        obs, state, cost, _ = dq.step(state, action)
        rewards.append(-cost.squeeze(1))
    loss = torch.zeros(1)
    returns = torch.zeros(batch)
    for t in reversed(range(T)):
        returns = rewards[t] + gamma * returns
        loss = loss - (log_probs[t] * returns).mean()
    loss.backward()
    grad = torch.cat([p.grad.view(-1).detach() for p in net.parameters()
                     if p.grad is not None])
    return F.cosine_similarity(grad.unsqueeze(0), gt_grad.unsqueeze(0)).item()


def compute_gt(policy_type, env_config, state_dict, seed_base):
    """Ground truth: 10 × 10K batch REINFORCE, averaged."""
    grads = []
    for i in range(GT_REPEATS):
        torch.manual_seed(seed_base + i)
        net = make_net(policy_type)
        net.load_state_dict(copy.deepcopy(state_dict))
        net.zero_grad()
        dq = load_env(env_config, temp=1.0, batch=GT_BATCH, seed=None, device='cpu')
        obs, state = dq.reset()
        log_probs, rewards = [], []
        for _ in range(T):
            queues, _ = obs
            probs = net(queues) * dq.network
            probs = torch.minimum(probs, queues.unsqueeze(1).repeat(1, dq.s, 1))
            mask = torch.all(probs == 0., dim=2).reshape(GT_BATCH, dq.s, 1)
            probs = probs + mask.repeat(1, 1, dq.q) * dq.network
            sm = torch.sum(probs, dim=-1, keepdim=True)
            sm = torch.where(sm == 0, torch.ones_like(sm), sm)
            probs = probs / sm
            dist = one_hot_sample.OneHotCategorical(probs=probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action).sum(dim=1))
            obs, state, cost, _ = dq.step(state, action)
            rewards.append(-cost.squeeze(1))
        loss = torch.zeros(1)
        returns = torch.zeros(GT_BATCH)
        for t in reversed(range(T)):
            returns = rewards[t] + 0.999 * returns
            loss = loss - (log_probs[t] * returns).mean()
        loss.backward()
        grads.append(torch.cat([p.grad.view(-1).detach()
                                for p in net.parameters() if p.grad is not None]))
    return torch.stack(grads).mean(dim=0)


def run():
    print("=" * 70)
    print("Experiment 2: Section 5.1 Gradient Cosine Similarity (criss-cross)")
    print("=" * 70)
    print(f"  NUM_CORES: {NUM_CORES}")
    print(f"  NUM_SAMPLES: {NUM_SAMPLES}")
    print(f"  T: {T}")
    print(f"  Ground truth: {GT_REPEATS} batches of {GT_BATCH} = {GT_REPEATS*GT_BATCH//1000}K total")

    with open(os.path.join(PROJECT_ROOT, 'configs/env/criss_cross_bh.yaml')) as f:
        env_config = yaml.safe_load(f)

    results = {}
    for policy_type in ['sPR', 'sMW']:
        print(f"\n  ===== Policy: {policy_type} =====")
        torch.manual_seed(42)
        net = make_net(policy_type)
        state_dict = copy.deepcopy(net.state_dict())

        t0 = time.time()
        print(f"    Computing ground truth...", end='', flush=True)
        gt_grad = compute_gt(policy_type, env_config, state_dict, seed_base=12345)
        gt_norm = float(torch.norm(gt_grad))
        print(f" norm={gt_norm:.4f} ({time.time()-t0:.1f}s)")

        t0 = time.time()
        pw_args = [(policy_type, env_config, state_dict, gt_grad, 1000+i)
                   for i in range(NUM_SAMPLES)]
        with mp.ProcessingPool(NUM_CORES) as pool:
            pw_sims = pool.map(pathwise_worker, pw_args)
        print(f"    PATHWISE (B=1, n={NUM_SAMPLES}): "
              f"cossim = {np.mean(pw_sims):.4f} ± {np.std(pw_sims):.4f} "
              f"({time.time()-t0:.1f}s)")

        t0 = time.time()
        rf_args = [(policy_type, env_config, state_dict, gt_grad, 1000, 0.999, 2000+i)
                   for i in range(NUM_SAMPLES)]
        with mp.ProcessingPool(NUM_CORES) as pool:
            rf_sims = pool.map(reinforce_worker, rf_args)
        print(f"    REINFORCE (B=1000, n={NUM_SAMPLES}): "
              f"cossim = {np.mean(rf_sims):.4f} ± {np.std(rf_sims):.4f} "
              f"({time.time()-t0:.1f}s)")

        results[policy_type] = {
            'gt_norm': gt_norm,
            'pathwise_mean': float(np.mean(pw_sims)),
            'pathwise_std': float(np.std(pw_sims)),
            'reinforce_mean': float(np.mean(rf_sims)),
            'reinforce_std': float(np.std(rf_sims)),
            'pathwise_samples': pw_sims,
            'reinforce_samples': rf_sims,
        }

    # Verification vs previous
    prev = {
        'sPR': {'pathwise_mean': 0.8000, 'reinforce_mean': -0.0000},
        'sMW': {'pathwise_mean': 0.1513, 'reinforce_mean': 0.0500},
    }

    print("\n" + "=" * 70)
    print("Verification vs previous result")
    print("=" * 70)
    print(f"{'Policy':>6} {'Method':>10} | {'Previous':>10} | {'Now':>15} | {'Diff':>8}")
    for pt in ['sPR', 'sMW']:
        for method, key in [('PATHWISE', 'pathwise_mean'), ('REINFORCE', 'reinforce_mean')]:
            p = prev[pt][key]
            n = results[pt][key]
            print(f"{pt:>6} {method:>10} | {p:>10.4f} | {n:>10.4f}     | {n-p:>+7.4f}")

    out_path = os.path.join(RESULTS_DIR, 'rerun_gradient.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return results


if __name__ == '__main__':
    run()
