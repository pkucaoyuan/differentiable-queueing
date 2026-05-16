"""
E1: STE Bias-Variance Analysis Beyond M/M/1

Extends the bias-variance characterization of the straight-through estimator (STE)
from the M/M/1 single-step setting (Theorem 1) to:
- Multiple environment types (M/M/1, criss-cross, multiclass, reentrant)
- Multi-step horizons (T = 1 to 1000)
- Multiple traffic intensities (rho = 0.9 to 0.99)
- Multiple STE temperatures (beta = 0.1 to 10.0)

Usage:
    cd experiments/
    python ste_bias_variance.py
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
from typing import Dict, Any
import pathos.multiprocessing as mp
import torch.nn.functional as F
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from queuetorch.env import load_env
from queuetorch.policies import SoftPriorityPolicy, SoftMaxWeightPolicy
from cmu_step_rules_PATHWISE import build_env_config

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Parameters ──────────────────────────────────────────────────────────────
HORIZONS = [1, 10, 100, 500, 1000]
RHO_VALUES = [0.9, 0.95, 0.99]
BETA_VALUES = [0.1, 0.5, 1.0, 2.0, 10.0]
POLICIES = ['sPR', 'sMW']
NUM_THETA = 20
NUM_SAMPLES = 200       # gradient samples per (theta, T, beta)
GT_BATCH = 100_000      # batch for ground truth REINFORCE
REINFORCE_BATCH = 1000
GAMMA = 0.999
NUM_CORES = int(os.environ.get("NSLOTS", min(4, os.cpu_count() or 1)))

# Environments
ENVIRONMENTS = {
    'mm1': {
        'config_path': 'configs/env/mm1.yaml',
        'rho_method': 'scale_lam',
    },
    'criss_cross': {
        'config_path': 'configs/env/criss_cross_IID.yaml',
        'rho_method': 'fixed',  # uses default rho
    },
    'multiclass_5': {
        'builder': 'build_env_config',
        'queue_class': 5,
        'gap': 0.5,
    },
    'reentrant_2': {
        'config_path': 'configs/env/reentrant_2.yaml',
        'rho_method': 'fixed',
    },
}


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_policy(policy_type, s, q):
    if policy_type == 'sPR':
        return SoftPriorityPolicy(s, q)
    elif policy_type == 'sMW':
        return SoftMaxWeightPolicy(s, q)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def get_env_config(env_name, rho):
    """Build env config for a given environment and traffic intensity."""
    env_spec = ENVIRONMENTS[env_name]

    if 'builder' in env_spec:
        return build_env_config(env_spec['queue_class'], rho, env_spec['gap'])

    config_path = os.path.join(PROJECT_ROOT, env_spec['config_path'])
    env_config = load_yaml(config_path)

    if env_spec.get('rho_method') == 'scale_lam' and env_config['lam_type'] == 'constant':
        # Scale arrival rates to achieve desired rho
        # For M/M/1: rho = lambda/mu, so lambda = rho * mu
        if env_config['lam_params'].get('val') is not None:
            base_lam = np.array(env_config['lam_params']['val'])
            if env_config.get('mu') is not None:
                mu = np.array(env_config['mu'])
                if mu.ndim == 2:
                    mu_diag = mu[0]
                else:
                    mu_diag = mu
                # Scale lambda to achieve target rho
                current_rho = np.sum(base_lam / mu_diag)
                if current_rho > 0:
                    scale = rho / current_rho
                    env_config['lam_params']['val'] = (base_lam * scale).tolist()

    return env_config


def get_env_dims(env_config):
    """Get (s, q) dimensions from env config."""
    if env_config.get('network') is None:
        env_type = env_config.get('env_type', env_config.get('name'))
        net = np.load(os.path.join(PROJECT_ROOT, f'env_data/{env_type}/{env_type}_network.npy'))
        return int(net.shape[0]), int(net.shape[1])
    else:
        net = np.array(env_config['network'])
        return int(net.shape[0]), int(net.shape[1])


def compute_pathwise_grad(net, env_config, T, device='cpu', temp=1.0):
    """Compute pathwise gradient with specified temperature."""
    net.zero_grad()
    dq = load_env(env_config, temp=temp, batch=1, seed=None, device=device)
    obs, state = dq.reset()
    total_cost = torch.zeros(1, device=device)

    for _ in range(T):
        queues, t = obs
        probs = net(queues)
        probs = probs * dq.network
        probs = torch.min(
            torch.stack((probs, queues.unsqueeze(1).repeat(1, dq.s, 1)), dim=3),
            dim=3
        ).values
        mask = torch.all(probs == 0., dim=2).reshape(1, dq.s, 1)
        probs = probs + mask.repeat(1, 1, dq.q) * dq.network
        sum_probs = torch.sum(probs, dim=-1, keepdim=True)
        sum_probs = torch.where(sum_probs == 0, torch.ones_like(sum_probs), sum_probs)
        probs = probs / sum_probs
        action = probs
        obs, state, cost, event_time = dq.step(state, action)
        total_cost = total_cost + cost.mean()

    loss = total_cost / T
    loss.backward()

    grads = []
    for param in net.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).detach().cpu())
        else:
            grads.append(torch.zeros_like(param.view(-1)).cpu())
    return torch.cat(grads)


def compute_reinforce_grad(net, env_config, T, batch_size=1000, gamma=0.999, device='cpu'):
    """Compute REINFORCE gradient."""
    import torch.distributions.one_hot_categorical as one_hot_sample
    net.zero_grad()
    dq = load_env(env_config, temp=1.0, batch=batch_size, seed=None, device=device)
    obs, state = dq.reset()

    log_probs = []
    rewards = []

    for _ in range(T):
        queues, t = obs
        probs = net(queues)
        probs = probs * dq.network
        probs = torch.minimum(probs, queues.unsqueeze(1).repeat(1, dq.s, 1))
        mask = torch.all(probs == 0., dim=2).reshape(batch_size, dq.s, 1)
        probs = probs + mask.repeat(1, 1, dq.q) * dq.network
        sum_probs = torch.sum(probs, dim=-1, keepdim=True)
        sum_probs = torch.where(sum_probs == 0, torch.ones_like(sum_probs), sum_probs)
        probs = probs / sum_probs

        dist = one_hot_sample.OneHotCategorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        log_probs.append(log_prob)

        obs, state, cost, event_time = dq.step(state, action)
        rewards.append(-cost.squeeze(1))

    policy_loss = torch.zeros(1, device=device)
    returns = torch.zeros(batch_size, device=device)
    for t_ in reversed(range(T)):
        returns = rewards[t_] + gamma * returns
        policy_loss = policy_loss - (log_probs[t_] * returns).mean()

    policy_loss.backward()

    grads = []
    for param in net.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).detach().cpu())
        else:
            grads.append(torch.zeros_like(param.view(-1)).cpu())
    return torch.cat(grads)


def cosine_sim(g1, g2):
    n1, n2 = torch.norm(g1), torch.norm(g2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()


def worker(job):
    """Process one (env, policy, rho, theta_idx) job.
    Computes GT once, then sweeps all (T, beta) combos."""
    torch.set_num_threads(1)
    env_name, policy_type, rho, theta_idx, seed = job

    env_config = get_env_config(env_name, rho)
    s, q = get_env_dims(env_config)

    # Create policy with random weights
    torch.manual_seed(seed)
    net = get_policy(policy_type, s, q)
    state_dict = copy.deepcopy(net.state_dict())

    # Compute ground truth gradient (large-batch REINFORCE at T=1000)
    net.load_state_dict(copy.deepcopy(state_dict))
    gt_grad = compute_reinforce_grad(net, env_config, T=1000,
                                      batch_size=GT_BATCH, gamma=GAMMA, device='cpu')

    gt_norm = float(torch.norm(gt_grad))
    results = []

    for T in HORIZONS:
        for beta in BETA_VALUES:
            # Collect PATHWISE gradient samples
            pw_grads = []
            for i in range(NUM_SAMPLES):
                net_pw = get_policy(policy_type, s, q)
                net_pw.load_state_dict(copy.deepcopy(state_dict))
                g = compute_pathwise_grad(net_pw, env_config, T=T, temp=beta)
                pw_grads.append(g)

            # Collect REINFORCE gradient samples
            rf_grads = []
            for i in range(min(NUM_SAMPLES, 50)):  # fewer RF samples (each is batch=1000)
                net_rf = get_policy(policy_type, s, q)
                net_rf.load_state_dict(copy.deepcopy(state_dict))
                g = compute_reinforce_grad(net_rf, env_config, T=T,
                                           batch_size=REINFORCE_BATCH, gamma=GAMMA)
                rf_grads.append(g)

            # Compute metrics
            pw_stack = torch.stack(pw_grads)
            pw_mean = pw_stack.mean(dim=0)
            pw_bias = float(torch.norm(pw_mean - gt_grad))
            pw_var = float(torch.mean(torch.sum((pw_stack - pw_mean) ** 2, dim=1)))
            pw_mse = pw_bias ** 2 + pw_var
            pw_cossim = float(np.mean([cosine_sim(g, gt_grad) for g in pw_grads]))

            rf_stack = torch.stack(rf_grads)
            rf_mean = rf_stack.mean(dim=0)
            rf_bias = float(torch.norm(rf_mean - gt_grad))
            rf_var = float(torch.mean(torch.sum((rf_stack - rf_mean) ** 2, dim=1)))
            rf_mse = rf_bias ** 2 + rf_var
            rf_cossim = float(np.mean([cosine_sim(g, gt_grad) for g in rf_grads]))

            results.append({
                'env': env_name, 'policy': policy_type, 'rho': rho,
                'theta_idx': theta_idx, 'T': T, 'beta': beta,
                'pw_bias': pw_bias, 'pw_variance': pw_var,
                'pw_mse': pw_mse, 'pw_cossim': pw_cossim,
                'rf_bias': rf_bias, 'rf_variance': rf_var,
                'rf_mse': rf_mse, 'rf_cossim': rf_cossim,
                'gt_norm': gt_norm,
            })

    return results


if __name__ == '__main__':
    start_time = time.time()

    # Build job list
    jobs = []
    base_seed = 42
    for env_name in ENVIRONMENTS:
        for policy in POLICIES:
            for rho in RHO_VALUES:
                for theta_idx in range(NUM_THETA):
                    seed = base_seed + theta_idx * 1000 + hash(env_name) % 10000
                    jobs.append((env_name, policy, rho, theta_idx, seed))

    total_jobs = len(jobs)
    print(f"{'=' * 60}")
    print(f"E1: STE Bias-Variance Analysis")
    print(f"  {len(ENVIRONMENTS)} environments x {len(POLICIES)} policies "
          f"x {len(RHO_VALUES)} rhos x {NUM_THETA} thetas = {total_jobs} jobs")
    print(f"  Each job sweeps {len(HORIZONS)} horizons x {len(BETA_VALUES)} betas")
    print(f"  Using {NUM_CORES} cores")
    print(f"{'=' * 60}\n")

    all_results = []
    completed = 0

    # Process in chunks to save incrementally
    chunk_size = NUM_CORES * 2
    for start in range(0, total_jobs, chunk_size):
        chunk = jobs[start:start + chunk_size]
        chunk_start = time.time()

        with mp.ProcessingPool(NUM_CORES) as pool:
            chunk_results = pool.map(worker, chunk)

        for result_list in chunk_results:
            all_results.extend(result_list)

        completed += len(chunk)
        elapsed = time.time() - start_time
        eta = (elapsed / completed) * (total_jobs - completed) if completed > 0 else 0

        print(f"[{completed}/{total_jobs}] "
              f"elapsed={elapsed/60:.1f}min, ETA={eta/60:.1f}min, "
              f"results={len(all_results)}")

        # Save incremental
        with open(os.path.join(RESULTS_DIR, 'E1_ste_bias_variance.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time / 3600:.1f} hours")
    print(f"Total results: {len(all_results)}")
    print("Done!")
