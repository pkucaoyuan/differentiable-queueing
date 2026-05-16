"""
Comprehensive reproduction of paper's main experiments.
Validates against existing reference data in cmu/ directory.

Sections covered:
- Section 5.1: Gradient cosine similarity (sPR, sMW)
- Section 5.2: CMU rule optimization (10-class, multiple alphas)
- Section 8: M/M/1 simulator validation

Note: All on CPU. Section 7 reentrant networks are run separately due to compute cost.
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from queuetorch.env import load_env
from queuetorch.policies import SoftPriorityPolicy, SoftMaxWeightPolicy
from cmu_step_rules_PATHWISE import (
    build_env_config, pathwise_cmu_step_rule, evaluate_iterate_fast
)
from cmu_rule_REINFORCE import reinforce_value_cmu

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CORES = int(os.environ.get('NSLOTS', min(4, os.cpu_count() or 1)))


# ============================================================================
# Section 8 / Sanity: M/M/1 Simulator Validation
# ============================================================================
def test_mm1_simulator():
    print("\n" + "=" * 70)
    print("M/M/1 Simulator Validation (sanity check)")
    print("=" * 70)

    with open(os.path.join(PROJECT_ROOT, 'configs/env/mm1.yaml')) as f:
        env_config = yaml.safe_load(f)

    rho = 0.9
    analytical = rho / (1 - rho)
    print(f"  Analytical E[Q] for M/M/1, rho={rho}: {analytical:.4f}")

    dq = load_env(env_config, temp=0.1, batch=50, seed=42, device='cpu')
    obs, state = dq.reset(seed=42)
    total_cost = torch.zeros(50)

    with torch.no_grad():
        for _ in range(200_000):
            queues, t = obs
            action = torch.ones(50, dq.s, dq.q) * dq.network
            action /= action.sum(dim=-1, keepdim=True)
            obs, state, cost, _ = dq.step(state, action)
            total_cost += cost.squeeze(1)

    sim = float(torch.mean(total_cost / state.time))
    err = abs(sim - analytical) / analytical * 100
    print(f"  Simulated E[Q]: {sim:.4f}  (error {err:.2f}%)")

    return {'analytical': analytical, 'simulated': sim, 'error_pct': err,
            'pass': err < 1.0}


# ============================================================================
# Section 5.1: Gradient Cosine Similarity
# ============================================================================
def gradient_comparison_worker(args):
    """Worker for one gradient sample."""
    torch.set_num_threads(1)
    method, policy_type, env_config, T, gt_grad, batch, gamma, seed = args
    torch.manual_seed(seed)

    if policy_type == 'sPR':
        net = SoftPriorityPolicy(2, 3)
    else:
        net = SoftMaxWeightPolicy(2, 3)

    if method == 'pathwise':
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
    else:  # reinforce
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

    cossim = F.cosine_similarity(grad.unsqueeze(0), gt_grad.unsqueeze(0)).item()
    return cossim


def test_gradient_comparison():
    print("\n" + "=" * 70)
    print("Section 5.1: Gradient Cosine Similarity (criss-cross)")
    print("=" * 70)

    with open(os.path.join(PROJECT_ROOT, 'configs/env/criss_cross_bh.yaml')) as f:
        env_config = yaml.safe_load(f)

    T = 1000
    NUM_SAMPLES = 50
    GT_BATCHES = [10000] * 10  # 10 batches of 10K each = 100K total for ground truth

    results = {}
    for policy_type in ['sPR', 'sMW']:
        print(f"\n  Policy: {policy_type}")

        torch.manual_seed(42)
        if policy_type == 'sPR':
            net = SoftPriorityPolicy(2, 3)
        else:
            net = SoftMaxWeightPolicy(2, 3)
        state_dict = copy.deepcopy(net.state_dict())

        # Ground truth: accumulate REINFORCE gradients over multiple batches
        print("    Computing ground truth (B=100K total)...", end='', flush=True)
        gt_args = [('reinforce', policy_type, env_config, T,
                    torch.zeros(1), gb, 0.999, 12345 + i)
                   for i, gb in enumerate(GT_BATCHES)]
        # Run sequentially to avoid OOM
        gt_grads = []
        for arg in gt_args:
            torch.set_num_threads(1)
            method_, pt, ec, T_, _, batch, gamma, seed = arg
            torch.manual_seed(seed)
            if pt == 'sPR':
                n = SoftPriorityPolicy(2, 3)
            else:
                n = SoftMaxWeightPolicy(2, 3)
            n.load_state_dict(copy.deepcopy(state_dict))
            n.zero_grad()
            dq = load_env(ec, temp=1.0, batch=batch, seed=None, device='cpu')
            obs, state = dq.reset()
            log_probs, rewards = [], []
            for _ in range(T_):
                queues, _ = obs
                probs = n(queues) * dq.network
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
            for t in reversed(range(T_)):
                returns = rewards[t] + gamma * returns
                loss = loss - (log_probs[t] * returns).mean()
            loss.backward()
            gt_grads.append(torch.cat([p.grad.view(-1).detach()
                                       for p in n.parameters() if p.grad is not None]))

        gt_grad = torch.stack(gt_grads).mean(dim=0)
        gt_norm = float(torch.norm(gt_grad))
        print(f" norm={gt_norm:.2f}")

        # Pathwise B=1 samples (parallel)
        pw_args = [('pathwise', policy_type, env_config, T, gt_grad, 1, 0.999, 1000+i)
                   for i in range(NUM_SAMPLES)]
        with mp.ProcessingPool(NUM_CORES) as pool:
            pw_sims = pool.map(gradient_comparison_worker, pw_args)

        # REINFORCE B=1000 samples (parallel)
        rf_args = [('reinforce', policy_type, env_config, T, gt_grad, 1000, 0.999, 2000+i)
                   for i in range(NUM_SAMPLES)]
        with mp.ProcessingPool(NUM_CORES) as pool:
            rf_sims = pool.map(gradient_comparison_worker, rf_args)

        pw_mean, pw_std = np.mean(pw_sims), np.std(pw_sims)
        rf_mean, rf_std = np.mean(rf_sims), np.std(rf_sims)
        print(f"    PATHWISE  (B=1):    cossim = {pw_mean:.4f} ± {pw_std:.4f}")
        print(f"    REINFORCE (B=1000): cossim = {rf_mean:.4f} ± {rf_std:.4f}")
        print(f"    Ratio (PW/RF): {pw_mean/abs(rf_mean) if rf_mean != 0 else float('inf'):.2f}")

        results[policy_type] = {
            'pathwise_cossim_mean': pw_mean, 'pathwise_cossim_std': pw_std,
            'reinforce_cossim_mean': rf_mean, 'reinforce_cossim_std': rf_std,
            'gt_norm': gt_norm,
            'pathwise_samples': pw_sims, 'reinforce_samples': rf_sims,
        }

    return results


# ============================================================================
# Section 5.2: CMU Rule (full reproduction)
# ============================================================================
def _run_pw(args):
    torch.set_num_threads(1)
    return pathwise_cmu_step_rule(**args)


def _run_rf(args):
    torch.set_num_threads(1)
    return reinforce_value_cmu(**args)


def test_cmu_rule():
    print("\n" + "=" * 70)
    print("Section 5.2: CMU Rule Optimization (10-class, rho=0.95)")
    print("=" * 70)

    QUEUE_CLASS = 10
    RHO = 0.95
    GAPS = [1.0, 0.5, 0.05, 0.01]
    PW_ALPHAS = [0.01, 0.1, 0.5, 1.0]
    RF_ALPHAS = [0.01, 0.1, 0.5, 1.0]
    NUM_TRIALS = 50  # match paper's standard
    NUM_ITER = 50
    T = 1000
    EVAL_T = 20_000
    TEMP = 1e-6

    seeds_file = os.path.join(PROJECT_ROOT, 'cmu', 'seeds_cmu_10class.json')
    if os.path.exists(seeds_file):
        with open(seeds_file) as f:
            seeds = json.load(f)
    else:
        seeds = [int.from_bytes(os.urandom(4), 'big') for _ in range(10000)]

    pw_results = {}
    rf_results = {}

    # PATHWISE — comprehensive sweep
    print(f"\n--- PATHWISE: {NUM_TRIALS} trials × {len(PW_ALPHAS)} alphas × {len(GAPS)} gaps ---")
    for alpha in PW_ALPHAS:
        pw_results[str(alpha)] = {}
        for gap in GAPS:
            env_config = build_env_config(QUEUE_CLASS, RHO, gap)
            jobs = [{'env_config': copy.deepcopy(env_config),
                     'seed': seeds[i],
                     'num_iter': NUM_ITER,
                     'step_rule_name': 'normalized_fixed',
                     'alpha': alpha, 'temp': TEMP, 'T': T, 'eval_T': EVAL_T}
                    for i in range(NUM_TRIALS)]

            t0 = time.time()
            with mp.ProcessingPool(NUM_CORES) as pool:
                trials = pool.map(_run_pw, jobs)
            costs = [r['avg_cost'] for r in trials]
            pw_results[str(alpha)][str(gap)] = trials
            print(f"  alpha={alpha:>5} gap={gap:>5}: {np.mean(costs):.3f}±{np.std(costs):.3f} ({time.time()-t0:.0f}s)")

            # Save incremental
            with open(os.path.join(RESULTS_DIR, 'reproduction_cmu_pathwise.json'), 'w') as f:
                json.dump(pw_results, f, indent=2)

    # REINFORCE
    print(f"\n--- REINFORCE: {NUM_TRIALS} trials × {len(RF_ALPHAS)} alphas × {len(GAPS)} gaps ---")
    for alpha in RF_ALPHAS:
        rf_results[str(alpha)] = {}
        for gap in GAPS:
            env_config = build_env_config(QUEUE_CLASS, RHO, gap)
            jobs = [{'env_config': copy.deepcopy(env_config),
                     'seed': seeds[i],
                     'num_iter': NUM_ITER,
                     'alpha': alpha, 'T': T, 'gamma': 0.99,
                     'batch': 100, 'eval_T': EVAL_T}
                    for i in range(NUM_TRIALS)]

            t0 = time.time()
            with mp.ProcessingPool(NUM_CORES) as pool:
                trials = pool.map(_run_rf, jobs)
            costs = [r['avg_cost'] for r in trials]
            rf_results[str(alpha)][str(gap)] = trials
            print(f"  alpha={alpha:>5} gap={gap:>5}: {np.mean(costs):.3f}±{np.std(costs):.3f} ({time.time()-t0:.0f}s)")

            with open(os.path.join(RESULTS_DIR, 'reproduction_cmu_reinforce.json'), 'w') as f:
                json.dump(rf_results, f, indent=2)

    return pw_results, rf_results


# ============================================================================
# Verification: Compare with reference data
# ============================================================================
def verify(pw_results, rf_results):
    print("\n" + "=" * 70)
    print("Verification vs Reference (cmu/*.json)")
    print("=" * 70)

    ref_pw = os.path.join(PROJECT_ROOT, 'cmu', 'pathwise_wc_cmu_multiclass10.json')
    ref_rf = os.path.join(PROJECT_ROOT, 'cmu', 'wc_reinforce_baseline_cmu_B100_multiclass10.json')

    if os.path.exists(ref_pw):
        with open(ref_pw) as f:
            ref = json.load(f)
        print("\nPATHWISE comparison:")
        print(f"  {'alpha':>6} {'gap':>5} | {'Ours':>20} | {'Reference':>20} | {'Diff%':>7}")
        for alpha_str in pw_results:
            if alpha_str not in ref:
                continue
            for gap_str in pw_results[alpha_str]:
                if gap_str not in ref[alpha_str]:
                    continue
                our_costs = [r['avg_cost'] for r in pw_results[alpha_str][gap_str]]
                ref_costs = [r['avg_cost'] for r in ref[alpha_str][gap_str]]
                om, os_ = np.mean(our_costs), np.std(our_costs)
                rm, rs = np.mean(ref_costs), np.std(ref_costs)
                diff = abs(om - rm) / rm * 100 if rm > 0 else 0
                marker = "OK" if diff < 7 else "WARN"
                print(f"  {alpha_str:>6} {gap_str:>5} | {om:>10.3f}±{os_:.3f} | {rm:>10.3f}±{rs:.3f} | {diff:>6.2f}% [{marker}]")

    if os.path.exists(ref_rf):
        with open(ref_rf) as f:
            ref = json.load(f)
        print("\nREINFORCE comparison:")
        print(f"  {'alpha':>6} {'gap':>5} | {'Ours':>20} | {'Reference':>20} | {'Diff%':>7}")
        for alpha_str in rf_results:
            if alpha_str not in ref:
                continue
            for gap_str in rf_results[alpha_str]:
                if gap_str not in ref[alpha_str]:
                    continue
                our_costs = [r['avg_cost'] for r in rf_results[alpha_str][gap_str]]
                ref_costs = [r['avg_cost'] for r in ref[alpha_str][gap_str]]
                om, os_ = np.mean(our_costs), np.std(our_costs)
                rm, rs = np.mean(ref_costs), np.std(ref_costs)
                diff = abs(om - rm) / rm * 100 if rm > 0 else 0
                marker = "OK" if diff < 7 else "WARN"
                print(f"  {alpha_str:>6} {gap_str:>5} | {om:>10.3f}±{os_:.3f} | {rm:>10.3f}±{rs:.3f} | {diff:>6.2f}% [{marker}]")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    start = time.time()
    print(f"Cores: {NUM_CORES}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. M/M/1 sanity
    mm1_result = test_mm1_simulator()
    with open(os.path.join(RESULTS_DIR, 'reproduction_mm1.json'), 'w') as f:
        json.dump(mm1_result, f, indent=2)

    # 2. Section 5.1 gradient comparison
    grad_results = test_gradient_comparison()
    with open(os.path.join(RESULTS_DIR, 'reproduction_gradient.json'), 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if 'samples' not in kk}
                   for k, v in grad_results.items()}, f, indent=2)

    # 3. Section 5.2 CMU rule
    pw_results, rf_results = test_cmu_rule()

    # 4. Verification
    verify(pw_results, rf_results)

    print(f"\n{'=' * 70}")
    print(f"Total time: {(time.time() - start) / 3600:.1f} hours")
    print(f"{'=' * 70}")
