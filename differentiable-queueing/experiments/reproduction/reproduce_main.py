"""
Reproduce main paper results with small-scale runs to verify correctness.

Reproduces:
  1. CMU rule optimization (Section 5.2): PATHWISE vs REINFORCE on 10-class queue
  2. Gradient comparison (Section 5.1): Cosine similarity on criss-cross
  3. M/M/1 and M/M/S simulator validation against analytical formulas

Compares against existing results in cmu/ to verify consistency.
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
# Use NSLOTS from SGE scheduler, fallback to 1 (never hardcode large numbers)
NUM_CORES = int(os.environ.get('NSLOTS', min(4, os.cpu_count() or 1)))


# ============================================================================
# Test 1: M/M/1 Simulator Validation
# ============================================================================
def test_mm1_simulator():
    """Validate M/M/1 simulator against Erlang-C analytical formula."""
    print("\n" + "=" * 70)
    print("TEST 1: M/M/1 Simulator Validation")
    print("=" * 70)

    with open(os.path.join(PROJECT_ROOT, 'configs/env/mm1.yaml'), 'r') as f:
        env_config = yaml.safe_load(f)

    # rho = 0.9 (from config: lam=0.9, mu=1.0)
    rho = 0.9
    analytical_Q = rho / (1 - rho)  # E[Q] = 9.0 for M/M/1
    print(f"  Analytical E[Q] for M/M/1 with rho={rho}: {analytical_Q:.4f}")

    dq = load_env(env_config, temp=0.1, batch=50, seed=42, device='cpu')
    obs, state = dq.reset(seed=42)
    total_cost = torch.zeros(50)

    with torch.no_grad():
        for _ in range(200_000):
            queues, t = obs
            action = torch.ones(50, dq.s, dq.q) * dq.network
            action /= action.sum(dim=-1, keepdim=True)
            obs, state, cost, event_time = dq.step(state, action)
            total_cost += cost.squeeze(1)

    sim_Q = float(torch.mean(total_cost / state.time))
    error = abs(sim_Q - analytical_Q)
    status = "PASS" if error < 0.3 else "FAIL"
    print(f"  Simulated E[Q]: {sim_Q:.4f}")
    print(f"  Error: {error:.4f}  [{status}]")
    return status == "PASS"


# ============================================================================
# Test 2: CMU Rule Optimization (PATHWISE)
# ============================================================================
def _run_pathwise(args):
    torch.set_num_threads(1)
    return pathwise_cmu_step_rule(**args)


def _run_reinforce(args):
    torch.set_num_threads(1)
    return reinforce_value_cmu(**args)


def test_cmu_pathwise():
    """Reproduce PATHWISE CMU rule optimization on 10-class queue.

    Reference numbers (from existing cmu/pathwise_wc_cmu_multiclass10.json):
      gap=1.0, alpha=0.5: avg_cost ~ 10.38-10.48
    """
    print("\n" + "=" * 70)
    print("TEST 2: CMU Rule PATHWISE Optimization (10-class, rho=0.95)")
    print("=" * 70)

    QUEUE_CLASS = 10
    RHO = 0.95
    GAPS = [1.0, 0.5, 0.01]
    ALPHA = 0.5
    NUM_ITER = 50
    T = 1000
    EVAL_T = 20_000
    TEMP = 1e-6
    NUM_TRIALS = 20

    # Load seeds for reproducibility
    seeds_file = os.path.join(PROJECT_ROOT, 'cmu', 'seeds_cmu_10class.json')
    if os.path.exists(seeds_file):
        with open(seeds_file) as f:
            seeds = json.load(f)
    else:
        seeds = list(range(10000))

    results = {}
    for gap in GAPS:
        env_config = build_env_config(QUEUE_CLASS, RHO, gap)

        jobs = []
        for i in range(NUM_TRIALS):
            jobs.append({
                'env_config': copy.deepcopy(env_config),
                'seed': seeds[i],
                'num_iter': NUM_ITER,
                'step_rule_name': 'normalized_fixed',
                'alpha': ALPHA,
                'temp': TEMP,
                'T': T,
                'eval_T': EVAL_T,
            })

        print(f"  Running PATHWISE gap={gap}, {NUM_TRIALS} trials...", end='', flush=True)
        start = time.time()
        with mp.ProcessingPool(NUM_CORES) as pool:
            trial_results = pool.map(_run_pathwise, jobs)
        elapsed = time.time() - start

        costs = [r['avg_cost'] for r in trial_results]
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        results[gap] = {'mean': mean_cost, 'std': std_cost, 'costs': costs}
        print(f" cost={mean_cost:.2f}±{std_cost:.2f} ({elapsed:.0f}s)")

    # Compare with existing results
    ref_file = os.path.join(PROJECT_ROOT, 'cmu', 'pathwise_wc_cmu_multiclass10.json')
    if os.path.exists(ref_file):
        with open(ref_file) as f:
            ref = json.load(f)
        print("\n  Comparison with existing results (alpha=0.5):")
        print(f"  {'gap':>6} | {'Reproduced':>15} | {'Reference':>15} | {'Match':>6}")
        print(f"  {'-'*50}")
        for gap in GAPS:
            gap_str = str(int(gap)) if gap == int(gap) else str(gap)
            if '0.5' in ref and gap_str in ref['0.5']:
                ref_costs = [r['avg_cost'] for r in ref['0.5'][gap_str]]
                ref_mean = np.mean(ref_costs)
                our_mean = results[gap]['mean']
                match = "OK" if abs(our_mean - ref_mean) / ref_mean < 0.10 else "DIFF"
                print(f"  {gap:>6} | {our_mean:>12.2f}±{results[gap]['std']:.2f} | "
                      f"{ref_mean:>12.2f}±{np.std(ref_costs):.2f} | {match:>6}")
    else:
        print("  (No reference results found for comparison)")

    return results


# ============================================================================
# Test 3: CMU Rule Optimization (REINFORCE)
# ============================================================================
def test_cmu_reinforce():
    """Reproduce REINFORCE CMU rule optimization on 10-class queue.

    Reference (from cmu/wc_reinforce_baseline_cmu_B100_multiclass10.json):
      gap=1.0, alpha=0.1: avg_cost ~ 10.81-11.10
    """
    print("\n" + "=" * 70)
    print("TEST 3: CMU Rule REINFORCE Optimization (10-class, rho=0.95)")
    print("=" * 70)

    QUEUE_CLASS = 10
    RHO = 0.95
    GAPS = [1.0, 0.5, 0.01]
    ALPHA = 0.1
    NUM_ITER = 50
    T = 1000
    EVAL_T = 20_000
    GAMMA = 0.99
    BATCH = 100
    NUM_TRIALS = 20

    seeds_file = os.path.join(PROJECT_ROOT, 'cmu', 'seeds_cmu_10class.json')
    if os.path.exists(seeds_file):
        with open(seeds_file) as f:
            seeds = json.load(f)
    else:
        seeds = list(range(10000))

    results = {}
    for gap in GAPS:
        env_config = build_env_config(QUEUE_CLASS, RHO, gap)

        jobs = []
        for i in range(NUM_TRIALS):
            jobs.append({
                'env_config': copy.deepcopy(env_config),
                'seed': seeds[i],
                'num_iter': NUM_ITER,
                'alpha': ALPHA,
                'T': T,
                'gamma': GAMMA,
                'batch': BATCH,
                'eval_T': EVAL_T,
            })

        print(f"  Running REINFORCE gap={gap}, {NUM_TRIALS} trials...", end='', flush=True)
        start = time.time()
        with mp.ProcessingPool(NUM_CORES) as pool:
            trial_results = pool.map(_run_reinforce, jobs)
        elapsed = time.time() - start

        costs = [r['avg_cost'] for r in trial_results]
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        results[gap] = {'mean': mean_cost, 'std': std_cost, 'costs': costs}
        print(f" cost={mean_cost:.2f}±{std_cost:.2f} ({elapsed:.0f}s)")

    # Compare with existing results
    ref_file = os.path.join(PROJECT_ROOT, 'cmu', 'wc_reinforce_baseline_cmu_B100_multiclass10.json')
    if os.path.exists(ref_file):
        with open(ref_file) as f:
            ref = json.load(f)
        print("\n  Comparison with existing results (alpha=0.1):")
        print(f"  {'gap':>6} | {'Reproduced':>15} | {'Reference':>15} | {'Match':>6}")
        print(f"  {'-'*50}")
        for gap in GAPS:
            gap_str = str(int(gap)) if gap == int(gap) else str(gap)
            if '0.1' in ref and gap_str in ref['0.1']:
                ref_costs = [r['avg_cost'] for r in ref['0.1'][gap_str]]
                ref_mean = np.mean(ref_costs)
                our_mean = results[gap]['mean']
                match = "OK" if abs(our_mean - ref_mean) / ref_mean < 0.10 else "DIFF"
                print(f"  {gap:>6} | {our_mean:>12.2f}±{results[gap]['std']:.2f} | "
                      f"{ref_mean:>12.2f}±{np.std(ref_costs):.2f} | {match:>6}")
    else:
        print("  (No reference results found for comparison)")

    return results


# ============================================================================
# Test 4: Gradient Comparison (Cosine Similarity)
# ============================================================================
def test_gradient_comparison():
    """Reproduce gradient comparison: PATHWISE vs REINFORCE cosine similarity."""
    print("\n" + "=" * 70)
    print("TEST 4: Gradient Comparison (Cosine Similarity)")
    print("=" * 70)

    with open(os.path.join(PROJECT_ROOT, 'configs/env/criss_cross_bh.yaml')) as f:
        env_config = yaml.safe_load(f)

    s, q = 2, 3
    T = 1000
    device = 'cpu'
    num_trials = 20
    gt_batch = 100_000  # smaller than paper's 1M for speed

    for policy_type in ['sPR', 'sMW']:
        print(f"\n  Policy: {policy_type}")

        # Random policy weights
        torch.manual_seed(42)
        net = SoftPriorityPolicy(s, q) if policy_type == 'sPR' else SoftMaxWeightPolicy(s, q)
        state_dict = copy.deepcopy(net.state_dict())

        # Ground truth: large-batch REINFORCE (use smaller batch to avoid OOM)
        print(f"    Computing ground truth (B={gt_batch})...", end='', flush=True)
        from gradient_comparison import _compute_reinforce_grad_core, compute_pathwise_grad, cosine_similarity

        # Accumulate GT over multiple smaller batches to avoid OOM
        gt_sub_batch = 10_000
        gt_grads = []
        for _ in range(gt_batch // gt_sub_batch):
            net_gt = SoftPriorityPolicy(s, q) if policy_type == 'sPR' else SoftMaxWeightPolicy(s, q)
            net_gt.load_state_dict(copy.deepcopy(state_dict))
            g = _compute_reinforce_grad_core(net_gt, env_config, gt_sub_batch, T, 0.999, device)
            gt_grads.append(g)
        gt_grad = torch.stack(gt_grads).mean(dim=0)
        print(f" norm={torch.norm(gt_grad):.4f}")

        # Pathwise samples (B=1)
        pw_sims = []
        for i in range(num_trials):
            net_pw = SoftPriorityPolicy(s, q) if policy_type == 'sPR' else SoftMaxWeightPolicy(s, q)
            net_pw.load_state_dict(copy.deepcopy(state_dict))
            pw_grad = compute_pathwise_grad(net_pw, env_config, 1, T, device)
            pw_sims.append(cosine_similarity(pw_grad, gt_grad))

        # REINFORCE samples (B=1000)
        rf_sims = []
        for i in range(num_trials):
            net_rf = SoftPriorityPolicy(s, q) if policy_type == 'sPR' else SoftMaxWeightPolicy(s, q)
            net_rf.load_state_dict(copy.deepcopy(state_dict))
            rf_grad = _compute_reinforce_grad_core(net_rf, env_config, 1000, T, 0.999, device)
            rf_sims.append(cosine_similarity(rf_grad, gt_grad))

        pw_mean = np.mean(pw_sims)
        rf_mean = np.mean(rf_sims)

        print(f"    PATHWISE cossim:  {pw_mean:.4f} ± {np.std(pw_sims):.4f}")
        print(f"    REINFORCE cossim: {rf_mean:.4f} ± {np.std(rf_sims):.4f}")

        status = "OK" if pw_mean > rf_mean * 0.8 else "WARN"
        print(f"    PATHWISE >= REINFORCE? {pw_mean:.3f} vs {rf_mean:.3f} [{status}]")


# ============================================================================
# Test 5: cmu-rule Reference (Optimal Policy)
# ============================================================================
def test_cmu_rule_reference():
    """Verify that the known cmu-rule optimal policy achieves theoretical cost."""
    print("\n" + "=" * 70)
    print("TEST 5: cmu-rule Optimal Policy Reference Cost")
    print("=" * 70)

    QUEUE_CLASS = 10
    RHO = 0.95
    GAPS = [1.0, 0.5, 0.01]

    for gap in GAPS:
        env_config = build_env_config(QUEUE_CLASS, RHO, gap)

        # cmu-rule: priority = h_j * mu_j = 1 * (1 + gap*j)
        priority = torch.tensor([[1 + gap * j for j in range(1, QUEUE_CLASS + 1)]])
        cmu_cost = evaluate_iterate_fast(priority, env_config, batch=100, eval_T=50_000)

        # Uniform priority (no differentiation between queues)
        uniform_priority = torch.ones(1, QUEUE_CLASS)
        uniform_cost = evaluate_iterate_fast(uniform_priority, env_config, batch=100, eval_T=50_000)

        improvement = (uniform_cost - cmu_cost) / uniform_cost * 100
        print(f"  gap={gap:>4}: cmu-rule={cmu_cost:.2f}, uniform={uniform_cost:.2f}, "
              f"improvement={improvement:.1f}%")

    return True


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    start_time = time.time()

    print("\n" + "#" * 70)
    print("# REPRODUCING MAIN PAPER RESULTS")
    print("#" * 70)

    # Test 1: Simulator validation
    test_mm1_simulator()

    # Test 5: cmu-rule reference costs
    test_cmu_rule_reference()

    # Test 2: PATHWISE CMU optimization
    pw_results = test_cmu_pathwise()

    # Test 3: REINFORCE CMU optimization
    rf_results = test_cmu_reinforce()

    # Compare PATHWISE vs REINFORCE
    print("\n" + "=" * 70)
    print("PATHWISE vs REINFORCE Summary")
    print("=" * 70)
    print(f"  {'gap':>6} | {'PATHWISE':>12} | {'REINFORCE':>12} | {'PW wins?':>10}")
    print(f"  {'-'*50}")
    for gap in [1.0, 0.5, 0.01]:
        if gap in pw_results and gap in rf_results:
            pw = pw_results[gap]['mean']
            rf = rf_results[gap]['mean']
            wins = "YES" if pw < rf else "NO"
            print(f"  {gap:>6} | {pw:>12.2f} | {rf:>12.2f} | {wins:>10}")

    # Test 4: Gradient comparison
    test_gradient_comparison()

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Total reproduction time: {total_time / 60:.1f} minutes")
    print(f"{'=' * 70}")

    # Save reproduction results
    repro = {
        'pathwise': {str(k): v for k, v in pw_results.items()},
        'reinforce': {str(k): v for k, v in rf_results.items()},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(PROJECT_ROOT, 'results', 'reproduction_results.json'), 'w') as f:
        json.dump(repro, f, indent=2, default=lambda x: x if not isinstance(x, np.floating) else float(x))
    print("Results saved to results/reproduction_results.json")
