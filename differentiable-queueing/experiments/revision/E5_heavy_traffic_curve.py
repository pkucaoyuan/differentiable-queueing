"""
E5: Heavy-Traffic Performance Curve (rho -> 1)

Sweeps traffic intensity rho from 0.80 to 0.99 and compares PATHWISE vs REINFORCE
policy optimization on the CMU rule multiclass queue problem.

Validates Theorem 2's prediction that PATHWISE's advantage grows as rho→1.

Usage:
    cd experiments/
    python heavy_traffic_curve.py
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
import copy
import sys
from collections import defaultdict
import pathos.multiprocessing as mp
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from queuetorch.env import load_env
from step_rules import make_step_rule
from cmu_step_rules_PATHWISE import build_env_config, pathwise_cmu_step_rule, evaluate_iterate_fast
from cmu_rule_REINFORCE import reinforce_value_cmu
import torch.distributions.one_hot_categorical as one_hot_sample

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Parameters ──────────────────────────────────────────────────────────────
RHO_GRID = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
QUEUE_CLASS = 10
GAPS = [0.01, 0.5, 1.0]
NUM_ITER = 50
T = 1000
EVAL_T = 20_000
NUM_TRIALS = 200
NUM_CORES = int(os.environ.get("NSLOTS", min(4, os.cpu_count() or 1)))

# Best hyperparameters (from existing experiments)
PATHWISE_ALPHA = 0.5
PATHWISE_TEMP = 1e-6
REINFORCE_ALPHA = 0.1
REINFORCE_BATCH = 100
REINFORCE_GAMMA = 0.99


def compute_cmu_optimal_cost(env_config, queue_class, gap):
    """Evaluate the known-optimal cmu-rule priority for this env."""
    priority = torch.tensor([[1 + gap * j for j in range(1, queue_class + 1)]])
    return evaluate_iterate_fast(priority, env_config, batch=100, eval_T=50_000)


def run_pathwise_trial(args):
    """Worker: one PATHWISE trial."""
    torch.set_num_threads(1)
    rho, gap, seed = args
    env_config = build_env_config(QUEUE_CLASS, rho, gap)
    result = pathwise_cmu_step_rule(
        env_config, seed=seed, num_iter=NUM_ITER,
        step_rule_name='normalized_fixed', alpha=PATHWISE_ALPHA,
        temp=PATHWISE_TEMP, T=T, eval_T=EVAL_T
    )
    return {
        'rho': rho, 'gap': gap, 'method': 'pathwise', 'seed': seed,
        'final_cost': result['avg_cost'],
    }


def run_reinforce_trial(args):
    """Worker: one REINFORCE trial."""
    torch.set_num_threads(1)
    rho, gap, seed = args
    env_config = build_env_config(QUEUE_CLASS, rho, gap)
    result = reinforce_value_cmu(
        env_config, seed=seed, num_iter=NUM_ITER,
        alpha=REINFORCE_ALPHA, T=T, gamma=REINFORCE_GAMMA,
        batch=REINFORCE_BATCH, eval_T=EVAL_T
    )
    return {
        'rho': rho, 'gap': gap, 'method': 'reinforce', 'seed': seed,
        'final_cost': result['avg_cost'],
    }


def run_cmu_baseline(args):
    """Worker: evaluate cmu-rule optimal cost."""
    torch.set_num_threads(1)
    rho, gap = args
    env_config = build_env_config(QUEUE_CLASS, rho, gap)
    cost = compute_cmu_optimal_cost(env_config, QUEUE_CLASS, gap)
    return {'rho': rho, 'gap': gap, 'cmu_cost': cost}


if __name__ == '__main__':
    seeds = list(range(NUM_TRIALS))
    all_results = []
    start_time = time.time()

    print(f"{'=' * 60}")
    print(f"E5: Heavy-Traffic Curve Experiment")
    print(f"  rho grid: {RHO_GRID}")
    print(f"  gaps: {GAPS}")
    print(f"  {NUM_TRIALS} trials per (rho, gap, method)")
    print(f"  Using {NUM_CORES} cores")
    print(f"{'=' * 60}\n")

    # Step 1: Compute cmu-rule baselines for all (rho, gap)
    print("Computing cmu-rule baselines...")
    cmu_jobs = [(rho, gap) for rho in RHO_GRID for gap in GAPS]
    with mp.ProcessingPool(NUM_CORES) as pool:
        cmu_results = pool.map(run_cmu_baseline, cmu_jobs)

    cmu_lookup = {}
    for r in cmu_results:
        cmu_lookup[(r['rho'], r['gap'])] = r['cmu_cost']
    print(f"  Done. {len(cmu_results)} baselines computed.\n")

    # Step 2: Run PATHWISE trials
    for gap in GAPS:
        print(f"\n--- Gap = {gap} ---")

        for rho in RHO_GRID:
            # PATHWISE jobs
            pw_jobs = [(rho, gap, seed) for seed in seeds]
            combo_start = time.time()

            with mp.ProcessingPool(NUM_CORES) as pool:
                pw_results = pool.map(run_pathwise_trial, pw_jobs)

            for r in pw_results:
                cmu_cost = cmu_lookup.get((rho, gap), 1.0)
                r['cmu_cost'] = cmu_cost
                r['normalized_cost'] = r['final_cost'] / cmu_cost if cmu_cost > 0 else float('inf')
                all_results.append(r)

            elapsed = time.time() - combo_start
            costs = [r['final_cost'] for r in pw_results]
            print(f"  PATHWISE rho={rho:.2f}: cost={np.mean(costs):.2f}±{np.std(costs):.2f} ({elapsed:.1f}s)")

            # Save incremental
            with open(os.path.join(RESULTS_DIR, 'E5_heavy_traffic_curve.json'), 'w') as f:
                json.dump(all_results, f, indent=2)

    # Step 3: Run REINFORCE trials
    for gap in GAPS:
        print(f"\n--- REINFORCE Gap = {gap} ---")

        for rho in RHO_GRID:
            rf_jobs = [(rho, gap, seed) for seed in seeds]
            combo_start = time.time()

            with mp.ProcessingPool(NUM_CORES) as pool:
                rf_results = pool.map(run_reinforce_trial, rf_jobs)

            for r in rf_results:
                cmu_cost = cmu_lookup.get((rho, gap), 1.0)
                r['cmu_cost'] = cmu_cost
                r['normalized_cost'] = r['final_cost'] / cmu_cost if cmu_cost > 0 else float('inf')
                all_results.append(r)

            elapsed = time.time() - combo_start
            costs = [r['final_cost'] for r in rf_results]
            print(f"  REINFORCE rho={rho:.2f}: cost={np.mean(costs):.2f}±{np.std(costs):.2f} ({elapsed:.1f}s)")

            # Save incremental
            with open(os.path.join(RESULTS_DIR, 'E5_heavy_traffic_curve.json'), 'w') as f:
                json.dump(all_results, f, indent=2)

    # Step 4: Aggregate and print summary
    print(f"\n{'=' * 80}")
    print("Summary: Mean Normalized Cost (cost / cmu_optimal)")
    print(f"{'=' * 80}")
    print(f"{'rho':>6} {'gap':>5} | {'PATHWISE':>12} {'REINFORCE':>12} {'Ratio(RF/PW)':>14}")
    print("-" * 60)

    aggregated = defaultdict(list)
    for r in all_results:
        key = (r['rho'], r['gap'], r['method'])
        aggregated[key].append(r['normalized_cost'])

    for gap in GAPS:
        for rho in RHO_GRID:
            pw_key = (rho, gap, 'pathwise')
            rf_key = (rho, gap, 'reinforce')
            pw_costs = aggregated.get(pw_key, [])
            rf_costs = aggregated.get(rf_key, [])
            if pw_costs and rf_costs:
                pw_mean = np.mean(pw_costs)
                rf_mean = np.mean(rf_costs)
                ratio = rf_mean / pw_mean if pw_mean > 0 else float('inf')
                print(f"{rho:>6.2f} {gap:>5.2f} | {pw_mean:>12.4f} {rf_mean:>12.4f} {ratio:>14.4f}")

    # Save final summary
    summary = []
    for (rho, gap, method), costs in aggregated.items():
        summary.append({
            'rho': rho, 'gap': gap, 'method': method,
            'mean_normalized_cost': float(np.mean(costs)),
            'std_normalized_cost': float(np.std(costs)),
            'n_trials': len(costs),
        })

    with open(os.path.join(RESULTS_DIR, 'E5_heavy_traffic_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time / 3600:.1f} hours")
    print("Done!")
