"""
E7: Hyperparameter Sensitivity Analysis

Demonstrates that PATHWISE is robust to hyperparameter choices while REINFORCE
is sensitive. Sweeps one hyperparameter at a time (others fixed at defaults).

Usage:
    cd experiments/
    python hyperparam_sensitivity.py
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
from collections import defaultdict
import pathos.multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cmu_step_rules_PATHWISE import build_env_config, pathwise_cmu_step_rule, evaluate_iterate_fast
from cmu_rule_REINFORCE import reinforce_value_cmu

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Fixed defaults ──────────────────────────────────────────────────────────
QUEUE_CLASS = 10
RHO = 0.95
GAP = 0.5
NUM_ITER = 50
EVAL_T = 20_000
NUM_TRIALS = 100
NUM_CORES = int(os.environ.get("NSLOTS", min(4, os.cpu_count() or 1)))

DEFAULTS = {
    'pathwise_alpha': 0.5,
    'pathwise_temp': 1e-6,
    'T': 1000,
    'reinforce_alpha': 0.1,
    'reinforce_batch': 100,
    'reinforce_gamma': 0.99,
}

# ── Sweep axes ──────────────────────────────────────────────────────────────
SWEEPS = {
    'temp': {
        'values': [1e-8, 1e-6, 1e-4, 0.01, 0.1],
        'methods': ['pathwise'],
        'param_key': 'pathwise_temp',
    },
    'alpha': {
        'values': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        'methods': ['pathwise', 'reinforce'],
        'param_key': {'pathwise': 'pathwise_alpha', 'reinforce': 'reinforce_alpha'},
    },
    'horizon_T': {
        'values': [100, 500, 1000, 2000, 5000],
        'methods': ['pathwise', 'reinforce'],
        'param_key': 'T',
    },
    'reinforce_batch': {
        'values': [10, 50, 100, 500, 1000],
        'methods': ['reinforce'],
        'param_key': 'reinforce_batch',
    },
}


def run_trial(args):
    """Worker for a single (sweep, value, method, seed) trial."""
    torch.set_num_threads(1)
    sweep_name, sweep_value, method, seed = args

    # Start from defaults, override the swept parameter
    params = dict(DEFAULTS)
    sweep_cfg = SWEEPS[sweep_name]
    param_key = sweep_cfg['param_key']
    if isinstance(param_key, dict):
        params[param_key[method]] = sweep_value
    else:
        params[param_key] = sweep_value

    env_config = build_env_config(QUEUE_CLASS, RHO, GAP)

    if method == 'pathwise':
        result = pathwise_cmu_step_rule(
            env_config, seed=seed, num_iter=NUM_ITER,
            step_rule_name='normalized_fixed',
            alpha=params['pathwise_alpha'],
            temp=params['pathwise_temp'],
            T=params['T'],
            eval_T=EVAL_T,
        )
    else:
        result = reinforce_value_cmu(
            env_config, seed=seed, num_iter=NUM_ITER,
            alpha=params['reinforce_alpha'],
            T=params['T'],
            gamma=params['reinforce_gamma'],
            batch=params['reinforce_batch'],
            eval_T=EVAL_T,
        )

    return {
        'sweep': sweep_name,
        'value': sweep_value,
        'method': method,
        'seed': seed,
        'cost': result['avg_cost'],
    }


if __name__ == '__main__':
    all_results = []
    start_time = time.time()

    # Count total jobs
    total_jobs = 0
    for sweep_name, sweep_cfg in SWEEPS.items():
        for value in sweep_cfg['values']:
            for method in sweep_cfg['methods']:
                total_jobs += NUM_TRIALS

    print(f"{'=' * 60}")
    print(f"E7: Hyperparameter Sensitivity Experiment")
    print(f"  {len(SWEEPS)} sweep axes, {total_jobs} total jobs")
    print(f"  Using {NUM_CORES} cores")
    print(f"{'=' * 60}\n")

    completed = 0

    for sweep_name, sweep_cfg in SWEEPS.items():
        print(f"\n--- Sweep: {sweep_name} ---")

        for value in sweep_cfg['values']:
            for method in sweep_cfg['methods']:
                jobs = [(sweep_name, value, method, seed) for seed in range(NUM_TRIALS)]
                combo_start = time.time()

                with mp.ProcessingPool(NUM_CORES) as pool:
                    results = pool.map(run_trial, jobs)

                all_results.extend(results)
                completed += len(results)
                elapsed = time.time() - combo_start
                total_elapsed = time.time() - start_time

                costs = [r['cost'] for r in results]
                eta = (total_elapsed / completed) * (total_jobs - completed) if completed > 0 else 0

                print(f"  {method:>10} {sweep_name}={value}: "
                      f"cost={np.mean(costs):.2f}±{np.std(costs):.2f} "
                      f"({elapsed:.1f}s) "
                      f"[{completed}/{total_jobs}, ETA {eta/60:.1f}min]")

                # Save incremental
                with open(os.path.join(RESULTS_DIR, 'E7_hyperparam_sensitivity.json'), 'w') as f:
                    json.dump(all_results, f, indent=2)

    # Aggregate
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")

    aggregated = defaultdict(list)
    for r in all_results:
        key = (r['sweep'], r['value'], r['method'])
        aggregated[key].append(r['cost'])

    summary = []
    for (sweep, value, method), costs in sorted(aggregated.items()):
        entry = {
            'sweep': sweep,
            'value': value,
            'method': method,
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
            'n_trials': len(costs),
        }
        summary.append(entry)
        print(f"  {sweep:>15} = {value:>8} | {method:>10}: {entry['mean_cost']:.2f} ± {entry['std_cost']:.2f}")

    with open(os.path.join(RESULTS_DIR, 'E7_hyperparam_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time / 3600:.1f} hours")
    print("Done!")
