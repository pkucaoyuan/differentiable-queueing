"""
Section 5.2: T (horizon) ablation reproduction

Reference: cmu/pathwise_ablation_T_{500,1000,2000,5000}.json
           cmu/reinforce_ablation_T_{500,1000,2000,5000}.json

Quick check: 1 alpha × 1 gap × 10 trials per T.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import json
import time
import copy
import sys
import statistics
import numpy as np
import pathos.multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cmu_step_rules_PATHWISE import build_env_config, pathwise_cmu_step_rule
from cmu_rule_REINFORCE import reinforce_value_cmu

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'reproduction')
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CORES = int(os.environ.get('NSLOTS', min(4, os.cpu_count() or 1)))

QUEUE_CLASS = 10
RHO = 0.95
GAP = 0.5
T_VALUES = [500, 1000, 2000, 5000]
PW_ALPHA = 0.5
RF_ALPHA = 0.1
NUM_TRIALS = 10
NUM_ITER = 50
EVAL_T = 20_000
TEMP = 1e-6


def _run_pw(args):
    torch.set_num_threads(1)
    return pathwise_cmu_step_rule(**args)


def _run_rf(args):
    torch.set_num_threads(1)
    return reinforce_value_cmu(**args)


def run():
    print("=" * 70)
    print("Section 5.2: T ablation reproduction (quick)")
    print("=" * 70)
    seeds_file = os.path.join(PROJECT_ROOT, 'cmu', 'seeds_cmu_10class.json')
    if os.path.exists(seeds_file):
        with open(seeds_file) as f:
            seeds = json.load(f)
    else:
        seeds = [int.from_bytes(os.urandom(4), 'big') for _ in range(10000)]

    pw_all = {}
    rf_all = {}
    env_config_base = build_env_config(QUEUE_CLASS, RHO, GAP)

    for T_val in T_VALUES:
        T_key = f"T_{T_val}"
        pw_all[T_key] = {str(PW_ALPHA): {}}
        rf_all[T_key] = {str(RF_ALPHA): {}}

        # PATHWISE
        jobs = [{'env_config': copy.deepcopy(env_config_base),
                 'seed': seeds[i],
                 'num_iter': NUM_ITER,
                 'step_rule_name': 'normalized_fixed',
                 'alpha': PW_ALPHA, 'temp': TEMP, 'T': T_val, 'eval_T': EVAL_T}
                for i in range(NUM_TRIALS)]
        t0 = time.time()
        with mp.ProcessingPool(NUM_CORES) as pool:
            trials = pool.map(_run_pw, jobs)
        costs = [r['avg_cost'] for r in trials]
        pw_all[T_key][str(PW_ALPHA)][str(GAP)] = trials
        print(f"  PW T={T_val:>4} gap={GAP}: {np.mean(costs):.3f}±{np.std(costs):.3f} ({time.time()-t0:.0f}s)")

        # REINFORCE
        jobs = [{'env_config': copy.deepcopy(env_config_base),
                 'seed': seeds[i],
                 'num_iter': NUM_ITER,
                 'alpha': RF_ALPHA, 'T': T_val, 'gamma': 0.99,
                 'batch': 100, 'eval_T': EVAL_T}
                for i in range(NUM_TRIALS)]
        t0 = time.time()
        with mp.ProcessingPool(NUM_CORES) as pool:
            trials = pool.map(_run_rf, jobs)
        costs = [r['avg_cost'] for r in trials]
        rf_all[T_key][str(RF_ALPHA)][str(GAP)] = trials
        print(f"  RF T={T_val:>4} gap={GAP}: {np.mean(costs):.3f}±{np.std(costs):.3f} ({time.time()-t0:.0f}s)")

    with open(os.path.join(RESULTS_DIR, 'T_ablation_pathwise.json'), 'w') as f:
        json.dump(pw_all, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'T_ablation_reinforce.json'), 'w') as f:
        json.dump(rf_all, f, indent=2)

    # Verification vs reference
    print("\n" + "=" * 70)
    print("Verification vs reference")
    print("=" * 70)
    for T_val in T_VALUES:
        T_key = f"T_{T_val}"
        for method_name, results_all, ref_pattern, alpha in [
            ('PATHWISE', pw_all, 'pathwise_ablation_T', PW_ALPHA),
            ('REINFORCE', rf_all, 'reinforce_ablation_T', RF_ALPHA),
        ]:
            ref_file = os.path.join(PROJECT_ROOT, 'cmu', f'{ref_pattern}_{T_val}.json')
            if not os.path.exists(ref_file):
                continue
            with open(ref_file) as f:
                ref = json.load(f)
            gap_key = str(GAP)
            if str(alpha) not in ref:
                continue
            ref_gap_key = gap_key if gap_key in ref[str(alpha)] else str(int(GAP)) if GAP == int(GAP) else gap_key
            if ref_gap_key not in ref[str(alpha)]:
                continue
            oc = [r['avg_cost'] for r in results_all[T_key][str(alpha)][gap_key]]
            rc = [r['avg_cost'] for r in ref[str(alpha)][ref_gap_key]]
            if len(oc) < 2 or len(rc) < 2:
                continue
            om = statistics.mean(oc)
            rm = statistics.mean(rc)
            diff = abs(om - rm) / rm * 100 if rm > 0 else 0
            marker = "OK" if diff < 7 else "WARN"
            print(f"  T={T_val:>4} {method_name:>10}: Ours {om:.3f} | Ref {rm:.3f} | {diff:.2f}% [{marker}]")


if __name__ == '__main__':
    start = time.time()
    run()
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
