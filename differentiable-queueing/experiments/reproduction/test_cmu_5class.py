"""
Phase B: 5-class CMU rule reproduction

Reference: cmu/pathwise_wc_cmu_multiclass5_all_eps_950_more_runs.json (1000 runs)
           cmu/wc_reinforce_baseline_cmu_B100_multiclass5_all_eps_950_more_runs.json

This verifies Section 5.2 on the smaller 5-class queue.

Settings (matching paper):
- 5-class queue, rho=0.99 (not 0.95 — that's 10-class default)
- 4 alphas × 4 gaps × 50 trials per method
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
import pathos.multiprocessing as mp
import statistics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cmu_step_rules_PATHWISE import build_env_config, pathwise_cmu_step_rule
from cmu_rule_REINFORCE import reinforce_value_cmu

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'reproduction')
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CORES = int(os.environ.get('NSLOTS', min(4, os.cpu_count() or 1)))

# 5-class queue settings — matching cmu_rule_REINFORCE.py main block (line 366-370)
QUEUE_CLASS = 5
RHO = 0.99
# Quick-check version: 2 alphas × 2 gaps × 10 trials = 40 trials per method
# (full version is 4×4×50=800 — too slow for first pass)
GAPS = [1.0, 0.05]  # 2 gaps to test high/low
PW_ALPHAS = [0.5]    # use the best-performing alpha
RF_ALPHAS = [0.1]    # use the best-performing alpha
NUM_TRIALS = 20      # quick check, will scale up if results look right
NUM_ITER = 50
T = 1000
EVAL_T = 20_000
TEMP = 1e-6

# Seed file: 5-class
SEEDS_FILE = os.path.join(PROJECT_ROOT, 'cmu', 'seeds_cmu_5class.json')


def _run_pw(args):
    torch.set_num_threads(1)
    return pathwise_cmu_step_rule(**args)


def _run_rf(args):
    torch.set_num_threads(1)
    return reinforce_value_cmu(**args)


def run():
    print("=" * 70)
    print("Phase B: 5-class CMU Rule Reproduction")
    print("=" * 70)
    print(f"  Queue class: {QUEUE_CLASS}")
    print(f"  rho: {RHO}")
    print(f"  gaps: {GAPS}")
    print(f"  PW alphas: {PW_ALPHAS}")
    print(f"  RF alphas: {RF_ALPHAS}")
    print(f"  trials per combo: {NUM_TRIALS}")
    print(f"  cores: {NUM_CORES}")

    # Load seeds matching existing 5-class reproduction
    if os.path.exists(SEEDS_FILE):
        with open(SEEDS_FILE) as f:
            seeds = json.load(f)
        print(f"  Using seeds from: {SEEDS_FILE}")
    else:
        seeds = [int.from_bytes(os.urandom(4), 'big') for _ in range(10000)]

    # PATHWISE
    pw_results = {}
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

            with open(os.path.join(RESULTS_DIR, 'reproduction_cmu5_pathwise.json'), 'w') as f:
                json.dump(pw_results, f, indent=2)

    # REINFORCE
    rf_results = {}
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

            with open(os.path.join(RESULTS_DIR, 'reproduction_cmu5_reinforce.json'), 'w') as f:
                json.dump(rf_results, f, indent=2)

    # Verification
    ref_pw = os.path.join(PROJECT_ROOT, 'cmu', 'pathwise_wc_cmu_multiclass5_all_eps_950_more_runs.json')
    ref_rf = os.path.join(PROJECT_ROOT, 'cmu', 'wc_reinforce_baseline_cmu_B100_multiclass5_all_eps_950_more_runs.json')

    print("\n" + "=" * 70)
    print("Verification vs Reference (5-class)")
    print("=" * 70)
    if os.path.exists(ref_pw):
        with open(ref_pw) as f:
            ref = json.load(f)
        print("\nPATHWISE:")
        print(f"  {'alpha':>6} {'gap':>5} | {'Ours':>20} | {'Reference':>20} | {'Diff%':>7}")
        for alpha in pw_results:
            if alpha not in ref:
                continue
            for gap in pw_results[alpha]:
                if gap not in ref[alpha]:
                    continue
                oc = [r['avg_cost'] for r in pw_results[alpha][gap]]
                rc = [r['avg_cost'] for r in ref[alpha][gap]]
                if len(oc) < 2 or len(rc) < 2:
                    continue
                om, os_ = statistics.mean(oc), statistics.stdev(oc)
                rm, rs = statistics.mean(rc), statistics.stdev(rc)
                diff = abs(om - rm) / rm * 100 if rm > 0 else 0
                marker = "OK" if diff < 7 else "WARN"
                print(f"  {alpha:>6} {gap:>5} | {om:>10.3f}±{os_:.3f} | {rm:>10.3f}±{rs:.3f} | {diff:>6.2f}% [{marker}]")

    if os.path.exists(ref_rf):
        with open(ref_rf) as f:
            ref = json.load(f)
        print("\nREINFORCE:")
        print(f"  {'alpha':>6} {'gap':>5} | {'Ours':>20} | {'Reference':>20} | {'Diff%':>7}")
        for alpha in rf_results:
            if alpha not in ref:
                continue
            for gap in rf_results[alpha]:
                if gap not in ref[alpha]:
                    continue
                oc = [r['avg_cost'] for r in rf_results[alpha][gap]]
                rc = [r['avg_cost'] for r in ref[alpha][gap]]
                if len(oc) < 2 or len(rc) < 2:
                    continue
                om, os_ = statistics.mean(oc), statistics.stdev(oc)
                rm, rs = statistics.mean(rc), statistics.stdev(rc)
                diff = abs(om - rm) / rm * 100 if rm > 0 else 0
                marker = "OK" if diff < 7 else "WARN"
                print(f"  {alpha:>6} {gap:>5} | {om:>10.3f}±{os_:.3f} | {rm:>10.3f}±{rs:.3f} | {diff:>6.2f}% [{marker}]")


if __name__ == '__main__':
    start = time.time()
    run()
    print(f"\nTotal time: {(time.time() - start) / 60:.1f} min")
