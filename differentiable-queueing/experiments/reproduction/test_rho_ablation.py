"""
Phase C: rho ablation reproduction

Reference: cmu/pathwise_ablation_rho_{0.9, 0.95, 0.99}.json
           cmu/reinforce_ablation_rho_{0.9, 0.95, 0.99}.json

Quick check: 1 alpha × 2 gaps × 10 trials = 20 trials per (rho, method).
Tests rho=0.9 and rho=0.99 (rho=0.95 already done in main reproduction).
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

QUEUE_CLASS = 10  # 10-class for ablation
RHOS = [0.9, 0.99]  # not 0.95 — already in main reproduction
GAPS = [1.0, 0.05]
PW_ALPHA = 0.5
RF_ALPHA = 0.1
NUM_TRIALS = 10
NUM_ITER = 50
T = 1000
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
    print("Phase C: rho ablation reproduction (quick)")
    print("=" * 70)

    seeds_file = os.path.join(PROJECT_ROOT, 'cmu', 'seeds_cmu_10class.json')
    if os.path.exists(seeds_file):
        with open(seeds_file) as f:
            seeds = json.load(f)
    else:
        seeds = [int.from_bytes(os.urandom(4), 'big') for _ in range(10000)]

    pw_all = {}
    rf_all = {}
    for rho in RHOS:
        rho_key = f"rho_{rho}"
        pw_all[rho_key] = {str(PW_ALPHA): {}}
        rf_all[rho_key] = {str(RF_ALPHA): {}}
        print(f"\n--- rho = {rho} ---")

        for gap in GAPS:
            env_config = build_env_config(QUEUE_CLASS, rho, gap)

            # PATHWISE
            jobs = [{'env_config': copy.deepcopy(env_config),
                     'seed': seeds[i],
                     'num_iter': NUM_ITER,
                     'step_rule_name': 'normalized_fixed',
                     'alpha': PW_ALPHA, 'temp': TEMP, 'T': T, 'eval_T': EVAL_T}
                    for i in range(NUM_TRIALS)]
            t0 = time.time()
            with mp.ProcessingPool(NUM_CORES) as pool:
                trials = pool.map(_run_pw, jobs)
            costs = [r['avg_cost'] for r in trials]
            pw_all[rho_key][str(PW_ALPHA)][str(gap)] = trials
            print(f"  PW alpha={PW_ALPHA} gap={gap}: {np.mean(costs):.3f}±{np.std(costs):.3f} ({time.time()-t0:.0f}s)")

            # REINFORCE
            jobs = [{'env_config': copy.deepcopy(env_config),
                     'seed': seeds[i],
                     'num_iter': NUM_ITER,
                     'alpha': RF_ALPHA, 'T': T, 'gamma': 0.99,
                     'batch': 100, 'eval_T': EVAL_T}
                    for i in range(NUM_TRIALS)]
            t0 = time.time()
            with mp.ProcessingPool(NUM_CORES) as pool:
                trials = pool.map(_run_rf, jobs)
            costs = [r['avg_cost'] for r in trials]
            rf_all[rho_key][str(RF_ALPHA)][str(gap)] = trials
            print(f"  RF alpha={RF_ALPHA} gap={gap}: {np.mean(costs):.3f}±{np.std(costs):.3f} ({time.time()-t0:.0f}s)")

        # Save per-rho results
        with open(os.path.join(RESULTS_DIR, f'rho_ablation_pathwise.json'), 'w') as f:
            json.dump(pw_all, f, indent=2)
        with open(os.path.join(RESULTS_DIR, f'rho_ablation_reinforce.json'), 'w') as f:
            json.dump(rf_all, f, indent=2)

    # Verification
    print("\n" + "=" * 70)
    print("Verification vs reference")
    print("=" * 70)
    for rho in RHOS:
        rho_str = f"{rho:.2f}".rstrip('0').rstrip('.')
        rho_key = f"rho_{rho}"

        for method_name, results_all, ref_pattern, alpha in [
            ('PATHWISE', pw_all, 'pathwise_ablation_rho', PW_ALPHA),
            ('REINFORCE', rf_all, 'reinforce_ablation_rho', RF_ALPHA),
        ]:
            ref_file = os.path.join(PROJECT_ROOT, 'cmu', f'{ref_pattern}_{rho_str}.json')
            if not os.path.exists(ref_file):
                # try with .0
                ref_file = os.path.join(PROJECT_ROOT, 'cmu', f'{ref_pattern}_{rho}.json')
            if not os.path.exists(ref_file):
                print(f"\n  rho={rho} {method_name}: ref file not found")
                continue
            with open(ref_file) as f:
                ref = json.load(f)

            for gap in GAPS:
                gap_ref_key = str(int(gap)) if gap == int(gap) else str(gap)
                if str(alpha) not in ref:
                    continue
                if gap_ref_key not in ref[str(alpha)]:
                    continue
                oc = [r['avg_cost'] for r in results_all[rho_key][str(alpha)][str(gap)]]
                rc = [r['avg_cost'] for r in ref[str(alpha)][gap_ref_key]]
                if len(oc) < 2 or len(rc) < 2:
                    continue
                om, os_ = statistics.mean(oc), statistics.stdev(oc)
                rm, rs = statistics.mean(rc), statistics.stdev(rc)
                diff = abs(om - rm) / rm * 100 if rm > 0 else 0
                marker = "OK" if diff < 7 else "WARN"
                print(f"  rho={rho:>5} gap={gap:>5} {method_name:>10}: "
                      f"Ours {om:.3f}±{os_:.3f} | Ref {rm:.3f}±{rs:.3f} | {diff:.2f}% [{marker}]")


if __name__ == '__main__':
    start = time.time()
    run()
    print(f"\nTotal time: {(time.time() - start)/60:.1f} min")
