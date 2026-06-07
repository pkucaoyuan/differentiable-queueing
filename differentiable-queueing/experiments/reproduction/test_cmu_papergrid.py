"""
Section 5.2: CμRule 10-class with PAPER-CORRECT gap grid

Per the deep-research-report (7).md, the paper Section 5.2 uses gap grid
{1.0, 0.5, 0.1, 0.05, 0.01} but upstream cmu_step_rules_*.py code only
iterates over [1, 0.5, 0.05, 0.01] — missing gap=0.1.

This script adds gap=0.1 to make the reproduction exactly match the paper's
grid (5 gaps × 4 alphas × 50 trials × 2 methods).

Designed for SGE: NSLOTS workers, ~6h wall time on 16 cores.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import json, time, copy, sys, statistics
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
# Paper-correct gap grid (added 0.1)
GAPS = [1.0, 0.5, 0.1, 0.05, 0.01]
PW_ALPHAS = [0.01, 0.1, 0.5, 1.0]
RF_ALPHAS = [0.01, 0.1, 0.5, 1.0]
NUM_TRIALS = 50
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
    print("§5.2 CμRule 10-class — PAPER GRID (gaps include 0.1)")
    print("=" * 70)
    print(f"  Gaps:    {GAPS}")
    print(f"  Alphas:  PW={PW_ALPHAS}, RF={RF_ALPHAS}")
    print(f"  Trials:  {NUM_TRIALS} | Cores: {NUM_CORES}")

    seeds_file = os.path.join(PROJECT_ROOT, 'cmu', 'seeds_cmu_10class.json')
    with open(seeds_file) as f:
        seeds = json.load(f)

    env_config_base = build_env_config(QUEUE_CLASS, RHO, gap=GAPS[0])

    pw_results = {}
    rf_results = {}

    # PATHWISE sweep
    print(f"\n--- PATHWISE: {NUM_TRIALS} × {len(PW_ALPHAS)} × {len(GAPS)} ---")
    for alpha in PW_ALPHAS:
        pw_results[str(alpha)] = {}
        for gap in GAPS:
            jobs = [{'env_config': copy.deepcopy(build_env_config(QUEUE_CLASS, RHO, gap)),
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
            print(f"  alpha={alpha} gap={gap}: {np.mean(costs):.3f}±{np.std(costs):.3f} ({time.time()-t0:.0f}s)")

    # REINFORCE sweep
    print(f"\n--- REINFORCE: {NUM_TRIALS} × {len(RF_ALPHAS)} × {len(GAPS)} ---")
    for alpha in RF_ALPHAS:
        rf_results[str(alpha)] = {}
        for gap in GAPS:
            jobs = [{'env_config': copy.deepcopy(build_env_config(QUEUE_CLASS, RHO, gap)),
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
            print(f"  alpha={alpha} gap={gap}: {np.mean(costs):.3f}±{np.std(costs):.3f} ({time.time()-t0:.0f}s)")

    with open(os.path.join(RESULTS_DIR, 'cmu_papergrid_pathwise.json'), 'w') as f:
        json.dump(pw_results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'cmu_papergrid_reinforce.json'), 'w') as f:
        json.dump(rf_results, f, indent=2)

    # Verification vs reference (where available)
    print("\n" + "=" * 70)
    print("Verification vs cmu/*.json reference")
    print("=" * 70)
    for alpha in PW_ALPHAS:
        for gap in GAPS:
            for method, results, ref_pattern, alphas in [
                ('PATHWISE', pw_results, 'pathwise_results', PW_ALPHAS),
                ('REINFORCE', rf_results, 'reinforce_results', RF_ALPHAS),
            ]:
                ref_file = os.path.join(PROJECT_ROOT, 'cmu',
                                        f'{ref_pattern}_{QUEUE_CLASS}_{RHO}.json')
                if not os.path.exists(ref_file):
                    continue
                with open(ref_file) as f:
                    ref = json.load(f)
                if str(alpha) not in ref or str(gap) not in ref[str(alpha)]:
                    continue
                oc = [r['avg_cost'] for r in results[str(alpha)][str(gap)]]
                rc = [r['avg_cost'] for r in ref[str(alpha)][str(gap)]]
                om, rm = statistics.mean(oc), statistics.mean(rc)
                diff = abs(om - rm) / rm * 100 if rm > 0 else 0
                marker = "OK" if diff < 7 else "WARN"
                print(f"  alpha={alpha:>5} gap={gap:>5} {method:>10}: "
                      f"Ours {om:.3f} | Ref {rm:.3f} | {diff:.2f}% [{marker}]")


if __name__ == '__main__':
    t0 = time.time()
    run()
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
