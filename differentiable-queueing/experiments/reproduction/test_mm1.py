"""
Experiment 1: M/M/1 Simulator Sanity Check

Validates that the queuetorch simulator produces correct steady-state queue length
for M/M/1 with rho=0.9. Analytical E[Q] = rho/(1-rho) = 9.0.

This is a SANITY CHECK, not a paper-section reproduction. If this fails, every
other experiment is suspect.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import sys
import json
import time
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from queuetorch.env import load_env

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run():
    print("=" * 70)
    print("Experiment 1: M/M/1 Simulator Sanity Check")
    print("=" * 70)

    with open(os.path.join(PROJECT_ROOT, 'configs/env/mm1.yaml')) as f:
        env_config = yaml.safe_load(f)

    rho = 0.9
    analytical = rho / (1 - rho)
    print(f"  Network: M/M/1, rho={rho}")
    print(f"  Analytical E[Q] = rho/(1-rho) = {analytical:.4f}")
    print(f"  Simulating 200,000 events with batch=50...")

    start = time.time()
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

    elapsed = time.time() - start
    sim = float(torch.mean(total_cost / state.time))
    error = abs(sim - analytical)
    error_pct = error / analytical * 100

    print(f"\n  Simulated E[Q] = {sim:.4f}")
    print(f"  Error = {error:.4f} ({error_pct:.3f}%)")
    print(f"  Wall time: {elapsed:.1f}s")

    # Previous result
    prev = 9.0282
    diff_from_prev = abs(sim - prev)
    print(f"\n  Previous result: {prev:.4f}")
    print(f"  Difference from previous: {diff_from_prev:.4f}")

    status = 'PASS' if error_pct < 1.0 else 'FAIL'
    print(f"\n  Status: {status}")

    result = {
        'experiment': 'mm1_sanity',
        'rho': rho,
        'analytical': analytical,
        'simulated': sim,
        'error': error,
        'error_pct': error_pct,
        'wall_time_sec': elapsed,
        'previous_simulated': prev,
        'diff_from_previous': diff_from_prev,
        'status': status,
    }

    out_path = os.path.join(RESULTS_DIR, 'rerun_mm1.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return result


if __name__ == '__main__':
    run()
