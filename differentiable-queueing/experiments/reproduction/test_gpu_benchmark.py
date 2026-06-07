"""
§4.3.1 GPU wall-clock benchmark — was blocked, now unlocked on GPU 3.

Measures PATHWISE simulation wall-clock vs batch size on CPU and GPU, on
the criss-cross environment used in §5.1. Demonstrates GPU speedup.

Run with `CUDA_VISIBLE_DEVICES=3` to pin to GPU 3 (must not touch other users).
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'

import sys, time, json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from queuetorch.env import load_env
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'reproduction')
os.makedirs(RESULTS_DIR, exist_ok=True)

ENV_NAME = 'criss_cross_bh'
T = 1000  # simulation horizon per run
BATCH_SIZES = [1, 16, 256, 1024, 4096, 16384]
WARMUP_RUNS = 1
TIMED_RUNS = 3


def load_env_at(batch, device):
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'env', f'{ENV_NAME}.yaml')
    with open(config_path) as f:
        env_config = yaml.safe_load(f)
    dq = load_env(env_config, temp=1e-3, batch=batch, seed=42, device=device)
    return dq


def run_simulation(dq, T, device):
    """Forward-only simulation: random softmax actions."""
    obs, state = dq.reset()
    s_dim = dq.s
    q_dim = dq.q
    # uniform softmax policy
    action_logits = torch.zeros((dq.batch, s_dim, q_dim), device=device)
    action = torch.nn.functional.softmax(action_logits, dim=-1)
    total_cost = torch.zeros(dq.batch, device=device)
    for _ in range(T):
        obs, state, cost, _ = dq.step(state, action)
        total_cost = total_cost + cost
    return total_cost.mean().item()


def benchmark(device_str):
    device = torch.device(device_str)
    print(f"\n══ Device: {device_str} ══")
    results = []
    for B in BATCH_SIZES:
        # Setup env
        try:
            dq = load_env_at(B, device)
        except Exception as e:
            print(f"  B={B:>4}: setup failed: {e}")
            continue

        # Warmup
        for _ in range(WARMUP_RUNS):
            _ = run_simulation(dq, T, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(TIMED_RUNS):
            t0 = time.time()
            cost = run_simulation(dq, T, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t0)

        median = float(np.median(times))
        throughput = (B * T) / median  # events per second
        results.append({
            'batch_size': B,
            'device': device_str,
            'median_time_s': median,
            'min_time_s': min(times),
            'max_time_s': max(times),
            'throughput_events_per_s': throughput,
            'final_cost': cost,
        })
        print(f"  B={B:>4}  median={median:.3f}s  throughput={throughput:.0f} ev/s")
    return results


def main():
    print("=" * 70)
    print("§4.3.1 GPU vs CPU benchmark (criss-cross, T=1000)")
    print("=" * 70)
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  Free memory: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print(f"  Warmup: {WARMUP_RUNS}, Timed: {TIMED_RUNS}")

    all_results = []
    cpu_results = benchmark('cpu')
    all_results.extend(cpu_results)

    if torch.cuda.is_available():
        gpu_results = benchmark('cuda')
        all_results.extend(gpu_results)

        # Speedup summary
        print("\n══ Speedup (GPU / CPU) ══")
        cpu_by_B = {r['batch_size']: r['median_time_s'] for r in cpu_results}
        for r in gpu_results:
            B = r['batch_size']
            if B in cpu_by_B:
                speedup = cpu_by_B[B] / r['median_time_s']
                tput_ratio = r['throughput_events_per_s'] / (cpu_by_B[B] and (B * T / cpu_by_B[B]))
                print(f"  B={B:>4}  speedup={speedup:.2f}×  "
                      f"CPU={cpu_by_B[B]:.3f}s  GPU={r['median_time_s']:.3f}s")

    with open(os.path.join(RESULTS_DIR, 'gpu_benchmark.json'), 'w') as f:
        json.dump({
            'env': ENV_NAME, 'T': T,
            'batch_sizes': BATCH_SIZES,
            'warmup_runs': WARMUP_RUNS, 'timed_runs': TIMED_RUNS,
            'results': all_results,
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }, f, indent=2)


if __name__ == '__main__':
    main()
