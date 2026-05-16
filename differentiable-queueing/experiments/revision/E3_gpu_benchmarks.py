"""
E3: GPU Wall-Clock Benchmarks

Benchmarks GPU vs CPU wall-clock time across batch sizes and network scales.
Demonstrates that the simulator uses only standard PyTorch ops — no custom CUDA kernels.

Usage (run on a GPU node):
    ssh researchgpu04
    eval "$(~/miniconda3/bin/conda shell.bash hook)" && conda activate gpu
    cd /user/yc4911/DJ_OR/differentiable-queueing/experiments
    python gpu_benchmarks.py
"""
import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from queuetorch.env import load_env

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def make_simple_policy(s, q, device):
    """Create a simple fixed-weight softmax policy for benchmarking."""
    weights = torch.randn(1, s, q, device=device, requires_grad=True)

    def policy_fn(queues):
        batch = queues.shape[0]
        return F.softmax(weights.expand(batch, -1, -1), dim=-1)

    return policy_fn, weights


def run_trajectory(dq, policy_fn, T, mode='forward_backward'):
    """Run T steps of simulation, optionally with backward pass."""
    obs, state = dq.reset()
    total_cost = torch.zeros(dq.batch, device=dq.device)

    for _ in range(T):
        queues, t = obs
        action = policy_fn(queues)
        # WC enforcement
        action = action * dq.network
        action = torch.min(
            torch.stack((action, queues.unsqueeze(1).repeat(1, dq.s, 1)), dim=3),
            dim=3
        ).values
        mask = torch.all(action == 0., dim=2)
        action = action + mask.unsqueeze(-1) * dq.network
        action = action / action.sum(dim=-1, keepdim=True)

        obs, state, cost, event_time = dq.step(state, action)
        total_cost = total_cost + cost.squeeze(1)

    loss = total_cost.mean()

    if mode == 'forward_backward':
        loss.backward()

    return loss.item()


def benchmark(env_config, batch, device, T, mode, num_warmup=5, num_timed=10):
    """Benchmark a single configuration. Returns timing dict."""
    torch.set_num_threads(1)

    dq = load_env(env_config, temp=0.1, batch=batch, device=device, seed=42)
    policy_fn, weights = make_simple_policy(dq.s, dq.q, dq.device)

    # Warmup
    for _ in range(num_warmup):
        if weights.grad is not None:
            weights.grad.zero_()
        if device == 'cuda':
            torch.cuda.synchronize()
        try:
            run_trajectory(dq, policy_fn, T=min(T, 100), mode=mode)
        except Exception:
            return None

    # Timed runs
    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(num_timed):
        if weights.grad is not None:
            weights.grad.zero_()
        dq = load_env(env_config, temp=0.1, batch=batch, device=device, seed=42)

        start = time.perf_counter()
        try:
            run_trajectory(dq, policy_fn, T, mode)
        except Exception as e:
            return None
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
    }


def get_operation_breakdown():
    """Document each operation in env.step() and its differentiability."""
    return [
        {'line': 173, 'operation': 'Action clipping (network compliance)',
         'differentiable': True, 'pytorch_op': 'element-wise multiply'},
        {'line': 177, 'operation': 'Work-conserving clipping (min with queues)',
         'differentiable': True, 'pytorch_op': 'torch.min(stack)'},
        {'line': 182, 'operation': 'Effective service times (service/action*mu)',
         'differentiable': True, 'pytorch_op': 'division, torch.clamp, torch.min'},
        {'line': 185, 'operation': 'Min service time over servers',
         'differentiable': True, 'pytorch_op': 'torch.min(dim=1)'},
        {'line': 188, 'operation': 'Event time concatenation',
         'differentiable': True, 'pytorch_op': 'torch.cat'},
        {'line': 191, 'operation': 'STE event selection (argmin fwd, softmax bwd)',
         'differentiable': 'surrogate', 'pytorch_op': 'F.one_hot(argmin) - softmax.detach() + softmax'},
        {'line': 194, 'operation': 'Queue update delta (outcome @ event_options)',
         'differentiable': True, 'pytorch_op': 'torch.matmul'},
        {'line': 198, 'operation': 'Inter-event time (min of event_times)',
         'differentiable': True, 'pytorch_op': 'torch.min(dim=1)'},
        {'line': 202, 'operation': 'Cost computation (time * queues @ h)',
         'differentiable': True, 'pytorch_op': 'torch.matmul'},
        {'line': 210, 'operation': 'Queue update (ReLU for non-negativity)',
         'differentiable': True, 'pytorch_op': 'F.relu'},
        {'line': 213, 'operation': 'Service time reduction',
         'differentiable': True, 'pytorch_op': 'subtraction'},
        {'line': 214, 'operation': 'Arrival time reduction',
         'differentiable': True, 'pytorch_op': 'subtraction'},
        {'line': '219-222', 'operation': 'New random draws (exogenous noise)',
         'differentiable': False, 'pytorch_op': 'draw_service(), draw_inter_arrivals()'},
    ]


if __name__ == '__main__':
    # Detect available devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU available. Running CPU-only benchmarks.")

    # Environment configs
    ENVS = {
        'mm1': 'configs/env/mm1.yaml',
        'criss_cross_IID': 'configs/env/criss_cross_IID.yaml',
    }

    # Check for reentrant configs
    for k in [5, 10]:
        fname = f'configs/env/reentrant_{k}.yaml'
        if os.path.exists(os.path.join(PROJECT_ROOT, fname)):
            ENVS[f'reentrant_{k}'] = fname

    BATCHES = [1, 10, 100, 1000]
    if 'cuda' in devices:
        BATCHES.append(10000)

    MODES = ['forward_only', 'forward_backward']
    T = 1000

    print(f"\n{'=' * 70}")
    print(f"E3: GPU Wall-Clock Benchmarks")
    print(f"  Environments: {list(ENVS.keys())}")
    print(f"  Batch sizes: {BATCHES}")
    print(f"  Devices: {devices}")
    print(f"  T = {T} steps")
    print(f"{'=' * 70}\n")

    results = []

    for env_name, config_path in ENVS.items():
        env_config = load_yaml(os.path.join(PROJECT_ROOT, config_path))
        print(f"\n--- {env_name} ---")

        for batch in BATCHES:
            for device in devices:
                for mode in MODES:
                    print(f"  B={batch:>5} {device:>4} {mode:>16} ... ", end='', flush=True)
                    timing = benchmark(env_config, batch, device, T, mode)

                    if timing is not None:
                        entry = {
                            'env': env_name,
                            'batch': batch,
                            'device': device,
                            'mode': mode,
                            **timing,
                        }
                        results.append(entry)
                        print(f"{timing['mean']:.3f}s ± {timing['std']:.3f}s")
                    else:
                        print("FAILED (OOM or error)")

    # Save results
    with open(os.path.join(RESULTS_DIR, 'E3_gpu_benchmarks.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save operation breakdown
    ops = get_operation_breakdown()
    with open(os.path.join(RESULTS_DIR, 'E3_operation_breakdown.json'), 'w') as f:
        json.dump(ops, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 90}")
    print("Wall-Clock Time (seconds) for T=1000 steps")
    print(f"{'=' * 90}")
    print(f"{'Env':<20} {'Batch':>6} | {'CPU fwd':>10} {'CPU fwd+bwd':>12} | "
          f"{'GPU fwd':>10} {'GPU fwd+bwd':>12} | {'Speedup':>8}")
    print("-" * 90)

    # Build lookup
    lookup = {}
    for r in results:
        key = (r['env'], r['batch'], r['device'], r['mode'])
        lookup[key] = r['mean']

    for env_name in ENVS:
        for batch in BATCHES:
            cpu_fwd = lookup.get((env_name, batch, 'cpu', 'forward_only'), float('nan'))
            cpu_fb = lookup.get((env_name, batch, 'cpu', 'forward_backward'), float('nan'))
            gpu_fwd = lookup.get((env_name, batch, 'cuda', 'forward_only'), float('nan'))
            gpu_fb = lookup.get((env_name, batch, 'cuda', 'forward_backward'), float('nan'))
            speedup = cpu_fb / gpu_fb if gpu_fb > 0 else float('nan')
            print(f"{env_name:<20} {batch:>6} | {cpu_fwd:>10.3f} {cpu_fb:>12.3f} | "
                  f"{gpu_fwd:>10.3f} {gpu_fb:>12.3f} | {speedup:>8.1f}x")

    print(f"\nOperation breakdown saved to results/E3_operation_breakdown.json")
    print("Done!")
