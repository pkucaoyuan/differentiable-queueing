"""
§4.3.1 — Extended GPU benchmark at large batch sizes (GPU-only sweep).

Continues from test_gpu_benchmark.py. Tests batch sizes up to 65k on GPU 3
where CPU would take too long to be a useful baseline.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'

import sys, time, json
import torch
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from queuetorch.env import load_env

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'reproduction')
os.makedirs(RESULTS_DIR, exist_ok=True)

ENV_NAME = 'criss_cross_bh'
T = 1000
BATCH_SIZES = [4096, 16384, 65536]
WARMUP = 1
TIMED = 3


def main():
    print(f"§4.3.1 large-batch GPU benchmark | CUDA: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("No CUDA; exiting", flush=True); return
    print(f"Device: {torch.cuda.get_device_name(0)}, free {torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)

    config_path = os.path.join(PROJECT_ROOT, 'configs', 'env', f'{ENV_NAME}.yaml')
    env_config = yaml.safe_load(open(config_path))
    device = torch.device('cuda')

    results = []
    for B in BATCH_SIZES:
        print(f"\nB={B}", flush=True)
        try:
            dq = load_env(env_config, temp=1e-3, batch=B, seed=42, device=device)
        except Exception as e:
            print(f"  setup failed: {e}", flush=True)
            continue

        action = torch.softmax(torch.zeros((B, dq.s, dq.q), device=device), dim=-1)

        # Warmup
        obs, state = dq.reset()
        for _ in range(T):
            obs, state, c, _ = dq.step(state, action)
        torch.cuda.synchronize()

        times = []
        for _ in range(TIMED):
            obs, state = dq.reset()
            t0 = time.time()
            for _ in range(T):
                obs, state, c, _ = dq.step(state, action)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

        med = float(np.median(times))
        tput = B * T / med
        mem_used = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9
        print(f"  median={med:.3f}s  throughput={tput:.0f} ev/s  mem={mem_used:.1f} GB", flush=True)
        results.append({'batch_size': B, 'median_time_s': med, 'throughput_events_per_s': tput,
                        'mem_used_gb': mem_used})
        # release GPU memory between batch sizes
        del dq, action, obs, state, c
        torch.cuda.empty_cache()

    out = {'env': ENV_NAME, 'T': T, 'results': results,
           'cuda_device_name': torch.cuda.get_device_name(0)}
    with open(os.path.join(RESULTS_DIR, 'gpu_benchmark_large.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {os.path.basename(__file__).replace('.py','.json')}", flush=True)


if __name__ == '__main__':
    main()
