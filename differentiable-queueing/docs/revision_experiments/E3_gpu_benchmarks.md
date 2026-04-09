# E3: GPU Wall-Clock Benchmarks

## Reviewer Requirement

**AE Major Comment 4**: "The statement regarding SmoothBackprop's efficient computation through reverse-mode auto-differentiation using PyTorch and Jax requires elaboration: (i) which operations enable auto-diff, (ii) how calculations can be parallelized on GPUs, (iii) whether custom operations are needed."

**Referee 2**: "GPU parallelism is mentioned early on in the abstract, but rarely comes up later. The simulator should be made publicly available; provide wall-clock performance measurements."

## Objective

1. Profile `env.step()` to identify all differentiable vs non-differentiable operations
2. Benchmark GPU vs CPU wall-clock time across batch sizes and network scales
3. Demonstrate no custom CUDA kernels are needed (all standard PyTorch ops)

## New File

`experiments/gpu_benchmarks.py`

## Required Code Modification

### `queuetorch/env.py`: GPU-native random sampling

**Problem**: Current `draw_service()` and `draw_inter_arrivals()` use `np.random.default_rng()` which is CPU-only. Tensors are created on CPU then moved to device.

**Fix**: Add option to use `torch.distributions` for GPU-native sampling.

```python
# Current (env.py ~line 92):
def draw_service(self):
    if self.service_type == 'exp':
        return torch.tensor(
            self.rng.exponential(1, size=(self.batch, self.q)),
            dtype=torch.float32
        ).to(self.device)

# Modified:
def draw_service(self):
    if self.service_type == 'exp':
        if self.device.type == 'cuda':
            dist = torch.distributions.Exponential(
                torch.ones(self.batch, self.q, device=self.device)
            )
            return dist.sample()
        else:
            return torch.tensor(
                self.rng.exponential(1, size=(self.batch, self.q)),
                dtype=torch.float32
            ).to(self.device)
```

Same pattern for `draw_inter_arrivals()` and hyper-exponential variants.

## Part (i): Operation Breakdown of `env.step()`

Document each operation in `step()` (env.py lines 152-231) and its differentiability:

| Line | Operation | Differentiable? | PyTorch Op |
|------|-----------|-----------------|------------|
| 177 | Action clipping (WC) | Yes (min is differentiable) | `torch.min()` |
| 182 | Effective service times | Yes | Division, `torch.clamp()` |
| 185 | Min over servers | Yes | `torch.min(dim=1)` |
| 188 | Event time concatenation | Yes | `torch.cat()` |
| 191 | **STE event selection** | **Surrogate** (forward: argmin, backward: softmax) | `F.one_hot(argmin) - softmax.detach() + softmax` |
| 195 | Inter-event time (min) | Yes | Element-wise min via `torch.matmul(event_times, outcome)` |
| 202 | Cost computation | Yes | `torch.matmul(queues * event_time, h)` |
| 210 | Queue update | Yes (ReLU for non-negativity) | `F.relu(queues + delta @ outcome)` |
| 213 | Service time reduction | Yes | Subtraction |
| 217 | Arrival time reduction | Yes | Subtraction |
| 220-228 | New service/arrival draws | **Detached** (exogenous noise) | `draw_service()`, `draw_inter_arrivals()` — no grad |

**Key insight**: The ONLY non-standard operation is line 191 (STE). Everything else is standard PyTorch. No custom autograd functions needed (except Sinkhorn in routing.py, which is optional).

## Part (ii): Benchmarking Script

```python
import torch
import time
import json
from queuetorch.env import load_env

def run_trajectory(dq, net, T, mode='forward_backward'):
    """Run T steps of simulation, optionally with backward pass."""
    obs, state = dq.reset()
    total_cost = torch.zeros(dq.batch, device=dq.device)
    
    for _ in range(T):
        queues, t = obs
        action = net(queues)
        # Apply WC constraints
        action = action * dq.network
        action = torch.minimum(action, queues.unsqueeze(1).repeat(1, dq.s, 1))
        mask = torch.all(action == 0, dim=2)
        action = action + mask.unsqueeze(-1) * dq.network
        action = action / action.sum(dim=-1, keepdim=True)
        
        obs, state, cost, event_time = dq.step(state, action)
        total_cost = total_cost + cost.squeeze(1)
    
    loss = total_cost.mean()
    
    if mode == 'forward_backward':
        loss.backward()
    
    return loss.item()

def benchmark(env_config, batch, device, T, mode, num_warmup=50, num_timed=10):
    """Benchmark a single configuration."""
    dq = load_env(env_config, temp=0.1, batch=batch, device=device)
    
    # Simple policy: softmax over fixed random weights
    q, s = dq.q, dq.s
    net = lambda queues: torch.softmax(
        torch.randn(1, s, q, device=device).expand(queues.shape[0], -1, -1), 
        dim=-1
    )
    
    # Warmup
    for _ in range(num_warmup):
        if device == 'cuda':
            torch.cuda.synchronize()
        run_trajectory(dq, net, T=min(T, 100), mode=mode)
    
    # Timed runs
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_timed):
        start = time.perf_counter()
        run_trajectory(dq, net, T, mode)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        'mean': sum(times) / len(times),
        'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        'min': min(times),
        'max': max(times)
    }

# Main experiment
ENVS = ['mm1', 'criss_cross_IID', 'reentrant_5', 'reentrant_10']
BATCHES = [1, 10, 100, 1000, 10000]
DEVICES = ['cpu', 'cuda']
MODES = ['forward_only', 'forward_backward']
T = 1000

results = []
for env_name in ENVS:
    env_config = load_yaml(f'configs/env/{env_name}.yaml')
    for batch in BATCHES:
        for device in DEVICES:
            for mode in MODES:
                timing = benchmark(env_config, batch, device, T, mode)
                results.append({
                    'env': env_name, 'batch': batch, 'device': device,
                    'mode': mode, **timing
                })
                print(f"{env_name} B={batch} {device} {mode}: {timing['mean']:.3f}s")

with open('results/gpu_benchmarks.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Part (iii): Custom Operations Inventory

Document in paper:

> All operations in the differentiable simulator use standard PyTorch functions:
> - `F.softmax`, `F.one_hot`, `torch.argmin` — event selection (STE)
> - `torch.matmul`, `torch.min`, `torch.clamp` — service time computation
> - `F.relu` — non-negativity of queue lengths
> - `torch.distributions.Exponential` — stochastic sampling (GPU-native)
>
> The only custom component is the straight-through estimator (line 191 of env.py), which is implemented as a standard PyTorch expression without a custom `autograd.Function`. The Sinkhorn routing module (routing.py) uses a custom `autograd.Function` with analytical backward pass, but this is optional and not required for the core gradient estimation.
>
> No custom CUDA kernels are needed. The entire simulation runs on standard PyTorch tensor operations, enabling seamless GPU execution via `.to('cuda')`.

## Parameters

```python
ENVS = {
    'mm1': {'q': 1, 's': 1},
    'criss_cross_IID': {'q': 3, 's': 2},
    'reentrant_5': {'q': 15, 's': 5},
    'reentrant_10': {'q': 30, 's': 10}
}
BATCHES = [1, 10, 100, 1000, 10000]
DEVICES = ['cpu', 'cuda']  # cpu uses torch.set_num_threads(1)
MODES = ['forward_only', 'forward_backward']
T = 1000
NUM_WARMUP = 50
NUM_TIMED = 10
```

## Expected Paper Output

### Table: Wall-Clock Time (seconds) for T=1000 steps

| Env (q,s) | Batch | CPU fwd | CPU fwd+bwd | GPU fwd | GPU fwd+bwd | Speedup |
|-----------|-------|---------|-------------|---------|-------------|---------|
| mm1 (1,1) | 1 | ... | ... | ... | ... | ... |
| mm1 (1,1) | 1000 | ... | ... | ... | ... | ... |
| reentrant_10 (30,10) | 1 | ... | ... | ... | ... | ... |
| reentrant_10 (30,10) | 1000 | ... | ... | ... | ... | ... |

### Figure: Speedup vs Batch Size
- x-axis: batch size (log scale)
- y-axis: GPU speedup ratio (CPU time / GPU time)
- 4 curves (one per environment)
- Expected: speedup increases with batch size; larger networks benefit more from GPU

### Text: Operation Breakdown
- Pie chart or breakdown table showing percentage of time in event selection, queue update, cost computation, random sampling

## Compute Estimate

~2 GPU-hours. Each configuration runs in seconds; total ~200 configurations.

## Hardware Required

- 1x NVIDIA GPU (A100, V100, or RTX 3090)
- CPU baseline: same machine, single-threaded
