# E8: Parameter Table and Writing Specifications

## Reviewer Requirement

**Referee 1**: "I am surprised that the parameters (service rates, arrival rates, cost, etc) of the numerical examples (reentrants, and criss-cross networks) are not specified. Moreover, several important implementation details are missing or unclear. For example, the discount factor gamma used in the REINFORCE method is not mentioned in Section 4.1. Additionally, it is unclear how the baseline term V_{pi_theta}(x_k) is constructed in the REINFORCE implementation."

**AE Minor 10**: "In both the Abstract and Introduction, the authors mention the proposed method works in 'non-stationary' systems. But there's no relevant discussion in the remaining of the paper."

## Deliverable: Appendix Parameter Tables

### Table A1: Environment Parameters

Source: Extract from `configs/env/*.yaml` and `env_data/*/` numpy arrays.

**Script to generate** (run once to extract all parameters):
```python
import numpy as np
import yaml
import os

envs = [
    ('criss_cross', 'configs/env/criss_cross_IID.yaml'),
    ('criss_cross_hyper', 'configs/env/criss_cross_hyper.yaml'),
]
# Reentrant networks load from env_data
for k in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    envs.append((f'reentrant_{k}', f'configs/env/reentrant_{k}.yaml'))
    if os.path.exists(f'configs/env/reentrant_{k}_hyper.yaml'):
        envs.append((f'reentrant_{k}_hyper', f'configs/env/reentrant_{k}_hyper.yaml'))

for name, config_path in envs:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    env_type = cfg.get('name', name)
    
    # Load network topology and rates
    if cfg.get('network') is None:
        # Load from env_data
        base = f'env_data/{env_type}'
        if not os.path.exists(base):
            # Try alternate paths
            base = f'env_data/reentrant/{env_type}' if 'reentrant' in env_type else base
        network = np.load(f'{base}/{env_type}_network.npy')
        mu = np.load(f'{base}/{env_type}_mu.npy')
        lam = np.load(f'{base}/{env_type}_lam.npy')
    else:
        network = np.array(cfg['network'])
        mu = np.array(cfg['mu'])
        lam = np.array(cfg['lam_params']['val']) if cfg['lam_type'] == 'constant' else 'varies'
    
    h = np.array(cfg['h'])
    
    s, q = network.shape
    print(f"\n=== {name} ===")
    print(f"  Queues: {q}, Servers: {s}")
    print(f"  Network:\n{network}")
    print(f"  Service rates (mu):\n{mu}")
    print(f"  Arrival rates (lambda): {lam}")
    print(f"  Holding costs (h): {h}")
    print(f"  Service type: {cfg.get('service_type', 'exp')}")
    print(f"  Train T: {cfg.get('train_T', 'N/A')}, Test T: {cfg.get('test_T', 'N/A')}")
    
    # Compute traffic intensity (approximate)
    if isinstance(lam, np.ndarray):
        total_arrival = sum(l for l in lam if l < 1e5)  # filter dummy values
        print(f"  Total arrival rate: {total_arrival:.4f}")
```

**Expected output format for paper**:

| Network | n (queues) | m (servers) | lambda | mu (selected) | h | rho | Service Dist |
|---------|-----------|-------------|--------|---------------|---|-----|-------------|
| Criss-cross | 3 | 2 | [0.25, 0.35, 0] | mu_11=1.0, mu_13=1.0, mu_22=0.85 | [1, 2, 1] | ~0.6 | Exp |
| Re-entrant 1 (6) | 6 | 2 | see text | see text | [1,...,1] | ~0.95 | Exp |
| Re-entrant 1 (6) HyperExp | 6 | 2 | same | same | [1,...,1] | ~0.95 | HyperExp(0.5) |
| ... | ... | ... | ... | ... | ... | ... | ... |
| Re-entrant 1 (30) | 30 | 10 | see text | see text | [1,...,1] | ~0.95 | Exp |

For large networks, reference a full parameter table in supplementary material or describe the systematic construction rule (e.g., "Service rates follow a 2-server, 3-queue block pattern repeated K times").

### Table A2: Training Hyperparameters

Source: `configs/model/ppg_softmax.yaml` and `PPO/configs/wc_softmax.yaml`

| Parameter | PATHWISE | PPO-WC | REINFORCE (gradient comparison) |
|-----------|----------|--------|-------------------------------|
| Policy architecture | 3-layer MLP, 128 hidden, ReLU | Same | N/A (soft priority vector) |
| Learning rate | 3e-4 (Adam, betas=[0.8, 0.9]) | 9e-4 policy, 3e-4 value | N/A |
| Gradient clipping | norm 1.0 | norm 1.0 | N/A |
| STE temperature beta | 0.1 (training), 10 (Section 7) | N/A | N/A |
| Discount factor gamma | N/A (undiscounted) | 0.998 | 0.999 |
| GAE lambda | N/A | 0.99 | N/A |
| PPO clip range | N/A | 0.2 | N/A |
| PPO epochs per batch | N/A | 3-4 | N/A |
| Entropy coefficient | N/A | varies by config | N/A |
| Actors (parallel envs) | 1 | 50 | N/A |
| Train horizon (per epoch) | 10k-30k (varies by network) | 50k per actor | 1000 |
| Evaluation horizon | 200k-400k | 200k | 1000 |
| Evaluation episodes | 100 | 100 | 100 |
| Training epochs | 100 | 100 | N/A |

### Table A3: REINFORCE Baseline Details

**For gradient comparison experiments** (Section 5.1):
```
Discount factor: gamma = 0.999
Batch size: B = 1000
No value baseline (raw REINFORCE with discounted returns)
```

**For CMU rule experiments** (Section 5.2):
```
Discount factor: gamma = 0.99
Batch size: B = 100
Value baseline: 2-layer MLP (128 hidden, ReLU)
  - Trained via MSE on discounted returns
  - 3 inner epochs per policy iteration
  - Learning rate: 3e-4 (Adam)
  - Return normalization: (return - mean) / std
```

**For Section 7 benchmarks (PPO-DG from Dai & Gluzman 2022)**:
```
Follow Dai & Gluzman's implementation exactly.
Includes: behavior cloning initialization, advantage normalization,
          value function clipping, learning rate annealing.
```

### Writing Fix: Non-Stationary Systems

**AE Minor 10**: Remove "non-stationary" claims from Abstract and Introduction, OR add a brief experiment/discussion showing the method works with time-varying arrival rates.

Option A (remove): Delete mentions of "non-stationary" from abstract and intro.

Option B (demonstrate): The config system already supports time-varying arrivals:
- `lam_type: 'step'` — arrival rate changes at time t_step
- `lam_type: 'sawtooth'` — periodic ramp in arrival rate

Add a small experiment or appendix figure showing PATHWISE works with `reentrant_2_varying.yaml` (step or sawtooth arrivals). This is a 1-paragraph addition with 1 figure.

**Recommendation**: Option B — it's a strength of the method and configs already exist.

### Writing Fix: Terminology

**AE Minor 1**: Change "generalized likelihood-ratio estimation [39]" → "likelihood-ratio estimation [39]". Add citation to Peng et al. (2018) for the actual GLR method.

**AE Minor 6**: Standardize to "RL" after first definition of "reinforcement learning".

**AE Minor 7**: Standardize to "discrete-event simulation" (with hyphen) throughout.

**AE Minor 4**: Page 26 Proposition 2 — remove duplicate "with".

**AE Minor 5**: Page 33 line 38 — fix "a infinitesimal" → "an infinitesimal".

## Compute

Zero. This is pure writing and data extraction.

## Files to Read

| File | What to Extract |
|------|----------------|
| `configs/env/criss_cross_IID.yaml` | lambda, mu, h for criss-cross |
| `configs/env/criss_cross_hyper.yaml` | HyperExp parameters |
| `configs/env/reentrant_K.yaml` (K=2..10) | env_type, train_T, test_T, service_type |
| `env_data/reentrant_K/reentrant_K_lam.npy` | Arrival rates |
| `env_data/reentrant_K/reentrant_K_mu.npy` | Service rates |
| `env_data/reentrant_K/reentrant_K_network.npy` | Network topology |
| `configs/model/ppg_softmax.yaml` | All training hyperparameters |
| `PPO/configs/wc_softmax.yaml` | PPO hyperparameters |
| `PPO/configs/softmax.yaml` | PPO baseline hyperparameters |
| `experiments/gradient_comparison.py` lines 80-82 | gamma=0.999 for REINFORCE |
| `experiments/cmu_rule_REINFORCE.py` lines 242-355 | Value baseline construction |
