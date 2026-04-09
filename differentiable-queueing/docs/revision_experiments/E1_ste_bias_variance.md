# E1: STE Bias-Variance Analysis Beyond M/M/1

## Reviewer Requirement

**AE Major Comment 1**: "The current bias-variance trade-off analysis on pages 18-19 provides good insights, but is limited to one-step transition analysis of M/M/1 systems. Extending this analysis to more complex systems would substantially enhance the theoretical framework."

## Objective

Extend the bias-variance characterization of the straight-through estimator (STE) from the M/M/1 single-step setting (Theorem 1) to:
- Multiple environment types (M/M/1, criss-cross, multiclass, reentrant)
- Multi-step horizons (T = 1 to 1000)
- Multiple traffic intensities (rho = 0.9 to 0.99)
- Multiple STE temperatures (beta = 0.1 to 10.0)

## New File

`experiments/ste_bias_variance.py`

## Code Reuse

| Source | Function | Usage |
|--------|----------|-------|
| `gradient_comparison.py:320` | `compute_gt_multiprocessing(...)` | Ground truth gradient via large-batch REINFORCE |
| `gradient_comparison.py:168` | `compute_pathwise_grad(net, env_config, B, T, device)` | Single pathwise gradient sample |
| `gradient_comparison.py:84` | `_compute_reinforce_grad_core(net, env_config, B, T, gamma, device)` | Single REINFORCE gradient sample |
| `gradient_comparison.py:237` | `cosine_similarity(g1, g2)` | Cosine similarity metric |
| `gradient_comparison.py:52` | `get_policy(policy_type, s, q)` | Policy factory (sPR, sMW, sMP) |
| `cmu_step_rules_PATHWISE.py:140` | `build_env_config(queue_class, rho, gap)` | Programmatic env config for multiclass |

## Algorithm

```
Input: environments, horizons, rho_values, beta_values, num_theta, num_samples, gt_batch

For each environment E in environments:
  For each policy P in [sPR, sMW]:
    For each rho in rho_values:
      env_config = load_or_build_config(E, rho)
      
      For theta_idx in range(num_theta):
        theta = sample_random_theta(policy_type=P)  # LogNormal(0,1)
        net = get_policy(P, s, q)
        net.load_state_dict(theta)
        
        # Step 1: Compute ground truth gradient
        g_true = compute_gt_multiprocessing(
            net, env_config, 
            total_batch=gt_batch,     # 1,000,000
            T=1000,                   # fixed horizon for GT
            gamma=0.999, 
            num_cores=NUM_CORES
        )
        
        For each T in horizons:
          For each beta in beta_values:
            # Step 2: Collect PATHWISE gradient samples
            pathwise_grads = []
            for i in range(num_samples):
              g_pw = compute_pathwise_grad(net, env_config, B=1, T=T, 
                                           device='cpu', temp=beta)
              pathwise_grads.append(g_pw)
            
            # Step 3: Collect REINFORCE gradient samples
            reinforce_grads = []
            for i in range(num_samples):
              g_rf = _compute_reinforce_grad_core(net, env_config, B=1000, 
                                                   T=T, gamma=0.999, device='cpu')
              reinforce_grads.append(g_rf)
            
            # Step 4: Compute metrics
            pw_mean = mean(pathwise_grads)
            rf_mean = mean(reinforce_grads)
            
            pw_bias = norm(pw_mean - g_true)
            pw_variance = mean([norm(g - pw_mean)**2 for g in pathwise_grads])
            pw_mse = pw_bias**2 + pw_variance / 1  # B=1
            pw_cossim = mean([cosine_similarity(g, g_true) for g in pathwise_grads])
            
            rf_bias = norm(rf_mean - g_true)  # should be ~0 (unbiased)
            rf_variance = mean([norm(g - rf_mean)**2 for g in reinforce_grads])
            rf_mse = rf_bias**2 + rf_variance / 1000  # B=1000
            rf_cossim = mean([cosine_similarity(g, g_true) for g in reinforce_grads])
            
            # Step 5: Save
            results.append({
              'env': E, 'policy': P, 'rho': rho, 'theta_idx': theta_idx,
              'T': T, 'beta': beta,
              'pw_bias': pw_bias, 'pw_variance': pw_variance, 
              'pw_mse': pw_mse, 'pw_cossim': pw_cossim,
              'rf_bias': rf_bias, 'rf_variance': rf_variance,
              'rf_mse': rf_mse, 'rf_cossim': rf_cossim,
              'gt_norm': norm(g_true)
            })

Save results to JSON
```

## Parallelization Strategy

The bottleneck is ground truth computation (10^6 REINFORCE trajectories per theta).

```python
# Outer loop: parallelize over (env, policy, rho, theta_idx) tuples
# Each job computes GT + all (T, beta) combinations for one theta
# This avoids recomputing GT for different T and beta

import pathos.multiprocessing as mp

def worker(job):
    torch.set_num_threads(1)
    env, policy, rho, theta_idx, seed = job
    # ... compute GT once, then sweep T and beta ...
    return results_for_this_theta

jobs = [(env, pol, rho, idx, seeds[idx]) 
        for env in envs for pol in policies 
        for rho in rhos for idx in range(num_theta)]

with mp.ProcessingPool(NUM_CORES) as pool:
    all_results = pool.map(worker, jobs)
```

**Note**: GT computation itself should NOT be parallelized within a worker (it needs sequential trajectory simulation). Instead, use large batch size within `_compute_reinforce_grad_core` which is already vectorized.

## Parameters

```python
# Environments
ENVIRONMENTS = {
    'mm1': {'config': 'configs/env/mm1.yaml', 'rho_param': 'lam_params.val'},
    'criss_cross': {'config': 'configs/env/criss_cross_IID.yaml', 'rho_param': 'scale'},
    'multiclass_5': {'builder': 'build_env_config', 'queue_class': 5, 'gap': 0.5},
    'reentrant_2': {'config': 'configs/env/reentrant_2.yaml', 'rho_param': 'scale'},
}

# Sweep parameters
HORIZONS = [1, 10, 100, 500, 1000]
RHO_VALUES = [0.9, 0.95, 0.99]
BETA_VALUES = [0.1, 0.5, 1.0, 2.0, 10.0]
POLICIES = ['sPR', 'sMW']
NUM_THETA = 20
NUM_SAMPLES = 500  # gradient samples per (theta, T, beta)
GT_BATCH = 1_000_000
REINFORCE_BATCH = 1000
GAMMA = 0.999
NUM_CORES = 100
```

## Modification to `gradient_comparison.py:compute_pathwise_grad`

Current function uses a fixed temperature from env config. Need to add `temp` parameter override:

```python
# Current (line 168):
def compute_pathwise_grad(net, env_config, batch_size, T, device):
    dq = load_env(env_config, temp=1.0, ...)  # hardcoded temp

# Modified:
def compute_pathwise_grad(net, env_config, batch_size, T, device, temp=1.0):
    dq = load_env(env_config, temp=temp, ...)  # parameterized
```

## Output Format

```json
// File: results/ste_bias_variance_{env}_{policy}.json
[
  {
    "env": "criss_cross",
    "policy": "sMW",
    "rho": 0.95,
    "theta_idx": 0,
    "T": 1000,
    "beta": 1.0,
    "pw_bias": 0.023,
    "pw_variance": 0.15,
    "pw_mse": 0.0005,
    "pw_cossim": 0.95,
    "rf_bias": 0.001,
    "rf_variance": 12.5,
    "rf_mse": 0.0125,
    "rf_cossim": 0.42,
    "gt_norm": 3.14
  },
  ...
]
```

## Expected Paper Output

### Figure 1: Bias and Variance vs Horizon T
- 4 panels (one per environment), 2 curves per panel (PATHWISE bias, PATHWISE variance)
- x-axis: T (log scale), y-axis: metric value
- Show that bias stays bounded as T grows (key concern from AE)
- Include REINFORCE variance as reference (should grow or stay high)

### Figure 2: Bias and Variance vs Temperature beta
- Same structure, x-axis = beta
- Extends paper's existing Figure 5 (left panel) to multiple environments
- Show beta in [0.5, 2] is a sweet spot across all environments

### Table: MSE Decomposition
- Rows: environments x rho
- Columns: PATHWISE (bias, var, MSE), REINFORCE (bias, var, MSE)
- At fixed T=1000, beta=1.0

## Compute Estimate

- Per (env, policy, rho, theta): 1M REINFORCE trajectories x 1000 steps = 10^9 events for GT
- Total theta instances: 4 envs x 2 policies x 3 rhos x 20 thetas = 480
- Per theta, gradient samples: 500 x (5 horizons x 5 betas) = 12,500 (but each is B=1, fast)
- **Bottleneck**: GT computation. 480 instances x ~10 min each (100-core parallel) = ~80 hours elapsed
- With 100 cores: ~500 core-hours total

## Sanity Checks

1. At T=1, results should match Theorem 1's analytical expressions for M/M/1
2. REINFORCE bias should be ~0 at all settings (unbiased estimator)
3. PATHWISE cossim should be >= REINFORCE cossim (reproducing Fig 4 of paper)
4. Higher rho should increase both methods' variance (harder estimation)
