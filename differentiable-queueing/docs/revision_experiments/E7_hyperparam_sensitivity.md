# E7: Hyperparameter Sensitivity

## Reviewer Requirement

**Referee 2**: "One of the key challenges of both PPO and REINFORCE is the sensitivity of the learned policy's performance to the choice of hyperparameters. It would be useful to see how sensitive the end-to-end policy optimization experiments are to the hyperparameters, in particular because I suspect that this will be a major strength of the proposed method relative to PPO / REINFORCE."

## Objective

Demonstrate that PATHWISE is robust to hyperparameter choices while REINFORCE is sensitive. Sweep one hyperparameter at a time (others fixed at defaults) and compare final policy quality.

## New File

`experiments/hyperparam_sensitivity.py`

## Code Reuse

Directly extend `experiments/cmu_step_rules_PATHWISE.py` pattern:
- `build_env_config(queue_class, rho, gap)` for env construction
- `pathwise_cmu_step_rule(env_config, seed, num_iter, step_rule, alpha, temp, T, eval_T)` for PATHWISE
- `reinforce_value_cmu(env_config, seed, num_iter, alpha, T, gamma, batch, eval_T)` for REINFORCE
- `evaluate_iterate_fast(priority, env_config, batch, eval_T)` for evaluation
- Parallelism via `pathos.multiprocessing.ProcessingPool`

## Algorithm

```python
import pathos.multiprocessing as mp
from cmu_step_rules_PATHWISE import build_env_config, pathwise_cmu_step_rule
from cmu_rule_REINFORCE import reinforce_value_cmu

# Fixed defaults
DEFAULTS = {
    'queue_class': 10,
    'rho': 0.95,
    'gap': 0.5,
    'num_iter': 50,
    'T': 1000,
    'eval_T': 20_000,
    'pathwise_alpha': 0.5,
    'pathwise_temp': 0.1,
    'reinforce_alpha': 0.1,
    'reinforce_batch': 100,
    'reinforce_gamma': 0.99,
}

# Axes to sweep (one at a time)
SWEEPS = {
    'temp': {
        'values': [0.001, 0.01, 0.1, 1.0, 10.0],
        'methods': ['pathwise'],  # only pathwise has temp
        'param_key': 'pathwise_temp',
    },
    'alpha': {
        'values': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        'methods': ['pathwise', 'reinforce'],
        'param_key': {'pathwise': 'pathwise_alpha', 'reinforce': 'reinforce_alpha'},
    },
    'horizon_T': {
        'values': [100, 500, 1000, 2000, 5000],
        'methods': ['pathwise', 'reinforce'],
        'param_key': 'T',
    },
    'batch_B': {
        'values': [1, 5, 10, 50],
        'methods': ['pathwise'],  # REINFORCE batch is separate concern
        'param_key': 'pathwise_batch',  # requires modification to pathwise function
    },
    'reinforce_batch': {
        'values': [10, 50, 100, 500, 1000],
        'methods': ['reinforce'],
        'param_key': 'reinforce_batch',
    },
}

NUM_TRIALS = 100
NUM_CORES = 60

def run_trial(args):
    torch.set_num_threads(1)
    sweep_name, sweep_value, method, seed = args
    
    # Start from defaults, override the swept parameter
    params = dict(DEFAULTS)
    if isinstance(SWEEPS[sweep_name]['param_key'], dict):
        params[SWEEPS[sweep_name]['param_key'][method]] = sweep_value
    else:
        params[SWEEPS[sweep_name]['param_key']] = sweep_value
    
    env_config = build_env_config(params['queue_class'], params['rho'], params['gap'])
    
    if method == 'pathwise':
        result = pathwise_cmu_step_rule(
            env_config, seed=seed, 
            num_iter=params['num_iter'],
            step_rule_name='normalized_fixed',
            alpha=params['pathwise_alpha'],
            temp=params['pathwise_temp'],
            T=params['T'],
            eval_T=params['eval_T']
        )
    else:  # reinforce
        result = reinforce_value_cmu(
            env_config, seed=seed,
            num_iter=params['num_iter'],
            alpha=params['reinforce_alpha'],
            T=params['T'],
            gamma=params['reinforce_gamma'],
            batch=params['reinforce_batch'],
            eval_T=params['eval_T']
        )
    
    return {
        'sweep': sweep_name, 'value': sweep_value, 
        'method': method, 'seed': seed,
        'cost': result['avg_cost']
    }

# Build all jobs
jobs = []
for sweep_name, sweep_cfg in SWEEPS.items():
    for value in sweep_cfg['values']:
        for method in sweep_cfg['methods']:
            for seed in range(NUM_TRIALS):
                jobs.append((sweep_name, value, method, seed))

# Run
with mp.ProcessingPool(NUM_CORES) as pool:
    all_results = pool.map(run_trial, jobs)

# Aggregate: for each (sweep, value, method), compute mean +/- std
# Save to results/hyperparam_sensitivity.json
```

## Parameters

```python
# Environment (fixed for all sweeps)
QUEUE_CLASS = 10
RHO = 0.95
GAP = 0.5
NUM_ITER = 50
EVAL_T = 20_000

# Sweep axes
TEMP_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]       # STE temperature
ALPHA_VALUES = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]  # learning rate
T_VALUES = [100, 500, 1000, 2000, 5000]             # simulation horizon
PW_BATCH_VALUES = [1, 5, 10, 50]                    # pathwise batch
RF_BATCH_VALUES = [10, 50, 100, 500, 1000]          # reinforce batch

NUM_TRIALS = 100
NUM_CORES = 60
```

## Output Format

```json
// results/hyperparam_sensitivity.json
[
  {
    "sweep": "alpha",
    "value": 0.01,
    "method": "pathwise",
    "mean_cost": 25.3,
    "std_cost": 1.2,
    "n_trials": 100
  },
  ...
]
```

## Expected Paper Output

### Figure: 4 Panels (one per sweep axis)

```
Panel 1: "STE Temperature (beta)"
  x-axis: beta values [0.001 ... 10.0] (log scale)
  y-axis: final average cost
  Curve: PATHWISE only (REINFORCE has no beta)
  Expected: flat curve — PATHWISE is insensitive to beta in [0.01, 1.0]

Panel 2: "Learning Rate (alpha)"
  x-axis: alpha values [0.001 ... 1.0] (log scale)
  y-axis: final average cost
  Curves: PATHWISE (blue, flat), REINFORCE (red, U-shaped)
  Expected: PATHWISE robust; REINFORCE has narrow optimal range

Panel 3: "Simulation Horizon (T)"
  x-axis: T values [100 ... 5000]
  y-axis: final average cost
  Curves: PATHWISE (flat after T=500), REINFORCE (needs T>2000)
  Expected: PATHWISE converges with shorter horizons

Panel 4: "Batch Size (B)"
  x-axis: batch size
  y-axis: final average cost
  Curves: PATHWISE B∈{1..50} (flat from B=1), REINFORCE B∈{10..1000} (decreasing)
  Expected: PATHWISE needs only B=1; REINFORCE needs B>>1
```

### Key Message for Paper

> PATHWISE achieves near-optimal performance across a wide range of hyperparameters
> (temperature, learning rate, horizon, batch size), whereas REINFORCE is sensitive to
> learning rate and requires large batch sizes. This robustness is a practical advantage:
> PATHWISE requires minimal hyperparameter tuning, unlike PPO/REINFORCE which
> require careful tuning of learning rate, clipping range, discount factor, and batch size.

## Compute Estimate

Total jobs:
- temp: 5 values x 1 method x 100 trials = 500
- alpha: 6 values x 2 methods x 100 trials = 1,200
- T: 5 values x 2 methods x 100 trials = 1,000
- pw_batch: 4 values x 1 method x 100 trials = 400
- rf_batch: 5 values x 1 method x 100 trials = 500
- **Total: 3,600 jobs**

REINFORCE jobs (~1,200) dominate compute at ~5M events each.
~200 core-hours total. On 60 cores: ~3.5 hours elapsed.

## Sanity Checks

1. Default parameters (alpha=0.5/0.1, T=1000, temp=0.1) should match existing results
2. Extreme values (alpha=0.001, T=100) should show worse performance for both methods
3. PATHWISE at B=1 should match B=50 (single trajectory sufficient)
4. REINFORCE at B=10 should be much worse than B=1000
