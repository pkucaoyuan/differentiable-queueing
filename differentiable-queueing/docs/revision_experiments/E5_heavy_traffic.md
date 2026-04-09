# E5: Heavy-Traffic Performance Curve (rho -> 1)

## Reviewer Requirement

**Referee 1**: "The theoretical analysis suggests that the heavy-traffic regime (i.e., rho -> 1) represents the most challenging case. I would be interested in seeing how the proposed method performs as rho increases."

## Objective

Plot PATHWISE vs REINFORCE policy optimization performance as a function of traffic intensity rho, validating Theorem 2's prediction that PATHWISE's advantage grows as rho->1 (variance ratio scales as (1-rho)^{-1}).

## New File

`experiments/heavy_traffic_curve.py`

## Code Reuse

This experiment directly extends the existing CMU rule optimization framework:

| Source | Function | Usage |
|--------|----------|-------|
| `cmu_step_rules_PATHWISE.py:140` | `build_env_config(queue_class, rho, gap)` | Build env config for any rho |
| `cmu_step_rules_PATHWISE.py:85` | `pathwise_cmu_step_rule(env_config, seed, ...)` | Pathwise optimization |
| `cmu_rule_REINFORCE.py:242` | `reinforce_value_cmu(env_config, seed, ...)` | REINFORCE with value baseline |
| `cmu_step_rules_PATHWISE.py:59` | `evaluate_iterate_fast(priority, env_config, ...)` | Fast evaluation |
| `step_rules.py:190` | `make_step_rule(name, alpha)` | Step rule factory |

## Algorithm

```python
import pathos.multiprocessing as mp
from cmu_step_rules_PATHWISE import build_env_config, pathwise_cmu_step_rule, evaluate_iterate_fast
from cmu_rule_REINFORCE import reinforce_value_cmu

# Parameters
RHO_GRID = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
QUEUE_CLASS = 10
GAPS = [0.01, 0.5, 1.0]
NUM_ITER = 50
T = 1000
EVAL_T = 20_000
NUM_TRIALS = 200
NUM_CORES = 80

# Best alphas (from existing experiments, or small grid search)
PATHWISE_ALPHA = 0.5
REINFORCE_ALPHA = 0.1
REINFORCE_BATCH = 100

def compute_cmu_optimal_cost(env_config, queue_class, gap):
    """Evaluate the known-optimal cmu-rule priority for this env."""
    # cmu-rule: priority proportional to h_j * mu_j = 1 * (1 + gap * j)
    priority = torch.tensor([[1 + gap * j for j in range(queue_class)]])
    return evaluate_iterate_fast(priority, env_config, batch=100, eval_T=50_000)

def run_single_trial(args):
    """Worker function for one (rho, gap, method, seed) combination."""
    torch.set_num_threads(1)
    rho, gap, method, seed = args
    env_config = build_env_config(QUEUE_CLASS, rho, gap)
    
    if method == 'pathwise':
        result = pathwise_cmu_step_rule(
            env_config, seed=seed, num_iter=NUM_ITER,
            step_rule_name='normalized_fixed', alpha=PATHWISE_ALPHA,
            temp=0.1, T=T, eval_T=EVAL_T
        )
    elif method == 'reinforce':
        result = reinforce_value_cmu(
            env_config, seed=seed, num_iter=NUM_ITER,
            alpha=REINFORCE_ALPHA, T=T, gamma=0.99,
            batch=REINFORCE_BATCH, eval_T=EVAL_T
        )
    
    # Also compute cmu-optimal for reference
    cmu_cost = compute_cmu_optimal_cost(env_config, QUEUE_CLASS, gap)
    
    return {
        'rho': rho, 'gap': gap, 'method': method, 'seed': seed,
        'final_cost': result['avg_cost'],
        'cmu_cost': cmu_cost,
        'normalized_cost': result['avg_cost'] / cmu_cost
    }

# Build job list
jobs = []
seeds = list(range(NUM_TRIALS))
for rho in RHO_GRID:
    for gap in GAPS:
        for method in ['pathwise', 'reinforce']:
            for seed in seeds:
                jobs.append((rho, gap, method, seed))

# Run in parallel
with mp.ProcessingPool(NUM_CORES) as pool:
    results = pool.map(run_single_trial, jobs)

# Aggregate
# For each (rho, gap, method): compute mean and std over seeds
aggregated = {}
for r in results:
    key = (r['rho'], r['gap'], r['method'])
    if key not in aggregated:
        aggregated[key] = []
    aggregated[key].append(r['normalized_cost'])

# Save
summary = []
for (rho, gap, method), costs in aggregated.items():
    summary.append({
        'rho': rho, 'gap': gap, 'method': method,
        'mean_normalized_cost': np.mean(costs),
        'std_normalized_cost': np.std(costs),
        'n_trials': len(costs)
    })

with open('results/heavy_traffic_curve.json', 'w') as f:
    json.dump(summary, f, indent=2)
```

## Parameters

```python
RHO_GRID = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
QUEUE_CLASS = 10          # 10-class single-server queue
GAPS = [0.01, 0.5, 1.0]  # service rate heterogeneity
NUM_ITER = 50             # gradient descent steps
T = 1000                  # simulation horizon per gradient step
EVAL_T = 20_000           # evaluation horizon
NUM_TRIALS = 200          # random seeds per (rho, gap, method)
NUM_CORES = 80

# Methods
PATHWISE: B=1, alpha=0.5, normalized_fixed step rule, temp=0.1
REINFORCE: B=100, alpha=0.1, value baseline (ValueNet 128 hidden), gamma=0.99
```

## Expected Paper Output

### Figure: Normalized Cost vs Traffic Intensity
```
x-axis: rho (0.80 to 0.99)
y-axis: avg_cost / cmu_optimal_cost (1.0 = matches cmu-rule)
Curves: PATHWISE (blue), REINFORCE (red), with shaded 95% CI
Panels: 3 panels for gap = 0.01, 0.5, 1.0
```

**Expected behavior**:
- Both methods achieve cost ratio < 1.0 at low rho (beating cmu-rule)
- As rho increases, both degrade (harder problem)
- **PATHWISE degrades more gracefully than REINFORCE** (smaller slope)
- Gap between methods widens as rho -> 1, consistent with Theorem 2

### Table: Cost Ratio at Key Traffic Intensities

| rho | gap | PATHWISE | REINFORCE | Ratio (RF/PW) |
|-----|-----|----------|-----------|---------------|
| 0.90 | 0.5 | ... | ... | ... |
| 0.95 | 0.5 | ... | ... | ... |
| 0.99 | 0.5 | ... | ... | ... |

The "Ratio" column should increase with rho, validating the theory.

## Compute Estimate

- Total jobs: 11 rhos x 3 gaps x 2 methods x 200 trials = 13,200 jobs
- Per PATHWISE job: 50 iter x 1000 steps = 50k events + eval 20k = 70k events
- Per REINFORCE job: 50 iter x 100 batch x 1000 steps = 5M events + eval 20k
- REINFORCE dominates compute: 6600 jobs x 5M = 33B events
- At ~1M events/sec/core: 33,000 core-seconds = ~550 core-minutes = ~9 core-hours
- With overhead (value net training etc): ~300 core-hours total
- On 80 cores: ~4 hours elapsed

## Sanity Checks

1. At rho=0.95 and gap=0.5, results should match existing paper results (Table 2, cmu_rule experiments)
2. cmu-rule cost should match analytical formula Q = rho/(1-rho) for single-class (gap=0 limit)
3. Normalized cost should approach 1.0 as rho increases (both methods struggle)
4. PATHWISE should always be <= REINFORCE (first-order dominates zeroth-order)

## Connection to Theory

This experiment directly visualizes the prediction from Theorem 2 (Section 8):
- PATHWISE variance ~ (1-rho)^{-3}
- REINFORCE variance ~ (1-rho)^{-4}
- Ratio: (1-rho)^{-1}, meaning REINFORCE needs (1-rho)^{-1} more samples

At rho=0.99: ratio = 100x. At rho=0.80: ratio = 5x. The performance gap curve should mirror this.
