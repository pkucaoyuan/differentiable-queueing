# E6: 3-Way Factorial Ablation

## Reviewer Requirement

**Referee 2**: "The proposed method actually introduces several changes relative to typical policy gradient approaches, which make it hard to tease out the real reason for the improvement: 1) The use of continuous vs. discrete control policy; 2) The use of deterministic rather than stochastic control policy; and 3) The use of first-order rather than zeroth-order gradients... The main additional ablation I would like to see is the comparison of a continuous, stochastic policy with first-order gradients against a continuous, stochastic policy with zeroth-order gradients."

Reference: Suh et al. (2022) "Do Differentiable Simulators Give Better Policy Gradients?"

## Objective

Disentangle three factors contributing to PATHWISE's improvement over REINFORCE:
1. **Continuous vs discrete** action space
2. **Deterministic vs stochastic** policy
3. **First-order vs zeroth-order** gradient estimation

## New File

`experiments/ablation_3way.py`

## The 6 Feasible Combinations

| # | Actions | Policy | Gradient | Name | Implementation |
|---|---------|--------|----------|------|----------------|
| 1 | Continuous | Deterministic | First-order | **PATHWISE** (paper method) | Soft actions, backprop through STE |
| 2 | Continuous | Deterministic | Zeroth-order | **Cont-Det-SPSA** | Soft actions, SPSA perturbation on theta |
| 3 | Continuous | Stochastic | First-order | **Cont-Stoch-Reparam** | Gaussian noise on theta, reparameterization trick |
| 4 | Continuous | Stochastic | Zeroth-order | **Cont-Stoch-RF** | Soft actions sampled, REINFORCE score function |
| 5 | Discrete | Stochastic | First-order | **Disc-Stoch-GumbelSTE** | Gumbel-Softmax STE |
| 6 | Discrete | Stochastic | Zeroth-order | **REINFORCE** (baseline) | Categorical sample, score function |

**Key comparison (R2's request)**: #3 vs #4 — same setup except gradient method.

**Note**: Discrete+Deterministic is degenerate (argmax has zero gradient), so combinations #5 uses Gumbel-Softmax which is stochastic.

## Implementation Details for Each Method

### Method 1: PATHWISE (existing)
```python
# From cmu_rule_REINFORCE.py:pathwise_cmu (line 107)
pr = F.softmax(priority / temp, dim=-1)
action = pr  # SOFT action, deterministic
# ... simulate T steps ...
loss = total_cost / T
loss.backward()  # backprop through STE in env.step()
grad = priority.grad
update = alpha * grad / grad.norm()
```

### Method 2: Cont-Det-SPSA (new)
```python
# Soft (continuous) actions, no backprop. SPSA gradient on theta.
eta = torch.randint(0, 2, priority.shape) * 2 - 1  # Rademacher {-1, +1}
perturbation = 0.1  # step size for finite difference

# Forward: J(theta + delta)
pr_plus = F.softmax((priority + perturbation * eta) / temp, dim=-1)
cost_plus = simulate_no_grad(env_config, pr_plus, T)

# Forward: J(theta - delta)
pr_minus = F.softmax((priority - perturbation * eta) / temp, dim=-1)
cost_minus = simulate_no_grad(env_config, pr_minus, T)

# SPSA gradient
spsa_grad = (cost_plus - cost_minus) / (2 * perturbation) * eta
update = alpha * spsa_grad / spsa_grad.norm()
```

### Method 3: Cont-Stoch-Reparam (new)
```python
# Add Gaussian noise to theta, backprop through reparameterization
noise_std = 0.1
epsilon = torch.randn_like(priority)  # reparameterization noise
perturbed_priority = priority + noise_std * epsilon  # reparameterized

pr = F.softmax(perturbed_priority / temp, dim=-1)
action = pr  # SOFT but stochastic (via noise)

# ... simulate T steps with STE ...
loss = total_cost / T
loss.backward()  # gradient flows through epsilon via reparam trick
grad = priority.grad
update = alpha * grad / grad.norm()
```

### Method 4: Cont-Stoch-RF (new)
```python
# Soft actions, but use score function gradient instead of backprop
noise_std = 0.1
epsilon = torch.randn_like(priority)
perturbed_priority = priority + noise_std * epsilon

pr = F.softmax(perturbed_priority / temp, dim=-1)
action = pr.detach()  # detach to prevent backprop through dynamics

# Simulate (no backprop needed through dynamics)
cost = simulate_no_grad(env_config, action, T)

# REINFORCE-style gradient via Gaussian score function
# d/d(theta) log N(theta + sigma*eps; theta, sigma^2 I) = eps / sigma
score = epsilon / noise_std
grad = cost * score  # REINFORCE with Gaussian parameterization
update = alpha * grad / grad.norm()
```

### Method 5: Disc-Stoch-GumbelSTE (new)
```python
# Gumbel-Softmax: sample discrete actions but use STE for gradient
gumbel_temp = 1.0
logits = priority / temp

# Forward: hard sample via Gumbel-max trick
gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
y_soft = F.softmax((logits + gumbel_noise) / gumbel_temp, dim=-1)
y_hard = F.one_hot(y_soft.argmax(dim=-1), num_classes=q).float()
action = y_hard - y_soft.detach() + y_soft  # STE: forward hard, backward soft

# ... simulate T steps with this action ...
loss = total_cost / T
loss.backward()
grad = priority.grad
update = alpha * grad / grad.norm()
```

### Method 6: REINFORCE (existing)
```python
# From cmu_rule_REINFORCE.py:reinforce_value_cmu (line 242)
pr = F.softmax(priority / policy_temp, dim=-1)
dist = torch.distributions.Categorical(probs=pr)
action_idx = dist.sample()  # discrete sample
log_prob = dist.log_prob(action_idx)
action = F.one_hot(action_idx, num_classes=q).float()

# Simulate
# ... accumulate cost and log_probs ...

# REINFORCE gradient with value baseline
advantage = returns - value_net(state)
policy_loss = (log_prob * advantage.detach()).sum()
policy_loss.backward()
grad = priority.grad
update = alpha * grad / grad.norm()
```

## Helper Functions

```python
def simulate_no_grad(env_config, action_fn_or_tensor, T, batch=1, seed=None):
    """Simulate T steps without gradient tracking. Returns average cost."""
    dq = load_env(env_config, temp=0.1, batch=batch, seed=seed)
    obs, state = dq.reset()
    total_cost = torch.zeros(batch)
    
    with torch.no_grad():
        for _ in range(T):
            queues, time = obs
            if callable(action_fn_or_tensor):
                action = action_fn_or_tensor(queues)
            else:
                action = action_fn_or_tensor
            # WC enforcement
            action = action * dq.network
            action = torch.minimum(action, queues.unsqueeze(1).repeat(1, dq.s, 1))
            mask = torch.all(action == 0, dim=2)
            action += mask.unsqueeze(-1) * dq.network
            action /= action.sum(dim=-1, keepdim=True)
            
            obs, state, cost, _ = dq.step(state, action)
            total_cost += cost.squeeze(1)
    
    return (total_cost / state.time).mean().item()
```

## Main Experiment Loop

```python
METHODS = {
    'pathwise':         {'actions': 'continuous', 'policy': 'deterministic', 'gradient': 'first_order'},
    'cont_det_spsa':    {'actions': 'continuous', 'policy': 'deterministic', 'gradient': 'zeroth_order'},
    'cont_stoch_reparam': {'actions': 'continuous', 'policy': 'stochastic', 'gradient': 'first_order'},
    'cont_stoch_rf':    {'actions': 'continuous', 'policy': 'stochastic', 'gradient': 'zeroth_order'},
    'gumbel_ste':       {'actions': 'discrete', 'policy': 'stochastic', 'gradient': 'first_order'},
    'reinforce':        {'actions': 'discrete', 'policy': 'stochastic', 'gradient': 'zeroth_order'},
}

QUEUE_CLASS = 10
RHO = 0.95
GAP = 0.5
ALPHAS = [0.01, 0.1, 0.5, 1.0]
NUM_ITER = 50
T = 1000
EVAL_T = 20_000
NUM_TRIALS = 200
NUM_CORES = 80

def run_trial(args):
    torch.set_num_threads(1)
    method_name, alpha, seed = args
    env_config = build_env_config(QUEUE_CLASS, RHO, GAP)
    
    # Initialize priority vector
    priority = torch.zeros(1, QUEUE_CLASS, requires_grad=True)
    running_sum = torch.zeros(1, QUEUE_CLASS)
    
    for iteration in range(NUM_ITER):
        if method_name == 'pathwise':
            grad = compute_pathwise_gradient(priority, env_config, T, seed+iteration)
        elif method_name == 'cont_det_spsa':
            grad = compute_spsa_gradient(priority, env_config, T, seed+iteration)
        elif method_name == 'cont_stoch_reparam':
            grad = compute_reparam_gradient(priority, env_config, T, seed+iteration)
        elif method_name == 'cont_stoch_rf':
            grad = compute_cont_reinforce_gradient(priority, env_config, T, seed+iteration)
        elif method_name == 'gumbel_ste':
            grad = compute_gumbel_ste_gradient(priority, env_config, T, seed+iteration)
        elif method_name == 'reinforce':
            grad = compute_reinforce_gradient(priority, env_config, T, seed+iteration)
        
        update = alpha * grad / (grad.norm() + 1e-8)
        priority = (priority.detach() - update).requires_grad_(True)
        running_sum += priority.detach()
    
    avg_priority = running_sum / NUM_ITER
    final_cost = evaluate_iterate_fast(avg_priority, env_config, batch=100, eval_T=EVAL_T)
    
    return {
        'method': method_name, 'alpha': alpha, 'seed': seed,
        'final_cost': final_cost
    }

# Build jobs
jobs = [(method, alpha, seed) 
        for method in METHODS 
        for alpha in ALPHAS 
        for seed in range(NUM_TRIALS)]

with mp.ProcessingPool(NUM_CORES) as pool:
    results = pool.map(run_trial, jobs)
```

## Parameters

```python
QUEUE_CLASS = 10
RHO = 0.95
GAP = 0.5
ALPHAS = [0.01, 0.1, 0.5, 1.0]
NUM_ITER = 50
T = 1000
EVAL_T = 20_000
NUM_TRIALS = 200
NUM_CORES = 80

# Method-specific
PATHWISE_TEMP = 0.1
SPSA_PERTURBATION = 0.1
GAUSSIAN_NOISE_STD = 0.1
GUMBEL_TEMP = 1.0
REINFORCE_BATCH = 100
REINFORCE_GAMMA = 0.99
POLICY_TEMP = 5.0  # for categorical sampling in REINFORCE
```

## Expected Paper Output

### Table: Ablation Results (Best Alpha per Method)

| # | Actions | Policy | Gradient | Best Cost | Std | Gap to Optimal |
|---|---------|--------|----------|-----------|-----|----------------|
| 1 | Cont | Det | 1st-order | **lowest** | low | ~0% |
| 2 | Cont | Det | 0th-order | medium | medium | ~5-10% |
| 3 | Cont | Stoch | 1st-order | low | low | ~2-5% |
| 4 | Cont | Stoch | 0th-order | medium | high | ~5-15% |
| 5 | Disc | Stoch | 1st-order | medium | medium | ~5-10% |
| 6 | Disc | Stoch | 0th-order | highest | highest | ~10-20% |

### Figure: Bar Chart
6 bars showing normalized cost for each method (with error bars). Group by action space and gradient order.

### Key Takeaway
Expected ranking: 1 > 3 > 5 > 2 > 4 > 6

Factor analysis:
- **First-order vs zeroth-order**: Compare (1 vs 2), (3 vs 4), (5 vs 6) — largest effect
- **Continuous vs discrete**: Compare (1 vs 5), (4 vs 6) — moderate effect
- **Deterministic vs stochastic**: Compare (1 vs 3), (2 vs 4) — smallest effect

## Compute Estimate

- 6 methods x 4 alphas x 200 trials = 4,800 jobs
- REINFORCE jobs (batch=100): ~5M events each → dominant cost
- PATHWISE/reparam jobs (batch=1): ~50k events each → fast
- Total: ~500 core-hours
- On 80 cores: ~6 hours elapsed

## Sanity Checks

1. Method #1 should reproduce existing pathwise results from paper
2. Method #6 should reproduce existing REINFORCE results from paper
3. Within each gradient order, methods should have similar performance (action space/policy type are secondary)
4. Across gradient orders, first-order should consistently dominate
