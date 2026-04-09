# E2: Generalized Likelihood Ratio (GLR) Comparison

## Reviewer Requirement

**AE Major Comment 3**: "The paper's treatment of likelihood ratio methods requires significant revision... A valuable addition to the paper would be a comparative analysis between the proposed method and the GLR method in a setting where both approaches are applicable."

**AE Minor 1**: "The statement 'Generalized likelihood-ratio estimation [39]...' misattributes the GLR method. Reference [39] (1987) describes the classic LR method, not GLR (Peng et al. 2018)."

## Scope Decision

GLR (Peng et al. 2018) requires known event-time distributions and bespoke per-network derivation. We implement GLR on **M/M/1 only** where it has a clean closed-form, and argue textually that PATHWISE is more scalable.

## Objective

1. Implement GLR gradient estimator for M/M/1 service rate control
2. Compare gradient MSE: PATHWISE vs GLR vs REINFORCE across rho and horizon T
3. Write discussion paragraph positioning PATHWISE as "generalized IPA" that handles non-differentiable sample paths via STE

## New File

`experiments/glr_comparison.py`

## Background: GLR for M/M/1

For the M/M/1 queue with arrival rate lambda, service rate mu (control parameter):
- Two event types: arrival (rate lambda) and departure (rate mu * 1{x>0})
- When x > 0, event rates: r_A = lambda, r_S = mu
- P(next event = departure | x > 0) = mu / (lambda + mu)
- P(next event = arrival | x > 0) = lambda / (lambda + mu)

**GLR score function** at event k (when x_k > 0):
```
If event k is a departure:
  score_k = d/dmu log(mu/(lambda+mu)) = lambda / (mu * (lambda + mu))
If event k is an arrival:
  score_k = d/dmu log(lambda/(lambda+mu)) = -1 / (lambda + mu)
If x_k = 0 (only arrivals possible):
  score_k = 0  (no dependence on mu)
```

**GLR gradient estimator**:
```
nabla_mu Q_N = (1/N) * sum_{k=0}^{N-1} score_k * cost_to_go_k
where cost_to_go_k = sum_{j=k}^{N-1} x_j * tau_{j+1}
```

## Algorithm

```python
def glr_gradient_mm1(lambda_rate, mu, T, seed):
    """GLR gradient estimator for M/M/1 steady-state queue length w.r.t. mu."""
    rng = np.random.default_rng(seed)
    
    x = 0  # queue length
    total_cost = 0.0
    scores = []
    costs_at_event = []
    
    for k in range(T):
        # Event rates
        rate_arrival = lambda_rate
        rate_service = mu if x > 0 else 0
        total_rate = rate_arrival + rate_service
        
        # Inter-event time
        tau = rng.exponential(1.0 / total_rate)
        total_cost += x * tau
        
        # Event selection
        if rng.random() < rate_arrival / total_rate:
            # Arrival
            x += 1
            if x > 1:  # was non-empty before
                score_k = -1.0 / (lambda_rate + mu)
            else:  # was empty, service rate irrelevant
                score_k = 0.0
        else:
            # Departure
            x -= 1
            score_k = lambda_rate / (mu * (lambda_rate + mu))
        
        scores.append(score_k)
        costs_at_event.append(x * tau)  # instantaneous cost contribution
    
    # Cost-to-go for each event
    cost_to_go = np.cumsum(costs_at_event[::-1])[::-1]
    
    # GLR gradient
    glr_grad = np.sum(np.array(scores) * cost_to_go) / T
    return glr_grad

def pathwise_gradient_mm1(lambda_rate, mu, T, seed):
    """IPA gradient for M/M/1 via Lindley recursion."""
    rng = np.random.default_rng(seed)
    
    W = 0.0       # waiting time
    dW_dmu = 0.0  # derivative of waiting time w.r.t. mu
    grad_sum = 0.0
    num_arrivals = 0
    
    for k in range(T):
        # Inter-arrival time
        inter_arrival = rng.exponential(1.0 / lambda_rate)
        # Workload
        S = rng.exponential(1.0)
        
        # Lindley recursion
        W_new = max(0, W - inter_arrival + S / mu)
        
        # Derivative recursion (IPA)
        if W_new > 0:
            dW_dmu = dW_dmu - S / (mu**2)
        else:
            dW_dmu = 0.0
        
        W = W_new
        grad_sum += dW_dmu
        num_arrivals += 1
    
    # By Little's law: dQ/dmu = lambda * (1/N) * sum dW_i/dmu
    return lambda_rate * grad_sum / num_arrivals

def reinforce_gradient_mm1(lambda_rate, mu, T, seed, h=0.1, theta=1.0):
    """REINFORCE gradient for M/M/1 with Beta(theta,1) service rate policy."""
    rng = np.random.default_rng(seed)
    
    # Sample service rate from policy: A = mu - h * Y, Y ~ Beta(theta, 1)
    Y = rng.beta(theta, 1)
    A = mu - h * Y  # actual service rate
    
    # Simulate M/M/1 with service rate A
    x = 0
    total_cost = 0.0
    for k in range(T):
        rate_total = lambda_rate + (A if x > 0 else 0)
        tau = rng.exponential(1.0 / rate_total)
        total_cost += x * tau
        if rng.random() < lambda_rate / rate_total:
            x += 1
        else:
            x = max(0, x - 1)
    
    avg_cost = total_cost / (T / (lambda_rate + A))  # normalize by time
    
    # Score function: d/dtheta log Beta(theta,1) = log(Y) + 1/theta
    score = np.log(Y) + 1.0 / theta
    
    return avg_cost * score


# Main experiment loop
for rho in [0.8, 0.9, 0.95, 0.99]:
    lambda_rate = rho  # fix mu = 1, lambda = rho
    mu = 1.0
    true_grad = -lambda_rate / (mu - lambda_rate)**2  # analytical: dQ/dmu
    
    for T in [100, 500, 1000, 5000]:
        glr_grads = [glr_gradient_mm1(lambda_rate, mu, T, seed=i) for i in range(NUM_TRIALS)]
        pw_grads = [pathwise_gradient_mm1(lambda_rate, mu, T, seed=i) for i in range(NUM_TRIALS)]
        rf_grads = [reinforce_gradient_mm1(lambda_rate, mu, T, seed=i) for i in range(NUM_TRIALS)]
        
        for name, grads in [('GLR', glr_grads), ('PATHWISE', pw_grads), ('REINFORCE', rf_grads)]:
            bias = mean(grads) - true_grad
            variance = var(grads)
            mse = bias**2 + variance
            save_result(rho, T, name, bias, variance, mse)
```

## Parameters

```python
RHO_VALUES = [0.8, 0.9, 0.95, 0.99]
HORIZONS = [100, 500, 1000, 5000]
NUM_TRIALS = 500
MU = 1.0  # fix service rate, vary lambda = rho * mu

# For REINFORCE: policy randomization range
H = 0.1  # service rate sampled from [mu-h, mu]
THETA = 1.0  # Beta(1,1) = Uniform

# Analytical ground truth
# dQ/dmu = -lambda / (mu - lambda)^2 = -rho / ((1-rho)^2 * mu)
```

## Output Format

```json
// File: results/glr_comparison.json
[
  {
    "rho": 0.95,
    "T": 1000,
    "method": "GLR",
    "bias": -0.12,
    "variance": 2.3,
    "mse": 2.31,
    "true_grad": -380.0
  },
  ...
]
```

## Expected Paper Output

### Table: Gradient MSE Comparison
| rho | T | PATHWISE MSE | GLR MSE | REINFORCE MSE | 
|-----|---|-------------|---------|---------------|
| 0.8 | 1000 | ... | ... | ... |
| 0.9 | 1000 | ... | ... | ... |
| 0.95 | 1000 | ... | ... | ... |
| 0.99 | 1000 | ... | ... | ... |

Expected: PATHWISE ~ GLR << REINFORCE. Both first-order methods should dominate.

### Figure: MSE vs Horizon T (at rho=0.95)
- 3 curves: PATHWISE, GLR, REINFORCE
- x-axis: T (log scale), y-axis: MSE (log scale)
- All should decrease with T, but REINFORCE much slower

### Discussion Text (for paper)
Key points to make:
1. GLR and PATHWISE achieve similar MSE on M/M/1 — both are first-order estimators
2. GLR requires: (a) known distributions (Exp), (b) per-network derivation of event probabilities, (c) case analysis of event orderings
3. PATHWISE requires: (a) a simulator (can be data-driven), (b) auto-differentiation library
4. For general networks (non-exponential, complex routing), GLR derivation is intractable; PATHWISE works out of the box
5. Position: PATHWISE is to IPA as GLR is to LR — both generalize their predecessors to non-differentiable sample paths, but via different mechanisms

## Compute Estimate

~20 core-hours. M/M/1 simulation is very fast (~0.01s per trajectory of T=5000).
500 trials x 4 rhos x 4 horizons x 3 methods = 24,000 runs, each ~0.01s = ~4 minutes single-core.
With parallelization: minutes.

## Writing Deliverables

1. Fix citation: change [39] from Glynn 1987 to "classic likelihood ratio method". Add Peng et al. 2018 as "generalized likelihood ratio (GLR)" with proper citation.
2. Add 1-2 paragraphs in Section 4 discussing GLR and the M/M/1 comparison.
3. Add comparison table/figure in appendix or Section 5.
