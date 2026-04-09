# E4: Non-Work-Conserving Policy on Criss-Cross IIB/IID

## Reviewer Requirement

**Referee 1**: "In the queueing literature, non-work-conserving policies can also be optimal under certain conditions. For instance, such policies are shown to be optimal in specific parameter regimes of criss-cross networks; see Cases IIB and IID in Martins et al. [1996] and Budhiraja et al. [2017]. Can the proposed method be generalized to handle these settings as well?"

**Referee 1 Remark**: "I believe the ability to leverage the known recursive form s_{k+1}=f(s_k,u_k,xi_{k+1}) is key to the success of the proposed method. Given this structure, I believe the method has the potential to discover non-work-conserving optimal policies as well."

## Background: Criss-Cross Network Cases

The criss-cross network (3 queues, 2 servers) has different optimal policy structures depending on parameters:

- **Server 1** serves queues 1 and 3
- **Server 2** serves queue 2 (dedicated)
- Jobs from queue 1 route to queue 2 after service; queues 2 and 3 exit

From Martins et al. (1996):
- **Case IIB**: Server 1 should **idle** (not serve queue 3) when queue 2 is congested, to avoid feeding more work downstream. The optimal policy is non-work-conserving.
- **Case IID**: Similar — idling is optimal in certain states.

Key insight: WC-Softmax forces server 1 to always serve some queue when non-empty queues exist. This prevents discovering the optimal idling policy.

## Objective

1. Define criss-cross IIB and IID parameter regimes
2. Train with and without WC constraint
3. Show that removing WC allows the method to discover idling policies
4. Compare final costs against known optimal policy structure

## New Files

- `experiments/criss_cross_nonwc.py` — main experiment script
- `configs/env/criss_cross_IIB.yaml` — IIB parameter regime
- `configs/env/criss_cross_IID.yaml` — IID parameter regime (if different from existing)

## Parameter Regimes

From Martins et al. (1996) Section 5, the criss-cross network parameters:

```python
# Common structure
n_queues = 3
n_servers = 2
network = [[1, 0, 1],   # server 1 serves queues 1, 3
           [0, 1, 0]]   # server 2 serves queue 2
R = [[-1, 0, 0],        # queue 1 completion: leave queue 1, enter queue 2
     [ 1,-1, 0],
     [ 0, 0,-1]]        # queue 3 completion: exit
h = [1, 1, 1]

# Case IIB: non-WC optimal
# Characterized by: mu_11 large, mu_13 small relative to lambda
# Server 1 should idle rather than serve queue 3 when queue 2 is long
criss_cross_IIB = {
    'lambda': [0.4, 0.0, 0.2],    # arrivals to queues 1 and 3
    'mu': [[2.0, 0, 0.5],         # server 1: fast for q1, slow for q3
           [0, 1.0, 0]],          # server 2: serves q2
    'h': [1, 1, 1],
}
# rho ~ lambda1/mu11 + lambda3/mu13 = 0.2 + 0.4 = 0.6 for server 1
# But feeding queue 1 → queue 2, server 2 is bottleneck

# Case IID: similar but with different rate structure  
criss_cross_IID = {
    'lambda': [0.3, 0.0, 0.35],
    'mu': [[1.5, 0, 0.6],
           [0, 0.8, 0]],
    'h': [1, 2, 1],  # higher cost for queue 2
}
```

**Note**: The exact parameters need to be calibrated to match Martins et al. (1996). The key property is that the heavy-traffic optimal policy involves server 1 idling in some states.

## Policy Architecture Modification

### Option A: Idle Action (Recommended)

Add an "idle" dimension to each server's softmax output:

```python
class PriorityNetWithIdle(nn.Module):
    """Policy network that can output idle action per server."""
    def __init__(self, q, s, layers=3, width=128):
        super().__init__()
        self.q = q
        self.s = s
        # Output: s * (q + 1) — extra dimension per server for idle
        self.fc_layers = nn.Sequential(
            nn.Linear(q, width), nn.ReLU(),
            *[nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(layers-2)],
            nn.Linear(width, s * (q + 1))
        )
    
    def forward(self, queues):
        batch = queues.shape[0]
        logits = self.fc_layers(queues).reshape(batch, self.s, self.q + 1)
        probs = F.softmax(logits, dim=-1)
        # First q dims = queue allocation, last dim = idle probability
        action = probs[:, :, :self.q]  # (batch, s, q)
        idle_prob = probs[:, :, self.q:]  # (batch, s, 1)
        # action already sums to < 1 (idle probability absorbed)
        return action  # capacity sharing: fractional allocation
```

### Option B: Remove WC Clipping Only (Simpler)

In `train_policy.py`, skip the WC enforcement:

```python
# Current (lines 423-428):
if train_policy == 'softmax':
    pr = pr * dq.network
    pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))  # WC clipping
    pr += 1*torch.all(pr == 0., dim=2).unsqueeze(-1).repeat(1,1,dq.q) * dq.network
    pr /= torch.sum(pr, dim=-1).unsqueeze(-1).repeat(1,1,dq.q)

# Non-WC version:
if train_policy == 'softmax':
    pr = pr * dq.network
    # Skip queue clipping — allow serving empty queues (= idling)
    pr /= torch.sum(pr, dim=-1).unsqueeze(-1).repeat(1,1,dq.q)
```

When server 1 allocates capacity to an empty queue, effective service rate = 0 for that allocation → server partially idles.

**Recommendation**: Use Option B for simplicity. The capacity sharing relaxation already allows fractional allocation; allocating to an empty queue is equivalent to idling.

## Algorithm

```python
METHODS = ['PATHWISE-WC', 'PATHWISE-nonWC']
ENVS = ['criss_cross_IIB', 'criss_cross_IID']
NUM_SEEDS = 10
TRAIN_T = 20_000
TEST_T = 200_000
NUM_EPOCHS = 100

for env_name in ENVS:
    env_config = load_yaml(f'configs/env/{env_name}.yaml')
    
    for method in METHODS:
        work_conserving = (method == 'PATHWISE-WC')
        
        for seed in range(NUM_SEEDS):
            # Initialize policy network
            net = PriorityNet(q=3, s=2, layers=3, width=128)
            optimizer = Adam(net.parameters(), lr=3e-4)
            
            for epoch in range(NUM_EPOCHS):
                # Training (STE)
                dq = load_env(env_config, temp=0.1, batch=1, seed=seed+epoch)
                obs, state = dq.reset()
                total_cost = 0
                
                for t in range(TRAIN_T):
                    queues, time = obs
                    pr = net(queues)
                    pr = pr * dq.network
                    
                    if work_conserving:
                        pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
                        mask = torch.all(pr == 0, dim=2)
                        pr += mask.unsqueeze(-1) * dq.network
                    
                    pr /= pr.sum(dim=-1, keepdim=True)
                    obs, state, cost, _ = dq.step(state, pr)
                    total_cost += cost
                
                loss = total_cost.mean() / TRAIN_T
                loss.backward()
                clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Evaluation every 5 epochs
                if epoch % 5 == 0:
                    eval_cost = evaluate(net, env_config, TEST_T, work_conserving)
                    log(env_name, method, seed, epoch, eval_cost)
            
            # Save final policy for analysis
            save_policy(net, env_name, method, seed)

# Post-hoc analysis: visualize learned policy
for env_name in ENVS:
    for method in METHODS:
        net = load_best_policy(env_name, method)
        # Create heatmap: for each (x1, x2, x3) state, show server 1 allocation
        plot_policy_heatmap(net, env_name, method)
```

## Policy Visualization

To show whether the learned policy idles:

```python
def plot_policy_heatmap(net, env_name, method):
    """Heatmap of server 1's allocation to queue 3 vs (x1, x2) state."""
    x1_range = range(0, 20)
    x2_range = range(0, 20)
    x3 = 5  # fixed
    
    alloc_q3 = np.zeros((len(x1_range), len(x2_range)))
    for i, x1 in enumerate(x1_range):
        for j, x2 in enumerate(x2_range):
            queues = torch.tensor([[x1, x2, x3]], dtype=torch.float32)
            pr = net(queues)
            pr = pr * network_mask
            pr /= pr.sum(dim=-1, keepdim=True)
            alloc_q3[i, j] = pr[0, 0, 2].item()  # server 1 → queue 3
    
    plt.imshow(alloc_q3, origin='lower', 
               extent=[0, 20, 0, 20], aspect='auto')
    plt.xlabel('Queue 2 length')
    plt.ylabel('Queue 1 length')
    plt.title(f'{method}: Server 1 allocation to Queue 3')
    plt.colorbar(label='Fraction of capacity')
```

For non-WC policy, we expect to see regions where allocation to queue 3 drops to ~0 when queue 2 is long (server 1 idles rather than feeding queue 2 via queue 1).

## Expected Paper Output

### Table: Final Average Holding Cost
| Environment | PATHWISE-WC | PATHWISE-nonWC | cmu-rule | MaxWeight | Known Optimal |
|-------------|-------------|----------------|----------|-----------|---------------|
| IIB | ... | ... (lower?) | ... | ... | Martins et al. |
| IID | ... | ... (lower?) | ... | ... | Martins et al. |

### Figure: Learning Curves
- 2 panels (IIB, IID), each with 2 curves (WC, nonWC) + error bands from 10 seeds

### Figure: Policy Heatmaps
- 4 panels: WC-IIB, nonWC-IIB, WC-IID, nonWC-IID
- Show that nonWC learns to idle in the correct states

## Compute Estimate

- Per (env, method, seed): 100 epochs x 20k steps = 2M events + 20 evals x 200k = 4M events
- Total: 2 envs x 2 methods x 10 seeds = 40 runs x 6M events = 240M events
- ~150 core-hours

## Key Risk

The exact parameters for Cases IIB and IID from Martins et al. (1996) need to be carefully verified. The paper defines cases by the relative magnitudes of service rates and arrival rates. If the parameter regime is wrong, the optimal policy might still be WC, and the experiment won't show the desired effect.

**Mitigation**: Read Martins et al. (1996) Section 5 carefully. Also check Budhiraja et al. (2017) for explicit parameter values used in their numerical examples.
