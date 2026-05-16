"""
E4: Non-Work-Conserving Criss-Cross Experiment

Compares work-conserving (WC) vs non-work-conserving (nonWC) pathwise policy
optimization on the criss-cross network (Case IIB from Martins et al. 1996).

In the criss-cross network:
  - Server 1 serves queues 1 and 3 (shared)
  - Server 2 serves queue 2 (dedicated)
  - Jobs from queue 1 route to queue 2 after service; queues 2 and 3 exit.

WC forces servers to work whenever any compatible queue is non-empty.
nonWC allows intentional idling, which can be beneficial when serving one
queue feeds work into a bottleneck downstream.

Usage:
    cd experiments/
    python criss_cross_nonwc.py [--device cpu] [--num_seeds 10]
"""
import os
# Set thread limits BEFORE importing torch/numpy to prevent oversubscription
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import numpy as np
import json
import time
import copy
import sys
import argparse
from collections import defaultdict
import pathos.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from queuetorch.env import load_env, Obs, EnvState

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# Experiment Parameters
# =============================================================================

METHODS = ['PATHWISE-WC', 'PATHWISE-nonWC']
ENVS = ['criss_cross_IIB']
NUM_SEEDS = 10
TRAIN_T = 20_000
TEST_T = 200_000
NUM_EPOCHS = 100
EVAL_EVERY = 5
HIDDEN_DIM = 128
NUM_HIDDEN_LAYERS = 1       # Total layers: input -> hidden x NUM_HIDDEN_LAYERS -> output = 3 layer MLP
LR = 3e-4
GRAD_CLIP_NORM = 1.0
STE_TEMP = 0.1
TRAIN_BATCH = 1
TEST_BATCH = 10
NUM_CORES = int(os.environ.get("NSLOTS", min(4, os.cpu_count() or 1)))
HEATMAP_MAX_QUEUE = 50
HEATMAP_BASE_LEVEL = 5


# =============================================================================
# Policy Network
# =============================================================================


class SchedulingMLP(nn.Module):
    """
    3-layer MLP policy for the criss-cross network.

    Input: queue lengths (batch, q)
    Output: softmax allocation per server (batch, s, q)
    """

    def __init__(self, s, q, hidden_dim=128, num_hidden=1):
        super().__init__()
        self.s = s
        self.q = q

        layers = [nn.Linear(q, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, s * q)

    def forward(self, queues):
        """
        Args:
            queues: (batch, q) tensor of queue lengths.

        Returns:
            (batch, s, q) tensor with softmax probabilities per server.
        """
        batch = queues.size(0)
        x = self.trunk(queues)
        x = self.head(x)
        x = x.view(batch, self.s, self.q)
        return F.softmax(x, dim=2)


# =============================================================================
# Action Processing (WC vs nonWC)
# =============================================================================


def apply_wc_action(pr, queues, network, batch, s, q):
    """
    Work-conserving action: mask empty queues, redistribute probability
    so that servers never idle when compatible non-empty queues exist.

    Args:
        pr: (batch, s, q) raw softmax output from policy.
        queues: (batch, q) current queue lengths.
        network: (batch, s, q) server-queue compatibility.
        batch: batch size.
        s: number of servers.
        q: number of queues.

    Returns:
        (batch, s, q) processed action tensor.
    """
    # Mask by network structure
    pr = pr * network

    # Clip to queue lengths (soft WC: can't allocate more than what's in queue)
    pr = torch.min(
        torch.stack((pr, queues.unsqueeze(1).expand(-1, s, -1)), dim=3),
        dim=3
    ).values

    # If all compatible queues are empty for a server, spread uniformly over
    # the network to avoid division by zero (the env will clip to zero anyway).
    all_zero = torch.all(pr == 0.0, dim=2).reshape(batch, s, 1).expand(-1, -1, q)
    pr = pr + all_zero.float() * network

    # Re-normalize per server
    pr = pr / (torch.sum(pr, dim=-1, keepdim=True) + 1e-8)
    return pr


def apply_nonwc_action(pr, queues, network, batch, s, q):
    """
    Non-work-conserving action: only mask by network structure, do NOT
    redistribute probability away from empty queues.  If the policy
    allocates effort to an empty queue the server effectively idles
    (the env clips action * queue to zero).

    Args:
        pr: (batch, s, q) raw softmax output from policy.
        queues: (batch, q) current queue lengths.
        network: (batch, s, q) server-queue compatibility.
        batch: batch size.
        s: number of servers.
        q: number of queues.

    Returns:
        (batch, s, q) processed action tensor.
    """
    # Mask by network structure only
    pr = pr * network

    # Re-normalize per server (preserves allocation to empty queues)
    pr = pr / (torch.sum(pr, dim=-1, keepdim=True) + 1e-8)
    return pr


# =============================================================================
# Train / Evaluate
# =============================================================================


def train_one_epoch(net, optimizer, env_config, method, seed, device,
                    init_queues=None):
    """
    Run one training epoch (single trajectory, pathwise STE gradient).

    Args:
        net: SchedulingMLP policy network.
        optimizer: torch optimizer.
        env_config: environment config dict.
        method: 'PATHWISE-WC' or 'PATHWISE-nonWC'.
        seed: random seed for this epoch's trajectory.
        device: torch device string.
        init_queues: optional warm-start queue state (batch, q).

    Returns:
        train_cost (float): average cost rate over the training trajectory.
        final_queues: detached queue state at end of trajectory for warm-start.
    """
    is_wc = (method == 'PATHWISE-WC')
    dq = load_env(env_config, temp=STE_TEMP, batch=TRAIN_BATCH,
                  seed=seed, device=device)

    optimizer.zero_grad()

    if init_queues is not None:
        obs, state = dq.reset(seed=seed, init_queues=init_queues)
    else:
        obs, state = dq.reset(seed=seed)

    total_cost = torch.zeros(TRAIN_BATCH, 1, device=device)

    for _ in range(TRAIN_T):
        queues, t = obs
        pr = net(queues)

        if is_wc:
            action = apply_wc_action(pr, queues, dq.network,
                                     TRAIN_BATCH, dq.s, dq.q)
        else:
            action = apply_nonwc_action(pr, queues, dq.network,
                                        TRAIN_BATCH, dq.s, dq.q)

        obs, state, cost, event_time = dq.step(state, action)
        total_cost = total_cost + cost

    avg_cost = torch.mean(total_cost / state.time)
    avg_cost.backward()

    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP_NORM)
    optimizer.step()

    final_queues = obs.queues.detach()
    return float(avg_cost.detach()), final_queues


def evaluate(net, env_config, method, device, seed=42):
    """
    Evaluate current policy over a long horizon with no gradient tracking.

    Args:
        net: SchedulingMLP policy network.
        env_config: environment config dict.
        method: 'PATHWISE-WC' or 'PATHWISE-nonWC'.
        device: torch device string.
        seed: evaluation seed.

    Returns:
        mean_cost (float): mean cost rate across test batch.
        std_cost (float): std of cost rate across test batch.
    """
    is_wc = (method == 'PATHWISE-WC')
    dq = load_env(env_config, temp=STE_TEMP, batch=TEST_BATCH,
                  seed=seed, device=device)
    obs, state = dq.reset(seed=seed)

    total_cost = torch.zeros(TEST_BATCH, 1, device=device)

    with torch.no_grad():
        for _ in range(TEST_T):
            queues, t = obs
            pr = net(queues)

            if is_wc:
                action = apply_wc_action(pr, queues, dq.network,
                                         TEST_BATCH, dq.s, dq.q)
            else:
                action = apply_nonwc_action(pr, queues, dq.network,
                                            TEST_BATCH, dq.s, dq.q)

            obs, state, cost, event_time = dq.step(state, action)
            total_cost = total_cost + cost

    cost_rates = (total_cost / state.time).squeeze(1)
    return float(cost_rates.mean()), float(cost_rates.std())


# =============================================================================
# Heatmap: Server 1 allocation to queue 3 as f(x1, x2)
# =============================================================================


def generate_heatmap(net, method, env_name, seed_id, device='cpu',
                     max_queue=HEATMAP_MAX_QUEUE, base_level=HEATMAP_BASE_LEVEL):
    """
    Create a heatmap of server 1's allocation to queue 3 as a function of
    (x1, x2) with x3 held at base_level.

    Server 1 (index 0) serves queues 1 and 3 (indices 0 and 2).  The
    allocation to queue 3 is pr[0, 0, 2] after softmax.

    Args:
        net: trained SchedulingMLP.
        method: 'PATHWISE-WC' or 'PATHWISE-nonWC'.
        env_name: environment name string (for filename).
        seed_id: seed identifier (for filename).
        device: torch device string.
        max_queue: grid size for the heatmap axes.
        base_level: fixed value for queue 3 (x3).

    Returns:
        Z: (max_queue, max_queue) numpy array of allocation values.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    Z = np.zeros((max_queue, max_queue))

    with torch.no_grad():
        for i in range(max_queue):
            for j in range(max_queue):
                obs = torch.tensor([[float(i), float(j), float(base_level)]],
                                   device=device)
                pr = net(obs)  # (1, s, q)
                # Server 1 (idx 0) allocation to queue 3 (idx 2)
                Z[i, j] = pr[0, 0, 2].item()

    # Save figure
    fig_dir = os.path.join(RESULTS_DIR, 'E4_heatmaps')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir,
                            f'{env_name}_{method}_seed{seed_id}.png')

    plt.figure(figsize=(7, 6))
    plt.imshow(Z, origin='lower', aspect='auto',
               extent=[0, max_queue, 0, max_queue])
    plt.colorbar(label='Server 1 alloc to Queue 3')
    plt.xlabel('Queue 2 length (x2)')
    plt.ylabel('Queue 1 length (x1)')
    plt.title(f'{method} | Server 1 -> Q3 | x3={base_level}')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return Z


# =============================================================================
# Single-seed worker (for pathos multiprocessing)
# =============================================================================


def run_single_seed(args):
    """
    Train and evaluate one (env, method, seed) combination.

    Args:
        args: dict with keys env_name, method, seed, device.

    Returns:
        dict with learning curve and final results.
    """
    torch.set_num_threads(1)

    env_name = args['env_name']
    method = args['method']
    seed = args['seed']
    device = args['device']

    # Load env config
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'env',
                               f'{env_name}.yaml')
    import yaml
    with open(config_path, 'r') as f:
        env_config = yaml.safe_load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build network
    s = len(env_config['network'])
    q = len(env_config['network'][0])
    net = SchedulingMLP(s, q, hidden_dim=HIDDEN_DIM,
                        num_hidden=NUM_HIDDEN_LAYERS).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # Training loop
    learning_curve = []
    init_queues = None

    for epoch in range(NUM_EPOCHS):
        # Use a different seed per epoch for trajectory randomness
        epoch_seed = seed + epoch * 1000

        train_cost, init_queues = train_one_epoch(
            net, optimizer, env_config, method, epoch_seed, device,
            init_queues=init_queues
        )

        # Evaluate periodically
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            eval_mean, eval_std = evaluate(net, env_config, method, device)
            learning_curve.append({
                'epoch': epoch,
                'train_cost': train_cost,
                'eval_mean': eval_mean,
                'eval_std': eval_std,
            })
            print(f"  [{env_name}][{method}][seed={seed}] "
                  f"epoch {epoch:3d} | train={train_cost:.4f} "
                  f"| eval={eval_mean:.4f} +/- {eval_std:.4f}", flush=True)

    # Final evaluation
    final_mean, final_std = evaluate(net, env_config, method, device)

    # Generate heatmap for this seed (on CPU)
    net_cpu = net.cpu()
    heatmap = generate_heatmap(net_cpu, method, env_name, seed,
                               device='cpu')

    return {
        'env': env_name,
        'method': method,
        'seed': seed,
        'learning_curve': learning_curve,
        'final_eval_mean': final_mean,
        'final_eval_std': final_std,
        'heatmap': heatmap.tolist() if heatmap is not None else None,
    }


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description='E4: Non-Work-Conserving Criss-Cross Experiment'
    )
    parser.add_argument('--device', type=str, default='cpu',
                        help='Computation device (cpu or cuda)')
    parser.add_argument('--num_seeds', type=int, default=NUM_SEEDS,
                        help='Number of random seeds')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Training epochs per seed')
    parser.add_argument('--num_cores', type=int, default=NUM_CORES,
                        help='Number of parallel workers')
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    num_seeds = args.num_seeds
    num_cores = args.num_cores

    global NUM_EPOCHS
    NUM_EPOCHS = args.num_epochs

    seeds = list(range(num_seeds))

    print(f"{'=' * 70}")
    print(f"E4: Non-Work-Conserving Criss-Cross Experiment")
    print(f"  Environments : {ENVS}")
    print(f"  Methods      : {METHODS}")
    print(f"  Seeds        : {num_seeds}")
    print(f"  Epochs       : {NUM_EPOCHS}")
    print(f"  Train T      : {TRAIN_T}")
    print(f"  Test T       : {TEST_T}")
    print(f"  Eval every   : {EVAL_EVERY}")
    print(f"  LR           : {LR}")
    print(f"  Hidden dim   : {HIDDEN_DIM}")
    print(f"  STE temp     : {STE_TEMP}")
    print(f"  Cores        : {num_cores}")
    print(f"  Device       : {device}")
    print(f"{'=' * 70}\n")

    all_results = []
    start_time = time.time()

    # Build job list
    jobs = []
    for env_name in ENVS:
        for method in METHODS:
            for seed in seeds:
                jobs.append({
                    'env_name': env_name,
                    'method': method,
                    'seed': seed,
                    'device': device,
                })

    total_jobs = len(jobs)
    print(f"Total jobs: {total_jobs}")

    # Run in parallel with pathos
    with mp.ProcessingPool(num_cores) as pool:
        results = pool.map(run_single_seed, jobs)

    all_results.extend(results)

    elapsed = time.time() - start_time
    print(f"\nAll jobs completed in {elapsed / 60:.1f} minutes.\n")

    # =================================================================
    # Summary statistics
    # =================================================================
    print(f"{'=' * 70}")
    print("Summary: Final Evaluation Cost (mean +/- std across seeds)")
    print(f"{'=' * 70}")
    print(f"{'Env':<25} {'Method':<18} {'Mean Cost':>12} {'Std':>10} {'Seeds':>6}")
    print("-" * 70)

    summary = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        summary[r['env']][r['method']].append(r['final_eval_mean'])

    summary_records = []
    for env_name in ENVS:
        for method in METHODS:
            costs = summary[env_name][method]
            if costs:
                mean_c = np.mean(costs)
                std_c = np.std(costs)
                n = len(costs)
                print(f"{env_name:<25} {method:<18} {mean_c:>12.4f} {std_c:>10.4f} {n:>6}")
                summary_records.append({
                    'env': env_name,
                    'method': method,
                    'mean_cost': float(mean_c),
                    'std_cost': float(std_c),
                    'n_seeds': n,
                })

    # Check if nonWC beats WC
    for env_name in ENVS:
        wc_costs = summary[env_name].get('PATHWISE-WC', [])
        nonwc_costs = summary[env_name].get('PATHWISE-nonWC', [])
        if wc_costs and nonwc_costs:
            wc_mean = np.mean(wc_costs)
            nonwc_mean = np.mean(nonwc_costs)
            pct = 100.0 * (wc_mean - nonwc_mean) / wc_mean if wc_mean != 0 else 0
            direction = "nonWC wins" if nonwc_mean < wc_mean else "WC wins"
            print(f"\n  {env_name}: WC={wc_mean:.4f}, nonWC={nonwc_mean:.4f}, "
                  f"diff={pct:+.2f}% ({direction})")

    # =================================================================
    # Save results
    # =================================================================
    output_path = os.path.join(RESULTS_DIR, 'E4_criss_cross_nonwc.json')
    output = {
        'experiment': 'E4_criss_cross_nonwc',
        'params': {
            'envs': ENVS,
            'methods': METHODS,
            'num_seeds': num_seeds,
            'num_epochs': NUM_EPOCHS,
            'train_T': TRAIN_T,
            'test_T': TEST_T,
            'eval_every': EVAL_EVERY,
            'lr': LR,
            'hidden_dim': HIDDEN_DIM,
            'grad_clip_norm': GRAD_CLIP_NORM,
            'ste_temp': STE_TEMP,
            'train_batch': TRAIN_BATCH,
            'test_batch': TEST_BATCH,
        },
        'summary': summary_records,
        'trials': all_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save learning curves as a separate lighter file
    curves_path = os.path.join(RESULTS_DIR, 'E4_learning_curves.json')
    curves = []
    for r in all_results:
        curves.append({
            'env': r['env'],
            'method': r['method'],
            'seed': r['seed'],
            'learning_curve': r['learning_curve'],
        })
    with open(curves_path, 'w') as f:
        json.dump(curves, f, indent=2)
    print(f"Learning curves saved to {curves_path}")

    total_elapsed = time.time() - start_time
    print(f"\nTotal wall time: {total_elapsed / 3600:.2f} hours")
    print("Done!")


if __name__ == '__main__':
    main()
