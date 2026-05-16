"""
E6: 3-way factorial ablation experiment.

Disentangles three binary factors contributing to PATHWISE vs REINFORCE gap:
    1. Action space:       continuous vs discrete
    2. Policy type:        deterministic vs stochastic
    3. Gradient estimator: first-order vs zeroth-order

The 2x2x2 cube has 8 corners but only 6 are sensible combinations:
    1. PATHWISE              (cont, det,   1st-order)  — existing baseline
    2. Cont-Det-SPSA         (cont, det,   0th-order)  — SPSA perturbation
    3. Cont-Stoch-Reparam    (cont, stoch, 1st-order)  — Gaussian reparameterization
    4. Cont-Stoch-RF         (cont, stoch, 0th-order)  — Gaussian REINFORCE
    5. Disc-Stoch-GumbelSTE  (disc, stoch, 1st-order)  — Gumbel-Softmax STE
    6. REINFORCE              (disc, stoch, 0th-order)  — existing baseline
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
import pathos.multiprocessing as mp
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from queuetorch.env import load_env
from cmu_step_rules_PATHWISE import build_env_config, evaluate_iterate_fast
import torch.distributions.one_hot_categorical as one_hot_sample

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
SEEDS_FILE = os.path.join(PROJECT_ROOT, 'cmu', 'seeds_cmu_10class.json')

# ── Experiment parameters ──
QUEUE_CLASS = 10
RHO = 0.95
GAP = 0.5
ALPHAS = [0.01, 0.1, 0.5, 1.0]
NUM_ITER = 50
T = 1000
EVAL_T = 20_000
NUM_TRIALS = 200
NUM_CORES = int(os.environ.get("NSLOTS", min(4, os.cpu_count() or 1)))

# ── Method-specific parameters ──
PATHWISE_TEMP = 1e-6
SPSA_PERTURBATION = 0.1
GAUSSIAN_NOISE_STD = 0.1
GUMBEL_TEMP = 1.0
REINFORCE_BATCH = 100
REINFORCE_GAMMA = 0.99
POLICY_TEMP = 5.0

# ── Method registry ──
METHODS = [
    'pathwise',
    'cont_det_spsa',
    'cont_stoch_reparam',
    'cont_stoch_rf',
    'disc_stoch_gumbel_ste',
    'reinforce',
]

OUTPUT_FILE = os.path.join(RESULTS_DIR, 'E6_ablation_3way.json')


# ─────────────────────────────────────────────────────────────────────────────
# Helper: discounted returns for REINFORCE-style methods
# ─────────────────────────────────────────────────────────────────────────────

def calculate_returns(costs, gamma, normalize=True):
    """Compute discounted returns from a list of per-step cost tensors."""
    returns = []
    R = 0
    for c in reversed(costs):
        R = c + R * gamma
        returns.insert(0, R)
    cat_returns = torch.cat(returns)
    if normalize and cat_returns.std() > 1e-8:
        for i in range(len(returns)):
            returns[i] = (returns[i] - cat_returns.mean()) / (cat_returns.std() + 1e-8)
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: PATHWISE  (continuous, deterministic, first-order)
# ─────────────────────────────────────────────────────────────────────────────

def run_pathwise(env_config, seed, alpha, num_iter, T, eval_T):
    """Standard pathwise gradient through the differentiable simulator."""
    torch.set_num_threads(1)

    dq = load_env(env_config, temp=PATHWISE_TEMP, batch=1, seed=seed, device='cpu')
    priority = torch.zeros((1, dq.q)).float()
    sum_priority = priority.clone()
    priority.requires_grad = True
    num = 1

    for i in range(num_iter):
        dq = load_env(env_config, temp=PATHWISE_TEMP, batch=1,
                       seed=seed + i, device='cpu')

        if i > 0:
            obs, state = dq.reset(seed=seed + i, init_queues=init_queues)
        else:
            obs, state = dq.reset(seed=seed)

        total_cost = torch.tensor([[0.]] * dq.batch)

        for _ in range(T):
            queues, t_sim = obs
            pr = F.softmax(priority.repeat(dq.batch, dq.s, 1), -1) * dq.network
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr += 1 * torch.all(pr == 0., dim=2).reshape(dq.batch, dq.s, 1) * dq.network
            pr /= torch.sum(pr, dim=-1).reshape(dq.batch, dq.s, 1)

            action = pr  # continuous, deterministic
            obs, state, cost, event_time = dq.step(state, action)
            total_cost += cost

        init_queues = queues.detach()
        avg_cost = torch.mean(total_cost / state.time)
        avg_cost.backward()

        grad = priority.grad.detach().clone()
        priority = priority.detach() - alpha * grad / (grad.norm() + 1e-8)

        sum_priority += priority.detach()
        num += 1
        priority.requires_grad = True

    avg_iterate = sum_priority / num
    final_cost = evaluate_iterate_fast(avg_iterate, env_config, eval_T=eval_T)
    return {'avg_cost': final_cost, 'last_iterate': avg_iterate.detach().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Cont-Det-SPSA  (continuous, deterministic, zeroth-order)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_cost_deterministic(priority, env_config, seed, T, init_queues=None):
    """Run a deterministic continuous-action trajectory, return scalar avg cost
    and final queues (both detached)."""
    dq = load_env(env_config, temp=PATHWISE_TEMP, batch=1, seed=seed, device='cpu')
    if init_queues is not None:
        obs, state = dq.reset(seed=seed, init_queues=init_queues)
    else:
        obs, state = dq.reset(seed=seed)

    total_cost = torch.tensor([[0.]] * dq.batch)
    with torch.no_grad():
        for _ in range(T):
            queues, t_sim = obs
            pr = F.softmax(priority.repeat(dq.batch, dq.s, 1), -1) * dq.network
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr += 1 * torch.all(pr == 0., dim=2).reshape(dq.batch, dq.s, 1) * dq.network
            pr /= torch.sum(pr, dim=-1).reshape(dq.batch, dq.s, 1)
            action = pr
            obs, state, cost, event_time = dq.step(state, action)
            total_cost += cost

    avg_cost = float(torch.mean(total_cost / state.time))
    return avg_cost, queues.detach()


def run_cont_det_spsa(env_config, seed, alpha, num_iter, T, eval_T):
    """SPSA: finite-difference gradient estimate via symmetric perturbation."""
    torch.set_num_threads(1)

    priority = torch.zeros((1, QUEUE_CLASS)).float()
    sum_priority = priority.clone()
    num = 1
    init_queues = None

    for i in range(num_iter):
        # Rademacher perturbation
        delta = (2 * torch.bernoulli(0.5 * torch.ones_like(priority)) - 1)
        c = SPSA_PERTURBATION

        p_plus = priority + c * delta
        p_minus = priority - c * delta

        # Use same seed for both perturbations so randomness cancels
        cost_plus, q_plus = _simulate_cost_deterministic(
            p_plus, env_config, seed + i, T, init_queues)
        cost_minus, q_minus = _simulate_cost_deterministic(
            p_minus, env_config, seed + i, T, init_queues)

        init_queues = q_plus  # carry forward queue state

        # SPSA gradient estimate: g_k = (f(x+c*delta) - f(x-c*delta)) / (2c * delta)
        grad = (cost_plus - cost_minus) / (2 * c) * (1.0 / delta)

        priority = priority - alpha * grad / (grad.norm() + 1e-8)

        sum_priority += priority.detach()
        num += 1

    avg_iterate = sum_priority / num
    final_cost = evaluate_iterate_fast(avg_iterate, env_config, eval_T=eval_T)
    return {'avg_cost': final_cost, 'last_iterate': avg_iterate.detach().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: Cont-Stoch-Reparam  (continuous, stochastic, first-order)
# ─────────────────────────────────────────────────────────────────────────────
# Noise is added in LOGIT space (before softmax) at each timestep.
# Gradient flows back through the noise via reparameterization trick.
# This is the first-order counterpart of Method 4.

def run_cont_stoch_reparam(env_config, seed, alpha, num_iter, T, eval_T):
    """Per-step Gaussian noise in logit space, gradient via reparameterization."""
    torch.set_num_threads(1)

    dq = load_env(env_config, temp=PATHWISE_TEMP, batch=1, seed=seed, device='cpu')
    priority = torch.zeros((1, dq.q)).float()
    sum_priority = priority.clone()
    priority.requires_grad = True
    num = 1

    for i in range(num_iter):
        dq = load_env(env_config, temp=PATHWISE_TEMP, batch=1,
                       seed=seed + i, device='cpu')

        if i > 0:
            obs, state = dq.reset(seed=seed + i, init_queues=init_queues)
        else:
            obs, state = dq.reset(seed=seed)

        torch.manual_seed(seed + i + 999999)
        total_cost = torch.tensor([[0.]] * dq.batch)

        for _ in range(T):
            queues, t_sim = obs

            # Noise in logit space (before softmax) — reparameterized
            logits = priority.repeat(dq.batch, dq.s, 1)
            noise = torch.randn_like(logits) * GAUSSIAN_NOISE_STD
            noisy_logits = logits + noise  # gradient flows through noise to priority

            pr = F.softmax(noisy_logits, -1) * dq.network
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr += 1 * torch.all(pr == 0., dim=2).reshape(dq.batch, dq.s, 1) * dq.network
            pr /= torch.sum(pr, dim=-1).reshape(dq.batch, dq.s, 1)

            action = pr  # continuous, stochastic (different each step)
            obs, state, cost, event_time = dq.step(state, action)
            total_cost += cost

        init_queues = queues.detach()
        avg_cost = torch.mean(total_cost / state.time)
        avg_cost.backward()

        grad = priority.grad.detach().clone()
        priority = priority.detach() - alpha * grad / (grad.norm() + 1e-8)

        sum_priority += priority.detach()
        num += 1
        priority.requires_grad = True

    avg_iterate = sum_priority / num
    final_cost = evaluate_iterate_fast(avg_iterate, env_config, eval_T=eval_T)
    return {'avg_cost': final_cost, 'last_iterate': avg_iterate.detach().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Method 4: Cont-Stoch-RF  (continuous, stochastic, zeroth-order)
# ─────────────────────────────────────────────────────────────────────────────
# Same randomization as Method 3 (per-step noise in logit space), but
# gradient estimated via score function instead of backprop.
# Uses batch of trajectories to reduce variance.

def run_cont_stoch_rf(env_config, seed, alpha, num_iter, T, eval_T):
    """Per-step Gaussian noise in logit space, gradient via score function."""
    torch.set_num_threads(1)

    priority = torch.zeros((1, QUEUE_CLASS)).float()
    sum_priority = priority.clone()
    num = 1
    init_queues = None

    for i in range(num_iter):
        dq = load_env(env_config, temp=PATHWISE_TEMP, batch=REINFORCE_BATCH,
                       seed=seed + i, device='cpu')

        if i > 0 and init_queues is not None:
            obs, state = dq.reset(seed=seed + i, init_queues=init_queues)
        else:
            obs, state = dq.reset(seed=seed)

        torch.manual_seed(seed + i + 777777)

        total_cost = torch.tensor([[0.]] * dq.batch)
        # Accumulate score across timesteps
        total_score = torch.zeros((REINFORCE_BATCH, QUEUE_CLASS))

        with torch.no_grad():
            for _ in range(T):
                queues, t_sim = obs

                # Per-step noise in logit space (same scheme as Method 3)
                logits = priority.repeat(dq.batch, dq.s, 1)
                noise = torch.randn(dq.batch, QUEUE_CLASS) * GAUSSIAN_NOISE_STD
                noisy_logits = logits + noise.unsqueeze(1).repeat(1, dq.s, 1)

                pr = F.softmax(noisy_logits, -1) * dq.network
                pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
                pr += 1 * torch.all(pr == 0., dim=2).reshape(dq.batch, dq.s, 1) * dq.network
                pr /= torch.sum(pr, dim=-1).reshape(dq.batch, dq.s, 1)

                action = pr
                obs, state, cost, event_time = dq.step(state, action)
                total_cost += cost

                # Accumulate Gaussian score: d/dmu log N(noise; 0, sigma^2) = noise / sigma^2
                # Since noisy_logits = priority + noise, score w.r.t. priority = noise / sigma^2
                total_score += noise / (GAUSSIAN_NOISE_STD ** 2)

        init_queues = queues[:1].detach()

        # Per-sample costs: (batch, 1)
        sample_costs = total_cost / state.time  # (batch, 1)

        # REINFORCE gradient: E[cost * cumulative_score / T]
        grad = torch.mean(sample_costs * total_score / T, dim=0, keepdim=True)

        priority = priority.detach() - alpha * grad / (grad.norm() + 1e-8)

        sum_priority += priority.detach()
        num += 1

    avg_iterate = sum_priority / num
    final_cost = evaluate_iterate_fast(avg_iterate.detach(), env_config, eval_T=eval_T)
    return {'avg_cost': final_cost, 'last_iterate': avg_iterate.detach().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Method 5: Disc-Stoch-GumbelSTE  (discrete, stochastic, first-order)
# ─────────────────────────────────────────────────────────────────────────────

def run_disc_stoch_gumbel_ste(env_config, seed, alpha, num_iter, T, eval_T):
    """Gumbel-Softmax with straight-through estimator for discrete actions."""
    torch.set_num_threads(1)

    dq = load_env(env_config, temp=PATHWISE_TEMP, batch=1, seed=seed, device='cpu')
    priority = torch.zeros((1, dq.q)).float()
    sum_priority = priority.clone()
    priority.requires_grad = True
    num = 1

    for i in range(num_iter):
        dq = load_env(env_config, temp=PATHWISE_TEMP, batch=1,
                       seed=seed + i, device='cpu')

        if i > 0:
            obs, state = dq.reset(seed=seed + i, init_queues=init_queues)
        else:
            obs, state = dq.reset(seed=seed)

        torch.manual_seed(seed + i + 555555)
        total_cost = torch.tensor([[0.]] * dq.batch)

        for _ in range(T):
            queues, t_sim = obs
            logits = priority.repeat(dq.batch, dq.s, 1)

            # Mask out empty queues by setting logits to -inf
            mask = (queues > 0.).unsqueeze(1).repeat(1, dq.s, 1) * dq.network
            masked_logits = logits + torch.log(mask + 1e-20)

            # Gumbel-Softmax: soft sample (differentiable)
            soft = F.gumbel_softmax(masked_logits, tau=GUMBEL_TEMP, hard=False)

            # Straight-through: hard forward, soft backward
            hard = F.one_hot(soft.argmax(dim=-1), num_classes=dq.q).float()
            action_ste = hard - soft.detach() + soft  # STE trick

            # Work-conserving adjustment
            action_ste = torch.minimum(action_ste,
                                        queues.unsqueeze(1).repeat(1, dq.s, 1))
            # Handle all-zero rows (idle server)
            idle_mask = torch.all(action_ste == 0., dim=2).reshape(dq.batch, dq.s, 1)
            action_ste = action_ste + idle_mask.float() * dq.network
            action_ste = action_ste / (torch.sum(action_ste, dim=-1, keepdim=True) + 1e-8)

            obs, state, cost, event_time = dq.step(state, action_ste)
            total_cost += cost

        init_queues = queues.detach()
        avg_cost = torch.mean(total_cost / state.time)
        avg_cost.backward()

        grad = priority.grad.detach().clone()
        priority = priority.detach() - alpha * grad / (grad.norm() + 1e-8)

        sum_priority += priority.detach()
        num += 1
        priority.requires_grad = True

    avg_iterate = sum_priority / num
    final_cost = evaluate_iterate_fast(avg_iterate, env_config, eval_T=eval_T)
    return {'avg_cost': final_cost, 'last_iterate': avg_iterate.detach().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Method 6: REINFORCE  (discrete, stochastic, zeroth-order)
# ─────────────────────────────────────────────────────────────────────────────

def run_reinforce(env_config, seed, alpha, num_iter, T, eval_T):
    """Discrete REINFORCE with baseline (discounted returns)."""
    torch.set_num_threads(1)

    dq_init = load_env(env_config, temp=1.0, batch=1, seed=seed, device='cpu')
    priority = torch.zeros((1, dq_init.q)).float()
    sum_priority = priority.clone()
    priority.requires_grad = True
    num = 1

    init_queues = None

    for i in range(num_iter):
        dq = load_env(env_config, temp=1.0, batch=REINFORCE_BATCH,
                       seed=seed + i, device='cpu')

        if i > 0 and init_queues is not None:
            obs, state = dq.reset(seed=seed + i, init_queues=init_queues)
        else:
            obs, state = dq.reset(seed=seed)

        torch.manual_seed(seed + i + 333333)

        log_prob_buffer = []
        costs = []

        for t_step in range(T):
            queues, t_sim = obs

            # Temperature-scaled softmax policy
            pr = F.softmax(POLICY_TEMP * priority.repeat(dq.batch, dq.s, 1), -1) * dq.network
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr += 1 * torch.all(pr == 0., dim=2).reshape(dq.batch, dq.s, 1) * dq.network
            pr /= torch.sum(pr, dim=-1).reshape(dq.batch, dq.s, 1)

            pr_dist = one_hot_sample.OneHotCategorical(probs=pr)
            action = pr_dist.sample()

            # Log probability for policy gradient
            log_prob = torch.sum(
                torch.log(torch.sum(action.detach() * pr, dim=2) + 1e-10),
                dim=1, keepdims=True
            )
            log_prob_buffer.append(log_prob)

            obs, state, cost, event_time = dq.step(state, action)
            costs.append(cost.detach())

        init_queues = queues[:1].detach()

        # Compute discounted returns
        returns = calculate_returns(costs, REINFORCE_GAMMA, normalize=True)

        # Policy gradient: sum_t [ return_t * log_pi(a_t|s_t) ]
        policy_loss = torch.tensor(0.)
        for t_step in range(len(returns)):
            policy_loss = policy_loss + returns[t_step] * log_prob_buffer[t_step]

        torch.mean(policy_loss).backward()

        grad = priority.grad.detach().clone()
        priority = priority.detach() - alpha * grad / (grad.norm() + 1e-8)

        sum_priority += priority.detach()
        num += 1
        priority.requires_grad = True

    avg_iterate = sum_priority / num
    final_cost = evaluate_iterate_fast(avg_iterate.detach(), env_config, eval_T=eval_T)
    return {'avg_cost': final_cost, 'last_iterate': avg_iterate.detach().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

METHOD_FN = {
    'pathwise':               run_pathwise,
    'cont_det_spsa':          run_cont_det_spsa,
    'cont_stoch_reparam':     run_cont_stoch_reparam,
    'cont_stoch_rf':          run_cont_stoch_rf,
    'disc_stoch_gumbel_ste':  run_disc_stoch_gumbel_ste,
    'reinforce':              run_reinforce,
}


def _run_job(kwargs):
    """Top-level worker entry point (must be picklable)."""
    torch.set_num_threads(1)
    method = kwargs.pop('method')
    return METHOD_FN[method](**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Seed management
# ─────────────────────────────────────────────────────────────────────────────

def load_seeds():
    if os.path.exists(SEEDS_FILE):
        with open(SEEDS_FILE, 'r') as f:
            seeds = json.load(f)
    else:
        os.makedirs(os.path.dirname(SEEDS_FILE), exist_ok=True)
        seeds = [int.from_bytes(os.urandom(4), 'big') for _ in range(10000)]
        with open(SEEDS_FILE, 'w') as f:
            json.dump(seeds, f)
    return seeds


# ─────────────────────────────────────────────────────────────────────────────
# Incremental save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_results):
    """Atomically write results JSON (write to tmp, then rename)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp_path = OUTPUT_FILE + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    os.replace(tmp_path, OUTPUT_FILE)


def load_existing_results():
    """Load previously saved results for resumption."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment():
    seeds = load_seeds()
    env_config = build_env_config(QUEUE_CLASS, RHO, GAP)

    # Build the full set of (method, alpha) combos
    combos = [(method, alpha) for method in METHODS for alpha in ALPHAS]
    total_combos = len(combos)
    total_jobs = total_combos * NUM_TRIALS

    print(f"{'=' * 70}", flush=True)
    print(f"E6: 3-way factorial ablation", flush=True)
    print(f"  QUEUE_CLASS={QUEUE_CLASS}  RHO={RHO}  GAP={GAP}", flush=True)
    print(f"  NUM_ITER={NUM_ITER}  T={T}  EVAL_T={EVAL_T}", flush=True)
    print(f"  {len(METHODS)} methods x {len(ALPHAS)} alphas x {NUM_TRIALS} trials"
          f" = {total_jobs} total jobs", flush=True)
    print(f"  Using {NUM_CORES} cores", flush=True)
    print(f"  Output: {OUTPUT_FILE}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Load any existing results for resumption
    all_results = load_existing_results()

    start_time = time.time()
    completed_combos = 0

    # Count already-completed combos
    for method, alpha in combos:
        alpha_str = str(alpha)
        if (method in all_results
                and alpha_str in all_results[method]
                and len(all_results[method][alpha_str]) >= NUM_TRIALS):
            completed_combos += 1

    skipped = completed_combos
    if skipped > 0:
        print(f"  Resuming: {skipped}/{total_combos} combos already complete.",
              flush=True)

    with mp.ProcessingPool(NUM_CORES) as pool:
        for method, alpha in combos:
            alpha_str = str(alpha)

            # Skip if already done
            if (method in all_results
                    and alpha_str in all_results[method]
                    and len(all_results[method][alpha_str]) >= NUM_TRIALS):
                continue

            # Build job list for this (method, alpha)
            jobs = []
            for trial_idx in range(NUM_TRIALS):
                jobs.append({
                    'method': method,
                    'env_config': copy.deepcopy(env_config),
                    'seed': seeds[trial_idx],
                    'alpha': alpha,
                    'num_iter': NUM_ITER,
                    'T': T,
                    'eval_T': EVAL_T,
                })

            combo_start = time.time()
            results = pool.map(_run_job, jobs)
            combo_elapsed = time.time() - combo_start

            # Store results
            if method not in all_results:
                all_results[method] = {}
            all_results[method][alpha_str] = results

            completed_combos += 1

            # Incremental save
            save_results(all_results)

            # Progress report
            total_elapsed = time.time() - start_time
            running_combos = completed_combos - skipped
            if running_combos > 0:
                avg_per_combo = total_elapsed / running_combos
                remaining = total_combos - completed_combos
                eta = avg_per_combo * remaining
            else:
                eta = 0

            # Summarize cost distribution for this combo
            costs = [r['avg_cost'] for r in results]
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)

            print(f"[{completed_combos}/{total_combos}] "
                  f"method={method:<25s} alpha={alpha:<5} "
                  f"| cost={mean_cost:.4f} +/- {std_cost:.4f} "
                  f"| combo: {combo_elapsed:.1f}s "
                  f"| total: {total_elapsed / 60:.1f}min "
                  f"| ETA: {eta / 60:.1f}min",
                  flush=True)

    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 70}", flush=True)
    print(f"E6 complete in {total_elapsed / 60:.1f} minutes", flush=True)
    print(f"Results saved to {OUTPUT_FILE}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Print summary table
    print(f"\n{'Method':<28s} {'Alpha':>6s} {'Mean Cost':>12s} {'Std':>10s}",
          flush=True)
    print('-' * 60, flush=True)
    for method in METHODS:
        if method not in all_results:
            continue
        for alpha_str in sorted(all_results[method].keys(),
                                key=lambda x: float(x)):
            costs = [r['avg_cost'] for r in all_results[method][alpha_str]]
            print(f"{method:<28s} {alpha_str:>6s} {np.mean(costs):>12.4f} "
                  f"{np.std(costs):>10.4f}", flush=True)


if __name__ == '__main__':
    run_experiment()
