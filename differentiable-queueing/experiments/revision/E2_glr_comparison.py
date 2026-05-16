"""
E2: GLR Gradient Estimator Comparison on M/M/1

Compares gradient MSE of three gradient estimators for the M/M/1 queue:
  1. PATHWISE (IPA via Lindley recursion)
  2. GLR (Generalized Likelihood Ratio)
  3. REINFORCE (score function with Gaussian perturbation)

The control parameter is the service rate mu. The objective is mean queue
length E[Q] = rho/(1-rho). The analytical gradient is:
    dE[Q]/dmu = -lambda / (mu - lambda)^2

Usage:
    cd experiments/
    python glr_comparison.py
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import json
import sys
import time
import numpy as np
import pathos.multiprocessing as mp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Parameters ───────────────────────────────────────────────────────────────
RHO_VALUES = [0.8, 0.9, 0.95, 0.99]
HORIZONS = [100, 500, 1000, 5000]
NUM_TRIALS = 500
MU = 1.0
REINFORCE_SIGMA = 0.05
NUM_CORES = int(os.environ.get("NSLOTS", min(4, os.cpu_count() or 1)))


# ── Analytical ground truth ──────────────────────────────────────────────────

def analytical_gradient(lam, mu):
    """dE[Q]/dmu for M/M/1 with holding cost h=1.

    E[Q] = rho/(1-rho) where rho = lam/mu.
    dE[Q]/dmu = -lam / (mu - lam)^2
    """
    return -lam / (mu - lam) ** 2


# ── M/M/1 simulation via CTMC (event-driven) ────────────────────────────────

def simulate_mm1_ctmc(lam, mu, T, rng):
    """Simulate an M/M/1 queue via CTMC for T events.

    Returns:
        queue_at_event: array of queue lengths just before each event (length T)
        event_types:    array of +1 (arrival) or -1 (departure) for each event
        inter_event_times: array of sojourn times between events (length T)
    """
    x = 0  # current queue length
    queue_at_event = np.empty(T, dtype=np.float64)
    event_types = np.empty(T, dtype=np.int32)
    inter_event_times = np.empty(T, dtype=np.float64)

    for k in range(T):
        queue_at_event[k] = x
        if x == 0:
            # Only arrivals possible
            dt = rng.exponential(1.0 / lam)
            inter_event_times[k] = dt
            event_types[k] = 1
            x = 1
        else:
            total_rate = lam + mu
            dt = rng.exponential(1.0 / total_rate)
            inter_event_times[k] = dt
            if rng.random() < lam / total_rate:
                event_types[k] = 1   # arrival
                x += 1
            else:
                event_types[k] = -1  # departure
                x -= 1

    return queue_at_event, event_types, inter_event_times


def time_average_queue_length(queue_at_event, inter_event_times):
    """Compute time-averaged queue length from CTMC trace."""
    total_time = inter_event_times.sum()
    if total_time == 0:
        return 0.0
    return np.dot(queue_at_event, inter_event_times) / total_time


# ── IPA / PATHWISE via Lindley recursion ─────────────────────────────────────

def ipa_gradient(lam, mu, T, rng):
    """IPA gradient of average waiting time w.r.t. mu via Lindley recursion.

    Lindley: W_{n+1} = max(0, W_n + S_n/mu - A_n)
    where S_n ~ Exp(1) (unit-rate service), A_n ~ Exp(lam).

    dW/dmu: if W_new > 0: dW_new = dW_prev - S_n / mu^2
            else:          dW_new = 0

    The average queue length relates to average wait via Little's law:
        E[Q] = lam * E[W]
    so dE[Q]/dmu = lam * dE[W]/dmu.
    """
    W = 0.0
    dW = 0.0
    sum_dW = 0.0

    for n in range(T):
        S = rng.exponential(1.0)   # unit-rate service time
        A = rng.exponential(1.0 / lam)  # inter-arrival time

        W_new = W + S / mu - A
        if W_new > 0:
            dW = dW - S / (mu ** 2)
        else:
            W_new = 0.0
            dW = 0.0

        W = W_new
        sum_dW += dW

    # dE[Q]/dmu = lam * (1/T) * sum(dW)
    return lam * sum_dW / T


# ── GLR (Generalized Likelihood Ratio) ──────────────────────────────────────

def glr_gradient(lam, mu, T, rng):
    """GLR gradient estimator for M/M/1 w.r.t. service rate mu.

    For an M/M/1 with arrival rate lam and service rate mu, at each event k:
      - If queue x > 0 and event is a departure:
            score_k = lam / (mu * (lam + mu))      [= d/dmu log(mu/(lam+mu)) / rate]
      - If queue x > 0 and event is an arrival:
            score_k = -1 / (lam + mu)
      - If queue x == 0 (only arrivals possible):
            score_k = 0

    The GLR gradient of the time-average cost is:
        (1/total_time) * sum_k [ score_k * cost_to_go_k ]

    where cost_to_go_k = integral of queue length from event k to the end.
    We approximate cost_to_go_k as the time-weighted sum of future queue
    lengths: sum_{j>=k} queue_at_event[j] * inter_event_time[j].
    """
    queue_at_event, event_types, inter_event_times = simulate_mm1_ctmc(
        lam, mu, T, rng
    )

    total_time = inter_event_times.sum()
    if total_time == 0:
        return 0.0

    # Compute cost-to-go for each event (reverse cumulative sum)
    weighted_costs = queue_at_event * inter_event_times
    cost_to_go = np.cumsum(weighted_costs[::-1])[::-1]

    # Compute GLR scores
    scores = np.zeros(T, dtype=np.float64)
    for k in range(T):
        x = queue_at_event[k]
        if x > 0:
            if event_types[k] == -1:  # departure
                scores[k] = lam / (mu * (lam + mu))
            else:  # arrival while x > 0
                scores[k] = -1.0 / (lam + mu)
            # if x == 0: score stays 0

    grad = np.dot(scores, cost_to_go) / total_time
    return grad


# ── REINFORCE (score function with Gaussian perturbation) ────────────────────

def reinforce_gradient(lam, mu, T, rng, sigma=REINFORCE_SIGMA):
    """REINFORCE gradient via Gaussian perturbation on mu.

    mu_perturbed = mu + sigma * epsilon,  epsilon ~ N(0,1)
    score = epsilon / sigma
    Simulate M/M/1 with mu_perturbed, return cost * score.
    """
    epsilon = rng.standard_normal()
    mu_pert = mu + sigma * epsilon

    # Ensure mu_pert > lam for stability; if not, clamp
    if mu_pert <= lam or mu_pert <= 0:
        mu_pert = lam + 1e-4

    queue_at_event, _, inter_event_times = simulate_mm1_ctmc(lam, mu_pert, T, rng)
    cost = time_average_queue_length(queue_at_event, inter_event_times)
    score = epsilon / sigma

    return cost * score


# ── Worker functions for parallel execution ──────────────────────────────────

def worker_ipa(args):
    """Worker: single IPA trial."""
    lam, mu, T, seed = args
    rng = np.random.default_rng(seed)
    return ipa_gradient(lam, mu, T, rng)


def worker_glr(args):
    """Worker: single GLR trial."""
    lam, mu, T, seed = args
    rng = np.random.default_rng(seed)
    return glr_gradient(lam, mu, T, rng)


def worker_reinforce(args):
    """Worker: single REINFORCE trial."""
    lam, mu, T, seed = args
    rng = np.random.default_rng(seed)
    return reinforce_gradient(lam, mu, T, rng)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(estimates, true_grad):
    """Compute bias, variance, and MSE of gradient estimates."""
    estimates = np.array(estimates)
    mean_est = np.mean(estimates)
    bias = mean_est - true_grad
    variance = np.var(estimates, ddof=0)
    mse = np.mean((estimates - true_grad) ** 2)
    return {
        'mean': float(mean_est),
        'bias': float(bias),
        'variance': float(variance),
        'mse': float(mse),
        'std': float(np.std(estimates)),
    }


# ── Main experiment ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    start_time = time.time()

    print("=" * 70)
    print("E2: GLR Gradient Estimator Comparison on M/M/1")
    print("=" * 70)
    print(f"  rho values:  {RHO_VALUES}")
    print(f"  horizons:    {HORIZONS}")
    print(f"  trials:      {NUM_TRIALS}")
    print(f"  mu:          {MU}")
    print(f"  sigma (RF):  {REINFORCE_SIGMA}")
    print(f"  cores:       {NUM_CORES}")
    print("=" * 70)
    print()

    all_results = []
    base_seed = 42

    total_combos = len(RHO_VALUES) * len(HORIZONS)
    combo_idx = 0

    for rho in RHO_VALUES:
        lam = rho * MU
        true_grad = analytical_gradient(lam, MU)

        for T in HORIZONS:
            combo_idx += 1
            combo_start = time.time()
            print(f"[{combo_idx}/{total_combos}] rho={rho}, T={T}, "
                  f"lam={lam:.4f}, true dE[Q]/dmu={true_grad:.6f}")

            # Build job lists with unique seeds
            seeds = [base_seed + i for i in range(NUM_TRIALS)]
            base_seed += NUM_TRIALS  # advance seed base

            ipa_jobs = [(lam, MU, T, s) for s in seeds]
            glr_jobs = [(lam, MU, T, s + 10_000_000) for s in seeds]
            rf_jobs  = [(lam, MU, T, s + 20_000_000) for s in seeds]

            # Run all three methods in parallel
            with mp.ProcessingPool(NUM_CORES) as pool:
                ipa_results = pool.map(worker_ipa, ipa_jobs)

            t_ipa = time.time()
            print(f"    IPA done      ({t_ipa - combo_start:.1f}s)")

            with mp.ProcessingPool(NUM_CORES) as pool:
                glr_results = pool.map(worker_glr, glr_jobs)

            t_glr = time.time()
            print(f"    GLR done      ({t_glr - t_ipa:.1f}s)")

            with mp.ProcessingPool(NUM_CORES) as pool:
                rf_results = pool.map(worker_reinforce, rf_jobs)

            t_rf = time.time()
            print(f"    REINFORCE done ({t_rf - t_glr:.1f}s)")

            # Compute metrics
            ipa_metrics = compute_metrics(ipa_results, true_grad)
            glr_metrics = compute_metrics(glr_results, true_grad)
            rf_metrics  = compute_metrics(rf_results, true_grad)

            entry = {
                'rho': rho,
                'T': T,
                'lam': float(lam),
                'mu': float(MU),
                'true_gradient': float(true_grad),
                'PATHWISE': ipa_metrics,
                'GLR': glr_metrics,
                'REINFORCE': rf_metrics,
            }
            all_results.append(entry)

            # Print summary for this combination
            print(f"    {'Method':<12} {'Mean':>12} {'Bias':>12} "
                  f"{'Variance':>12} {'MSE':>12}")
            print(f"    {'-' * 60}")
            for method_name in ['PATHWISE', 'GLR', 'REINFORCE']:
                m = entry[method_name]
                print(f"    {method_name:<12} {m['mean']:>12.6f} {m['bias']:>12.6f} "
                      f"{m['variance']:>12.6f} {m['mse']:>12.6f}")

            elapsed = time.time() - combo_start
            print(f"    Total: {elapsed:.1f}s")
            print()

            # Incremental save
            out_path = os.path.join(RESULTS_DIR, 'E2_glr_comparison.json')
            with open(out_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    # ── Final summary table ──────────────────────────────────────────────────
    total_time = time.time() - start_time
    print()
    print("=" * 100)
    print("FINAL SUMMARY: MSE by (rho, T)")
    print("=" * 100)
    header = (f"{'rho':>6} {'T':>6} {'True Grad':>12} | "
              f"{'PW MSE':>12} {'GLR MSE':>12} {'RF MSE':>12} | "
              f"{'Best':>10}")
    print(header)
    print("-" * 100)

    for entry in all_results:
        pw_mse  = entry['PATHWISE']['mse']
        glr_mse = entry['GLR']['mse']
        rf_mse  = entry['REINFORCE']['mse']
        mse_dict = {'PATHWISE': pw_mse, 'GLR': glr_mse, 'REINFORCE': rf_mse}
        best = min(mse_dict, key=mse_dict.get)
        print(f"{entry['rho']:>6.2f} {entry['T']:>6d} {entry['true_gradient']:>12.4f} | "
              f"{pw_mse:>12.6f} {glr_mse:>12.6f} {rf_mse:>12.6f} | "
              f"{best:>10}")

    print()
    print("=" * 100)
    print("FINAL SUMMARY: Bias by (rho, T)")
    print("=" * 100)
    header = (f"{'rho':>6} {'T':>6} {'True Grad':>12} | "
              f"{'PW Bias':>12} {'GLR Bias':>12} {'RF Bias':>12}")
    print(header)
    print("-" * 90)

    for entry in all_results:
        print(f"{entry['rho']:>6.2f} {entry['T']:>6d} {entry['true_gradient']:>12.4f} | "
              f"{entry['PATHWISE']['bias']:>12.6f} "
              f"{entry['GLR']['bias']:>12.6f} "
              f"{entry['REINFORCE']['bias']:>12.6f}")

    print()
    print("=" * 100)
    print("FINAL SUMMARY: Variance by (rho, T)")
    print("=" * 100)
    header = (f"{'rho':>6} {'T':>6} {'True Grad':>12} | "
              f"{'PW Var':>12} {'GLR Var':>12} {'RF Var':>12}")
    print(header)
    print("-" * 90)

    for entry in all_results:
        print(f"{entry['rho']:>6.2f} {entry['T']:>6d} {entry['true_gradient']:>12.4f} | "
              f"{entry['PATHWISE']['variance']:>12.6f} "
              f"{entry['GLR']['variance']:>12.6f} "
              f"{entry['REINFORCE']['variance']:>12.6f}")

    out_path = os.path.join(RESULTS_DIR, 'E2_glr_comparison.json')
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("Done!")
