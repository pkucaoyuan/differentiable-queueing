"""
Section 8: Theorem 2 numerical validation

Verify the variance scaling predictions on M/M/1:
- PATHWISE variance ~ (1 - rho)^{-3}
- REINFORCE variance ~ (1 - rho)^{-4}
- Ratio ~ (1 - rho)^{-1}

We measure variance of the PATHWISE (IPA) and REINFORCE gradient estimators on M/M/1
across rho values and fit log-log slope to verify scaling.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import json
import time
import numpy as np
import pathos.multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'reproduction')
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CORES = int(os.environ.get('NSLOTS', min(4, os.cpu_count() or 1)))

RHOS = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
T = 1000  # horizon
NUM_TRIALS = 500
MU = 1.0


def ipa_gradient_mm1(lam, mu, T, seed):
    """IPA gradient for M/M/1 via Lindley recursion."""
    rng = np.random.default_rng(seed)
    W = 0.0
    dW_dmu = 0.0
    grad_sum = 0.0
    for _ in range(T):
        inter_arr = rng.exponential(1.0 / lam)
        S = rng.exponential(1.0)
        W_new = max(0, W - inter_arr + S / mu)
        if W_new > 0:
            dW_dmu = dW_dmu - S / (mu ** 2)
        else:
            dW_dmu = 0.0
        W = W_new
        grad_sum += dW_dmu
    return lam * grad_sum / T


def reinforce_gradient_mm1(lam, mu, T, seed, sigma=0.05):
    """REINFORCE gradient: Gaussian perturbation on mu, score function."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal()
    mu_p = max(mu + sigma * eps, lam + 0.001)  # keep stable
    x = 0
    total_cost = 0.0
    for _ in range(T):
        rate_arr = lam
        rate_svc = mu_p if x > 0 else 0
        total_rate = rate_arr + rate_svc
        if total_rate <= 0:
            break
        tau = rng.exponential(1.0 / total_rate)
        total_cost += x * tau
        if rng.random() < rate_arr / total_rate:
            x += 1
        else:
            x = max(0, x - 1)
    avg_cost = total_cost / T
    score = eps / sigma
    return avg_cost * score


def trial_pw(args):
    lam, mu, T, seed = args
    return ipa_gradient_mm1(lam, mu, T, seed)


def trial_rf(args):
    lam, mu, T, seed = args
    return reinforce_gradient_mm1(lam, mu, T, seed)


def run():
    print("=" * 70)
    print("Section 8: Theorem 2 Variance Scaling Validation")
    print("=" * 70)
    print(f"  Rhos: {RHOS}")
    print(f"  T: {T}")
    print(f"  Trials per rho: {NUM_TRIALS}")
    print(f"  Cores: {NUM_CORES}")

    results = {}
    for rho in RHOS:
        lam = rho * MU
        t0 = time.time()
        pw_args = [(lam, MU, T, i + 1000) for i in range(NUM_TRIALS)]
        rf_args = [(lam, MU, T, i + 2000) for i in range(NUM_TRIALS)]

        with mp.ProcessingPool(NUM_CORES) as pool:
            pw_grads = pool.map(trial_pw, pw_args)
        with mp.ProcessingPool(NUM_CORES) as pool:
            rf_grads = pool.map(trial_rf, rf_args)

        pw_var = np.var(pw_grads)
        rf_var = np.var(rf_grads)
        ratio = rf_var / pw_var if pw_var > 0 else float('inf')
        gap = 1 - rho
        analytical_grad = -lam / (MU - lam) ** 2

        results[rho] = {
            'rho': rho,
            'gap': gap,
            'analytical_grad': analytical_grad,
            'pw_mean': float(np.mean(pw_grads)),
            'pw_var': float(pw_var),
            'rf_mean': float(np.mean(rf_grads)),
            'rf_var': float(rf_var),
            'ratio': float(ratio),
            'time': time.time() - t0,
        }
        print(f"  rho={rho:.2f} gap={gap:.3f} | "
              f"true={analytical_grad:.2f}, PW var={pw_var:.2e}, RF var={rf_var:.2e}, "
              f"ratio={ratio:.2f} ({time.time()-t0:.1f}s)")

    # Log-log fit
    rhos = np.array([r for r in RHOS])
    gaps = 1 - rhos
    pw_vars = np.array([results[r]['pw_var'] for r in RHOS])
    rf_vars = np.array([results[r]['rf_var'] for r in RHOS])

    log_gap = np.log(gaps)
    pw_slope = np.polyfit(log_gap, np.log(pw_vars), 1)[0]
    rf_slope = np.polyfit(log_gap, np.log(rf_vars), 1)[0]

    print()
    print("=" * 70)
    print("Log-Log Slope Fit (predicted: PW slope=-3, RF slope=-4)")
    print("=" * 70)
    print(f"  PATHWISE variance log-log slope: {pw_slope:.3f}  (predicted: -3)")
    print(f"  REINFORCE variance log-log slope: {rf_slope:.3f}  (predicted: -4)")
    print(f"  Difference: {rf_slope - pw_slope:.3f}  (predicted: -1)")

    results['fit'] = {
        'pw_slope': float(pw_slope),
        'rf_slope': float(rf_slope),
        'predicted_pw_slope': -3,
        'predicted_rf_slope': -4,
    }
    with open(os.path.join(RESULTS_DIR, 'theorem2_validation.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    start = time.time()
    run()
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
