"""
§5.1 Figure 4 cossim — GPU version, paper-canonical samples.

Uses GPU directly (no multiprocessing) since the bottleneck is the gt_batch=1M
simulation, which fits entirely on a single GPU. This is the missing piece for
§5.1 paper reproduction.

Run with CUDA_VISIBLE_DEVICES=0 (pin to GPU 0, the free one).
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'

import sys, time, json
import torch
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradient_comparison import (
    get_policy, _compute_reinforce_grad_core, compute_pathwise_grad, cosine_similarity,
    POLICY_TYPES, DEFAULT_GAMMA, DEFAULT_REINFORCE_BATCH,
)
from queuetorch.env import load_env

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'reproduction')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run(env_name, rhos, gt_batch, num_samples, estimators_per_sample, horizon, device):
    """Returns dict[rho] -> dict[policy_type] -> {pathwise_cosines, reinforce_cosines}."""
    print(f"\n══ §5.1 cossim GPU: {env_name}, device={device} ══", flush=True)
    print(f"   rhos={rhos}, gt_batch={gt_batch}, num_samples={num_samples}, "
          f"est_per_sample={estimators_per_sample}, T={horizon}", flush=True)

    base_cfg_path = os.path.join(PROJECT_ROOT, 'configs', 'env', f'{env_name}.yaml')
    with open(base_cfg_path) as f:
        base_config = yaml.safe_load(f)

    all_results = {}
    for rho in rhos:
        print(f"\n── rho={rho} ──", flush=True)
        # Scale arrival rates by rho
        cfg = yaml.safe_load(open(base_cfg_path))
        # criss_cross_bh: lam_type is constant, lam_params.val is a list
        # Scale by rho/0.95 (paper's default rho is 0.95)
        orig_rho = 0.95
        if 'lam_params' in cfg and 'val' in cfg['lam_params']:
            cfg['lam_params']['val'] = [v * rho / orig_rho for v in cfg['lam_params']['val']]
        all_results[rho] = {}

        for policy_type in POLICY_TYPES:
            print(f"  policy={policy_type}", flush=True)
            t_policy = time.time()
            pw_cosines, rf_cosines = [], []

            for sample_idx in range(num_samples):
                # Fresh random policy
                from queuetorch.env import load_env as _le
                dq_dim = _le(cfg, temp=1.0, batch=1, seed=42, device='cpu')
                s, q = dq_dim.s, dq_dim.q
                del dq_dim

                torch.manual_seed(20260623 + sample_idx)
                net = get_policy(policy_type, s, q).to(device)
                state_dict = net.state_dict()

                # GT gradient with large batch
                t_gt = time.time()
                gt_net = get_policy(policy_type, s, q).to(device)
                gt_net.load_state_dict(state_dict)
                gt_grad = _compute_reinforce_grad_core(
                    gt_net, cfg, batch_size=gt_batch, T=horizon, gamma=DEFAULT_GAMMA, device=device)

                # Skip if GT has NaN or zero norm (numerical issues at heavy traffic)
                if torch.isnan(gt_grad).any() or torch.norm(gt_grad) < 1e-10:
                    continue

                # Multiple estimator samples
                for est_idx in range(estimators_per_sample):
                    torch.manual_seed(31415 + est_idx)
                    # PATHWISE B=1
                    pw_net = get_policy(policy_type, s, q).to(device)
                    pw_net.load_state_dict(state_dict)
                    pw_grad = compute_pathwise_grad(pw_net, cfg, batch_size=1, T=horizon, device=device)

                    # REINFORCE B=1000
                    rf_net = get_policy(policy_type, s, q).to(device)
                    rf_net.load_state_dict(state_dict)
                    rf_grad = _compute_reinforce_grad_core(
                        rf_net, cfg, batch_size=DEFAULT_REINFORCE_BATCH, T=horizon,
                        gamma=DEFAULT_GAMMA, device=device)

                    # NaN-resistant cosine
                    if torch.isnan(pw_grad).any() or torch.norm(pw_grad) < 1e-10:
                        pass  # skip
                    else:
                        pw_cosines.append(cosine_similarity(pw_grad, gt_grad))
                    if torch.isnan(rf_grad).any() or torch.norm(rf_grad) < 1e-10:
                        pass
                    else:
                        rf_cosines.append(cosine_similarity(rf_grad, gt_grad))

                if sample_idx % max(1, num_samples // 5) == 0:
                    print(f"    sample {sample_idx}/{num_samples} done ({time.time()-t_gt:.1f}s/sample)",
                          flush=True)

            n_pw = len(pw_cosines) if pw_cosines else 0
            n_rf = len(rf_cosines) if rf_cosines else 0
            all_results[rho][policy_type] = {
                'pathwise_cosines': pw_cosines,
                'reinforce_cosines': rf_cosines,
                'pathwise_mean': float(np.mean(pw_cosines)) if n_pw else float('nan'),
                'pathwise_std':  float(np.std(pw_cosines)) if n_pw else float('nan'),
                'reinforce_mean': float(np.mean(rf_cosines)) if n_rf else float('nan'),
                'reinforce_std':  float(np.std(rf_cosines)) if n_rf else float('nan'),
                'n_pw_kept': n_pw,
                'n_rf_kept': n_rf,
                'n_attempted': num_samples * estimators_per_sample,
            }
            print(f"    PW: {np.mean(pw_cosines):+.3f}±{np.std(pw_cosines):.3f}  "
                  f"RF: {np.mean(rf_cosines):+.3f}±{np.std(rf_cosines):.3f}  "
                  f"(policy took {(time.time()-t_policy)/60:.1f} min)", flush=True)

    return all_results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"Free mem: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)
    else:
        print("WARN: CUDA not available, falling back to CPU", flush=True)

    # Tractable canonical: GPU 0, ~4-5h
    # 20 samples x 15 estimators @ gt_batch=100K
    # Paper: 100 x 100 x 1M (we are 50× less in samples × est)
    # Quick run was 5 x 20 x 100K = 100 cells (we are 60× more)
    res = run(
        env_name='criss_cross_bh',
        rhos=[0.8, 0.9, 0.95, 0.99],
        gt_batch=100_000,
        num_samples=20,
        estimators_per_sample=15,
        horizon=1000,
        device=device,
    )

    out_path = os.path.join(RESULTS_DIR, 'gradient_gpu_canonical_v2.json')
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n══ Summary ══")
    print(f"{'rho':>6s} {'policy':>5s} {'PW mean':>10s} {'PW std':>8s} {'RF mean':>10s} {'RF std':>8s}  PW>RF?")
    for rho, pols in res.items():
        for pol, d in pols.items():
            marker = '✓' if d['pathwise_mean'] > d['reinforce_mean'] else '✗'
            print(f"  {rho:.2f}  {pol:>5s}  {d['pathwise_mean']:>+8.3f}  {d['pathwise_std']:>8.3f}  "
                  f"{d['reinforce_mean']:>+8.3f}  {d['reinforce_std']:>8.3f}  {marker}")
