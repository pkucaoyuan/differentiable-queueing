"""
§5.1 cossim — single-env variant, takes env name from CLI.
Used to fan out across GPU 5/6/7 on different networks.

Usage:  CUDA_VISIBLE_DEVICES=N python test_gradient_gpu_env.py <env_name>
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
    print(f"\n══ §5.1 cossim: {env_name}, device={device} ══", flush=True)
    print(f"   rhos={rhos}, gt_batch={gt_batch}, num_samples={num_samples}, "
          f"est_per_sample={estimators_per_sample}, T={horizon}", flush=True)

    cfg_path = os.path.join(PROJECT_ROOT, 'configs', 'env', f'{env_name}.yaml')
    base_cfg = yaml.safe_load(open(cfg_path))

    all_results = {}
    for rho in rhos:
        print(f"\n── rho={rho} ──", flush=True)
        cfg = yaml.safe_load(open(cfg_path))
        # Scale arrival rates by rho if the config has explicit lam_params.val
        # (reentrant configs use npy files via lam=null and skip this)
        orig_rho = base_cfg.get('rho', 0.95) if 'rho' in base_cfg else 0.95
        lp = cfg.get('lam_params', {})
        if isinstance(lp, dict) and 'val' in lp and lp['val'] is not None:
            cfg['lam_params']['val'] = [v * rho / orig_rho for v in lp['val']]

        all_results[rho] = {}
        for policy_type in POLICY_TYPES:
            print(f"  policy={policy_type}", flush=True)
            t_policy = time.time()
            pw_cosines, rf_cosines = [], []

            for sample_idx in range(num_samples):
                dq_dim = load_env(cfg, temp=1.0, batch=1, seed=42, device='cpu')
                s, q = dq_dim.s, dq_dim.q
                del dq_dim

                torch.manual_seed(20260628 + sample_idx)
                net = get_policy(policy_type, s, q).to(device)
                state_dict = net.state_dict()

                gt_net = get_policy(policy_type, s, q).to(device)
                gt_net.load_state_dict(state_dict)
                gt_grad = _compute_reinforce_grad_core(
                    gt_net, cfg, batch_size=gt_batch, T=horizon, gamma=DEFAULT_GAMMA, device=device)
                if torch.isnan(gt_grad).any() or torch.norm(gt_grad) < 1e-10:
                    continue

                for est_idx in range(estimators_per_sample):
                    torch.manual_seed(31415 + est_idx)
                    pw_net = get_policy(policy_type, s, q).to(device)
                    pw_net.load_state_dict(state_dict)
                    pw_grad = compute_pathwise_grad(pw_net, cfg, batch_size=1, T=horizon, device=device)

                    rf_net = get_policy(policy_type, s, q).to(device)
                    rf_net.load_state_dict(state_dict)
                    rf_grad = _compute_reinforce_grad_core(
                        rf_net, cfg, batch_size=DEFAULT_REINFORCE_BATCH, T=horizon,
                        gamma=DEFAULT_GAMMA, device=device)

                    if not (torch.isnan(pw_grad).any() or torch.norm(pw_grad) < 1e-10):
                        pw_cosines.append(cosine_similarity(pw_grad, gt_grad))
                    if not (torch.isnan(rf_grad).any() or torch.norm(rf_grad) < 1e-10):
                        rf_cosines.append(cosine_similarity(rf_grad, gt_grad))

                if sample_idx % max(1, num_samples // 5) == 0:
                    print(f"    sample {sample_idx}/{num_samples}", flush=True)

            n_pw = len(pw_cosines); n_rf = len(rf_cosines)
            all_results[rho][policy_type] = {
                'pathwise_cosines': pw_cosines,
                'reinforce_cosines': rf_cosines,
                'pathwise_mean': float(np.mean(pw_cosines)) if n_pw else float('nan'),
                'pathwise_std':  float(np.std(pw_cosines))  if n_pw else float('nan'),
                'reinforce_mean': float(np.mean(rf_cosines)) if n_rf else float('nan'),
                'reinforce_std':  float(np.std(rf_cosines))  if n_rf else float('nan'),
                'n_pw_kept': n_pw, 'n_rf_kept': n_rf,
            }
            print(f"    PW: {all_results[rho][policy_type]['pathwise_mean']:+.3f}±"
                  f"{all_results[rho][policy_type]['pathwise_std']:.3f}  "
                  f"RF: {all_results[rho][policy_type]['reinforce_mean']:+.3f}±"
                  f"{all_results[rho][policy_type]['reinforce_std']:.3f}  "
                  f"({(time.time()-t_policy)/60:.1f} min)", flush=True)

    return all_results


if __name__ == '__main__':
    env_name = sys.argv[1] if len(sys.argv) > 1 else 'criss_cross_bh'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}, "
              f"free {torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)

    # For reentrant networks (lam from npy), rho scaling is no-op; use single 'default' rho
    cfg_path = os.path.join(PROJECT_ROOT, 'configs', 'env', f'{env_name}.yaml')
    base_cfg = yaml.safe_load(open(cfg_path))
    lp = base_cfg.get('lam_params', {})
    if isinstance(lp, dict) and lp.get('val') is not None:
        rhos = [0.8, 0.9, 0.95, 0.99]
    else:
        rhos = [0.95]  # single default rho; reentrant uses npy-loaded lambda

    res = run(
        env_name=env_name,
        rhos=rhos,
        gt_batch=20_000,
        num_samples=20,
        estimators_per_sample=15,
        horizon=1000,
        device=device,
    )

    out_path = os.path.join(RESULTS_DIR, f'gradient_gpu_{env_name}.json')
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {out_path}")
