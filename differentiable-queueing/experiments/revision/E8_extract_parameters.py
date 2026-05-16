"""
E8: Extract parameter tables from configs and env_data for paper appendix.
Outputs JSON files with all environment and training parameters.
Zero compute — pure data extraction.
"""
import os
import sys
import json
import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def extract_env_parameters():
    """Extract all environment parameters into a structured dict."""
    configs_dir = os.path.join(PROJECT_ROOT, 'configs', 'env')
    env_data_dir = os.path.join(PROJECT_ROOT, 'env_data')

    # Environments used in the paper
    env_files = [
        'criss_cross_IID.yaml',
        'criss_cross_hyper.yaml',
        'criss_cross_bh.yaml',
    ]
    # Add reentrant configs
    for k in range(2, 11):
        fname = f'reentrant_{k}.yaml'
        if os.path.exists(os.path.join(configs_dir, fname)):
            env_files.append(fname)
        fname_hyper = f'reentrant_{k}_hyper.yaml'
        if os.path.exists(os.path.join(configs_dir, fname_hyper)):
            env_files.append(fname_hyper)

    # Add other relevant configs
    for extra in ['mm1.yaml', 'multiclass.yaml', 'multiclass_5.yaml', 'n_model.yaml']:
        if os.path.exists(os.path.join(configs_dir, extra)):
            env_files.append(extra)

    results = []
    for fname in env_files:
        fpath = os.path.join(configs_dir, fname)
        if not os.path.exists(fpath):
            continue
        cfg = load_yaml(fpath)
        name = cfg.get('name', fname.replace('.yaml', ''))
        env_type = cfg.get('env_type', name)

        entry = {
            'config_file': fname,
            'name': name,
            'env_type': env_type,
            'lam_type': cfg.get('lam_type'),
            'service_type': cfg.get('service_type', 'exp'),
            'h': cfg.get('h'),
            'init_queues': cfg.get('init_queues'),
            'train_T': cfg.get('train_T'),
            'test_T': cfg.get('test_T'),
            'num_pool': cfg.get('num_pool', 1),
        }

        # Load network topology
        if cfg.get('network') is None:
            npy_base = os.path.join(env_data_dir, env_type)
            net_file = os.path.join(npy_base, f'{env_type}_network.npy')
            if os.path.exists(net_file):
                network = np.load(net_file)
                entry['network'] = network.tolist()
                entry['n_servers'] = int(network.shape[0])
                entry['n_queues'] = int(network.shape[1])
            else:
                entry['network'] = None
                entry['n_servers'] = None
                entry['n_queues'] = len(cfg.get('h', []))
        else:
            network = np.array(cfg['network'])
            entry['network'] = network.tolist()
            entry['n_servers'] = int(network.shape[0])
            entry['n_queues'] = int(network.shape[1])

        # Load service rates
        if cfg.get('mu') is None:
            mu_file = os.path.join(env_data_dir, env_type, f'{env_type}_mu.npy')
            if os.path.exists(mu_file):
                mu = np.load(mu_file)
                entry['mu'] = mu.tolist()
            else:
                entry['mu'] = None
        else:
            entry['mu'] = np.array(cfg['mu']).tolist() if cfg['mu'] is not None else None

        # Load arrival rates
        lam_params = cfg.get('lam_params', {})
        if cfg.get('lam_type') == 'constant':
            if lam_params.get('val') is None:
                lam_file = os.path.join(env_data_dir, env_type, f'{env_type}_lam.npy')
                if os.path.exists(lam_file):
                    lam = np.load(lam_file)
                    entry['lambda'] = lam.tolist()
                else:
                    entry['lambda'] = None
            else:
                entry['lambda'] = lam_params['val']
        else:
            entry['lam_params'] = lam_params

        # Load queue event options
        qeo = cfg.get('queue_event_options')
        if qeo == 'custom':
            delta_file = os.path.join(env_data_dir, env_type, f'{env_type}_delta.npy')
            if os.path.exists(delta_file):
                delta = np.load(delta_file)
                entry['queue_event_options'] = delta.tolist()
                entry['has_routing'] = True
            else:
                entry['has_routing'] = False
        elif qeo is not None:
            entry['queue_event_options'] = qeo
            entry['has_routing'] = True
        else:
            entry['has_routing'] = False

        # Compute traffic intensity (approximate)
        if entry.get('lambda') is not None and entry.get('mu') is not None:
            try:
                lam_arr = np.array(entry['lambda']).flatten()
                mu_arr = np.array(entry['mu'])
                # For single server: rho = sum(lambda_i / mu_i)
                if mu_arr.ndim == 2 and mu_arr.shape[0] == 1:
                    mu_diag = mu_arr[0]
                    nonzero = mu_diag > 0
                    rho = np.sum(lam_arr[nonzero] / mu_diag[nonzero])
                    entry['approx_rho'] = float(rho)
                else:
                    entry['approx_rho'] = None
            except Exception:
                entry['approx_rho'] = None

        results.append(entry)

    return results


def extract_model_parameters():
    """Extract training hyperparameters from model configs."""
    configs_dir = os.path.join(PROJECT_ROOT, 'configs', 'model')
    results = {}

    for fname in os.listdir(configs_dir):
        if not fname.endswith('.yaml') or 'template' in fname:
            continue
        cfg = load_yaml(os.path.join(configs_dir, fname))
        results[fname] = cfg

    # Also check PPO configs
    ppo_configs_dir = os.path.join(PROJECT_ROOT, 'PPO', 'configs')
    if os.path.exists(ppo_configs_dir):
        for fname in os.listdir(ppo_configs_dir):
            if fname.endswith('.yaml'):
                cfg = load_yaml(os.path.join(ppo_configs_dir, fname))
                results[f'PPO/{fname}'] = cfg

    return results


def extract_reinforce_details():
    """Extract REINFORCE implementation details from experiment scripts."""
    details = {
        'gradient_comparison': {
            'source': 'experiments/gradient_comparison.py',
            'gamma': 0.999,
            'batch_size': 1000,
            'baseline': 'none (raw REINFORCE with discounted returns)',
            'normalization': 'cosine similarity for gradient comparison',
        },
        'cmu_rule_reinforce': {
            'source': 'experiments/cmu_rule_REINFORCE.py',
            'gamma': 0.99,
            'batch_size': 100,
            'baseline': '2-layer MLP (64 hidden, ReLU)',
            'baseline_training': {
                'inner_epochs': 3,
                'optimizer': 'Adam',
                'lr': 0.001,
                'batch_size': 1024,
                'return_normalization': True,
            },
            'policy': 'softmax priority vector (1 x q)',
            'update_rule': 'normalized SGD: theta -= alpha * (grad / ||grad||)',
        },
        'cmu_step_rules_pathwise': {
            'source': 'experiments/cmu_step_rules_PATHWISE.py',
            'temperature': 1e-6,
            'batch_size': 1,
            'horizon_T': 1000,
            'eval_T': 20000,
            'num_iterations': 100,
            'rho': 0.95,
            'queue_class': 10,
            'step_rules_tested': ['normalized_fixed', 'adam', 'rmsprop'],
        },
    }
    return details


if __name__ == '__main__':
    print("Extracting environment parameters...")
    env_params = extract_env_parameters()
    with open(os.path.join(RESULTS_DIR, 'E8_env_parameters.json'), 'w') as f:
        json.dump(env_params, f, indent=2, default=str)
    print(f"  Saved {len(env_params)} environments to results/E8_env_parameters.json")

    print("Extracting model parameters...")
    model_params = extract_model_parameters()
    with open(os.path.join(RESULTS_DIR, 'E8_model_parameters.json'), 'w') as f:
        json.dump(model_params, f, indent=2, default=str)
    print(f"  Saved {len(model_params)} model configs to results/E8_model_parameters.json")

    print("Extracting REINFORCE details...")
    rf_details = extract_reinforce_details()
    with open(os.path.join(RESULTS_DIR, 'E8_reinforce_details.json'), 'w') as f:
        json.dump(rf_details, f, indent=2)
    print("  Saved to results/E8_reinforce_details.json")

    # Print summary table
    print("\n" + "=" * 80)
    print("TABLE A1: Environment Parameters Summary")
    print("=" * 80)
    print(f"{'Name':<25} {'q':>3} {'s':>3} {'Service':>10} {'Arrival':>10} {'rho':>6} {'Train T':>8}")
    print("-" * 80)
    for e in env_params:
        q = e.get('n_queues', '?')
        s = e.get('n_servers', '?')
        svc = e.get('service_type', 'exp')
        arr = e.get('lam_type', '?')
        rho = f"{e['approx_rho']:.3f}" if e.get('approx_rho') else '?'
        train_t = e.get('train_T', '?')
        print(f"{e['name']:<25} {q:>3} {s:>3} {svc:>10} {arr:>10} {rho:>6} {train_t:>8}")

    print("\n" + "=" * 80)
    print("TABLE A2: Training Hyperparameters")
    print("=" * 80)
    for mname, mcfg in model_params.items():
        print(f"\n--- {mname} ---")
        if isinstance(mcfg, dict):
            for section, vals in mcfg.items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        print(f"  {section}.{k}: {v}")
                else:
                    print(f"  {section}: {vals}")

    print("\nDone!")
