"""
§7 avg-iterate / Polyak evaluation — closes the report's gap that we used
last-iterate (min cost over epochs) instead of paper's avg-iterate evaluation.

Approach: load the last 50 epoch checkpoints, average their parameters
(Polyak averaging), evaluate the averaged network on the env with 100 episodes
× T=200,000 horizon (paper protocol). Compare to our previously reported
"min test cost" (last-iterate) numbers.

This is post-hoc: uses the 1100 .pt checkpoints we already saved.
Does NOT retrain — just evaluates Polyak-averaged policies.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'

import sys, json, time, glob, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'train')))

# IMPORTANT: must re-define PriorityNet here so torch.load can find it
class PriorityNet(nn.Module):
    def __init__(self, s, q, layers, hidden_dim, f_time=False, x_stats=None, t_stats=None):
        super().__init__()
        self.s = s; self.q = q
        self.x_stats = x_stats; self.t_stats = t_stats
        self.layers = layers; self.hidden_dim = hidden_dim
        self.f_time = f_time
        if self.f_time:
            self.input_fc = nn.Linear(self.q + 1, hidden_dim)
        else:
            self.input_fc = nn.Linear(self.q, hidden_dim)
        self.layers_fc = nn.ModuleList()
        for _ in range(layers):
            self.layers_fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_fc = nn.Linear(hidden_dim, self.s * self.q)

    def forward(self, x, t=0):
        batch = x.size()[0]
        if self.x_stats is not None:
            x = (x - self.x_stats[0]) / self.x_stats[1]
        if self.t_stats is not None:
            t = (t - self.t_stats[0]) / self.t_stats[1]
        if self.f_time:
            x = torch.cat((x, t), 1)
        x = F.relu(self.input_fc(x))
        for layer in self.layers_fc:
            x = F.relu(layer(x))
        x = self.output_fc(x)
        x = x.view(batch, self.s, self.q)
        return F.softmax(x, dim=-1)


# inject PriorityNet into __main__ so torch.load can find it
import __main__
__main__.PriorityNet = PriorityNet

from queuetorch.env import load_env

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'reproduction')

# Per-env config: list of (env_name, env_yaml, model_yaml)
ENVS = [
    ('criss_cross_bh', 'criss_cross_bh.yaml', 'ppg_softmax.yaml'),
    ('reentrant_2',    'reentrant_2.yaml',    'ppg_softmax.yaml'),
    ('reentrant_3',    'reentrant_3.yaml',    'ppg_softmax.yaml'),
    ('reentrant_4',    'reentrant_4.yaml',    'ppg_softmax.yaml'),
    ('reentrant_5',    'reentrant_5.yaml',    'ppg_softmax.yaml'),
    ('reentrant_6',    'reentrant_6.yaml',    'ppg_softmax.yaml'),
    ('reentrant_7',    'reentrant_7.yaml',    'ppg_softmax.yaml'),
    ('reentrant_8',    'reentrant_8.yaml',    'ppg_softmax.yaml'),
    ('reentrant_9',    'reentrant_9.yaml',    'ppg_softmax.yaml'),
    ('reentrant_10',   'reentrant_10.yaml',   'ppg_softmax.yaml'),
]

POLYAK_WINDOW = 50   # last N epochs to average
EVAL_BATCH = 100     # paper uses 100 episodes
EVAL_T = 50000       # paper uses 200K; we use 50K (still 25× our 2K training horizon)


def average_state_dicts(state_dicts):
    """Average parameters across a list of state_dicts."""
    avg = {}
    for k in state_dicts[0].keys():
        avg[k] = torch.stack([sd[k].float() for sd in state_dicts]).mean(dim=0)
    return avg


def evaluate_policy(net, env_config, batch, T, device='cpu'):
    """Run policy on env for `T` steps × `batch` episodes, return mean cost."""
    dq = load_env(env_config, temp=1e-3, batch=batch, seed=42, device=device)
    obs, state = dq.reset()
    total_cost = torch.zeros(batch, device=device)
    with torch.no_grad():
        for _ in range(T):
            queues, t = obs
            action = net(queues, t.detach() if t is not None else None)
            obs, state, cost, _ = dq.step(state, action)
            total_cost = total_cost + cost
    return float((total_cost / dq.state_time(state)).mean()) if hasattr(dq, 'state_time') else float((total_cost / T).mean())


def get_avg_cost(net, env_config, batch, T):
    """Time-averaged cost with paper's WC eval protocol."""
    dq = load_env(env_config, temp=1e-3, batch=batch, seed=42, device='cpu')
    obs, state = dq.reset()
    total_cost = torch.zeros(batch, 1)
    with torch.no_grad():
        for _ in range(T):
            queues, t = obs
            pr = net(queues, t.detach() if t is not None else None)  # (B,s,q) softmax
            # Apply paper's eval protocol (matches train_policy.py 269-273)
            pr = pr * dq.network
            pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
            pr = F.one_hot(torch.argmax(pr, dim=2), num_classes=dq.q).float()
            action = torch.round(pr)
            obs, state, cost, _ = dq.step(state, action)
            total_cost = total_cost + cost
    # Normalize by actual time elapsed
    time_elapsed = state.time.squeeze(-1)
    return float((total_cost.squeeze(-1) / time_elapsed).mean())


def main():
    results = {}
    for env_name, env_yaml, model_yaml in ENVS:
        ckpt_dir = os.path.join(PROJECT_ROOT, 'models', env_name)
        if not os.path.isdir(ckpt_dir):
            print(f"SKIP {env_name}: no checkpoint dir")
            continue
        # Find available checkpoint epochs
        files = glob.glob(os.path.join(ckpt_dir, 'ppg_softmax_*.pt'))
        if not files:
            print(f"SKIP {env_name}: no .pt files")
            continue
        epoch_files = []
        for f in files:
            m = re.search(r'ppg_softmax_(\d+)\.pt$', f)
            if m:
                epoch_files.append((int(m.group(1)), f))
        epoch_files.sort()
        epochs = [e for e, _ in epoch_files]
        max_ep = max(epochs)
        print(f"\n══ {env_name}: {len(epochs)} checkpoints, epochs 0..{max_ep} ══")

        # Load last POLYAK_WINDOW epochs, get state_dict from each
        window_start = max(0, max_ep - POLYAK_WINDOW + 1)
        window_files = [f for e, f in epoch_files if e >= window_start]
        print(f"  Polyak window: epochs {window_start}..{max_ep} ({len(window_files)} ckpts)")

        nets = []
        for f in window_files:
            try:
                n = torch.load(f, map_location='cpu')
                nets.append(n)
            except Exception as e:
                print(f"  load {f} failed: {e}")
        if not nets:
            print(f"  no nets loaded — skip"); continue

        # Build avg net by averaging state_dicts
        avg_sd = average_state_dicts([n.state_dict() for n in nets])
        last_net = nets[-1]
        avg_net = PriorityNet(last_net.s, last_net.q, last_net.layers, last_net.hidden_dim,
                              f_time=last_net.f_time)
        avg_net.load_state_dict(avg_sd)
        avg_net.eval()
        last_net.eval()

        # Load env config
        config_path = os.path.join(PROJECT_ROOT, 'configs', 'env', env_yaml)
        with open(config_path) as fh:
            env_config = yaml.safe_load(fh)

        # Evaluate both: last-iterate (max_ep) and Polyak-avg
        t0 = time.time()
        last_cost = get_avg_cost(last_net, env_config, EVAL_BATCH, EVAL_T)
        avg_cost  = get_avg_cost(avg_net,  env_config, EVAL_BATCH, EVAL_T)
        elapsed = time.time() - t0
        diff_pct = (avg_cost - last_cost) / last_cost * 100
        print(f"  last (ep {max_ep}): {last_cost:.3f}")
        print(f"  Polyak avg (last {len(window_files)}): {avg_cost:.3f}  (diff {diff_pct:+.2f}%)")
        print(f"  eval time: {elapsed:.0f}s @ B={EVAL_BATCH}, T={EVAL_T}")

        results[env_name] = {
            'epochs_total': max_ep + 1,
            'polyak_window_epochs': len(window_files),
            'eval_batch': EVAL_BATCH,
            'eval_T': EVAL_T,
            'last_iterate_cost': last_cost,
            'polyak_avg_cost': avg_cost,
            'diff_pct': diff_pct,
            'eval_time_s': elapsed,
        }

    out_path = os.path.join(RESULTS_DIR, 'polyak_eval.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n══ Summary ══")
    print(f"{'env':<20s} {'last':>9s} {'polyak':>9s} {'diff%':>8s}")
    for env, r in results.items():
        print(f"  {env:<18s} {r['last_iterate_cost']:>9.3f} {r['polyak_avg_cost']:>9.3f} {r['diff_pct']:>+7.2f}%")


if __name__ == '__main__':
    main()
