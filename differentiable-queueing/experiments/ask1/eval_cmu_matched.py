"""cmu-rule baseline at the SAME evaluation protocol as the PPO runs
(reentrant_2_ppofast: 25 test envs, seeds 42..66, eval_t=2500, cost = total
accumulated holding cost / elapsed time, mean over envs) — the shipped
PPO/cmu_results.json was computed at test_T=10000 and is not horizon-matched.
"""
import json
import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'PPO'))
os.chdir(os.path.join(REPO, 'PPO'))

from utils.env import load_sb_p_env

DEV = 'cuda'
N_ENVS = 25
EVAL_T = 2500
TEST_SEED = 42

env_config = yaml.safe_load(open('../configs/env/reentrant_2_ppofast.yaml'))
envs = [load_sb_p_env(env_config=env_config, temp=1.0, batch=1, seed=s,
                      policy_name='WC', device=torch.device(DEV))
        for s in range(TEST_SEED, TEST_SEED + N_ENVS)]

mu = envs[0].mu[0]          # (s, q)
network = envs[0].network[0]
s_dim, q_dim = mu.shape

obs_list, state_list = [], []
for e in envs:
    ob = e.reset()
    obs_list.append(ob if isinstance(ob, tuple) else (ob,))

total_cost = [torch.zeros(1, device=DEV) for _ in envs]
times = [None] * len(envs)

with torch.no_grad():
    for tt in range(EVAL_T):
        for i, e in enumerate(envs):
            raw = obs_list[i][0]
            if not torch.is_tensor(raw):
                raw = torch.as_tensor(np.asarray(raw)).float()
            queues = raw.reshape(-1)[:q_dim].to(DEV)
            # cmu rule (h=1): each server serves its nonempty compatible queue
            # with the highest service rate mu_ij
            score = mu.clone()
            score[network == 0] = -1
            score[:, :] = torch.where(queues.unsqueeze(0) > 0, score,
                                      torch.full_like(score, -1))
            action = torch.zeros(s_dim, q_dim, device=DEV)
            for si in range(s_dim):
                if (score[si] > -1).any():
                    action[si, torch.argmax(score[si])] = 1.0
            _, _, _, _, info = e.step(action)
            obs_list[i] = info['obs']
            total_cost[i] = total_cost[i] + info['cost']
            times[i] = info['state'].time

costs = torch.cat([total_cost[i] / times[i].reshape(-1)[0] for i in range(len(envs))])
out = {'avg_cost': float(costs.mean()), 'std_error': float(costs.std() / np.sqrt(N_ENVS)),
       'protocol': f'{N_ENVS} envs x eval_t={EVAL_T}, seeds {TEST_SEED}..{TEST_SEED+N_ENVS-1}'}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(REPO, 'ppo_runs', 'cmu_matched.json'), 'w'))
