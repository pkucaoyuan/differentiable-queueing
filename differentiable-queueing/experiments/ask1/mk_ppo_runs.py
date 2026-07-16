"""Generate isolated PPO run directories (Ask-2, Fig 12 multi-seed reproduction).

Each run dir <repo>/ppo_runs/<variant>_s<seed>/ contains a copy of train.py,
symlinks to PPO/utils and PPO/env_data, and its own configs/<variant>.yaml.
Run dirs sit at PPO's level so train.py's '../configs/env/...' resolves.

Reduced-but-honest spec (full paper config is ~days/run on this pipeline):
  reentrant_2 (paper Fig 12 net), episode_steps/num_epochs set via CLI,
  eval on 25 test envs x test_T=2500 (env variant reentrant_2_ppofast).
"""
import os
import shutil
import sys

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PPO = os.path.join(REPO, 'PPO')
RUNS = os.path.join(REPO, 'ppo_runs')

EPISODE_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
NUM_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 40

VARIANTS = {
    'WC':      ('wc_softmax.yaml', dict()),
    'vanilla': ('softmax.yaml', dict()),
    'bc':      ('vanilla_bc.yaml', dict()),
}
PLAN = [('WC', 0), ('WC', 1), ('WC', 2), ('WC', 3),
        ('vanilla', 0), ('vanilla', 1), ('bc', 0), ('bc', 1)]


def main():
    os.makedirs(RUNS, exist_ok=True)
    cmds = []
    for gpu, (var, seed) in enumerate(PLAN):
        rid = f'{var}_s{seed}'
        rd = os.path.join(RUNS, rid)
        os.makedirs(os.path.join(rd, 'configs'), exist_ok=True)
        shutil.copy(os.path.join(PPO, 'train.py'), os.path.join(rd, 'train.py'))
        for link in ['utils', 'env_data']:
            dst = os.path.join(rd, link)
            if not os.path.islink(dst):
                os.symlink(os.path.join(PPO, link), dst)
        base, _ = VARIANTS[var]
        cfg = yaml.safe_load(open(os.path.join(PPO, 'configs_gpu', base)))
        cfg['env']['device'] = 'cuda'
        cfg['env']['model_seed'] = 100 + 17 * seed
        cfg['env']['train_seed'] = 3003 + 1000 * seed
        cfg['env']['test_seed'] = 42          # same test envs across runs
        cfg['training']['episode_steps'] = EPISODE_STEPS
        cfg['training']['num_epochs'] = NUM_EPOCHS
        cfg['training']['test_batch'] = 25
        yaml.safe_dump(cfg, open(os.path.join(rd, 'configs', 'run.yaml'), 'w'))
        cmds.append(
            f'cd {rd} && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} '
            f'setsid nohup /opt/conda/bin/python train.py run reentrant_2_ppofast '
            f'> run.log 2>&1 < /dev/null &')
    with open(os.path.join(RUNS, 'launch.sh'), 'w') as f:
        f.write('#!/bin/bash\n' + '\n'.join(cmds) + '\n')
    print(f'{len(PLAN)} run dirs ready, episode_steps={EPISODE_STEPS}, '
          f'epochs={NUM_EPOCHS}; launch: bash {RUNS}/launch.sh')


if __name__ == '__main__':
    main()
