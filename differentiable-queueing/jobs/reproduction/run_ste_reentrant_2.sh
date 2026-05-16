#!/bin/bash
# STE training on reentrant_2 — first reentrant network reproduction
# Section 7 main benchmark (smallest reentrant network)
# Submit: grid_run --grid_submit=batch --grid_mem=8G ./jobs/run_ste_reentrant_2.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing

echo "=== STE Training on reentrant_2 (Section 7) ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"

python train/train_policy.py -e=reentrant_2.yaml -m=ppg_softmax.yaml --algo ste

echo "=== Done ==="
echo "Date: $(date)"
