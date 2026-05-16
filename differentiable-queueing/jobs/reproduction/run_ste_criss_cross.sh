#!/bin/bash
# Step 1: Reproduce STE (PATHWISE) training on criss-cross network
# Submit: grid_run --grid_submit=batch --grid_mem=8G ./jobs/run_ste_criss_cross.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing

echo "=== STE Training on criss-cross ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"

# train_policy.py uses relative paths ./configs/env/ and ./env_data/
python train/train_policy.py -e=criss_cross_bh.yaml -m=ppg_softmax.yaml --algo ste

echo "=== Done ==="
echo "Date: $(date)"
