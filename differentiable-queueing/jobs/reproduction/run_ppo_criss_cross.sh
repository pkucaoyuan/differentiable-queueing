#!/bin/bash
# Step 2: Reproduce PPO training on criss-cross network
# Submit: grid_run --grid_submit=batch --grid_mem=16G --grid_ncpus=4 ./jobs/run_ppo_criss_cross.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing/PPO

echo "=== PPO Training on criss-cross ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "NSLOTS: ${NSLOTS:-1}"

# PPO/train.py: arg1 adds .yaml if missing; arg2 always gets .yaml appended
python train.py wc_softmax criss_cross_bh

echo "=== Done ==="
echo "Date: $(date)"
