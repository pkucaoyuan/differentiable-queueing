#!/bin/bash
# STE training on reentrant_6
eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu
cd /user/yc4911/DJ_OR/differentiable-queueing
echo "=== STE Training on reentrant_6 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
python train/train_policy.py -e=reentrant_6.yaml -m=ppg_softmax.yaml --algo ste
echo "=== Done ==="
echo "Date: $(date)"
