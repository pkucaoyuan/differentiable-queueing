#!/bin/bash
# Phase D: STE training on reentrant_3
# Extends Section 7 reproduction to a larger reentrant network (9 queues, 3 servers)
# Submit: grid_run --grid_submit=batch --grid_mem=8G ./jobs/reproduction/run_ste_reentrant_3.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing

echo "=== Phase D: STE Training on reentrant_3 (Section 7) ==="
echo "Host: $(hostname)"
echo "Date: $(date)"

python train/train_policy.py -e=reentrant_3.yaml -m=ppg_softmax.yaml --algo ste

echo "=== Done ==="
echo "Date: $(date)"
