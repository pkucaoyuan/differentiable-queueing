#!/bin/bash
# Experiment 2: Section 5.1 gradient cossim
# Submit: grid_run --grid_submit=batch --grid_mem=30G --grid_ncpus=16 ./jobs/run_test_gradient.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/reproduction

echo "=== Test 2: Section 5.1 Gradient Cossim ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"

python test_gradient.py

echo "=== Done ==="
echo "Date: $(date)"
