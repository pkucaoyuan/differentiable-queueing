#!/bin/bash
# Experiment 1: M/M/1 sanity check
# Submit: grid_run --grid_submit=batch --grid_mem=4G ./jobs/run_test_mm1.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/reproduction

echo "=== Test 1: M/M/1 Sanity ==="
echo "Host: $(hostname)"
echo "Date: $(date)"

python test_mm1.py

echo "=== Done ==="
echo "Date: $(date)"
