#!/bin/bash
# Phase B (quick): 5-class CMU rule reproduction
# Quick check: 1 alpha × 2 gaps × 20 trials = 40 trials per method
# Submit: grid_run --grid_submit=batch --grid_mem=20G --grid_ncpus=16 ./jobs/reproduction/run_cmu_5class.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/reproduction

echo "=== Phase B: 5-class CMU Rule (quick) ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"

python test_cmu_5class.py

echo "=== Done ==="
echo "Date: $(date)"
