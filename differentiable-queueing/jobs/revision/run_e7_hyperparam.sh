#!/bin/bash
# E7: Hyperparameter sensitivity analysis
# Compute-heavy (~3.5h with 16 cores)
# Submit: grid_run --grid_submit=batch --grid_mem=30G --grid_ncpus=16 ./jobs/run_e7_hyperparam.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/revision

echo "=== E7: Hyperparameter Sensitivity ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"

python hyperparam_sensitivity.py

echo "=== Done ==="
echo "Date: $(date)"
