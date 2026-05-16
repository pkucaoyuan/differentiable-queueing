#!/bin/bash
# Phase C: rho ablation reproduction (quick)
# Submit: grid_run --grid_submit=batch --grid_mem=20G --grid_ncpus=16 ./jobs/reproduction/run_rho_ablation.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/reproduction

echo "=== Phase C: rho ablation (quick) ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"

python test_rho_ablation.py

echo "=== Done ==="
echo "Date: $(date)"
