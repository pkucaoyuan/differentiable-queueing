#!/bin/bash
# Comprehensive reproduction: M/M/1 + Section 5.1 + Section 5.2
# Submit: grid_run --grid_submit=batch --grid_mem=50G --grid_ncpus=16 ./jobs/run_full_reproduction.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/reproduction

echo "=== Full Reproduction ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"

python full_reproduction.py

echo "=== Done ==="
echo "Date: $(date)"
