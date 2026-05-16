#!/bin/bash
# E6: 3-way factorial ablation (6 methods)
# Submit: grid_run --grid_submit=batch --grid_mem=30G --grid_ncpus=16 ./jobs/run_e6_ablation.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/revision

echo "=== E6: 3-Way Ablation ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"

python ablation_3way.py

echo "=== Done ==="
echo "Date: $(date)"
