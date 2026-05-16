#!/bin/bash
# E2: GLR vs PATHWISE vs REINFORCE gradient comparison on M/M/1
# Fast experiment (~20 min with 16 cores)
# Submit: grid_run --grid_submit=batch --grid_mem=8G --grid_ncpus=16 ./jobs/run_e2_glr.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/revision

echo "=== E2: GLR Comparison ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"

python glr_comparison.py

echo "=== Done ==="
echo "Date: $(date)"
