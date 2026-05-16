#!/bin/bash
# Step 3: Reproduce CMU rule experiment (PATHWISE vs REINFORCE)
# This is the paper's core result (Section 5.2 / Figure 9)
# Submit: grid_run --grid_submit=batch --grid_mem=30G --grid_ncpus=16 ./jobs/run_cmu_reproduce.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/reproduction

echo "=== CMU Rule Reproduction ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"

python reproduce_main.py

echo "=== Done ==="
echo "Date: $(date)"
