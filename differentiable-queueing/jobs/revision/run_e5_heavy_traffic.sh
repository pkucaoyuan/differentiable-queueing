#!/bin/bash
# E5: Heavy-traffic performance curve (rho 0.80 -> 0.99)
# Compute-heavy (~4h with 16 cores)
# Submit: grid_run --grid_submit=batch --grid_mem=30G --grid_ncpus=16 ./jobs/run_e5_heavy_traffic.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/revision

echo "=== E5: Heavy Traffic Curve ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"

python heavy_traffic_curve.py

echo "=== Done ==="
echo "Date: $(date)"
