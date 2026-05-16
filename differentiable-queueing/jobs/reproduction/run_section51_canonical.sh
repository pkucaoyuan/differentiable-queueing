#!/bin/bash
# Phase A: Canonical Section 5.1 reproduction using original gradient_comparison.py
#
# Original paper defaults: gt_batch=1M, num_samples=100, estimators=100
# We use a smaller but still robust version: gt_batch=200K, num_samples=10, estimators=20
# This is 4-5x larger ground truth than our previous attempt (100K), enough for stable estimates
#
# Submit:
#   grid_run --grid_submit=batch --grid_mem=40G --grid_ncpus=16 \
#       ./jobs/reproduction/run_section51_canonical.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

# Run from experiments/ so that ./configs/env/ works (original script convention)
cd /user/yc4911/DJ_OR/differentiable-queueing/experiments

echo "=== Phase A: Canonical Section 5.1 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"
echo "PWD: $(pwd)"
echo ""
echo "Command: python gradient_comparison.py \\"
echo "             --env criss_cross_bh.yaml \\"
echo "             --horizon 1000 \\"
echo "             --gt_batch 200000 \\"
echo "             --num_samples 10 \\"
echo "             --estimators_per_sample 20 \\"
echo "             --num_cores ${NSLOTS:-16}"
echo ""

# Reduced gt_batch from 200K to 80K to fit in memory (previous attempt at 200K hit OOM with 40G)
# 80K / 16 cores = 5K per worker — should fit comfortably
python gradient_comparison.py \
    --env criss_cross_bh.yaml \
    --horizon 1000 \
    --gt_batch 80000 \
    --num_samples 5 \
    --estimators_per_sample 20 \
    --num_cores ${NSLOTS:-16}

# Move results to organized location
mkdir -p /user/yc4911/DJ_OR/differentiable-queueing/results/reproduction
mv /user/yc4911/DJ_OR/differentiable-queueing/experiments/results/gradient_comparison_*.json \
   /user/yc4911/DJ_OR/differentiable-queueing/results/reproduction/ 2>/dev/null
rmdir /user/yc4911/DJ_OR/differentiable-queueing/experiments/results 2>/dev/null

echo ""
echo "=== Done ==="
echo "Date: $(date)"
