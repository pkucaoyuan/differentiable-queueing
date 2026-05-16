#!/bin/bash
# Section 8: Theorem 2 retry with T=10000 (v1 had T=1000 — wrong slopes)
eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu
cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/reproduction
echo "=== Section 8 Theorem 2 v2 (T=10000, 1000 trials) ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"
python test_theorem2_scaling_v2.py
echo "=== Done ==="
echo "Date: $(date)"
