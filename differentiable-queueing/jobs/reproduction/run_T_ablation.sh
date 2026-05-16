#!/bin/bash
# Section 5.2: T ablation reproduction
eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu
cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/reproduction
echo "=== Section 5.2 T ablation ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"
python test_T_ablation.py
echo "=== Done ==="
echo "Date: $(date)"
