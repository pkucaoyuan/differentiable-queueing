#!/bin/bash
# §5.2 CμRule 10-class with paper-correct gap grid (adds gap=0.1)
eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu
cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/reproduction
echo "=== §5.2 CμRule 10-class (paper grid with gap=0.1) ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"
python test_cmu_papergrid.py
echo "=== Done ==="
echo "Date: $(date)"
