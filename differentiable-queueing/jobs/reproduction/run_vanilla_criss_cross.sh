#!/bin/bash
# Section 6: Vanilla softmax (no WC) on criss-cross
# For comparison with WC-Softmax (already done in run_ste_criss_cross.sh)
eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu
cd /user/yc4911/DJ_OR/differentiable-queueing
echo "=== Section 6: Vanilla Softmax on criss-cross ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
python train/train_policy.py -e=criss_cross_bh.yaml -m=ppg_vanilla.yaml --algo ste
echo "=== Done ==="
echo "Date: $(date)"
