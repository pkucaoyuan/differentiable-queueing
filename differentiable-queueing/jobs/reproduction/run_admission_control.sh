#!/bin/bash
# Section 5.3: Admission control — PATHWISE vs SPSA
# Submit: grid_run --grid_submit=batch --grid_mem=16G --grid_ncpus=4 ./jobs/reproduction/run_admission_control.sh

eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

# env.py uses ./env_data/ relative path — must run from project root
cd /user/yc4911/DJ_OR/differentiable-queueing

echo "=== Section 5.3: Admission Control ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "NSLOTS: ${NSLOTS:-1}"

python experiments/admission_control.py --device cpu --num_trials 5 --policy MaxWeight

echo "=== Done ==="
echo "Date: $(date)"
