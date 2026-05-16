#!/bin/bash
#$ -q debian.q
#$ -pe threaded 16
#$ -cwd
#$ -o /user/yc4911/DJ_OR/differentiable-queueing/results/reproduce.log
#$ -e /user/yc4911/DJ_OR/differentiable-queueing/results/reproduce.err
#$ -N reproduce_main

# Activate conda environment
eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu

# Run from experiments directory (required for relative path imports)
cd /user/yc4911/DJ_OR/differentiable-queueing/experiments/reproduction

python reproduce_main.py
