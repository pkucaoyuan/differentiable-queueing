#!/bin/bash
# Overnight pipeline per GPU (CUDA_VISIBLE_DEVICES set by launcher):
#   phase 1: batch-size sweep (advisor Ask-1 B-axis), reuses existing GT
#   phase 2: stage2 full-spec wave1 (6/9-class nets, GT=1e5)
#   phase 3: stage2 full-spec wave2 (12/15-class nets, GT=5e4)
# Each phase retried up to 2x (workers resume via claims; OOM cells stay claimed).
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/opt/conda/bin/python

run() {
  for attempt in 1 2; do
    echo "=== phase: $* (attempt $attempt) $(date) ==="
    $PY experiments/ask1/run_cossim.py "$@" && break
    sleep 10
  done
}

run --stage sweep --scaling paper
run --stage stage2 --scaling paper --no-control-gt --gt-trajs 100000 \
    --nets reentrant_2,reentrant_3,re-reentrant_2,re-reentrant_3
run --stage stage2 --scaling paper --no-control-gt --gt-trajs 50000 \
    --nets reentrant_4,reentrant_5,re-reentrant_4,re-reentrant_5
echo "=== overnight pipeline finished $(date) ==="
