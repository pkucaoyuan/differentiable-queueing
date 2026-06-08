#!/bin/bash
# Orchestrator round 3: waits for E5 + Polyak v2 + cμ baseline.
set -u
cd /user/yc4911/DJ_OR/differentiable-queueing

LOG_DIR=logs/direct
PYTHON=/user/yc4911/miniconda3/envs/gpu/bin/python

stage() {
    echo "════════════════════════════════════════════════════════════════"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "════════════════════════════════════════════════════════════════"
}

wait_for_pid() {
    local pid=$1; local name=$2
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] $name (PID $pid) already finished"; return 0
    fi
    echo "[$(date '+%H:%M:%S')] Waiting for $name (PID $pid)..."
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%H:%M:%S')] $name (PID $pid) finished"
}

stage "Stage 1: wait for E5 + Polyak v2 + cμ baseline"
[ -f $LOG_DIR/E5_heavy_traffic.pid ] && wait_for_pid "$(cat $LOG_DIR/E5_heavy_traffic.pid)" "E5"
[ -f $LOG_DIR/polyak_eval.pid ]      && wait_for_pid "$(cat $LOG_DIR/polyak_eval.pid)"      "polyak_v2"
[ -f $LOG_DIR/cmu_baseline.pid ]     && wait_for_pid "$(cat $LOG_DIR/cmu_baseline.pid)"     "cmu_baseline"

stage "Stage 2: rebuild all figures"
$PYTHON reports/build_figures.py 2>&1 || true
$PYTHON reports/build_fig_papergrid.py 2>&1 || true
$PYTHON reports/build_fig_polyak.py 2>&1 || true
$PYTHON reports/build_fig_queue_ordering.py 2>&1 || true
# Build cμ-vs-STE figure (will write below)
[ -f reports/build_fig_ste_vs_cmu.py ] && $PYTHON reports/build_fig_ste_vs_cmu.py 2>&1 || true
[ -f reports/build_fig_heavy_traffic.py ] && $PYTHON reports/build_fig_heavy_traffic.py 2>&1 || true

stage "Stage 3: refresh artifact matrix"
$PYTHON repro/render_matrix.py

stage "Stage 4: commit + push"
cd /user/yc4911/DJ_OR
git add differentiable-queueing/results/ \
        differentiable-queueing/reports/ \
        differentiable-queueing/repro/ \
        differentiable-queueing/experiments/reproduction/test_*.py \
        differentiable-queueing/logs/direct/*.out 2>/dev/null

if git diff --cached --quiet; then
    echo "[$(date '+%H:%M:%S')] No changes to commit"
else
    git commit -m "Round 3: Polyak v2 + cμ baseline + E5 heavy traffic + queue ordering fig

- results/reproduction/polyak_eval.json — Polyak vs last-iterate w/ correct
  time normalization (state.time, not T steps)
- results/reproduction/ste_vs_cmu_benchmark.json — §7 Tables 1-5 closer
- results/revision/E5_heavy_traffic_*.json — heavy-traffic curve
- reports/figures/fig_section52_queue_ordering.{png,pdf} — Fig 9 left
- reports/figures/fig_section7_polyak_vs_last.{png,pdf} — Polyak comparison
- reports/figures/fig_section7_ste_vs_cmu.{png,pdf} — STE/cμ benchmark
- Refreshed status matrix

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
    git push origin main 2>&1 | tail -5
fi
stage "Round 3 done at $(date)"
