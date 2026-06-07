#!/bin/bash
# Background orchestrator: waits for current jobs, refreshes figures, commits, pushes.
# Runs additional follow-up experiments after current ones finish.
# Safe to detach with nohup — survives terminal disconnect.

set -u  # error on undefined; allow command failures for individual stages
cd /user/yc4911/DJ_OR/differentiable-queueing

LOG_DIR=logs/direct
PYTHON=/user/yc4911/miniconda3/envs/gpu/bin/python
PYTHON_GPU=/user/yc4911/miniconda3/envs/gpu-cu/bin/python

stage() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "════════════════════════════════════════════════════════════════"
}

wait_for_pid() {
    local pid=$1; local name=$2
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] $name (PID $pid) already finished"
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] Waiting for $name (PID $pid)..."
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%H:%M:%S')] $name (PID $pid) finished"
}

# ────────────────────────────────────────────────────────────────────
# STAGE 1: Wait for currently-running jobs
# ────────────────────────────────────────────────────────────────────
stage "Stage 1: wait for cmu_papergrid + section51_mid"
[ -f $LOG_DIR/cmu_papergrid.pid ] && wait_for_pid "$(cat $LOG_DIR/cmu_papergrid.pid)" "cmu_papergrid"
[ -f $LOG_DIR/section51_mid.pid ] && wait_for_pid "$(cat $LOG_DIR/section51_mid.pid)" "section51_mid"

# ────────────────────────────────────────────────────────────────────
# STAGE 2: Rebuild figures with the new data
# ────────────────────────────────────────────────────────────────────
stage "Stage 2: regenerate all figures"
$PYTHON reports/build_figures.py || echo "WARN: build_figures.py failed"
$PYTHON_GPU reports/build_fig_gpu.py || echo "WARN: build_fig_gpu.py failed"

# Also build a paper-grid figure if the new JSON appeared
if [ -f results/reproduction/cmu_papergrid_pathwise.json ]; then
    stage "Stage 2b: build §5.2 papergrid figure"
    $PYTHON reports/build_fig_papergrid.py 2>&1 || echo "WARN: papergrid fig failed"
fi

if [ -f results/reproduction/gradient_comparison_criss_cross_bh_rho0.95.json ]; then
    # Already exists from quick run; mid run overwrites — that's fine
    stage "Stage 2c: rebuild §5.1 figure (mid-canonical refresh)"
    $PYTHON reports/build_fig_section51.py 2>&1 || echo "WARN: §5.1 fig failed"
fi

# ────────────────────────────────────────────────────────────────────
# STAGE 3: Refresh status.json + matrix
# ────────────────────────────────────────────────────────────────────
stage "Stage 3: refresh artifact matrix"
$PYTHON repro/render_matrix.py

# ────────────────────────────────────────────────────────────────────
# STAGE 4: Commit and push
# ────────────────────────────────────────────────────────────────────
stage "Stage 4: git add / commit / push"
cd /user/yc4911/DJ_OR
git add differentiable-queueing/results/reproduction/cmu_papergrid_*.json 2>/dev/null
git add differentiable-queueing/results/reproduction/gradient_comparison_*.json 2>/dev/null
git add differentiable-queueing/reports/figures/ 2>/dev/null
git add differentiable-queueing/reports/build_fig_*.py 2>/dev/null
git add differentiable-queueing/repro/*.{json,csv,md} 2>/dev/null
git add differentiable-queueing/logs/direct/*.{out,err} 2>/dev/null

if git diff --cached --quiet; then
    echo "[$(date '+%H:%M:%S')] No changes to commit"
else
    git commit -m "$(cat <<EOF
Auto-orchestrated refresh after §5.2 papergrid + §5.1 mid finished

- New: results/reproduction/cmu_papergrid_*.json (§5.2 with gap=0.1)
- New: results/reproduction/gradient_comparison_*.json (§5.1 mid-canonical 30×30)
- Refreshed figures from the new data
- Updated status.json / matrix / summary

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
    git push origin main 2>&1 | tail -5
fi

stage "Orchestrator done at $(date)"
