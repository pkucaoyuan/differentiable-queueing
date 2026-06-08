#!/bin/bash
# Orchestrator round 2: waits for Polyak + §5.1 v3 + E5, then refresh + push.
set -u
cd /user/yc4911/DJ_OR/differentiable-queueing

LOG_DIR=logs/direct
PYTHON=/user/yc4911/miniconda3/envs/gpu/bin/python

stage() {
    echo ""
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

stage "Stage 1: wait for Polyak + §5.1 v3 + E5"
[ -f $LOG_DIR/polyak_eval.pid ]     && wait_for_pid "$(cat $LOG_DIR/polyak_eval.pid)"     "polyak_eval"
[ -f $LOG_DIR/section51_mid.pid ]   && wait_for_pid "$(cat $LOG_DIR/section51_mid.pid)"   "section51_v3"
[ -f $LOG_DIR/E5_heavy_traffic.pid ] && wait_for_pid "$(cat $LOG_DIR/E5_heavy_traffic.pid)" "E5_heavy_traffic"

stage "Stage 2: rebuild figures"
$PYTHON reports/build_figures.py || true
$PYTHON reports/build_fig_papergrid.py 2>&1 || true
$PYTHON reports/build_fig_section51.py 2>&1 || echo "WARN: §5.1 fig failed"
[ -f reports/build_fig_polyak.py ]   && $PYTHON reports/build_fig_polyak.py 2>&1 || true
[ -f reports/build_fig_heavy_traffic.py ] && $PYTHON reports/build_fig_heavy_traffic.py 2>&1 || true

stage "Stage 3: refresh artifact matrix"
$PYTHON repro/render_matrix.py

stage "Stage 4: git add + commit + push"
cd /user/yc4911/DJ_OR
git add differentiable-queueing/results/ \
        differentiable-queueing/reports/figures/ \
        differentiable-queueing/reports/build_fig_*.py \
        differentiable-queueing/repro/ \
        differentiable-queueing/logs/direct/*.out 2>/dev/null

if git diff --cached --quiet; then
    echo "[$(date '+%H:%M:%S')] No changes to commit"
else
    git commit -m "$(cat <<EOF
Round 2: §7 Polyak eval + §5.1 v3 + E5 heavy traffic curve

- results/reproduction/polyak_eval.json — avg-iterate vs last-iterate per env
- results/reproduction/gradient_comparison_*.json — §5.1 with 15×20×50K
- results/revision/E5_heavy_traffic_*.json — heavy-traffic regime sweep
- Refreshed figures and status matrix

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
    git push origin main 2>&1 | tail -5
fi

stage "Round 2 orchestrator done at $(date)"
