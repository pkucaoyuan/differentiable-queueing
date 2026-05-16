#!/bin/bash
# Background monitor: logs job progress every 10 min until both jobs complete
LOG=/user/yc4911/DJ_OR/differentiable-queueing/logs/monitor.log
mkdir -p $(dirname $LOG)
> $LOG

source /opt/n1ge/default/common/settings.sh 2>/dev/null

while true; do
    echo "============================================" >> $LOG
    echo "Time: $(date +'%Y-%m-%d %H:%M:%S')" >> $LOG
    echo "============================================" >> $LOG

    JOBS=$(qstat 2>&1 | grep -E "8631656|8631657")
    if [ -z "$JOBS" ]; then
        echo "All jobs done." >> $LOG
        echo "" >> $LOG
        echo "=== Final reentrant_2 last 10 test costs ===" >> $LOG
        grep "test cost" /user/yc4911/DJ_OR/differentiable-queueing/run_ste_reentrant_2.sh.o8631657 2>/dev/null | tail -10 >> $LOG
        echo "" >> $LOG
        echo "=== Final full_reproduction tail ===" >> $LOG
        tail -50 /user/yc4911/DJ_OR/differentiable-queueing/run_full_reproduction.sh.o8631656 2>/dev/null >> $LOG
        break
    fi

    echo "$JOBS" >> $LOG
    echo "" >> $LOG

    # full_reproduction progress
    LOG_F=/user/yc4911/DJ_OR/differentiable-queueing/run_full_reproduction.sh.o8631656
    if [ -f "$LOG_F" ]; then
        SIZE=$(stat -c%s "$LOG_F")
        echo "[full_repro] log size: ${SIZE}B" >> $LOG
        # Latest milestone
        LATEST=$(grep -E "alpha=|gap=|TEST|Section|cossim|Verification|Total time" "$LOG_F" 2>/dev/null | tail -3)
        if [ -n "$LATEST" ]; then
            echo "  Latest:" >> $LOG
            echo "$LATEST" | sed 's/^/    /' >> $LOG
        fi
    fi

    # reentrant_2 progress
    LOG_R=/user/yc4911/DJ_OR/differentiable-queueing/run_ste_reentrant_2.sh.o8631657
    if [ -f "$LOG_R" ]; then
        N_EPOCHS=$(grep -c "test cost" "$LOG_R" 2>/dev/null)
        LAST_TEST=$(grep "test cost" "$LOG_R" 2>/dev/null | tail -1 | awk '{print $NF}')
        FIRST_TEST=$(grep "test cost" "$LOG_R" 2>/dev/null | head -1 | awk '{print $NF}')
        echo "[reentrant_2] epochs done: $N_EPOCHS / 100" >> $LOG
        echo "  first test cost: $FIRST_TEST" >> $LOG
        echo "  last test cost:  $LAST_TEST" >> $LOG
    fi

    echo "" >> $LOG

    sleep 600
done
