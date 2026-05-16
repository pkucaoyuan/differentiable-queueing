#!/bin/bash
# Auto-monitor: tracks all currently-running yc4911 jobs, archives finished logs,
# emits one line per state change (start, finish-success, finish-fail).
# Note: This runs as a Bash run_in_background with grep --line-buffered,
# every line is a notification.

source /opt/n1ge/default/common/settings.sh 2>/dev/null
PROJECT=/user/yc4911/DJ_OR/differentiable-queueing
mkdir -p $PROJECT/logs/reproduction

declare -A SEEN_STATE
declare -A JOB_NAME

# poll every 60s
while true; do
    QSTAT=$(qstat 2>&1)
    NOW=$(date '+%Y-%m-%d %H:%M:%S')

    # Track currently running jobs
    declare -A CURRENT
    while IFS= read -r line; do
        if [[ "$line" =~ ^[[:space:]]*([0-9]+)[[:space:]]+([0-9.]+)[[:space:]]+([^[:space:]]+)[[:space:]]+yc4911[[:space:]]+([rqtwd]+) ]]; then
            jid="${BASH_REMATCH[1]}"
            name="${BASH_REMATCH[3]}"
            state="${BASH_REMATCH[4]}"
            CURRENT[$jid]="$state"
            JOB_NAME[$jid]="$name"
            # detect state change
            if [ "${SEEN_STATE[$jid]:-none}" != "$state" ]; then
                echo "[$NOW] job $jid ($name) state: ${SEEN_STATE[$jid]:-new}->$state"
                SEEN_STATE[$jid]="$state"
            fi
        fi
    done <<< "$QSTAT"

    # Detect finished jobs (in SEEN but not in CURRENT)
    for jid in "${!SEEN_STATE[@]}"; do
        if [ -z "${CURRENT[$jid]}" ]; then
            name="${JOB_NAME[$jid]}"
            # Check exit status
            log_o=$(ls $PROJECT/*.o$jid 2>/dev/null)
            log_e=$(ls $PROJECT/*.e$jid 2>/dev/null)

            if [ -n "$log_o" ]; then
                if grep -q "=== Done ===" "$log_o"; then
                    # Check stderr for errors
                    has_err="false"
                    if [ -n "$log_e" ] && grep -qE "Traceback|Error|Fail" "$log_e" 2>/dev/null; then
                        has_err="true"
                    fi
                    if [ "$has_err" = "true" ]; then
                        # Get error type
                        err=$(grep -E "Error|RuntimeError|FileNotFoundError" "$log_e" 2>/dev/null | tail -1 | cut -c1-100)
                        echo "[$NOW] job $jid ($name) FINISHED-WITH-ERROR: $err"
                    else
                        # Try to extract one summary number
                        summary=$(grep -E "Total time|min test|cost=" "$log_o" 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-150)
                        echo "[$NOW] job $jid ($name) FINISHED-OK: $summary"
                    fi
                else
                    # job ended without "Done" — killed by scheduler
                    qac=$(qacct -j $jid 2>/dev/null | grep -E "failed|exit_status|maxvmem" | tr '\n' ' ' | cut -c1-150)
                    echo "[$NOW] job $jid ($name) DIED: $qac"
                fi
                # Archive logs
                mv $PROJECT/*.{o,e,po,pe}$jid $PROJECT/logs/reproduction/ 2>/dev/null
            else
                echo "[$NOW] job $jid ($name) GONE (no log found)"
            fi
            unset SEEN_STATE[$jid]
            unset JOB_NAME[$jid]
        fi
    done

    # If no jobs at all left, also clean up
    if [ ${#CURRENT[@]} -eq 0 ] && [ ${#SEEN_STATE[@]} -eq 0 ]; then
        echo "[$NOW] no jobs in queue — auto-monitor idle"
    fi

    unset CURRENT

    sleep 60
done
