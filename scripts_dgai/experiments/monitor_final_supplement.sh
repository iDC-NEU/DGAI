#!/bin/bash
# Monitor a final_update_supplement process group and pause it on unsafe storage state.

set -u

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <master_pid> <master_log> <monitor_log>" >&2
    exit 1
fi

MASTER_PID="$1"
MASTER_LOG="$2"
MONITOR_LOG="$3"
INTERVAL="${MONITOR_INTERVAL_SECONDS:-600}"
MIN_FREE_GB="${MIN_FREE_GB:-150}"
TARGET_FS="${MONITOR_TARGET_FS:-${DISKANN_INDEX_PATH}}"

mkdir -p "$(dirname "$MONITOR_LOG")"

while kill -0 "$MASTER_PID" 2>/dev/null; do
    {
        echo "===== $(date '+%F %T') ====="
        ps -o pid,ppid,stat,pcpu,pmem,etime,cmd --forest -g "$MASTER_PID" 2>/dev/null || true
        free -m | sed -n '1,3p'
        df -h "$TARGET_FS" | tail -n 1
        echo "--- master tail ---"
        tail -n 12 "$MASTER_LOG" 2>/dev/null || true
    } >>"$MONITOR_LOG"

    avail_kb="$(df -Pk "$TARGET_FS" | awk 'NR==2 {print $4}')"
    min_kb=$((MIN_FREE_GB * 1024 * 1024))
    mount_options="$(findmnt -T "$TARGET_FS" -n -o OPTIONS)"
    if [ "$avail_kb" -lt "$min_kb" ] || printf '%s' "$mount_options" | tr ',' '\n' | grep -qx ro; then
        echo "$(date '+%F %T') PAUSE: free=$((avail_kb / 1024 / 1024))GiB options=${mount_options}" >>"$MONITOR_LOG"
        kill -STOP "-$MASTER_PID" 2>/dev/null || kill -STOP "$MASTER_PID" 2>/dev/null || true
        exit 2
    fi
    sleep "$INTERVAL"
done

echo "$(date '+%F %T') master process exited" >>"$MONITOR_LOG"
