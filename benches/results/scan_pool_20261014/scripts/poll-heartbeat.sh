#!/usr/bin/env bash
# benches/scripts/poll-heartbeat.sh
#
# Parent-side companion to benches/scripts/lib/with-heartbeat.sh.
# Watches a heartbeat log file from outside (e.g. over ssh) and
# returns one of:
#
#   - 0  with the wrapped command's exit code printed if the log
#        contains a `[heartbeat] DONE rc=N` line.
#   - 1  with "STILL_RUNNING" if the mtime is fresh.
#   - 2  with "HUNG" if the mtime is older than 3*HEARTBEAT_SECS.
#
# Usage:
#   bash poll-heartbeat.sh <log-path> [<heartbeat_secs>]
#
# Example (parent shell, polling an arnold job):
#   ssh arnold "bash $REPO/benches/scripts/poll-heartbeat.sh /scratch/install.log 60"

set -uo pipefail

if [ $# -lt 1 ]; then
    echo "usage: poll-heartbeat.sh <log-path> [<heartbeat_secs>]" >&2
    exit 64
fi

LOG=$1
HEARTBEAT_SECS=${2:-${HEARTBEAT_SECS:-60}}
HUNG_THRESHOLD=$(( HEARTBEAT_SECS * 3 ))

if [ ! -f "$LOG" ]; then
    echo "NO_LOG"
    exit 1
fi

# Look for the DONE line first — completed jobs are unambiguous.
done_line=$(grep -E '^\[heartbeat\] DONE rc=' "$LOG" | tail -1)
if [ -n "$done_line" ]; then
    rc=$(printf '%s' "$done_line" | grep -oE 'rc=[0-9]+' | head -1 | cut -d= -f2)
    echo "DONE rc=$rc"
    exit "$rc"
fi

# Not done. Check mtime. `stat -c %Y` returns the seconds-since-epoch
# of last modification; compare against `date +%s` for "how stale".
now=$(date +%s)
mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
age=$(( now - mtime ))

if [ "$age" -gt "$HUNG_THRESHOLD" ]; then
    last_line=$(tail -1 "$LOG" 2>/dev/null)
    printf 'HUNG age=%ss threshold=%ss last=%q\n' \
           "$age" "$HUNG_THRESHOLD" "$last_line"
    exit 2
fi

last_line=$(tail -1 "$LOG" 2>/dev/null)
printf 'STILL_RUNNING age=%ss last=%q\n' "$age" "$last_line"
exit 1
