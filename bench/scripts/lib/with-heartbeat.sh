#!/usr/bin/env bash
# bench/scripts/lib/with-heartbeat.sh
#
# Wrap a long-running command so it emits a heartbeat line to a
# log file every $HEARTBEAT_SECS (default 60) seconds. The wrapped
# command's stdout/stderr are still captured to the same log,
# interleaved with the heartbeats.
#
# This solves the failure mode where a sub-agent dispatches a
# multi-minute build/benchmark and can't tell from the outside
# whether the underlying job is alive or hung. With this wrapper,
# polling `tail -1 <log>` or `stat -c %Y <log>` from the agent
# tells you when the command last emitted ANY signal of life,
# which is enough to distinguish "running" from "hung".
#
# Usage:
#   bash with-heartbeat.sh <log-path> <cmd> [<args>...]
#
# Environment:
#   HEARTBEAT_SECS   — interval between '.' emissions (default 60)
#   HEARTBEAT_LABEL  — identifying string on each heartbeat line
#                       (default = the command name)
#
# Exit code: the wrapped command's exit code.
#
# Example:
#   bash with-heartbeat.sh /tmp/install.log \
#       cargo pgrx install --release \
#           --features "pg17 experimental_index_am relfile_storage"
#
#   # In another shell, the agent polls:
#   stat -c %Y /tmp/install.log     # mtime — if older than ~3*HEARTBEAT_SECS, hung
#   tail -1 /tmp/install.log         # last heartbeat or output line

set -uo pipefail

if [ $# -lt 2 ]; then
    echo "usage: with-heartbeat.sh <log-path> <cmd> [<args>...]" >&2
    exit 2
fi

LOG=$1; shift
HEARTBEAT_SECS=${HEARTBEAT_SECS:-60}
HEARTBEAT_LABEL=${HEARTBEAT_LABEL:-$(basename "$1")}

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

# Write an initial banner so callers see the wrapped command.
{
    printf '[heartbeat] starting: %s\n' "$*"
    printf '[heartbeat] interval: %ss\n' "$HEARTBEAT_SECS"
    printf '[heartbeat] label:    %s\n' "$HEARTBEAT_LABEL"
} >> "$LOG"

# Background heartbeat loop. Writes a single '.' line with a
# timestamp every HEARTBEAT_SECS seconds. PID is tracked so we
# can kill it on exit.
(
    while true; do
        sleep "$HEARTBEAT_SECS"
        printf '. [%s %s elapsed]\n' "$HEARTBEAT_LABEL" "$(date +%H:%M:%S)" >> "$LOG"
    done
) &
HEARTBEAT_PID=$!

# Make sure the heartbeat dies if we get signalled or exit.
trap 'kill -9 "$HEARTBEAT_PID" 2>/dev/null; wait "$HEARTBEAT_PID" 2>/dev/null' EXIT TERM INT

# Run the wrapped command, sending its output to the same log.
"$@" >> "$LOG" 2>&1
RC=$?

# Final status banner — include the exit code so polling agents
# can tell completion from "still running".
{
    if [ "$RC" -eq 0 ]; then
        printf '[heartbeat] DONE rc=0 (clean exit)\n'
    else
        printf '[heartbeat] DONE rc=%s (failed)\n' "$RC"
    fi
} >> "$LOG"

exit "$RC"
