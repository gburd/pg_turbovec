---
name: long-running-bench
description: Conventions for dispatching sub-agents that run long benchmarks or builds on remote hosts. Wraps commands in a heartbeat emitter so the parent can tell "running" from "hung".
---

# Dispatch convention for long-running sub-agents

A consistent failure mode in this repo's bench history: a sub-agent dispatches a 5–30 minute command on `arnold` (`cargo pgrx install --release`, a 50-query latency sweep, a 1 M-row index build) and the parent has no way to distinguish "still running" from "hung in some dead tool call". This has cost real wall-clock — three sub-agents needed manual takeover after wedging at flat tool-counts for an hour or more.

This skill encodes the heartbeat pattern that prevents the recurrence.

## The wrapper

`bench/scripts/lib/with-heartbeat.sh` (~80 lines of bash) wraps any command. It:

1. Writes a banner line to `<log>` listing the command + interval.
2. Forks a background loop that appends `. [label HH:MM:SS elapsed]` to `<log>` every `$HEARTBEAT_SECS` (default 60).
3. Runs the command with stdout/stderr appended to the same log.
4. Writes a `[heartbeat] DONE rc=N` line on exit.
5. Kills the heartbeat fork in its `trap` so a Ctrl-C from the parent leaves no orphan.

Returns the wrapped command's exit code unchanged.

## Dispatch instructions for the agent

When the dispatched task includes any single command expected to run > 60 seconds, the dispatch prompt MUST instruct the agent to:

1. **Wrap with `with-heartbeat.sh`**:

   ```bash
   ssh gburd@arnold 'nohup bash bench/scripts/lib/with-heartbeat.sh /scratch/install.log \
        cargo pgrx install --release \
            --features "pg17 experimental_index_am relfile_storage" \
        > /dev/null 2>&1 &'
   ```

2. **Poll the log's mtime, not the command itself**:

   ```bash
   # Liveness check from the parent agent's perspective.
   # If the log mtime hasn't moved in 3*HEARTBEAT_SECS, the
   # wrapped command is hung; otherwise it's making progress.
   ssh gburd@arnold 'stat -c %Y /scratch/install.log'
   ```

3. **Read completion via the DONE line**:

   ```bash
   ssh gburd@arnold 'grep -E "^\[heartbeat\] DONE" /scratch/install.log'
   # exits 0 with a line if done, exits 1 if still running.
   ```

## Dispatch prompt template

Insert this block into any prompt that dispatches a long-running command:

> **Heartbeat convention.** Every command expected to run > 60 seconds MUST be wrapped in `bench/scripts/lib/with-heartbeat.sh <log-path> <cmd> <args...>`. The parent will poll `stat -c %Y <log-path>` to verify liveness; if mtime hasn't moved in 3 minutes the parent will assume hang and steer/abort. Set `HEARTBEAT_SECS` lower for noisier-but-quicker feedback (e.g. 30 for builds), higher for low-noise long jobs (e.g. 300 for an overnight ANN build).

## Why mtime not tail

Tail content is the obvious choice but mtime is cheaper, faster, and works regardless of whether the wrapped command produces any output of its own. A `cargo build` that only prints "Finished" at the end would otherwise look hung between heartbeats.

## When to skip

- Single commands that always finish in < 60 s (no point).
- Commands that already emit progress every 10 s (e.g. `pg_basebackup` with `--progress`, `pgbench`).
- Test runs invoked via `cargo pgrx test` — pgrx already prints progress per test.

## Anti-patterns (caught the hard way)

- **Piping the wrapped command through `nvim` / `less` / `tail -f`**. Those block on tty input even when run inside a non-interactive shell. The wrapper redirects stdout/stderr to the log file, but if the command itself spawns a pager you'll still hang. Always pass `--no-pager` to git, `PAGER=cat` to anything that respects it, and never `… | nvim`.
- **Forgetting `nohup` on the outer ssh**. Without it, the agent's ssh disconnect kills the wrapped command. Use `nohup … &` or `tmux new-session -d`.
- **Polling more often than HEARTBEAT_SECS**. The mtime granularity is the heartbeat interval; faster polling wastes ssh round-trips.
