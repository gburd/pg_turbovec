---
name: long-running-bench
description: Conventions for dispatching sub-agents that run long benchmarks or builds on remote hosts. Wraps commands in a heartbeat emitter so the parent can tell "running" from "hung".
---

# Dispatch convention for long-running sub-agents

A consistent failure mode in this repo's bench history: a sub-agent dispatches a 5–30 minute command on `arnold` (`cargo pgrx install --release`, a 50-query latency sweep, a 1 M-row index build) and the parent has no way to distinguish "still running" from "hung in some dead tool call". This has cost real wall-clock — three sub-agents needed manual takeover after wedging at flat tool-counts for an hour or more.

This skill encodes the heartbeat pattern that prevents the recurrence.

## The wrapper

`benches/scripts/lib/with-heartbeat.sh` (~80 lines of bash) wraps any command. It:

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
   ssh gburd@arnold 'nohup bash benches/scripts/lib/with-heartbeat.sh /scratch/install.log \
        cargo pgrx install --release \
            --features "pg17 experimental_index_am relfile_storage" \
        > /dev/null 2>&1 &'
   ```

2. **Poll via the companion `poll-heartbeat.sh`** — returns one of `STILL_RUNNING` (rc=1), `HUNG` (rc=2), or `DONE rc=N` (rc=N):

   ```bash
   ssh gburd@arnold 'bash /scratch/pg_turbovec-bench/pg_turbovec/benches/scripts/poll-heartbeat.sh /scratch/install.log 60'
   ```

3. **Drive the agent loop off the rc** — `STILL_RUNNING` keeps polling, `HUNG` triggers a steer / abort, `DONE` lets the agent move on with the wrapped command's exit code.

## Dispatch prompt template

Insert this block into any prompt that dispatches a long-running command:

> **Heartbeat convention.** Every command expected to run > 60 seconds MUST be wrapped in `benches/scripts/lib/with-heartbeat.sh <log-path> <cmd> <args...>`. The parent will poll `stat -c %Y <log-path>` to verify liveness; if mtime hasn't moved in 3 minutes the parent will assume hang and steer/abort. Set `HEARTBEAT_SECS` lower for noisier-but-quicker feedback (e.g. 30 for builds), higher for low-noise long jobs (e.g. 300 for an overnight ANN build).

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
