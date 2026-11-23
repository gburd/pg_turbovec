#!/usr/bin/env bash
# 04_savepoint_tid_reuse on an IVF index.  VACUUM tombstones an IVF index's
# dead slots instead of removing them, so the freed TIDs are still on disk
# (tombstoned) when the rolled-back inserts reuse them; the rollback must
# leave them tombstoned, not live with a rolled-back row's vector.
TR_WITH="bit_width = 4, lists = 16" exec bash "$(dirname "$0")/04_savepoint_tid_reuse.sh"
