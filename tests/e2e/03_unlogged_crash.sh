#!/usr/bin/env bash
# An unlogged table's index must be usable after a crash: reset to empty,
# then accept inserts and answer queries.  2.16.3 (and contrib/bloom) reset it
# to a zeroed page when the crash came before a checkpoint ("bad magic on
# meta page" until REINDEX).  Control: a logged table's index survives the
# same crash with its rows.
. "$(dirname "$0")/lib.sh"
mk ul UNLOGGED
mk lg
crash; start
expect "unlogged rows after crash" "$(q -c "SELECT count(*) FROM ul")" "0"
q -c "INSERT INTO ul SELECT g, $(vec g) FROM generate_series(1, 100) g"
expect "unlogged index after crash + insert" "$(found ul $(seq -s ' ' 1 100) | wc -w)" "100"
expect "logged index after crash" "$(found lg 1 500 1000 2000)" "1 500 1000 2000"
