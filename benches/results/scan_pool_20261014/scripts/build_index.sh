#!/usr/bin/env bash
# build_index.sh <db> <label> <pre-SQL (SETs)> <CREATE INDEX sql>
# Records wall, WAL bytes, index size, data-dir delta, peak summed RSS (1 Hz) of all postgres procs, progress phases.
set -uo pipefail
export PATH=/opt/pg18/bin:$PATH
DB=$1; L=$2; PRE=$3; SQL=$4; R=~/results/q/build; mkdir -p $R
PS="psql -h /tmp -X -v ON_ERROR_STOP=1 -d $DB"
PM=$(head -1 /data/pgdata/postmaster.pid)
( mx=0; while true; do s=0; for p in $(pgrep -P $PM) ; do r=$(awk '/RssAnon/{print $2}' /proc/$p/status 2>/dev/null); s=$((s+${r:-0})); done
    [ $s -gt $mx ] && mx=$s; echo "$(date +%s) anon_kb=$s max_anon_kb=$mx" >> $R/$L.rss; 
    psql -h /tmp -X -d $DB -Atc "SELECT now()::time, phase, blocks_done, blocks_total, tuples_done, tuples_total FROM pg_stat_progress_create_index" >> $R/$L.progress 2>/dev/null; sleep 1; done ) &
SP=$!
LSN0=$($PS -Atc "SELECT pg_current_wal_lsn()"); DU0=$(du -sb /data/pgdata | cut -f1)
T0=$(date +%s.%N)
$PS -c "$PRE" -c "$SQL" > $R/$L.out 2>&1; RC=$?
T1=$(date +%s.%N)
kill $SP; wait $SP 2>/dev/null
$PS -c "CHECKPOINT" >/dev/null
LSN1=$($PS -Atc "SELECT pg_current_wal_lsn()"); DU1=$(du -sb /data/pgdata | cut -f1)
IDX=$(echo "$SQL" | grep -oiE 'INDEX +[a-z_0-9]+' | head -1 | awk '{print $2}')
{
echo "label=$L db=$DB rc=$RC"; echo "pre: $PRE"; echo "sql: $SQL"
echo "wall_s=$(echo "$T1-$T0" | bc)"
echo "wal_bytes=$($PS -Atc "SELECT pg_wal_lsn_diff('$LSN1','$LSN0')") (includes the post-build CHECKPOINT record)"
echo "index_bytes=$($PS -Atc "SELECT pg_relation_size('$IDX')") total_index_bytes=$($PS -Atc "SELECT pg_total_relation_size('$IDX')")"
echo "datadir_delta_bytes=$((DU1-DU0)) (includes WAL)"
echo "peak_summed_RssAnon_kb=$(tail -1 $R/$L.rss | sed 's/.*max_anon_kb=//') (private anon of all postmaster children, 1 Hz; excludes shared_buffers)"
cat $R/$L.out
} > $R/$L.summary
cat $R/$L.summary
exit $RC
