#!/usr/bin/env bash
# Repro: a cancelled VACUUM in the index-vacuum phase of a FLAT turbovec
# index leaves a duplicate .tvim id (operator report
# pg_turbovec-cancelled-vacuum-dup-id-2026-11-13).
#
# The flat swap-remove loop moves each live tail slot into a dead hole via
# per-slot GenericXLog records (each durable on its own), then lowers
# n_vectors in the meta ONLY after the loop. A cancel raised by
# check_for_interrupts!() between slots aborts the vacuum txn but the
# already-written page moves survive, while n_vectors still counts the old
# tail -> a moved live row is now present at both its new slot and its old
# tail slot = "id N appears in more than one slot".
#
# Exit 2 = reproduced (is_corrupt=true). Exit 0 = clean (fixed).
# Needs a pgrx PG16 install with pg_turbovec. Standalone cluster on :28991.
set -uo pipefail
BIN=$(dirname "${PG_CONFIG:-$(ls -d ~/.pgrx/16.*/pgrx-install/bin/pg_config | head -1)}")
DATA=/tmp/cvac_pgdata; PORT=28991; LOG=/tmp/cvac_pg.log
DIM=768; N=${N:-60000}
export PGHOST=/tmp PGPORT=$PORT PGDATABASE=postgres
pkill -9 -f cvac_pgdata 2>/dev/null || true
if [ -d "$DATA" ]; then mv "$DATA" "/tmp/ocv_$$" && find "/tmp/ocv_$$" -mindepth 1 -delete && rmdir "/tmp/ocv_$$"; fi
"$BIN/initdb" -D "$DATA" -A trust >/dev/null 2>&1
printf "shared_preload_libraries = 'pg_turbovec'\nunix_socket_directories = '/tmp'\n" >> "$DATA/postgresql.conf"
"$BIN/pg_ctl" -D "$DATA" -o "-p $PORT" -l "$LOG" -w start >/dev/null
trap '"$BIN/pg_ctl" -D "$DATA" -m fast -w stop >/dev/null 2>&1' EXIT
P="$BIN/psql -X -q -At -v ON_ERROR_STOP=1"

$P <<SQL
DROP TABLE IF EXISTS rep3;
CREATE EXTENSION IF NOT EXISTS pg_turbovec;
CREATE TABLE rep3 (id bigint PRIMARY KEY, emb turbovec.vector);
ALTER TABLE rep3 SET (autovacuum_enabled = off);
CREATE INDEX rep3_tv ON rep3 USING turbovec (emb turbovec.vec_cosine_ops)
  WITH (dim='$DIM', bit_width='4') WHERE emb IS NOT NULL;
INSERT INTO rep3
SELECT i, ('[' || string_agg(((hashint8(i * $DIM + g) % 1000) / 1000.0)::text, ',') || ']')::turbovec.vector
FROM generate_series(1, $N) i, generate_series(1, $DIM) g GROUP BY i;
DELETE FROM rep3 WHERE id % 3 = 1;
SQL
echo "before: $($P -c "SELECT is_corrupt FROM turbovec.turbovec_check('rep3_tv'::regclass)")"

( PGAPPNAME=rep3_vac $P -c "VACUUM rep3" ) &
vpid=$!
seen=0
until [ "$($P -c "SELECT phase FROM pg_stat_progress_vacuum WHERE relid='rep3'::regclass" 2>/dev/null)" = "vacuuming indexes" ]; do
  kill -0 $vpid 2>/dev/null || { echo "vacuum finished before index phase"; break; }
  sleep 0.005
done
sleep 0.05
$P -c "SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE application_name='rep3_vac'" >/dev/null 2>&1 || true
wait $vpid 2>/dev/null || true

res=$($P -c "SELECT is_corrupt||'|'||COALESCE(reason,'') FROM turbovec.turbovec_check('rep3_tv'::regclass)")
echo "after: $res"
# an INSERT to show the write-blocking symptom (don't fail the script on it)
$P -v ON_ERROR_STOP=0 -c "INSERT INTO rep3 VALUES (${N}0, (SELECT emb FROM rep3 WHERE emb IS NOT NULL LIMIT 1))" 2>&1 | grep -iE 'ERROR|INSERT' | head -1 || true

corrupt=${res%%|*}
case "$corrupt" in
  t|true)  echo "RESULT: REPRODUCED (is_corrupt=true)"; exit 2 ;;
  f|false) echo "RESULT: CLEAN (is_corrupt=false)"; exit 0 ;;
  *)       echo "RESULT: INCONCLUSIVE ($res)"; exit 3 ;;
esac
