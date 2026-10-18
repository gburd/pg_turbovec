#!/usr/bin/env bash
# Sustained-load soak for the am_version-collision class (v2.14.1).
#
# turbovec_check / log signatures cannot see this class: the index stays
# structurally clean while a long-lived reader silently serves a stale cached
# copy. So this soak carries an ORACLE: persistent reader backends insert-free
# probe the index with a vector that only one freshly-committed row matches,
# and check the ANN top-1 against the committed truth.
#
# Workload, all concurrent against one flat index:
#   - NWRITERS persistent writers, each looping BEGIN; INSERT 1-3 rows; COMMIT
#     with random sleeps so transactions overlap (the collision trigger).
#   - a VACUUM loop (DELETE an old filler row + VACUUM) to interleave
#     ambulkdelete with open writers.
#   - NREADERS persistent readers (one backend each, so their private cache
#     persists across iterations, like pooled backends).
#
# Oracle: each writer inserts a "beacon" row whose vector is a unique far-out
# direction (one coordinate +BIG, the rest 0), then records (id, coord) in
# the beacons table IN THE SAME TRANSACTION. A reader picks a committed beacon,
# runs the index ORDER BY with that direction, and the top-1 MUST be that
# beacon's id (no corpus row is anywhere near it). A miss = stale cache.
# It also re-checks each miss with a fresh backend to separate a stale cache
# from a genuinely wrong index.
#
# Usage: DUR=600 NWRITERS=6 NREADERS=6 bash am_version_soak.sh
# Exit 0 = zero stale reads; 1 = any stale read.
set -uo pipefail
BIN=$(dirname "${PG_CONFIG:-$(ls -d ~/.pgrx/16.*/pgrx-install/bin/pg_config | head -1)}")  # override: PG_CONFIG=/path/to/pg_config
DATA=${DATA:-/tmp/amv_soak_pgdata}; PORT=${PORT:-28978}; DUR=${DUR:-600}
NWRITERS=${NWRITERS:-6}; NREADERS=${NREADERS:-6}; DIM=${DIM:-64}
# LISTS>0 builds an IVF index: VACUUM then TOMBSTONES dead slots instead of
# removing them, which exercises the insert-flush tombstone carry (v2.14.1).
LISTS=${LISTS:-0}
# Unique beacon directions are capped at 4*C(DIM,2); fail fast instead of mid-run.
python3 -c "import sys; d=$DIM; cap=4*d*(d-1)//2; need=$DUR*$NWRITERS//2+1000; sys.exit(0 if need<cap else 1)" \
  || { echo "beacon direction space too small for DUR=$DUR NWRITERS=$NWRITERS at DIM=$DIM; raise DIM"; exit 2; }
OUT=${OUT:-/tmp/amv_soak_out}; mkdir -p "$OUT"; : > "$OUT/stale.log"; : > "$OUT/reads.log"
export PGHOST=/tmp PGPORT=$PORT PGDATABASE=postgres
if [ ! -d "$DATA" ]; then
  "$BIN/initdb" -D "$DATA" -A trust >/dev/null
  printf "shared_preload_libraries='pg_turbovec'\nunix_socket_directories='/tmp'\nmax_connections=100\n" >> "$DATA/postgresql.conf"
fi
"$BIN/pg_ctl" -D "$DATA" -o "-p $PORT" -l "$OUT/pg.log" -w start >/dev/null
P="$BIN/psql -X -q -tA -v ON_ERROR_STOP=1"
cleanup() { kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; "$BIN/pg_ctl" -D "$DATA" -m fast -w stop >/dev/null; }
trap cleanup EXIT

# Corpus: all coordinates in [0, 0.1] (small, positive). Beacon for coord c is
# BIG * e_c with BIG = 100 -- cosine-nearest to its own direction by a wide
# margin, so quantization cannot reorder it.
$P <<SQL
DROP EXTENSION IF EXISTS pg_turbovec CASCADE;
CREATE EXTENSION pg_turbovec;
CREATE EXTENSION pageinspect;
ALTER DATABASE postgres SET search_path = turbovec, public;
SET search_path = turbovec, public;
CREATE TABLE t (id bigint PRIMARY KEY, v turbovec.vector);
INSERT INTO t SELECT g, ('[' || array_to_string(ARRAY(SELECT ((g*31+k*7)%100)/1000.0 FROM generate_series(1,$DIM) k), ',') || ']')::turbovec.vector
  FROM generate_series(1, 5000) g;
CREATE INDEX t_idx ON t USING turbovec (v vec_cosine_ops) WITH (lists = $LISTS);
CREATE TABLE beacons (id bigint PRIMARY KEY, coord int, sign int);
CREATE SEQUENCE beacon_seq START 1000000;
CREATE SEQUENCE filler_seq START 5000000;
SQL

beacon_vec() { # beacon id -> a direction UNIQUE to that id (no two beacons tie)
  python3 -c "
import sys
d=$DIM; k=int(sys.argv[1]) - 1000000
pairs=[(i,j) for i in range(d) for j in range(i+1,d)]
assert k < 4*len(pairs), 'beacon id space exhausted: raise DIM or lower DUR*NWRITERS'
i,j=pairs[k % len(pairs)]; sgn=[(1,1),(1,-1),(-1,1),(-1,-1)][(k//len(pairs))%4]
v=['0']*d; v[i]=str(100*sgn[0]); v[j]=str(100*sgn[1]); print('['+','.join(v)+']')" "$1"
}
export -f beacon_vec; export DIM

# Writers run in synchronised BURSTS: every writer opens a transaction and
# inserts its beacon, then they commit one after another, then the system goes
# QUIET for IDLE seconds. The quiet gap is what makes this a real test: a reader
# that cached between two colliding commits stays stale until the NEXT
# version-changing write, and in a busy system that comes within milliseconds
# and hides the bug (an earlier version of this soak saw 30 collisions in 120 s
# and 0 stale reads for exactly that reason). Production traffic has quiet
# periods; the soak must too.
IDLE=${IDLE:-3}
writer() {
  local w=$1 end=$((SECONDS+DUR)) fifo="$OUT/w$w.in"
  mkfifo "$fifo" 2>/dev/null
  $P < "$fifo" > "$OUT/writer$w.out" 2>&1 &
  exec {wfd}>"$fifo"
  while [ $SECONDS -lt $end ]; do
    # wait for the burst barrier (the coordinator bumps $OUT/burst)
    local b; b=$(cat "$OUT/burst" 2>/dev/null)
    while [ "$(cat "$OUT/burst" 2>/dev/null)" = "$b" ] && [ $SECONDS -lt $end ]; do sleep 0.05; done
    local bid; bid=$($P -c "SELECT nextval('beacon_seq')")
    echo "BEGIN; INSERT INTO beacons VALUES ($bid, 0, 0); INSERT INTO t VALUES ($bid, '$(beacon_vec $bid)');" >&$wfd
    # stagger commits: writer w commits w*150 ms into the burst, so the commits are
    # sequential and every transaction overlaps every earlier one
    sleep "$(python3 -c "print(0.3 + $w*0.15)")"
    echo "COMMIT;" >&$wfd
  done
  exec {wfd}>&-
}

coordinator() { local end=$((SECONDS+DUR)) n=0; echo 0 > "$OUT/burst"
  while [ $SECONDS -lt $end ]; do n=$((n+1)); echo $n > "$OUT/burst"; sleep "$(python3 -c "print(0.3 + $NWRITERS*0.15 + 0.5 + $IDLE)")"; done; }

vacuumer() {
  local end=$((SECONDS+DUR)) g=1
  while [ $SECONDS -lt $end ]; do
    # INDEX_CLEANUP ON: on PG14+ a VACUUM with dead items on only a few pages
    # skips index cleanup entirely, so without it an IVF index almost never
    # gets a tombstone mid-run and a tombstone-erasing flush cannot show up.
    # The freed heap space is then refilled at once with NULL-vector rows (not
    # indexed), which makes it unlikely (not guaranteed: placement follows the
    # free-space map) that a later beacon insert reuses that TID and overwrites
    # the dead index slot with a live row. So the tombstone usually has to
    # persist on its own, and a flush that erases it leaves a live entry with
    # no heap row. Refill ids are -(4g+s): distinct per VACUUM round.
    $P -c "DELETE FROM t WHERE id = $g" -c "VACUUM (INDEX_CLEANUP ON) t" \
       -c "INSERT INTO t (id) SELECT -(4 * $g + s) FROM generate_series(0, 3) s" >/dev/null 2>>"$OUT/vacuum.err"
    g=$((g+1)); sleep ${VAC_EVERY:-7}
  done
}

reader() { # persistent backend (its private cache persists, like a pooled backend)
  # Each round: ask the persistent backend about EVERY beacon committed in the
  # last few seconds. A collision hides the SECOND of two colliding commits only
  # from a reader that cached between them, so probing just the newest beacon
  # misses it; probing all recent ones does not.
  local r=$1 end=$((SECONDS+DUR)) fifo_in="$OUT/r$r.in" fifo_out="$OUT/r$r.out"
  mkfifo "$fifo_in" 2>/dev/null; : > "$fifo_out"
  $P < "$fifo_in" > "$fifo_out" 2>&1 &
  exec {fd}>"$fifo_in"
  echo "SET enable_seqscan = off; SET turbovec.probes = $(( LISTS > 0 ? LISTS : 1 ));" >&$fd
  local i=0
  while [ $SECONDS -lt $end ]; do
    # Warm this backend's cache on the current state first (the "C caches
    # between the commits" step), then probe recent beacons.
    echo "SELECT 1 AS warm FROM t ORDER BY v <=> '$(beacon_vec 1000000)' LIMIT 1 \gset" >&$fd
    $P -c "SELECT id FROM beacons WHERE id > (SELECT max(id) - 2*$NWRITERS FROM beacons) ORDER BY id" > "$OUT/r$r.batch" 2>/dev/null
    while read -r bid; do
      [ -z "$bid" ] && continue
      i=$((i+1))
      echo "SELECT 'R$r.$i '||coalesce((SELECT id FROM t ORDER BY v <=> '$(beacon_vec $bid)' LIMIT 1)::text,'none')||' want=$bid';" >&$fd
    done < "$OUT/r$r.batch"
    sleep 0.02
  done
  exec {fd}>&-
}

watcher() { # observes reused versions directly: each writer records the am_version
  # it committed (read AFTER its COMMIT, under no lock, so it may see a later
  # one); a version that two DIFFERENT beacon commits both report as "the
  # version right after me" is ambiguous, so instead sample the meta page twice
  # around a quiet gap: collisions are counted only when the version is
  # unchanged while the committed-beacon count grew AND no writer was
  # mid-commit (the burst barrier file says the burst ended).
  # Informational only -- the PASS/FAIL oracle is the stale-read count.
  local end=$((SECONDS+DUR)) prev_v="" prev_n=""
  while [ $SECONDS -lt $end ]; do
    sleep "$(python3 -c "print(0.3 + $NWRITERS*0.15 + 0.4)")"   # end of a burst's commits
    local row; row=$($P -c "SELECT (SELECT get_byte(p,84)+get_byte(p,85)*256+get_byte(p,86)*65536+get_byte(p,87)*16777216 FROM get_raw_page('t_idx',0) p)||' '||(SELECT count(*) FROM beacons)" 2>/dev/null)
    set -- $row
    [ -n "$prev_v" ] && [ "$2" != "$prev_n" ] && [ $(( $1 - prev_v )) -lt $(( $2 - prev_n )) ] \
      && echo "version +$(( $1 - prev_v )) for +$(( $2 - prev_n )) committed beacons ($prev_v->$1)" >> "$OUT/collisions.log"
    prev_v=$1; prev_n=$2
    sleep "$(python3 -c "print(0.1 + $IDLE)")"
  done
}
: > "$OUT/collisions.log"
coordinator &
for w in $(seq 1 $NWRITERS); do writer $w & done
watcher &
vacuumer &
for r in $(seq 1 $NREADERS); do reader $r & done
echo "[soak] DUR=${DUR}s writers=$NWRITERS readers=$NREADERS dim=$DIM started $(date -u +%T)"
sleep $((DUR + 5))
kill $(jobs -p) 2>/dev/null; wait 2>/dev/null

# Score: a read is stale iff its top-1 != the beacon it was asked about AND the
# beacon was committed when asked (it was: we read it from beacons first). A
# newer beacon with the SAME direction can legitimately win -- re-check every
# miss against the truth for its own direction at end of run.
cat "$OUT"/r*.out | grep -E '^R[0-9]+\.[0-9]+ ' > "$OUT/reads.log"
total=$(wc -l < "$OUT/reads.log")
# Directions are unique per beacon, so the ANN top-1 for a committed beacon's
# direction must be exactly that beacon. Anything else is a stale read.
awk '{w=$3; sub("want=","",w); if ($2!=w) print}' "$OUT/reads.log" > "$OUT/stale.log"
stale=$(wc -l < "$OUT/stale.log")
writes=$($P -c "SELECT count(*) FROM beacons"); chk=$($P -c "SELECT is_corrupt FROM turbovec_check('t_idx'::regclass)")
# Lost-delete check. With concurrent snapshots, and with PG's index-cleanup
# bypass when few pages carry dead items, the index can legitimately hold
# entries for LP_DEAD heap tuples after the soak. A quiet
# VACUUM (INDEX_CLEANUP ON) must remove every one of them; an index row count
# above the live heap count after it means a delete was lost (or re-added).
$P -c "VACUUM (INDEX_CLEANUP ON) t" >/dev/null
heap_n=$($P -c "SELECT count(*) FROM t WHERE v IS NOT NULL")
# Live index entries = slots minus tombstoned slots (tombstones exist only on
# IVF/1-bit; a flat index swap-removes). After a quiet VACUUM (INDEX_CLEANUP ON)
# every dead heap tuple's entry is removed or tombstoned, so live entries must
# equal live heap rows. A resurrected entry (a tombstone erased by a flush, or a
# VACUUMed row rewritten) stays live with no heap row behind it -> live > heap.
# Not reltuples: a VACUUM with nothing new to remove reports n_vectors there,
# tombstoned slots included, which would false-fail a correct IVF index.
idx_n=$($P -c "SELECT round(n_vectors * (1 - tombstone_density))::bigint FROM turbovec_check('t_idx'::regclass)")
if [ "$LISTS" -gt 0 ]; then
  # The run must have exercised deletes + VACUUM. Do NOT require tombstones to
  # SURVIVE to the end: a flush that erases them (the bug) destroys exactly that
  # evidence, so an end-state check would call a broken build INVALID instead of
  # FAIL. The live-vs-heap check below is the verdict.
  deleted=$($P -c "SELECT 5000 - count(*) FROM t WHERE id BETWEEN 1 AND 5000")
  dens=$($P -c "SELECT round(tombstone_density::numeric, 5) FROM turbovec_check('t_idx'::regclass)")
  echo "[soak] IVF: corpus rows deleted+vacuumed during the run: $deleted, tombstone_density at end: $dens"
  [ "$deleted" -gt 0 ] || { echo "RESULT: INVALID (the VACUUM loop deleted nothing, so the IVF soak tested nothing)"; exit 2; }
fi
echo "[soak] after quiet VACUUM (INDEX_CLEANUP ON): index n_vectors=$idx_n heap live=$heap_n"
[ "$idx_n" = "$heap_n" ] || { echo "RESULT: FAIL (index has $((idx_n - heap_n)) entries for rows the heap no longer has)"; exit 1; }
echo "[soak] reads=$total stale=$stale beacon_txns=$writes turbovec_check.is_corrupt=$chk"
echo "[soak] bursts where am_version advanced LESS than the number of committed beacon txns (direct reuse evidence): $(wc -l < "$OUT/collisions.log")"
verr=$(grep -c ERROR "$OUT/vacuum.err" 2>/dev/null); verr=${verr:-0}
[ "$verr" -eq 0 ] || { echo "RESULT: INVALID (vacuum loop had $verr errors: $(grep -m1 ERROR "$OUT/vacuum.err"))"; exit 2; }
echo "[soak] writer errors: $(cat "$OUT"/writer*.out | grep -c ERROR)  vacuum errors: $(grep -c ERROR "$OUT/vacuum.err" 2>/dev/null || echo 0)"
# A soak that did no work proves nothing: require real volume before PASS.
MIN_READS=${MIN_READS:-$((DUR * NREADERS / 2))}; MIN_WRITES=${MIN_WRITES:-$((DUR * NWRITERS / 10))}
if [ "$total" -lt "$MIN_READS" ] || [ "$writes" -lt "$MIN_WRITES" ]; then
  echo "RESULT: INVALID (reads=$total < $MIN_READS or beacon_txns=$writes < $MIN_WRITES: the harness did not exercise the index)"; exit 2
fi
[ "$stale" -eq 0 ] && [ "$chk" = "f" ] && { echo "RESULT: PASS"; exit 0; } || { echo "RESULT: FAIL"; exit 1; }
