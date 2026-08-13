#!/bin/bash
# Comprehensive sustained-load corruption repro for pg_turbovec (v2 harness).
#
# Against a ~1.8M-row 768d FLAT turbovec index, concurrently:
#   - VACUUM every 3s (backend V)             -> ambulkdelete swap-remove shrink
#   - NWRITERS writers, each a SINGLE long-lived psql running a loop of
#     one-transaction-per-line batches (16 then 128 rows) via
#     INSERT ... ON CONFLICT DO UPDATE. Every RECONNECT_EVERY cycles the
#     writer's psql EXITS and a fresh one starts -> the fresh-backend
#     first-insert cold-load path is still exercised, but connection count
#     stays BOUNDED at ~NWRITERS (no unbounded fresh-backend spawn that
#     saturated max_connections in the v1 harness). Writers also fully
#     restart their loop periodically (writer-restart churn).
#   - 3 concurrent scanner loops (ORDER BY emb <-> q LIMIT k), persistent
#     connections.
#   - a PAUSE-gated monitor: every CHECK_EVERY seconds it (a) always scans
#     the PG log for the DEFINITIVE corruption signatures (dup-id XX001,
#     SIGABRT, recovery mode) at zero DB cost, and (b) periodically raises a
#     PAUSE flag, lets writers+VACUUM quiesce for a few seconds so the
#     shared rewrite lock drains, runs an UNCONTENDED turbovec_check
#     (fast + reliable, no starvation), records is_corrupt, then clears
#     PAUSE. This gives real is_corrupt readings throughout WITHOUT the
#     reader-starvation that made the v1 in-flight check time out.
#
# ZERO tolerance: any dup-id / SIGABRT / recovery / is_corrupt=true = FAIL.
set -u
PGBIN=/data/.pgrx/18.4/pgrx-install/bin
PORT=28818; DB=postgres
PSQL="$PGBIN/psql -p $PORT -d $DB -v ON_ERROR_STOP=0 -tAq"
DUR=${DUR:-4200}
NWRITERS=${NWRITERS:-4}
RECONNECT_EVERY=${RECONNECT_EVERY:-8}     # writer reconnects (fresh backend) every N cycles
CHECK_EVERY=${CHECK_EVERY:-60}            # seconds between quiesced turbovec_checks
OUT=${OUT:-/data/repro/out-$(date +%s)}
PGLOG=${PGLOG:-/data/pg.log}
mkdir -p "$OUT"
echo "[run] OUT=$OUT DUR=$DUR NWRITERS=$NWRITERS reconnect_every=$RECONNECT_EVERY check_every=$CHECK_EVERY start=$(date -u +%FT%TZ)" | tee "$OUT/run.meta"

MAXID=$($PSQL -c "SELECT COALESCE(max(id),0) FROM corpus;")
echo "[run] starting maxid=$MAXID" | tee -a "$OUT/run.meta"
$PSQL -c "DROP SEQUENCE IF EXISTS repro_idseq; CREATE SEQUENCE repro_idseq START $((MAXID+1));" >/dev/null

STOP="$OUT/STOP"; PAUSE="$OUT/PAUSE"
rm -f "$STOP" "$PAUSE"
PIDS=()

BATCH_SQL='INSERT INTO corpus (id, emb)
SELECT nextval('"'"'repro_idseq'"'"') AS id,
       turbovec.array_to_vec((SELECT array_agg((sin(gg*0.7+s*0.01)+random())::real) FROM generate_series(1,768) AS s))
FROM generate_series(1,8) AS gg
UNION ALL
SELECT (1 + floor(random()*1700000))::bigint AS id,
       turbovec.array_to_vec((SELECT array_agg((sin(gg*1.3+s*0.02)+random())::real) FROM generate_series(1,768) AS s))
FROM generate_series(1,8) AS gg
ON CONFLICT (id) DO UPDATE SET emb = EXCLUDED.emb;
INSERT INTO corpus (id, emb)
SELECT nextval('"'"'repro_idseq'"'"') AS id,
       turbovec.array_to_vec((SELECT array_agg((sin(gg*0.9+s*0.03)+random())::real) FROM generate_series(1,768) AS s))
FROM generate_series(1,64) AS gg
UNION ALL
SELECT (1 + floor(random()*1700000))::bigint AS id,
       turbovec.array_to_vec((SELECT array_agg((sin(gg*1.7+s*0.04)+random())::real) FROM generate_series(1,768) AS s))
FROM generate_series(1,64) AS gg
ON CONFLICT (id) DO UPDATE SET emb = EXCLUDED.emb;'

# ---- VACUUM loop (every 3s), pauses when PAUSE flag is up -----------------
(
  while [ ! -f "$STOP" ]; do
    if [ ! -f "$PAUSE" ]; then
      $PGBIN/psql -p $PORT -d $DB -c "VACUUM corpus;" >>"$OUT/vacuum.log" 2>&1
    fi
    sleep 3
  done
) & PIDS+=($!)

# ---- writers: bounded persistent connections, reconnect for fresh backend -
for w in $(seq 1 $NWRITERS); do
  (
    while [ ! -f "$STOP" ]; do
      # One fresh backend for a run of RECONNECT_EVERY batches (each on its
      # own transaction => one relfile flush each). Then the psql exits and
      # the loop makes a NEW one (fresh-backend cold-load path again).
      {
        for c in $(seq 1 $RECONNECT_EVERY); do
          [ -f "$STOP" ] && break
          # honour PAUSE: skip issuing new work while the monitor checks
          while [ -f "$PAUSE" ] && [ ! -f "$STOP" ]; do sleep 0.3; done
          echo "$BATCH_SQL"
        done
      } | $PGBIN/psql -p $PORT -d $DB -v ON_ERROR_STOP=0 >>"$OUT/writer-$w.log" 2>&1
    done
  ) & PIDS+=($!)
done

# ---- scanner loops (3), persistent-ish (fresh conn per query is fine) ------
for s in 1 2 3; do
  (
    while [ ! -f "$STOP" ]; do
      while [ -f "$PAUSE" ] && [ ! -f "$STOP" ]; do sleep 0.3; done
      $PGBIN/psql -p $PORT -d $DB -v ON_ERROR_STOP=0 -tAq >>"$OUT/scanner-$s.log" 2>&1 <<SQL
SET search_path = turbovec, public;
SET turbovec.probes = 16;
SELECT id FROM corpus
ORDER BY emb <-> turbovec.array_to_vec((SELECT array_agg((sin(x*0.5)+random())::real) FROM generate_series(1,768) AS x))
LIMIT 10;
SQL
    done
  ) & PIDS+=($!)
done

# ---- monitor -------------------------------------------------------------
STATUS="$OUT/STATUS"; VERDICT="$OUT/verdict.txt"; : > "$STATUS"
CORRUPT=0
START=$(date +%s)
LOGSTART=$(( $(wc -l < "$PGLOG" 2>/dev/null || echo 0) + 1 ))
last_check=0
while [ ! -f "$STOP" ]; do
  now=$(date +%s); el=$(( now - START ))
  [ $el -ge $DUR ] && break
  # Definitive continuous signatures (zero DB cost).
  dup=$(tail -n +$LOGSTART "$PGLOG" 2>/dev/null | grep -c "more than one slot" || true)
  xx=$(tail -n +$LOGSTART "$PGLOG" 2>/dev/null | grep -c "duplicate ids" || true)
  sig=$(tail -n +$LOGSTART "$PGLOG" 2>/dev/null | grep -Ec "signal 6|SIGABRT|was terminated by signal|database system is in recovery" || true)
  isc="f"; chk="(no-check)"; cnt="-"
  if [ $(( now - last_check )) -ge $CHECK_EVERY ]; then
    # Quiesce briefly so the rewrite lock drains, then run an
    # UNCONTENDED turbovec_check (fast + reliable).
    touch "$PAUSE"; sleep 5
    chk=$(timeout 120 $PGBIN/psql -p $PORT -d $DB -tAq -c \
      "SELECT n_vectors||'|'||slot_count||'|'||count_matches||'|'||COALESCE(duplicate_id::text,'-')||'|'||is_corrupt FROM turbovec.turbovec_check('corpus_emb_idx'::regclass);" 2>>"$OUT/check-err.log")
    cnt=$(timeout 30 $PSQL -c "SELECT count(*) FROM corpus;" 2>/dev/null)
    rm -f "$PAUSE"
    isc=$(echo "$chk" | cut -d'|' -f5)
    last_check=$now
  fi
  echo "$(date -u +%FT%TZ) el=${el}s rows=$cnt check=[$chk] dup_log=$dup xx_log=$xx sigabrt=$sig" | tee -a "$STATUS"
  if [ "$isc" = "t" ] || [ "${dup:-0}" -gt 0 ] || [ "${xx:-0}" -gt 0 ] || [ "${sig:-0}" -gt 0 ]; then
    CORRUPT=1
    echo "CORRUPTION DETECTED at el=${el}s: check=[$chk] dup_log=$dup xx_log=$xx sigabrt=$sig" | tee -a "$STATUS"
    rm -f "$PAUSE"
    break
  fi
  sleep 5
done

touch "$STOP"; rm -f "$PAUSE"; sleep 2
for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
pkill -P $$ 2>/dev/null; sleep 1

# Let the lock drain, then final uncontended check.
sleep 8
finalchk=$(timeout 180 $PGBIN/psql -p $PORT -d $DB -tAq -c \
  "SELECT n_vectors||'|'||slot_count||'|'||count_matches||'|'||COALESCE(duplicate_id::text,'-')||'|'||is_corrupt FROM turbovec.turbovec_check('corpus_emb_idx'::regclass);" 2>>"$OUT/check-err.log")
finalrows=$(timeout 30 $PSQL -c "SELECT count(*) FROM corpus;" 2>/dev/null)
dup=$(tail -n +$LOGSTART "$PGLOG" 2>/dev/null | grep -c "more than one slot" || true)
xx=$(tail -n +$LOGSTART "$PGLOG" 2>/dev/null | grep -c "duplicate ids" || true)
sig=$(tail -n +$LOGSTART "$PGLOG" 2>/dev/null | grep -Ec "signal 6|SIGABRT|was terminated by signal|database system is in recovery" || true)
el=$(( $(date +%s) - START ))
# total inserts issued (approx) = final sequence value minus start
seqval=$($PSQL -c "SELECT last_value FROM repro_idseq;" 2>/dev/null)
{
  echo "duration_s=$el"
  echo "final_rows=$finalrows"
  echo "final_check=$finalchk"
  echo "seq_last_value=$seqval (start was $((MAXID+1)))"
  echo "dup_id_log=$dup"
  echo "duplicate_ids_log=$xx"
  echo "sigabrt_recovery_log=$sig"
  if [ "$CORRUPT" = "1" ] || [ "${dup:-0}" -gt 0 ] || [ "${xx:-0}" -gt 0 ] || [ "${sig:-0}" -gt 0 ] || [ "$(echo "$finalchk"|cut -d'|' -f5)" = "t" ]; then
    echo "VERDICT=FAIL"
  else
    echo "VERDICT=PASS"
  fi
} | tee "$VERDICT"
echo "[run] done el=${el}s"
