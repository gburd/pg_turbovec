#!/bin/bash
# Minimal crash repro: 2 writers + VACUUM every 1s. Captures each
# writer backend's stderr so a Rust panic message is visible.
set -u
PGBIN=/data/.pgrx/18.4/pgrx-install/bin
PORT=28818; DB=postgres
OUT=${OUT:-/data/repro/mini-$(date +%s)}
mkdir -p "$OUT"
DUR=${DUR:-300}
export RUST_BACKTRACE=full
MAXID=$($PGBIN/psql -p $PORT -d $DB -tAq -c "SELECT COALESCE(max(id),0) FROM corpus;")
$PGBIN/psql -p $PORT -d $DB -tAq -c "DROP SEQUENCE IF EXISTS mini_idseq; CREATE SEQUENCE mini_idseq START $((MAXID+1));" >/dev/null
STOP="$OUT/STOP"; rm -f "$STOP"
# VACUUM every 1s
( while [ ! -f "$STOP" ]; do $PGBIN/psql -p $PORT -d $DB -c "VACUUM corpus;" >>"$OUT/vacuum.log" 2>&1; sleep 1; done ) &
VP=$!
# 2 writers, fresh backend each cycle, stderr captured
for w in 1 2; do
( while [ ! -f "$STOP" ]; do
  $PGBIN/psql -p $PORT -d $DB -v ON_ERROR_STOP=0 2>>"$OUT/writer-$w.stderr" >>"$OUT/writer-$w.log" <<SQL
INSERT INTO corpus (id, emb)
SELECT nextval('mini_idseq'),
       turbovec.array_to_vec((SELECT array_agg((sin(gg*0.7+s*0.01)+random())::real) FROM generate_series(1,768) AS s))
FROM generate_series(1,64) AS gg
UNION ALL
SELECT (1 + floor(random()*1700000))::bigint,
       turbovec.array_to_vec((SELECT array_agg((sin(gg*1.3+s*0.02)+random())::real) FROM generate_series(1,768) AS s))
FROM generate_series(1,64) AS gg
ON CONFLICT (id) DO UPDATE SET emb = EXCLUDED.emb;
SQL
done ) &
done
START=$(date +%s)
SIG0=$(grep -cE "signal 6|terminated by signal" /data/pg.log)
while [ ! -f "$STOP" ]; do
  el=$(( $(date +%s) - START )); [ $el -ge $DUR ] && break
  chk=$($PGBIN/psql -p $PORT -d $DB -tAq -c "SELECT COALESCE(duplicate_id::text,'-')||'/'||is_corrupt FROM turbovec.turbovec_check('corpus_emb_idx'::regclass);" 2>/dev/null)
  sig=$(( $(grep -cE "signal 6|terminated by signal" /data/pg.log) - SIG0 ))
  echo "el=${el}s check=$chk sig=$sig" | tee -a "$OUT/STATUS"
  if [ "${chk#*/}" = "t" ] || [ "$sig" -gt 0 ]; then echo "CORRUPT el=$el" | tee -a "$OUT/STATUS"; break; fi
  sleep 2
done
touch "$STOP"; sleep 2; kill $VP 2>/dev/null; pkill -P $$ 2>/dev/null
echo "DONE $OUT"
