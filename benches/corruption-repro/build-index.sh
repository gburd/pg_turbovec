#!/bin/bash
set -e
PSQL="/data/.pgrx/18.4/pgrx-install/bin/psql -p 28818 -d postgres -v ON_ERROR_STOP=1"
echo "[idx] building flat turbovec index on corpus(emb)"
time $PSQL -c "DROP INDEX IF EXISTS corpus_emb_idx;"
time $PSQL -c "CREATE INDEX corpus_emb_idx ON corpus USING turbovec (emb turbovec.vec_l2_ops);"
echo "[idx] turbovec_check:"
$PSQL -c "SELECT * FROM turbovec.turbovec_check('corpus_emb_idx'::regclass);"
