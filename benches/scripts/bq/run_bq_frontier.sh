#!/usr/bin/env bash
# 1-bit sign-BQ recall/storage/latency frontier — runbook driver.
#
# Phases (run `all`, or one at a time):
#   preflight  report the host's SIMD tier and say plainly whether this host
#              may publish latency (AVX2+) or recall/storage/build only.
#   queryset   materialise bq_query_set (qid, qvec) from a held-out table, or
#              -- with BQ_QUERY_PROVENANCE=in_corpus -- from corpus members.
#   sweep      the bit_width in {1,2,4} sweep (benches/scripts/bq/bq_frontier.py)
#   all        preflight, queryset, sweep
#
# Env (all overridable):
#   BQ_DSN              libpq DSN of the bench cluster            [required]
#   BQ_TABLE            corpus table                              [docs]
#   BQ_VEC_EXPR         indexed expression, verbatim              [embt]
#   BQ_DIM              corpus dimensionality                     [required]
#   BQ_OPCLASS          vec_cosine_ops | vec_l2_ops | vec_ip_ops  [vec_cosine_ops]
#   BQ_OPERATOR         matching operator                         [turbovec.<=>]
#   BQ_HELDOUT_TABLE    held-out query source (id, <vec>)         [<unset>]
#   BQ_QUERY_PROVENANCE held_out | in_corpus                      [held_out]
#   BQ_N_QUERIES        query count                               [200]
#   BQ_OUT              output JSON path                          [required]
#   BQ_IVF_LISTS        also try an IVF arm with N lists          [0 = skip]
#   BQ_EXTRA            extra args forwarded to bq_frontier.py    []
#
# Heartbeat: the sweep and the ground-truth pass both run for many minutes,
# so this script wraps itself per .pi/skills/long-running-bench/SKILL.md.
# Poll with benches/scripts/poll-heartbeat.sh "$BQ_LOG".
#
# See docs/BQ_RECALL_BENCH.md for the full methodology and what the results
# do and do not license you to claim.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${BQ_TABLE:=docs}"
: "${BQ_VEC_EXPR:=embt}"
: "${BQ_OPCLASS:=vec_cosine_ops}"
: "${BQ_OPERATOR:=turbovec.<=>}"
: "${BQ_QUERY_PROVENANCE:=held_out}"
: "${BQ_N_QUERIES:=200}"
: "${BQ_IVF_LISTS:=0}"
: "${BQ_EXTRA:=}"
# Namespace this run's shared state. Index names are SCHEMA-scoped and the
# query-set / GT tables have fixed default names, so two concurrent arms in
# one database silently corrupt each other (2026-09-09: one arm's 256-d
# query set replaced another's 1024-d one mid-sweep, and a DROP TABLE
# bq_gt destroyed 860s of ground truth). Set BQ_RUN_ID per arm -- or better,
# give each arm its own database.
: "${BQ_RUN_ID:=}"
if [ -n "$BQ_RUN_ID" ]; then
    : "${BQ_QUERY_TABLE:=bq_query_set_${BQ_RUN_ID}}"
    : "${BQ_GT_TABLE:=bq_gt_${BQ_RUN_ID}}"
else
    : "${BQ_QUERY_TABLE:=bq_query_set}"
    : "${BQ_GT_TABLE:=bq_gt}"
fi
: "${BQ_LOG:=/tmp/bq_frontier.log}"
: "${BQ_HELDOUT_TABLE:=}"

phase=${1:-all}

ts(){ date -u +'%H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }

need(){ [ -n "${!1:-}" ] || die "$1 is required (see the header of $0)"; }

psql_q(){ psql -X -q -P pager=off -v ON_ERROR_STOP=1 -d "$BQ_DSN" "$@"; }

# --------------------------------------------------------------------------
# preflight: the AVX2 gate, stated out loud before anything is measured.
# --------------------------------------------------------------------------
preflight() {
    local flags tier
    flags=$(grep -m1 -E '^(flags|Features)' /proc/cpuinfo 2>/dev/null || echo "")
    if echo "$flags" | grep -qw avx512f; then tier=avx512
    elif echo "$flags" | grep -qw avx2;   then tier=avx2
    else                                       tier=scalar
    fi
    log "host       : $(hostname)"
    log "cpu        : $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ //')"
    log "kernel tier: $tier"
    if [ "$tier" = scalar ]; then
        cat <<'EOF'
  !! NO AVX2 ON THIS HOST.
  !! turbovec dispatches to its SCALAR fallback here: correct results, but
  !! ~1000x slower for a full-corpus scan (AGENTS.md, "Bench hosts").
  !! This run measures RECALL, STORAGE, BUILD TIME and MEMORY only.
  !! A warm p50 from this host is NOT a publishable number. bq_frontier.py
  !! will refuse to emit one (see --force-scalar-latency if you want the
  !! scalar floor recorded under an explicitly-unpublishable key).
  !! Publish latency from `arnold` (AVX2) instead.
EOF
    else
        log "AVX2+ present: latency (warm p50/p95/p99, qps) is publishable from this host."
    fi
    log "postgres   : $(psql_q -tAc 'SHOW server_version' 2>/dev/null || echo '<unreachable>')"
    log "pg_turbovec: $(psql_q -tAc "SELECT extversion FROM pg_extension WHERE extname='pg_turbovec'" 2>/dev/null || echo '<not installed>')"
    python3 "$HERE/bq_frontier.py" --self-check
}

# --------------------------------------------------------------------------
# queryset: bq_query_set (qid int, qvec turbovec.vector)
# --------------------------------------------------------------------------
queryset() {
    if [ "$BQ_QUERY_PROVENANCE" = held_out ]; then
        [ -n "$BQ_HELDOUT_TABLE" ] || die \
"BQ_QUERY_PROVENANCE=held_out needs BQ_HELDOUT_TABLE (a table of vectors NOT
in $BQ_TABLE). Held-out queries are strongly preferred: with in-corpus queries
rank 1 is trivially the query itself, so R@k carries a 1/k floor for ANY index
and the 1-bit-vs-2-bit gap is compressed toward zero. If you must use corpus
members, set BQ_QUERY_PROVENANCE=in_corpus -- the caveat is then recorded in
the artefact."
        log "queryset: $BQ_N_QUERIES held-out queries from $BQ_HELDOUT_TABLE"
        psql_q <<SQL
DROP TABLE IF EXISTS "$BQ_QUERY_TABLE";
CREATE TABLE "$BQ_QUERY_TABLE" AS
SELECT row_number() OVER (ORDER BY id)::int AS qid,
       ${BQ_VEC_EXPR} AS qvec
FROM ${BQ_HELDOUT_TABLE}
ORDER BY id
LIMIT ${BQ_N_QUERIES};
CREATE INDEX ON "$BQ_QUERY_TABLE" (qid);
SELECT count(*) AS n_queries FROM "$BQ_QUERY_TABLE";
SQL
    else
        log "queryset: $BQ_N_QUERIES IN-CORPUS queries from $BQ_TABLE (R@k has a 1/k floor)"
        psql_q <<SQL
DROP TABLE IF EXISTS "$BQ_QUERY_TABLE";
CREATE TABLE "$BQ_QUERY_TABLE" AS
SELECT row_number() OVER (ORDER BY id)::int AS qid,
       ${BQ_VEC_EXPR} AS qvec
FROM ${BQ_TABLE}
ORDER BY id
LIMIT ${BQ_N_QUERIES};
CREATE INDEX ON "$BQ_QUERY_TABLE" (qid);
SELECT count(*) AS n_queries FROM "$BQ_QUERY_TABLE";
SQL
    fi
}

# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
sweep() {
    need BQ_OUT
    mkdir -p "$(dirname "$BQ_OUT")"
    log "sweep -> $BQ_OUT"
    # shellcheck disable=SC2086
    python3 "$HERE/bq_frontier.py" \
        --dsn "$BQ_DSN" \
        --table "$BQ_TABLE" \
        --vec-expr "$BQ_VEC_EXPR" \
        --dim "$BQ_DIM" \
        --opclass "$BQ_OPCLASS" \
        --operator "$BQ_OPERATOR" \
        --query-table "$BQ_QUERY_TABLE" \
        --gt-table "$BQ_GT_TABLE" \
        ${BQ_RUN_ID:+--run-id "$BQ_RUN_ID"} \
        --query-provenance "$BQ_QUERY_PROVENANCE" \
        --ivf-lists "$BQ_IVF_LISTS" \
        --out "$BQ_OUT" \
        $BQ_EXTRA
}

need BQ_DSN
case "$phase" in
    preflight) preflight ;;
    queryset)  need BQ_DIM; queryset ;;
    sweep)     need BQ_DIM; sweep ;;
    all)       need BQ_DIM; preflight && queryset && sweep ;;
    *) die "usage: $0 {preflight|queryset|sweep|all}" ;;
esac
log "phase=$phase done"
