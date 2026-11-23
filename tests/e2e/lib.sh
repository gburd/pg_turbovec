# shellcheck shell=bash
# Shared helpers for tests/e2e cases (sourced).  Every case creates its own
# table, so cases are independent and may run in any order.
set -euo pipefail
export PGOPTIONS="-c search_path=public,turbovec"
q() { psql -X -q -A -t -v ON_ERROR_STOP=1 "$@"; }
vec() { # vec <seed>: a deterministic 32-d vector literal
    echo "('[' || array_to_string(array(SELECT sin($1 * 0.37 + s)::float8 FROM generate_series(1, 32) s), ',') || ']')::turbovec.vector"
}
mk() { # mk <table> [UNLOGGED] [reloptions]: 2000 rows + an index (default: flat 4-bit)
    q -c "DROP TABLE IF EXISTS $1" -c "CREATE ${2:-} TABLE $1 (id int PRIMARY KEY, emb turbovec.vector)" \
      -c "INSERT INTO $1 SELECT g, $(vec g) FROM generate_series(1, 2000) g" \
      -c "CREATE INDEX $1_idx ON $1 USING turbovec (emb turbovec.vec_cosine_ops) WITH (${3:-bit_width = 4})"
}
# found <table> <ids...>: the ids that the index returns as their own top-1
found() {
    local t=$1; shift; local list; list=$(IFS=,; echo "$*")
    q -c "SET enable_seqscan = off" -c "SELECT coalesce(string_agg(b.id::text, ' ' ORDER BY b.id), '') FROM $t b
           WHERE b.id IN ($list) AND (SELECT x.id FROM $t x ORDER BY x.emb <=> b.emb LIMIT 1) = b.id"
}
expect() { # expect <what> <got> <want>
    if [ "$2" != "$3" ]; then echo "$1: got [$2], want [$3]"; exit 1; fi
}
