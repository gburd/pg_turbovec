#!/bin/bash
# One-time: dedicated non-assert PG cluster on port 5440 for the step-5 A/B,
# load the 500k corpus into two tables (default storage, STORAGE MAIN), build
# identical flat 4-bit indexes. Run with the OLD arm installed.
set -euxo pipefail
B=/work/pg16ab/bin; D=/work/ab/data
[ -d /work/pg16ab ] || cp -a /work/pg16rel /work/pg16ab
[ -d $D ] || $B/initdb -D $D -U admin >/dev/null
grep -q ab-conf $D/postgresql.conf || cat >> $D/postgresql.conf <<C
# ab-conf
shared_buffers = 8GB
maintenance_work_mem = 4GB
max_parallel_maintenance_workers = 8
max_parallel_workers_per_gather = 0
jit = off
synchronous_commit = off
max_wal_size = 16GB
unix_socket_directories = '/tmp'
port = 5440
listen_addresses = ''
shared_preload_libraries = 'pg_turbovec'
C
$B/pg_ctl -D $D -l /work/ab/pg.log -w start
$B/createdb -h /tmp -p 5440 bench || true
$B/psql -h /tmp -p 5440 -d bench -v ON_ERROR_STOP=1 -qc "CREATE EXTENSION IF NOT EXISTS pg_turbovec"
/work/venv/bin/python - <<'P'
import numpy as np, psycopg
A = np.fromfile("/work/corpus/base.f32", dtype=np.float32).reshape(-1, 1024)
c = psycopg.connect("host=/tmp port=5440 dbname=bench", autocommit=True)
for t, stor in (("docs_ext", None), ("docs_main", "MAIN")):
    c.execute(f"DROP TABLE IF EXISTS {t}"); c.execute(f"CREATE TABLE {t}(id bigint primary key, tv turbovec.vector)")
    if stor: c.execute(f"ALTER TABLE {t} ALTER COLUMN tv SET STORAGE {stor}")
    with c.cursor().copy(f"COPY {t}(id, tv) FROM STDIN") as cp:
        for i in range(len(A)): cp.write_row((i, "[" + ",".join(f"{x:.6f}" for x in A[i]) + "]"))
    c.execute(f"VACUUM ANALYZE {t}")
    c.execute("SET maintenance_work_mem='4GB'")
    c.execute(f"CREATE INDEX {t}_tv ON {t} USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width=4)")
    print(t, c.execute(f"SELECT pg_size_pretty(pg_relation_size('{t}')), pg_size_pretty(pg_relation_size(reltoastrelid)), pg_size_pretty(pg_relation_size('{t}_tv')) FROM pg_class WHERE relname='{t}'").fetchone(), flush=True)
P
echo AB_SETUP_OK
