[heartbeat] starting: bash /home/fedora/tail_all.sh
[heartbeat] interval: 60s
[heartbeat] label:    bash
+ mkdir -p /home/fedora/diag
+ sudo dnf -y -q install gcc git clang clang-devel llvm-devel openblas-devel readline-devel zlib-devel libicu-devel bison flex perl openssl-devel python3-pip perf
. [bash 11:30:43 elapsed]
+ curl -sSf https://sh.rustup.rs
+ sh -s -- -y -q --profile minimal --default-toolchain 1.98.0
+ . /home/fedora/.cargo/env
++ case ":${PATH}:" in
++ export PATH=/home/fedora/.cargo/bin:/home/fedora/.local/bin:/home/fedora/bin:/usr/local/bin:/usr/bin
++ PATH=/home/fedora/.cargo/bin:/home/fedora/.local/bin:/home/fedora/bin:/usr/local/bin:/usr/bin
+ cargo install -q --locked cargo-pgrx --version 0.19.3
. [bash 11:31:44 elapsed]
+ cargo pgrx init --pg16 download
+ git clone -q --branch fix/scan-concurrency-2.12.2 https://github.com/gburd/pg_turbovec.git /home/fedora/pgtv
+ cd /home/fedora/pgtv
+ git log --oneline -1
++ head -1
++ ls -d /home/fedora/.pgrx/16.15/pgrx-install/bin
+ PGB=/home/fedora/.pgrx/16.15/pgrx-install/bin
+ export PATH=/home/fedora/.pgrx/16.15/pgrx-install/bin:/home/fedora/.cargo/bin:/home/fedora/.local/bin:/home/fedora/bin:/usr/local/bin:/usr/bin
+ PATH=/home/fedora/.pgrx/16.15/pgrx-install/bin:/home/fedora/.cargo/bin:/home/fedora/.local/bin:/home/fedora/bin:/usr/local/bin:/usr/bin
+ cd /home/fedora/pgtv
+ cargo pgrx install --release --pg-config /home/fedora/.pgrx/16.15/pgrx-install/bin/pg_config
. [bash 11:32:44 elapsed]
. [bash 11:33:44 elapsed]
+ python3 -m pip install -q --user numpy pyarrow huggingface_hub 'psycopg[binary]'
+ D=/home/fedora/pgdata
+ initdb -D /home/fedora/pgdata -U fedora
initdb: warning: enabling "trust" authentication for local connections
initdb: hint: You can change this by editing pg_hba.conf or using the option -A, or --auth-local and --auth-host, the next time you run initdb.
+ cat
+ pg_ctl -D /home/fedora/pgdata -l /home/fedora/pg.log -w start
waiting for server to start.... done
server started
+ createdb -h /tmp bench
createdb: error: connection to server on socket "/tmp/.s.PGSQL.28816" failed: No such file or directory
	Is the server running locally and accepting connections on that socket?
+ psql -h /tmp -d bench -qc 'CREATE EXTENSION pg_turbovec; CREATE TABLE docs(id bigint PRIMARY KEY, tv turbovec.vector)'
psql: error: connection to server on socket "/tmp/.s.PGSQL.28816" failed: No such file or directory
	Is the server running locally and accepting connections on that socket?
+ mkdir -p /home/fedora/corpus
+ python3 /home/fedora/load_corpus.py 1000000 5200 bench
+ psql -h /tmp -d bench -qc 'VACUUM ANALYZE docs'
psql: error: connection to server on socket "/tmp/.s.PGSQL.28816" failed: No such file or directory
	Is the server running locally and accepting connections on that socket?
+ psql -h /tmp -d bench -c 'CREATE INDEX docs_ivf ON docs USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4, lists = 1000)'
psql: error: connection to server on socket "/tmp/.s.PGSQL.28816" failed: No such file or directory
	Is the server running locally and accepting connections on that socket?
+ echo '=== step A: whole vs ooc per query, planes ON (default)'
=== step A: whole vs ooc per query, planes ON (default)
+ python3 /home/fedora/tail_diag.py 400 /home/fedora/diag/A_planes_on.jsonl
+ cat /home/fedora/diag/A.txt
Traceback (most recent call last):
  File "/home/fedora/tail_diag.py", line 7, in <module>
    Q = np.load('/home/fedora/corpus/corpus_1000000/q.npy')
  File "/home/fedora/.local/lib/python3.14/site-packages/numpy/lib/_npyio_impl.py", line 454, in load
    fid = stack.enter_context(open(os.fspath(file), "rb"))
                              ~~~~^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/fedora/corpus/corpus_1000000/q.npy'
+ echo '=== step B: same, postmaster with TURBOVEC_4BIT_PLANES=0'
=== step B: same, postmaster with TURBOVEC_4BIT_PLANES=0
+ pg_ctl -D /home/fedora/pgdata -m fast -w stop
waiting for server to shut down.... done
server stopped
+ TURBOVEC_4BIT_PLANES=0
+ pg_ctl -D /home/fedora/pgdata -l /home/fedora/pg.log -w start
waiting for server to start.... done
server started
+ python3 /home/fedora/tail_diag.py 400 /home/fedora/diag/B_planes_off.jsonl
+ cat /home/fedora/diag/B.txt
Traceback (most recent call last):
  File "/home/fedora/tail_diag.py", line 7, in <module>
    Q = np.load('/home/fedora/corpus/corpus_1000000/q.npy')
  File "/home/fedora/.local/lib/python3.14/site-packages/numpy/lib/_npyio_impl.py", line 454, in load
    fid = stack.enter_context(open(os.fspath(file), "rb"))
                              ~~~~^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/fedora/corpus/corpus_1000000/q.npy'
+ pg_ctl -D /home/fedora/pgdata -m fast -w stop
waiting for server to shut down.... done
server stopped
+ pg_ctl -D /home/fedora/pgdata -l /home/fedora/pg.log -w start
waiting for server to start.... done
server started
+ echo '=== step C: knobs on whole-load (planes on): hi_dim_rerank off; scan_parallelism 1'
=== step C: knobs on whole-load (planes on): hi_dim_rerank off; scan_parallelism 1
+ python3 /home/fedora/tail_knobs.py
+ cat /home/fedora/diag/C.txt
Traceback (most recent call last):
  File "/home/fedora/tail_knobs.py", line 2, in <module>
    Q = np.load('/home/fedora/corpus/corpus_1000000/q.npy'); N = 400
  File "/home/fedora/.local/lib/python3.14/site-packages/numpy/lib/_npyio_impl.py", line 454, in load
    fid = stack.enter_context(open(os.fspath(file), "rb"))
                              ~~~~^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/fedora/corpus/corpus_1000000/q.npy'
+ echo '=== step D: perf profile of the 30 slowest whole-load queries'
=== step D: perf profile of the 30 slowest whole-load queries
+ python3 /home/fedora/tail_perf.py
+ cat /home/fedora/diag/D.txt
Traceback (most recent call last):
  File "/home/fedora/tail_perf.py", line 2, in <module>
    Q = np.load('/home/fedora/corpus/corpus_1000000/q.npy')
  File "/home/fedora/.local/lib/python3.14/site-packages/numpy/lib/_npyio_impl.py", line 454, in load
    fid = stack.enter_context(open(os.fspath(file), "rb"))
                              ~~~~^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/fedora/corpus/corpus_1000000/q.npy'
+ echo ALL_DONE
ALL_DONE
[heartbeat] DONE rc=0 (clean exit)
