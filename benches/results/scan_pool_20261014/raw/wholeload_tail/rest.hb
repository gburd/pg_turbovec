[heartbeat] starting: bash /home/fedora/tail_rest.sh
[heartbeat] interval: 60s
[heartbeat] label:    bash
+ . /home/fedora/.cargo/env
++ case ":${PATH}:" in
+ export PGPORT=5432
+ PGPORT=5432
++ ls -d /home/fedora/.pgrx/16.15/pgrx-install/bin
++ head -1
+ PGB=/home/fedora/.pgrx/16.15/pgrx-install/bin
+ export PATH=/home/fedora/.pgrx/16.15/pgrx-install/bin:/home/fedora/.cargo/bin:/home/fedora/.local/bin:/home/fedora/bin:/usr/local/bin:/usr/bin
+ PATH=/home/fedora/.pgrx/16.15/pgrx-install/bin:/home/fedora/.cargo/bin:/home/fedora/.local/bin:/home/fedora/bin:/usr/local/bin:/usr/bin
+ D=/home/fedora/pgdata
+ createdb -h /tmp bench
+ psql -h /tmp -d bench -qc 'CREATE EXTENSION pg_turbovec; CREATE TABLE docs(id bigint PRIMARY KEY, tv turbovec.vector)'
+ mkdir -p /home/fedora/corpus
+ python3 /home/fedora/load_corpus.py 1000000 5200 bench
. [bash 11:37:19 elapsed]
. [bash 11:38:19 elapsed]
. [bash 11:39:19 elapsed]
. [bash 11:40:19 elapsed]
. [bash 11:41:19 elapsed]
. [bash 11:42:19 elapsed]
. [bash 11:43:19 elapsed]
. [bash 11:44:19 elapsed]
+ psql -h /tmp -d bench -qc 'VACUUM ANALYZE docs'
+ psql -h /tmp -d bench -c 'CREATE INDEX docs_ivf ON docs USING turbovec (tv turbovec.vec_cosine_ops) WITH (bit_width = 4, lists = 1000)'
. [bash 11:45:19 elapsed]
. [bash 11:46:19 elapsed]
. [bash 11:47:19 elapsed]
CREATE INDEX
+ echo '=== step A: whole vs ooc per query, planes ON (default)'
=== step A: whole vs ooc per query, planes ON (default)
+ python3 /home/fedora/tail_diag.py 400 /home/fedora/diag/A_planes_on.jsonl
. [bash 11:48:19 elapsed]
. [bash 11:49:19 elapsed]
+ cat /home/fedora/diag/A.txt
whole p50 74.2 p95 133.0 slow>60 400/400 | ooc p50 75.8 p95 135.4 slow>60 400
same top-10 set whole vs ooc: 400 / 400
+ echo '=== step B: same, postmaster with TURBOVEC_4BIT_PLANES=0'
=== step B: same, postmaster with TURBOVEC_4BIT_PLANES=0
+ pg_ctl -D /home/fedora/pgdata -m fast -w stop
waiting for server to shut down.......................................................... [bash 11:50:19 elapsed]
...... failed
pg_ctl: server does not shut down
+ TURBOVEC_4BIT_PLANES=0
+ pg_ctl -D /home/fedora/pgdata -l /home/fedora/pg.log -w start
pg_ctl: another server might be running; trying to start server anyway
waiting for server to start.... stopped waiting
pg_ctl: could not start server
Examine the log output.
+ python3 /home/fedora/tail_diag.py 400 /home/fedora/diag/B_planes_off.jsonl
+ cat /home/fedora/diag/B.txt
Traceback (most recent call last):
  File "/home/fedora/tail_diag.py", line 11, in <module>
    conn = psycopg.connect("host=/tmp dbname=bench", autocommit=True, prepare_threshold=0); cur = conn.cursor()
  File "/home/fedora/.local/lib/python3.14/site-packages/psycopg/connection.py", line 126, in connect
    raise last_ex.with_traceback(None)
psycopg.OperationalError: connection failed: connection to server on socket "/tmp/.s.PGSQL.5432" failed: FATAL:  the database system is shutting down
+ pg_ctl -D /home/fedora/pgdata -m fast -w stop
waiting for server to shut down...................... done
server stopped
+ pg_ctl -D /home/fedora/pgdata -l /home/fedora/pg.log -w start
waiting for server to start.... done
server started
+ echo '=== step C: knobs on whole-load (planes on): hi_dim_rerank off; scan_parallelism 1'
=== step C: knobs on whole-load (planes on): hi_dim_rerank off; scan_parallelism 1
+ python3 /home/fedora/tail_knobs.py
+ cat /home/fedora/diag/C.txt
/home/fedora/.local/lib/python3.14/site-packages/numpy/_core/fromnumeric.py:3862: RuntimeWarning: Mean of empty slice
  return _methods._mean(a, axis=axis, dtype=dtype,
/home/fedora/.local/lib/python3.14/site-packages/numpy/_core/_methods.py:142: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
slow qids (whole>60ms): 400 of 400
default                slow-set median   67.5 ms (max  138.0) | other median    nan ms
hi_dim_rerank=off      slow-set median   13.8 ms (max   14.3) | other median    nan ms
scan_parallelism=1     slow-set median   62.7 ms (max  135.0) | other median    nan ms
search_k=10,hdr=off    slow-set median   12.8 ms (max   13.3) | other median    nan ms
probes=64              slow-set median   63.4 ms (max   67.8) | other median    nan ms
+ echo '=== step D: perf profile of the 30 slowest whole-load queries'
=== step D: perf profile of the 30 slowest whole-load queries
+ python3 /home/fedora/tail_perf.py
. [bash 11:51:19 elapsed]
+ cat /home/fedora/diag/D.txt
--- slow30
    52.37%  [.] turbovec::search::avx512_post_flush_heap_update
                --2.95%--0
    27.17%  [.] randomize_mem
            |--22.40%--0xcb08
             --4.47%--0x6000
     4.95%  [.] pg_turbovec::index::ivf::materialize_rotation_matrix
             --0.55%--0
     1.17%  [.] __memmove_avx512_unaligned_erms
             --0.77%--0x2ffffffff
     1.15%  [.] hash_search_with_hash_value
             --1.14%--hash_search_with_hash_value
                        --0.53%--0
     1.00%  [.] <serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::recursion_checked::<<serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::parse_array<<alloc::vec::Vec<_> as serde_core::de::Deserialize>::deserialize::VecVisitor<f32>>::{closure#0}, alloc::vec::Vec<f32>>
     0.90%  [.] <serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::parse_value::<<f32 as serde_core::de::Deserialize>::deserialize::PrimitiveVisitor>
     0.64%  [.] core::slice::sort::unstable::quicksort::quicksort::<(f32, u64), <[(f32, u64)]>::sort_unstable_by<turbovec::search::scan_with_luts::{closure#5}::{closure#0}>::{closure#0}>
     0.58%  [.] _bt_check_natts
     0.56%  [.] turbovec::rotation::wht_block_avx512
     0.44%  [.] pg_turbovec::index::ivf::coarse_probe
     0.40%  [.] __memset_avx512_unaligned_erms
     0.40%  [.] pg_turbovec::index::ivf::rotate_query
     0.38%  [.] heap_hot_search_buffer
     0.37%  [.] turbovec::search::rank_first_keep
     0.35%  [.] pg_turbovec::index::relfile::read_coarse_centroids
     0.33%  [.] heap_page_prune_opt
     0.31%  [.] PinBuffer
     0.30%  [.] turbovec::search::search_single_query_vnni_blk2
     0.30%  [.] turbovec::rotation::permute_gather_avx512::<1>
     0.22%  [.] <turbovec::TurboQuantIndex>::try_search_with_mask
     0.21%  [.] turbovec::search::rerank_legacy::{closure#0}::{closure#1}
     0.21%  [.] turbovec::search::avx2_post_flush_heap_update
     0.20%  [.] LWLockRelease
     0.20%  [.] pg_turbovec::kernels::lane_sum::<pg_turbovec::kernels::dot::{closure#0}>
     0.19%  [.] turbovec::search::rank_full_known
     0.18%  [.] LWLockAttemptLock
     0.18%  [.] turbovec::search::low_dot_avx512
     0.17%  [.] __memcmp_evex_movbe
     0.15%  [.] AllocSetCheck
     0.15%  [.] _bt_compare
     0.12%  [.] pg_turbovec::kernels::cosine_distance_with_qnorm
     0.12%  [.] _bt_checkpage
     0.11%  [.] AllocSetAlloc
     0.10%  [k] rep_movs_alternative
     0.09%  [.] _bt_moveright
     0.08%  [.] turbovec::search::plane_terms
     0.08
--- fast30
    60.28%  [.] randomize_mem
            |--49.68%--0xcb08
             --9.77%--0x6000
    10.58%  [.] pg_turbovec::index::ivf::materialize_rotation_matrix
             --1.39%--0
     2.14%  [.] <serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::recursion_checked::<<serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::parse_array<<alloc::vec::Vec<_> as serde_core::de::Deserialize>::deserialize::VecVisitor<f32>>::{closure#0}, alloc::vec::Vec<f32>>
     1.96%  [.] <serde_cbor::de::Deserializer<serde_cbor::read::SliceRead>>::parse_value::<<f32 as serde_core::de::Deserialize>::deserialize::PrimitiveVisitor>
     1.80%  [.] __memmove_avx512_unaligned_erms
             --1.37%--0x2ffffffff
     1.65%  [.] hash_search_with_hash_value
             --1.60%--hash_search_with_hash_value
                        --0.72%--0
     1.55%  [.] turbovec::search::search
     1.24%  [.] turbovec::rotation::wht_block_avx512
             --0.78%--0
     1.20%  [.] core::slice::sort::unstable::quicksort::quicksort::<(f32, i64), <[(f32, i64)]>::sort_unstable_by<turbovec::search::rerank_legacy::{closure#0}::{closure#8}>::{closure#0}>
     1.06%  [.] _bt_check_natts
             --0.64%--BTreeTupleIsPivot (inlined)
                        --0.62%--_bt_check_natts
     0.91%  [.] heap_hot_search_buffer
             --0.90%--heap_hot_search_buffer
     0.86%  [.] pg_turbovec::index::ivf::coarse_probe
     0.86%  [.] pg_turbovec::index::ivf::rotate_query
     0.78%  [.] turbovec::search::exact4_sum_avx512
     0.62%  [.] __memset_avx512_unaligned_erms
     0.61%  [.] turbovec::rotation::permute_gather_avx512::<1>
     0.57%  [.] turbovec::search::rerank_legacy::{closure#0}::{closure#1}
     0.54%  [.] heap_page_prune_opt
             --0.54%--heap_page_prune_opt
     0.51%  [.] PinBuffer
     0.50%  [.] pg_turbovec::index::relfile::read_coarse_centroids
     0.49%  [.] <turbovec::TurboQuantIndex>::try_search_with_mask
     0.47%  [.] LWLockAttemptLock
     0.38%  [.] LWLockRelease
     0.37%  [.] __memcmp_evex_movbe
     0.36%  [.] pg_turbovec::kernels::lane_sum::<pg_turbovec::kernels::dot::{closure#0}>
     0.32%  [.] _bt_compare
     0.30%  [.] AllocSetCheck
     0.27%  [.] pg_turbovec::kernels::cosine_distance_with_qnorm
     0.23%  [.] _bt_checkpage
     0.21%  [.] AllocSetAlloc
     0.21%  [.] _int_free_chunk
     0.20%  [.] hash_bytes
     0.19%  [.] _bt_moveright
     0.18%  [.] rayon_core::scope::scope::<turbovec::search::pool_map_spin<alloc::vec::Vec<(f32, 
+ echo ALL_DONE
ALL_DONE
[heartbeat] DONE rc=0 (clean exit)
