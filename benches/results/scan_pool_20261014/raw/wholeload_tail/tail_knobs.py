import json, time, numpy as np, psycopg
Q = np.load('/home/fedora/corpus/corpus_1000000/q.npy'); N = 400
lit = ["[" + ",".join(f"{x:.6f}" for x in q) + "]" for q in Q[:N]]
A = {json.loads(l)['qid']: json.loads(l) for l in open('/home/fedora/diag/A_planes_on.jsonl')}
slow = [q for q, r in A.items() if r['whole_ms'] > 60]
print(f"slow qids (whole>60ms): {len(slow)} of {N}")
sql = "SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) %s::turbovec.vector LIMIT 10"
base = ["SET search_path = turbovec, public", "SET enable_seqscan = off", "SET turbovec.out_of_core = off", "SET turbovec.cache_size_mb = 4096", "SET turbovec.probes = 16"]
for name, extra in [("default", []), ("hi_dim_rerank=off", ["SET turbovec.hi_dim_rerank = off"]),
                    ("scan_parallelism=1", ["SET turbovec.scan_parallelism = 1"]),
                    ("search_k=10,hdr=off", ["SET turbovec.hi_dim_rerank = off", "SET turbovec.search_k = 10"]),
                    ("probes=64", ["SET turbovec.probes = 64"])]:
    conn = psycopg.connect("host=/tmp dbname=bench", autocommit=True, prepare_threshold=0); cur = conn.cursor()
    for s in base + extra: cur.execute(s)
    for i in range(20): cur.execute(sql, (lit[i],)).fetchall()
    ts = []
    for q in slow[:60] + [q for q in range(N) if q not in slow][:60]:
        t = time.perf_counter_ns(); cur.execute(sql, (lit[q],)).fetchall(); ts.append(((time.perf_counter_ns() - t) / 1e6, q in slow))
    s = [t for t, w in ts if w]; f = [t for t, w in ts if not w]
    print(f"{name:22s} slow-set median {np.median(s):6.1f} ms (max {max(s):6.1f}) | other median {np.median(f):6.1f} ms")
    conn.close()
