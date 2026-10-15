import json, subprocess, threading, time, numpy as np, psycopg
Q = np.load('/home/fedora/corpus/corpus_1000000/q.npy')
A = [json.loads(l) for l in open('/home/fedora/diag/A_planes_on.jsonl')]
slow = [r['qid'] for r in sorted(A, key=lambda r: -r['whole_ms'])[:30]]
fast = [r['qid'] for r in sorted(A, key=lambda r: r['whole_ms'])[:30]]
sql = "SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) %s::turbovec.vector LIMIT 10"
def prof(qids, label):
    conn = psycopg.connect("host=/tmp dbname=bench", autocommit=True, prepare_threshold=0); cur = conn.cursor()
    for s in ["SET search_path = turbovec, public", "SET enable_seqscan = off", "SET turbovec.out_of_core = off", "SET turbovec.cache_size_mb = 4096", "SET turbovec.probes = 16"]: cur.execute(s)
    lit = ["[" + ",".join(f"{x:.6f}" for x in Q[q]) + "]" for q in qids]
    for v in lit[:5]: cur.execute(sql, (v,)).fetchall()
    pid = cur.execute("SELECT pg_backend_pid()").fetchone()[0]
    p = subprocess.Popen(["sudo", "perf", "record", "-q", "-g", "-F", "1999", "-p", str(pid), "-o", f"/tmp/perf_{label}.data"])
    time.sleep(1.0)
    for _ in range(5):
        for v in lit: cur.execute(sql, (v,)).fetchall()
    p.send_signal(2); p.wait()
    rep = subprocess.run(["sudo", "perf", "report", "-q", "-i", f"/tmp/perf_{label}.data", "--stdio", "--no-children", "--sort", "sym"], capture_output=True, text=True).stdout
    print(f"--- {label}"); print("\n".join(l for l in rep.splitlines() if l.strip().endswith(']') is False and '%' in l)[:2500])
prof(slow, "slow30"); prof(fast, "fast30")
