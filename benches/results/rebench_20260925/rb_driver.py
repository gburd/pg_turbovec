#!/usr/bin/env python3
"""End-to-end matched-recall sweep, corrected.

Fixes over the flawed harness:
 * latency = top-level EXPLAIN(ANALYZE) **Execution Time** (whole query incl.
   the xs_recheckorderby reorder-queue recheck), measured IDENTICALLY for both
   engines. NOT the Index-Scan-node time (which excludes turbovec's recheck).
 * ALL queries of an arm run in ONE psql -f session (avoids the ~460ms
   per-backend cold cache reload the prior per-query psql paid every time).
 * seq-fallback is flagged ONLY for a Seq Scan on `docs` (the InitPlan seqscan
   on the 100-row query_set is harmless and must not disqualify an arm).
 * recall from the SAME arm's returned ids.
"""
import json, subprocess, sys, time, statistics, os
DB=os.environ.get("DB","bench")
def loadavg(): return float(open("/proc/loadavg").read().split()[0])
def run_arm(setup, query_tmpl, qids, n_warm, n_timed, lit):
    """One psql script: warm-ups (discarded) then timed EXPLAIN(ANALYZE JSON) +
    the plain query for ids, all in ONE backend. lit maps qid->vector literal,
    inlined (no subquery) to avoid the ~90ms InitPlan overhead measured on both
    engines."""
    lines=[setup]
    for i in range(n_warm):
        lines.append(f"EXPLAIN (ANALYZE, TIMING ON, BUFFERS OFF) {query_tmpl % lit[qids[i%len(qids)]]}")
    for i in range(n_timed):
        v=lit[qids[i%len(qids)]]
        lines.append(r"\echo @@T")
        lines.append(f"EXPLAIN (ANALYZE, FORMAT JSON, TIMING ON, BUFFERS OFF) {query_tmpl % v}")
        lines.append(r"\echo @@I")
        lines.append(query_tmpl % v)
    script="\n".join(lines)+"\n"
    r=subprocess.run(["psql","-d",DB,"-v","ON_ERROR_STOP=1","-qAt","-F","|"],
                     input=script,capture_output=True,text=True)
    if r.returncode!=0: raise RuntimeError(r.stderr.strip()[-500:])
    return r.stdout
def parse(out):
    """Yield (exec_ms, ids, seq_on_docs) per timed query."""
    blocks=out.split("@@T\n")[1:]
    for b in blocks:
        jpart, rest = b.split("@@I\n",1)
        plan=json.loads(jpart)[0]
        et=plan["Execution Time"]
        seq_docs=[False]
        def walk(n):
            if n.get("Node Type")=="Seq Scan" and n.get("Relation Name")=="docs": seq_docs[0]=True
            for c in n.get("Plans",[]): walk(c)
        walk(plan["Plan"])
        rlines=[x for x in rest.strip().splitlines() if x.strip()]
        ids=[int(x) for x in rlines if x.strip().lstrip("-").isdigit()]
        yield et, ids, seq_docs[0]
def recall(pred, truth): return len(set(pred)&set(truth))/10.0
def main():
    gate=float(os.environ.get("LOAD_GATE","6.0"))
    n_timed=int(os.environ.get("N_TIMED","40")); n_warm=int(os.environ.get("N_WARM","5"))
    qids=[int(x) for x in subprocess.run(["psql","-d",DB,"-qAt","-c","SELECT qid FROM public.query_set ORDER BY qid"],capture_output=True,text=True).stdout.split() if x.strip()]
    gt={}
    for line in subprocess.run(["psql","-d",DB,"-qAt","-F","|","-c","SELECT qid, hit_id FROM public.gt_top10"],capture_output=True,text=True).stdout.splitlines():
        if "|" not in line: continue
        q,h=line.split("|"); gt.setdefault(int(q),[]).append(int(h))
    assert all(len(gt.get(q,[]))==10 for q in qids), "GT incomplete"
    # Prefetch each query's vector literal ONCE (emb for HNSW/vector, tv for
    # turbovec). Inlining as a literal is what a real client does with a bound
    # param; a subquery in the ORDER BY adds ~90ms of InitPlan/materialize
    # overhead OUTSIDE the index scan and inflates every arm (measured).
    qemb={}; qtv={}
    for q in qids:
        qemb[q]=subprocess.run(["psql","-d",DB,"-qAt","-c",f"SELECT emb FROM public.query_set WHERE qid={q}"],capture_output=True,text=True).stdout.strip()
        qtv[q]=subprocess.run(["psql","-d",DB,"-qAt","-c",f"SELECT tv::text FROM public.query_set WHERE qid={q}"],capture_output=True,text=True).stdout.strip()
    arms=[]
    for ef in (20,40,80,120,200,400,800):
        arms.append(("hnsw",f"ef{ef}",f"SET hnsw.ef_search={ef}; SET enable_seqscan=off;",
            "SELECT id FROM public.docs ORDER BY emb <=> '%s'::vector LIMIT 10;","emb"))
    for tag in ("flat_bw4","flat_bw1"):
        for k in (32,100,256,400,800,1024,2000):
            arms.append(("tv",f"{tag}_k{k}",
                f"SET search_path=turbovec,public; SET enable_seqscan=off; SET turbovec.search_k={k}; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off;",
                "SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '%s'::turbovec.vector LIMIT 10;","tv"))
    for tag in ("ivf_bw4","ivf_bw1"):
        for probes in (8,16,32,64,128):
            for k in (100,256,800):
                arms.append(("tv",f"{tag}_p{probes}_k{k}",
                    f"SET search_path=turbovec,public; SET enable_seqscan=off; SET turbovec.probes={probes}; SET turbovec.search_k={k}; SET turbovec.oversample=1.0; SET turbovec.hi_dim_rerank=off;",
                    "SELECT id FROM public.docs ORDER BY tv OPERATOR(turbovec.<=>) '%s'::turbovec.vector LIMIT 10;","tv"))
    fam=os.environ.get("ARM_FAMILY","")
    def keep(kind,label):
        if not fam: return True
        return kind=="hnsw" if fam=="hnsw" else label.startswith(fam)
    arms=[a for a in arms if keep(a[0],a[1])]
    litmap={"emb":qemb,"tv":qtv}
    prev=json.load(open(os.environ["OUT"]))["results"] if os.path.exists(os.environ.get("OUT","")) else []
    results=[]
    for kind,label,setup,tmpl,vecset in arms:
        la0=loadavg()
        out=run_arm(setup,tmpl,qids,n_warm,n_timed,litmap[vecset])
        la1=loadavg(); obs=max(la0,la1)
        parsed=list(parse(out))
        times=[et for et,_,_ in parsed]
        recalls=[recall(ids, gt[qids[i%len(qids)]]) for i,(_,ids,_) in enumerate(parsed)]
        seq_any=any(s for _,_,s in parsed)
        row={"kind":kind,"label":label,
             "p50_ms":round(statistics.median(times),3),
             "p95_ms":round(sorted(times)[min(int(0.95*len(times)),len(times)-1)],3),
             "recall":round(statistics.mean(recalls),4),
             "n":len(times),"loadavg":round(obs,2),
             "contended":obs>gate,"seq_fallback":seq_any}
        results.append(row)
        print(f"  {label}: R@10={row['recall']} p50={row['p50_ms']}ms p95={row['p95_ms']} "
              f"seq_on_docs={seq_any} load={obs:.2f} contended={row['contended']}",flush=True)
    out_obj={"meta":{"host":os.uname().nodename,"corpus":"cohere-wiki en 1M x 1024d",
                 "latency_basis":"top_level_EXPLAIN_ANALYZE_Execution_Time_whole_query_one_session",
                 "gate":gate,"n_timed":n_timed,"ts":time.strftime("%Y-%m-%dT%H:%M:%S%z")},
         "results":prev+results}
    json.dump(out_obj, open(os.environ.get("OUT","/tmp/rb_result.json"),"w"), indent=1)
    print("wrote", os.environ.get("OUT"),flush=True)
if __name__=="__main__": main()
