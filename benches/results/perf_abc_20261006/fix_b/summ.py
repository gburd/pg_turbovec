import json,statistics as st,collections,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
d=collections.defaultdict(list)
for r in rows:
    ms=[float(x) for x in r["ms"].split()]
    d[(r["tab"],r["op"],r["v"])].append(st.median(ms))
N=20000
print("tab  op              base_us/row  new_us/row   delta   per-round medians (ms) base | new")
for tab in ["t","tp"]:
    for op in ["<=>","<->","<#>","<+>","vector_dims","pg_column_size"]:
        b=d[(tab,op,"base")]; n=d[(tab,op,"new")]
        bm=st.median(b)*1000/N; nm=st.median(n)*1000/N
        print("%-4s %-15s %11.3f %11.3f %7.3f   %s | %s" % (tab,op,bm,nm,nm-bm,[round(x,1) for x in b],[round(x,1) for x in n]))
