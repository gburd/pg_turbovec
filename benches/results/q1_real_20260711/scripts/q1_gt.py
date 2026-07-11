#!/usr/bin/env python3
"""Recompute exact top-100 L2 GT (excluding self) for an already-loaded table.
Keeps a 200-wide buffer so after self-removal we still have a full 100.
Usage: q1_gt.py <table> <dim>
"""
import sys, time
import numpy as np
import psycopg2

SOCK = "/mnt/nvme/pg"; DB = "vecbench"; DATA = "/mnt/nvme/data"


def connect():
    c = psycopg2.connect(host=SOCK, dbname=DB, user="ec2-user", port=5432)
    cur = c.cursor(); cur.execute("SET search_path=public,turbovec"); c.commit(); cur.close()
    return c


def main():
    tab, dim = sys.argv[1], int(sys.argv[2])
    Q = np.load(f"{DATA}/{tab}_queries.npy").astype(np.float32)
    nq = Q.shape[0]
    # query ids are 0..nq-1 (the loader reserves the first nq rows as queries)
    qids = np.arange(nq, dtype=np.int64)
    BUF = 200; K = 100
    best_d = np.full((nq, BUF), np.inf, dtype=np.float32)
    best_i = np.full((nq, BUF), -1, dtype=np.int64)
    c = connect(); cur = c.cursor()
    cur.execute(f"DECLARE gtc CURSOR FOR SELECT id, embt::text FROM {tab} ORDER BY id")
    CH = 50000; seen = 0; t0 = time.time()
    while True:
        cur.execute(f"FETCH {CH} FROM gtc")
        rows = cur.fetchall()
        if not rows:
            break
        ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        mat = np.asarray([[float(x) for x in r[1].strip("[]").split(",")] for r in rows],
                         dtype=np.float32)
        d = (mat * mat).sum(1)[None, :] - 2.0 * (Q @ mat.T)
        cand_d = np.concatenate([best_d, d], axis=1)
        cand_i = np.concatenate([best_i, np.broadcast_to(ids, (nq, len(ids)))], axis=1)
        part = np.argpartition(cand_d, BUF, axis=1)[:, :BUF]
        best_d = np.take_along_axis(cand_d, part, axis=1)
        best_i = np.take_along_axis(cand_i, part, axis=1)
        seen += len(rows)
        if seen % 200000 < CH:
            print(f"  GT {seen} ({time.time()-t0:.0f}s)", flush=True)
    cur.execute("CLOSE gtc"); c.close()
    order = np.argsort(best_d, axis=1)
    gt_sorted = np.take_along_axis(best_i, order, axis=1)
    gt_out = np.full((nq, K), -1, dtype=np.int64)
    for r in range(nq):
        row = [x for x in gt_sorted[r] if x != qids[r] and x >= 0][:K]
        gt_out[r, :len(row)] = row
    np.save(f"{DATA}/{tab}_gt.npy", gt_out)
    valid = (gt_out >= 0).sum(1)
    print(f"GT rewritten: valid-per-row min/med/max {valid.min()}/{int(np.median(valid))}/{valid.max()} "
          f"in {time.time()-t0:.0f}s", flush=True)
    print("GT_DONE", flush=True)


if __name__ == "__main__":
    main()
