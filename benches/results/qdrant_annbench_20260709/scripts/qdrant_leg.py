#!/usr/bin/env python3
"""Qdrant leg: load a corpus into Qdrant, build HNSW m=32/efc=256 + int8
scalar quant (Qdrant's own continuous-benchmark profile, oversampling=2,
rescore=true), sweep hnsw_ef; measure recall@10 vs the SAME published HDF5
GT, single-conn p50 latency + parallel-8 RPS, and on-disk collection size.

Emits the same JSON row shape as leg3_driver so the frontiers join.

Usage: qdrant_leg.py <corpus> [collection_suffix]
  corpus in {sift1m, gist1m, gist10m}
"""
import sys, os, time, json, statistics, threading
import numpy as np
import h5py
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

RESULTS = "/mnt/nvme/results"
SPECS = {
    "sift1m": ("/mnt/nvme/data/sift-128-euclidean.hdf5", 128),
    "gist1m": ("/mnt/nvme/data/gist-960-euclidean.hdf5", 960),
    "gist10m": ("/mnt/nvme/data/gist10m.npy", 960),  # semi-synthetic, GT below
}


def load_train(corpus):
    path, dim = SPECS[corpus]
    if corpus == "gist10m":
        return np.load(path, mmap_mode="r"), dim
    with h5py.File(path, "r") as h:
        return np.asarray(h["train"][:], dtype=np.float32), dim


def load_gt(corpus, qcap=1000):
    path, dim = SPECS[corpus]
    if corpus == "gist10m":
        test = np.load("/mnt/nvme/data/gist10m_test.npy")[:qcap]
        gt = np.load("/mnt/nvme/data/gist10m_gt.npy")[:qcap]
        return test.astype(np.float32), gt.astype(np.int64)
    with h5py.File(path, "r") as h:
        return np.asarray(h["test"][:qcap], np.float32), np.asarray(h["neighbors"][:qcap], np.int64)


def build_collection(client, coll, dim, train, m=32, efc=256):
    if client.collection_exists(coll):
        client.delete_collection(coll)
    client.create_collection(
        collection_name=coll,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.EUCLID,
                                        on_disk=False),
        hnsw_config=qm.HnswConfigDiff(m=m, ef_construct=efc),
        quantization_config=qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8,
                                               quantile=0.99, always_ram=True)),
        optimizers_config=qm.OptimizersConfigDiff(default_segment_number=8),
    )
    n = train.shape[0]
    batch = 2048
    t0 = time.time()
    for start in range(0, n, batch):
        end = min(start + batch, n)
        chunk = np.asarray(train[start:end], dtype=np.float32)
        client.upsert(
            collection_name=coll, wait=False,
            points=qm.Batch(ids=list(range(start, end)),
                            vectors=chunk.tolist()))
        if start % 200000 == 0:
            print(f"  qdrant upsert {end}/{n} ({time.time()-t0:.0f}s)", flush=True)
    # wait for indexing to finish (status green + indexed_vectors ~ n)
    print("  waiting for green + full index ...", flush=True)
    while True:
        info = client.get_collection(coll)
        if info.status == qm.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= n * 0.99:
            break
        time.sleep(5)
    build_s = time.time() - t0
    # on-disk size of the collection storage dir
    coll_dir = f"/mnt/nvme/qdrant_storage/collections/{coll}"
    sz = 0
    for root, _, files in os.walk(coll_dir):
        for f in files:
            try:
                sz += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    print(f"  qdrant built build_s={build_s:.1f} idx_MB={sz/1e6:.0f}", flush=True)
    return {"build_s": round(build_s, 2), "idx_bytes": sz, "idx_total_bytes": sz}


def _search_one(client, coll, q, k, ef, oversample):
    return client.query_points(
        collection_name=coll, query=q.tolist(), limit=k, with_payload=False,
        search_params=qm.SearchParams(
            hnsw_ef=ef,
            quantization=qm.QuantizationSearchParams(
                ignore=False, rescore=True, oversampling=oversample)),
    ).points


def measure_latency(client, coll, k, test, gt, ef, oversample, repeats=3):
    # warm
    for q in test:
        _search_one(client, coll, q, k, ef, oversample)
    recall = None
    best = None
    for r in range(repeats):
        hits = 0; total = 0; lats = []
        for i, q in enumerate(test):
            t0 = time.perf_counter()
            res = _search_one(client, coll, q, k, ef, oversample)
            lats.append((time.perf_counter() - t0) * 1000.0)
            if r == 0:
                ids = set(int(p.id) for p in res)
                truth = set(int(x) for x in gt[i][:k])
                hits += len(truth & ids); total += k
        if r == 0:
            recall = hits / total
        lats.sort()
        m = statistics.mean(lats)
        cand = {"p50": lats[len(lats)//2], "p95": lats[int(len(lats)*0.95)],
                "mean": m, "qps_1conn": 1000.0/m}
        if best is None or cand["mean"] < best["mean"]:
            best = cand
    best["recall"] = round(recall, 4)
    for kk in ("p50", "p95", "mean", "qps_1conn"):
        best[kk] = round(best[kk], 3)
    return best


def measure_qps(coll, k, test, ef, oversample, nconn=8, duration=8.0):
    stop = time.time() + duration
    counts = [0] * nconn
    def worker(wid):
        c = QdrantClient(host="127.0.0.1", port=6333, prefer_grpc=True, timeout=60)
        idx = wid; n = 0
        while time.time() < stop:
            q = test[idx % len(test)]; idx += 1
            _search_one(c, coll, q, k, ef, oversample)
            n += 1
        counts[wid] = n
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(nconn)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    return round(sum(counts) / (time.time() - t0), 1)


def run(corpus, suffix=""):
    train, dim = load_train(corpus)
    test, gt = load_gt(corpus)
    coll = f"{corpus}{suffix}"
    client = QdrantClient(host="127.0.0.1", port=6333, prefer_grpc=True, timeout=600)
    print(f"[qdrant {corpus}] n={train.shape[0]} dim={dim} q={len(test)}", flush=True)
    b = build_collection(client, coll, dim, train)
    rows = []
    oversample = 2.0
    for ef in [50, 100, 150, 256, 400, 800]:
        r = measure_latency(client, coll, 10, test, gt, ef, oversample)
        qps8 = measure_qps(coll, 10, test, ef, oversample, nconn=8)
        row = {"engine": "qdrant", "variant": "m32_efc256_int8", "hnsw_ef": ef,
               "oversample": oversample, "rescore": True, **b, **r, "qps_8conn": qps8}
        rows.append(row)
        print(f"  ef={ef}: R@10={r['recall']} p50={r['p50']}ms qps1={r['qps_1conn']} qps8={qps8}", flush=True)
        with open(f"{RESULTS}/qdrant_{corpus}.json", "w") as f:
            json.dump(rows, f, indent=2)
    print(f"  wrote {RESULTS}/qdrant_{corpus}.json", flush=True)
    return rows


if __name__ == "__main__":
    corpus = sys.argv[1]
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    run(corpus, suffix)
    print("QDRANT_LEG_DONE", flush=True)
