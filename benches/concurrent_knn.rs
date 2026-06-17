//! Concurrent in-process bench for the cache-lookup hot path.
//!
//! This bench is the in-process counterpart to `bench/concurrent.sh`.
//! Where the bash script measures *aggregate* QPS across N
//! independent PostgreSQL backends (and therefore exercises N
//! independent copies of the `Mutex<HashMap>` cache), this bench
//! hammers a *single* `Mutex<HashMap>` from N native threads inside
//! one process. That isolates the question "would the cache mutex
//! become a bottleneck if we ever shared the cache between
//! concurrent threads in the same backend?" from CPU saturation,
//! per-backend SPI overhead, and PostgreSQL's own internal
//! LWLocks (snapshots, ProcArrayLock, the lock manager…), all of
//! which are what `bench/concurrent.sh` actually ends up
//! measuring.
//!
//! Run with:
//!
//! ```bash
//! cargo bench --bench concurrent_knn --no-default-features --features pg16
//! ```
//!
//! Output is JSON on stdout plus a copy under
//! `benches/results/concurrent_knn_inproc_<ts>.json`.
//!
//! Why this is *not* a Criterion `bench_function`: Criterion's
//! per-iteration timing model is a poor fit for a benchmark whose
//! whole point is wall-clock throughput at varying thread counts
//! over a fixed time budget.  We declare `harness = false` and run
//! a plain `fn main()` instead, mirroring what people do with
//! pgbench.

#![allow(clippy::too_many_lines, clippy::map_unwrap_or)]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use rand::prelude::*;
use rand::rngs::StdRng;
use turbovec::IdMapIndex;

use pg_turbovec::kernels;

/// Composite cache key — bit-for-bit identical to `src/cache.rs`.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
struct CacheKey {
    rel_oid: u32,
    attnum: i16,
    bit_width: u8,
    dim: u32,
}

/// Cache entry — a faithful copy of the production layout, minus
/// the `bytes` field (we don't exercise the LRU eviction here; a
/// single entry is enough to reproduce the lookup contention
/// pattern).
struct Entry {
    index: Arc<IdMapIndex>,
    relfilenode: u32,
    n_rows: i64,
    seq: u64,
}

/// The full cache type as declared in `src/cache.rs`.
type Cache = Mutex<HashMap<CacheKey, Entry>>;

/// Mirror of `cache::lookup`. The only behavioural deviation is the
/// `next_seq` step: the production code uses a *second* mutex for
/// the global sequence counter, so we do the same here to avoid
/// understating contention.
fn lookup(
    cache: &Cache,
    seq_counter: &Mutex<u64>,
    key: CacheKey,
    expected_relfile: u32,
    expected_n_rows: i64,
) -> Option<Arc<IdMapIndex>> {
    let mut g = cache.lock();
    let entry = g.get_mut(&key)?;
    if entry.relfilenode != expected_relfile || entry.n_rows != expected_n_rows {
        g.remove(&key);
        return None;
    }
    // Same two-mutex layout as production (cache.rs `next_seq`).
    let mut s = seq_counter.lock();
    *s += 1;
    entry.seq = *s;
    Some(entry.index.clone())
}

/// Build a deterministic 10 000-row, 384-dim corpus and the
/// corresponding `IdMapIndex` at 4-bit width.
fn build_index(n: usize, dim: usize, bit_width: usize, seed: u64) -> IdMapIndex {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
    let mut ids: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n {
        let raw: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
        let unit = kernels::normalise_to_vec(&raw);
        flat.extend_from_slice(&unit);
        ids.push(i as u64);
    }
    let mut idx = IdMapIndex::new(dim, bit_width).expect("IdMapIndex::new (dim, bit_width)");
    idx.add_with_ids(&flat, &ids).expect("add_with_ids");
    idx
}

/// Build a fixed pool of unit-norm query vectors so the workload is
/// steady-state.  Threads pick from this pool with a per-thread
/// PRNG.
fn build_query_pool(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let raw: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
            kernels::normalise_to_vec(&raw)
        })
        .collect()
}

#[derive(serde::Serialize)]
struct Run {
    threads: usize,
    duration_s: f64,
    total_ops: u64,
    qps_total: f64,
    qps_per_thread: f64,
    speedup_vs_n1: f64,
    avg_latency_us: f64,
}

#[derive(serde::Serialize)]
struct Report {
    timestamp_utc: String,
    host: String,
    n_cpu: usize,
    corpus_rows: usize,
    dim: usize,
    bit_width: usize,
    k: usize,
    duration_s_per_run: f64,
    mode: String,
    notes: &'static str,
    runs: Vec<Run>,
}

fn main() {
    // Parameters — match `bench/concurrent.sh`.
    const N_CORPUS: usize = 10_000;
    const DIM: usize = 384;
    const BIT_WIDTH: usize = 4;
    const K: usize = 10;
    const QUERY_POOL: usize = 256;
    let duration = Duration::from_secs(
        std::env::var("DURATION_S")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(5),
    );
    let thread_counts: Vec<usize> = std::env::var("THREADS")
        .ok()
        .map(|s| {
            s.split(|c: char| c == ',' || c.is_whitespace())
                .filter(|t| !t.is_empty())
                .map(|t| t.parse::<usize>().expect("threads must be integers"))
                .collect()
        })
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16]);

    let mode = std::env::var("MODE").unwrap_or_else(|_| "mutex".to_string());
    eprintln!(
        "[setup] building {} x {}-dim corpus at bit_width={} (seed=42), mode={}",
        N_CORPUS, DIM, BIT_WIDTH, mode
    );
    let idx = build_index(N_CORPUS, DIM, BIT_WIDTH, 42);
    let queries = build_query_pool(QUERY_POOL, DIM, 13);

    let arc_idx: Arc<IdMapIndex> = Arc::new(idx);
    let key = CacheKey {
        rel_oid: 16384,
        attnum: 1,
        bit_width: BIT_WIDTH as u8,
        dim: DIM as u32,
    };
    let cache: Arc<Cache> = Arc::new(Mutex::new(HashMap::new()));
    let seq: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    cache.lock().insert(
        key,
        Entry {
            index: Arc::clone(&arc_idx),
            relfilenode: 1,
            n_rows: N_CORPUS as i64,
            seq: 1,
        },
    );

    let queries = Arc::new(queries);

    let mut runs: Vec<Run> = Vec::new();
    let mut baseline_qps: Option<f64> = None;

    for &n in &thread_counts {
        eprintln!("[run] threads={n}, duration={:?}", duration);
        let stop = Arc::new(AtomicBool::new(false));
        let total_ops = Arc::new(AtomicU64::new(0));
        let total_lat_ns = Arc::new(AtomicU64::new(0));

        let mut handles = Vec::with_capacity(n);
        let start_barrier = Arc::new(std::sync::Barrier::new(n + 1));
        for tid in 0..n {
            let cache = Arc::clone(&cache);
            let seq = Arc::clone(&seq);
            let arc_idx_t = Arc::clone(&arc_idx);
            let stop = Arc::clone(&stop);
            let total_ops = Arc::clone(&total_ops);
            let total_lat_ns = Arc::clone(&total_lat_ns);
            let queries = Arc::clone(&queries);
            let barrier = Arc::clone(&start_barrier);
            let mode = mode.clone();
            handles.push(thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(0x00c0_ffee + tid as u64);
                barrier.wait();
                let mut local_ops: u64 = 0;
                let mut local_ns: u64 = 0;
                while !stop.load(Ordering::Relaxed) {
                    let qid = rng.gen_range(0..queries.len());
                    let q = &queries[qid];
                    let t0 = Instant::now();
                    let arc = match mode.as_str() {
                        // "nolock": skip the cache entirely.  This
                        // is the upper bound — the speedup curve
                        // here tells us how much *non-cache* cost
                        // (mostly `IdMapIndex::search`) is on the
                        // hot path.  Comparing it to "mutex" mode
                        // isolates the cache mutex contribution.
                        "nolock" => Arc::clone(&arc_idx_t),
                        _ => lookup(&cache, &seq, key, 1, N_CORPUS as i64)
                            .expect("warm cache should always hit"),
                    };
                    let (_scores, _ids) = arc.search(q, K);
                    let elapsed = t0.elapsed().as_nanos() as u64;
                    local_ops += 1;
                    local_ns += elapsed;
                }
                total_ops.fetch_add(local_ops, Ordering::Relaxed);
                total_lat_ns.fetch_add(local_ns, Ordering::Relaxed);
            }));
        }

        // Release threads simultaneously, time the wall-clock window.
        start_barrier.wait();
        let started = Instant::now();
        thread::sleep(duration);
        stop.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }
        let elapsed = started.elapsed().as_secs_f64();

        let ops = total_ops.load(Ordering::Relaxed);
        let lat_ns = total_lat_ns.load(Ordering::Relaxed);
        let qps_total = ops as f64 / elapsed;
        let qps_per_thread = qps_total / n as f64;
        let avg_lat_us = if ops > 0 {
            (lat_ns as f64 / ops as f64) / 1_000.0
        } else {
            0.0
        };
        if baseline_qps.is_none() {
            baseline_qps = Some(qps_total);
        }
        let speedup = qps_total / baseline_qps.unwrap();
        eprintln!(
            "    threads={:<3} qps_total={:>10.0}  qps/thread={:>9.0}  speedup={:.3}x  avg_lat={:.2} us",
            n, qps_total, qps_per_thread, speedup, avg_lat_us
        );
        runs.push(Run {
            threads: n,
            duration_s: elapsed,
            total_ops: ops,
            qps_total,
            qps_per_thread,
            speedup_vs_n1: speedup,
            avg_latency_us: avg_lat_us,
        });
    }

    let report = Report {
        timestamp_utc: chrono_like_utc(),
        host: hostname(),
        n_cpu: num_cpus_best_effort(),
        corpus_rows: N_CORPUS,
        dim: DIM,
        bit_width: BIT_WIDTH,
        k: K,
        duration_s_per_run: duration.as_secs_f64(),
        mode: mode.clone(),
        notes: "in-process Mutex<HashMap> contention bench — \
                see bench/concurrent.sh for the cross-backend \
                pgbench sweep",
        runs,
    };
    let json = serde_json::to_string_pretty(&report).unwrap();
    println!("{}", json);

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/results");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join(format!(
        "concurrent_knn_inproc_{}_{}.json",
        report.mode, report.timestamp_utc
    ));
    if let Err(e) = fs::write(&path, &json) {
        eprintln!("[warn] failed to write {}: {e}", path.display());
    } else {
        eprintln!("[done] wrote {}", path.display());
    }
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| {
            std::process::Command::new("uname")
                .arg("-n")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn num_cpus_best_effort() -> usize {
    thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
}

/// `YYYYMMDDTHHMMSSZ` UTC timestamp using only stdlib.  Avoids
/// pulling in `chrono` for what is essentially a debug filename.
fn chrono_like_utc() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = dur.as_secs() as i64;
    // Civil-from-days, ported from Howard Hinnant's chrono primer:
    // <https://howardhinnant.github.io/date_algorithms.html#civil_from_days>
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let hour = sod / 3_600;
    let minute = (sod % 3_600) / 60;
    let second = sod % 60;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = y + i64::from(m <= 2);
    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        y, m, d, hour, minute, second
    )
}
