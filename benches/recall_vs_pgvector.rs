//! Recall vs pgvector head-to-head benchmark for `pg_turbovec`.
//!
//! Unlike `benches/recall.rs` (which uses synthetic random vectors)
//! this bench drives the `turbovec::IdMapIndex` against a pre-built
//! real-embedding fixture (e.g. GloVe-100 from ann-benchmarks). It
//! reports R@1 / R@10 / R@100 against an *exact* ground-truth file,
//! plus per-query latency at LIMIT=10 (p50/p95/p99), at bit_width
//! 4 and 2.
//!
//! The pgvector HNSW comparison is driven by
//! `benches/scripts/run_recall_vs_pgvector.py`, which talks SQL to a
//! cluster that has both extensions loaded. This bench measures the
//! pure-Rust kernel cost (the upper bound for what `pg_turbovec`'s
//! SQL surface can deliver) and — because it shares the bit_width
//! and the ground-truth file with the SQL driver — its R@k numbers
//! are directly comparable to pgvector HNSW running against the
//! exact same corpus.
//!
//! # Inputs
//!
//! Set `TURBOVEC_FIXTURE_DIR` to a directory containing:
//!
//!   corpus.bin         — `<u32 dim><u32 n><f32 dim*n>`
//!   queries.bin        — `<u32 dim><u32 n><f32 dim*n>`
//!   ground_truth.bin   — `<u32 k><u32 n><u32 k*n>`
//!
//! Both `corpus.bin` and `queries.bin` should already be unit-normalised.
//! See `benches/scripts/prepare_glove_fixture.py`.
//!
//! Optional env vars:
//!
//!   TURBOVEC_BIT_WIDTHS  — comma-separated list (default `4,2`)
//!   TURBOVEC_RESULTS     — output JSON path (default `benches/results/recall_vs_pgvector_<DATE>.json`)
//!   TURBOVEC_LIMIT       — k for latency measurement (default 10)
//!
//! # Run
//!
//! ```bash
//! TURBOVEC_FIXTURE_DIR=fixtures/glove-100 \
//!   cargo bench --bench recall_vs_pgvector --no-default-features
//! ```

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use turbovec::IdMapIndex;

#[derive(serde::Serialize)]
struct PerConfig {
    bit_width: usize,
    n_corpus: usize,
    n_queries: usize,
    dim: usize,
    r_at_1: f64,
    r_at_10: f64,
    r_at_100: f64,
    /// Wall time per query, in microseconds. LIMIT-K = `limit_k`.
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    mean_us: f64,
    limit_k: usize,
    /// Bytes the serialized .tvim index occupies (whole-index
    /// footprint, including id-map side-tables).
    index_bytes: u64,
    /// Bytes per row, i.e. `index_bytes / n_corpus`.
    bytes_per_row: f64,
    build_secs: f64,
}

#[derive(serde::Serialize)]
struct ExactBaseline {
    n_corpus: usize,
    n_queries: usize,
    dim: usize,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    mean_us: f64,
    limit_k: usize,
    /// FP32 storage of the corpus (no quantization).
    bytes_per_row: f64,
}

#[derive(serde::Serialize)]
struct Report {
    fixture_dir: String,
    fixture_dim: usize,
    fixture_corpus_n: usize,
    fixture_queries_n: usize,
    ground_truth_k: usize,
    /// One block per `bit_width`.
    pg_turbovec: Vec<PerConfig>,
    exact_brute_force: ExactBaseline,
    /// Wall-clock timestamp (RFC 3339).
    timestamp: String,
    host: String,
}

fn read_f32_matrix(path: &Path) -> std::io::Result<(usize, Vec<f32>)> {
    let mut f = std::fs::File::open(path)?;
    let mut header = [0u8; 8];
    f.read_exact(&mut header)?;
    let dim = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
    let mut buf = vec![0u8; dim * n * 4];
    f.read_exact(&mut buf)?;
    let floats: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    debug_assert_eq!(floats.len(), dim * n);
    Ok((dim, floats))
}

fn read_u32_matrix(path: &Path) -> std::io::Result<(usize, Vec<u32>)> {
    let mut f = std::fs::File::open(path)?;
    let mut header = [0u8; 8];
    f.read_exact(&mut header)?;
    let k = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
    let mut buf = vec![0u8; k * n * 4];
    f.read_exact(&mut buf)?;
    let ints: Vec<u32> = buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((k, ints))
}

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if samples.is_empty() {
        return 0.0;
    }
    let idx = ((samples.len() - 1) as f64 * p).round() as usize;
    samples[idx]
}

fn recall_at_k(brute: &[u32], indexed: &[u64], k: usize) -> f64 {
    let take = k.min(brute.len()).min(indexed.len());
    if take == 0 {
        return 0.0;
    }
    let bset: HashSet<u32> = brute.iter().take(take).copied().collect();
    let hits = indexed
        .iter()
        .take(take)
        .filter(|&&i| bset.contains(&(i as u32)))
        .count();
    hits as f64 / take as f64
}

/// Brute-force exact top-`k` over the corpus by inner product
/// (== cosine on unit-norm input).
fn exact_top_k(corpus: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<u32> {
    let n = corpus.len() / dim;
    let mut scored: Vec<(f32, u32)> = (0..n)
        .map(|i| {
            let row = &corpus[i * dim..(i + 1) * dim];
            let mut s = 0.0_f32;
            for j in 0..dim {
                s += row[j] * query[j];
            }
            (s, i as u32)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

fn main() {
    let fixture_dir = std::env::var("TURBOVEC_FIXTURE_DIR").unwrap_or_else(|_| {
        eprintln!(
            "TURBOVEC_FIXTURE_DIR not set; nothing to bench. Build a \
             fixture with benches/scripts/prepare_glove_fixture.py."
        );
        std::process::exit(2);
    });
    let fixture_dir = PathBuf::from(fixture_dir);

    let bit_widths: Vec<usize> = std::env::var("TURBOVEC_BIT_WIDTHS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![4, 2]);

    let limit_k: usize = std::env::var("TURBOVEC_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let timestamp = chrono_like_now();

    let default_results = format!(
        "benches/results/recall_vs_pgvector_{}.json",
        timestamp.replace([':', 'T'], "_").trim_end_matches('Z')
    );
    let results_path =
        std::env::var("TURBOVEC_RESULTS").unwrap_or_else(|_| default_results.clone());

    eprintln!("recall_vs_pgvector: fixture_dir = {:?}", fixture_dir);
    eprintln!("recall_vs_pgvector: bit_widths  = {:?}", bit_widths);
    eprintln!("recall_vs_pgvector: limit_k     = {}", limit_k);
    eprintln!("recall_vs_pgvector: results     = {}", results_path);

    let (dim_c, corpus) =
        read_f32_matrix(&fixture_dir.join("corpus.bin")).expect("read corpus.bin");
    let (dim_q, queries) =
        read_f32_matrix(&fixture_dir.join("queries.bin")).expect("read queries.bin");
    assert_eq!(dim_c, dim_q, "corpus/queries dim mismatch");
    let (gt_k, ground_truth) =
        read_u32_matrix(&fixture_dir.join("ground_truth.bin")).expect("read ground_truth.bin");

    let dim = dim_c;
    let n_corpus = corpus.len() / dim;
    let n_queries = queries.len() / dim;
    let gt_n = ground_truth.len() / gt_k;
    assert_eq!(gt_n, n_queries, "ground_truth row count mismatch");

    eprintln!(
        "fixture: dim={} corpus={} queries={} gt_k={}",
        dim, n_corpus, n_queries, gt_k
    );

    // turbovec requires `dim % 8 == 0`. GloVe-100 (and other public
    // datasets) violate this; we pad with zeros, which is exactly
    // identity-preserving for cosine on unit-norm input. The padded
    // dim is what we feed IdMapIndex; the *original* dim is what we
    // report (pgvector sees the original, see SQL driver). We keep
    // the un-padded copy for the exact brute-force baseline so its
    // latency reflects the fixture's true dim, not 104.
    let raw_dim = dim;
    let padded_dim = raw_dim.div_ceil(8) * 8;
    let corpus_raw = corpus.clone();
    let queries_raw = queries.clone();
    let (corpus, queries) = if padded_dim != raw_dim {
        eprintln!(
            "padding dim {} -> {} for IdMapIndex (turbovec requires dim % 8 == 0)",
            raw_dim, padded_dim
        );
        let pad_matrix = |src: &[f32], n: usize| -> Vec<f32> {
            let mut out = vec![0.0_f32; n * padded_dim];
            for i in 0..n {
                out[i * padded_dim..i * padded_dim + raw_dim]
                    .copy_from_slice(&src[i * raw_dim..(i + 1) * raw_dim]);
            }
            out
        };
        (
            pad_matrix(&corpus, n_corpus),
            pad_matrix(&queries, n_queries),
        )
    } else {
        (corpus, queries)
    };
    let dim = padded_dim;

    // --- Exact brute-force baseline (used for latency comparison and
    // as a sanity check against the on-disk ground-truth file).
    eprintln!("running exact brute-force latency baseline...");
    let mut exact_lat = Vec::with_capacity(n_queries);
    let mut sanity_mismatches = 0usize;
    for qi in 0..n_queries {
        let q = &queries_raw[qi * raw_dim..(qi + 1) * raw_dim];
        let t0 = Instant::now();
        let top = exact_top_k(&corpus_raw, raw_dim, q, limit_k);
        let dt = t0.elapsed().as_micros() as f64;
        exact_lat.push(dt);
        // Cross-check against the file (top-1 must match; ann-benchmarks
        // ties can shuffle the rest).
        let gt_row = &ground_truth[qi * gt_k..qi * gt_k + 1];
        if !top.is_empty() && top[0] != gt_row[0] {
            sanity_mismatches += 1;
        }
    }
    if sanity_mismatches > 0 {
        eprintln!(
            "warning: {sanity_mismatches}/{n_queries} top-1 mismatches between \
             our brute-force and the ground_truth.bin file (ties are normal, \
             but more than a handful is suspicious)"
        );
    }
    let mean_exact = exact_lat.iter().copied().sum::<f64>() / exact_lat.len() as f64;
    let baseline = ExactBaseline {
        n_corpus,
        n_queries,
        dim: raw_dim,
        p50_us: percentile(&mut exact_lat.clone(), 0.50),
        p95_us: percentile(&mut exact_lat.clone(), 0.95),
        p99_us: percentile(&mut exact_lat.clone(), 0.99),
        mean_us: mean_exact,
        limit_k,
        bytes_per_row: (raw_dim * 4) as f64,
    };

    // --- pg_turbovec at each bit width.
    let mut blocks = Vec::new();
    for &bw in &bit_widths {
        eprintln!("building IdMapIndex at bit_width={}", bw);
        let t_build = Instant::now();
        let mut idx = IdMapIndex::new(dim, bw).expect("IdMapIndex::new (dim, bit_width)");
        let ids: Vec<u64> = (0..n_corpus as u64).collect();
        idx.add_with_ids(&corpus, &ids).expect("add_with_ids");
        idx.prepare();
        let build_secs = t_build.elapsed().as_secs_f64();
        eprintln!("  build took {:.3}s", build_secs);

        // Recall.
        let mut sum_r1 = 0.0;
        let mut sum_r10 = 0.0;
        let mut sum_r100 = 0.0;
        let k_max = 100.min(gt_k);
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let (_scores, ids_out) = idx.search(q, k_max);
            let gt_row = &ground_truth[qi * gt_k..(qi + 1) * gt_k];
            sum_r1 += recall_at_k(gt_row, &ids_out, 1);
            sum_r10 += recall_at_k(gt_row, &ids_out, 10);
            sum_r100 += recall_at_k(gt_row, &ids_out, k_max);
        }
        let nf = n_queries as f64;

        // Latency at LIMIT=limit_k. Single-threaded, single warm cache.
        // Run twice and discard the first pass.
        for _ in 0..n_queries.min(50) {
            let q = &queries[..dim];
            let _ = idx.search(q, limit_k);
        }
        let mut lat = Vec::with_capacity(n_queries);
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let t0 = Instant::now();
            let _ = idx.search(q, limit_k);
            lat.push(t0.elapsed().as_micros() as f64);
        }
        let mean = lat.iter().copied().sum::<f64>() / lat.len() as f64;

        // Bytes/row: serialise the index to a temp file and stat it.
        let tmp = std::env::temp_dir().join(format!(
            "pg_turbovec_recall_bw{}_{}.tvim",
            bw,
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        idx.write(&tmp).expect("idx.write");
        let index_bytes = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
        let _ = std::fs::remove_file(&tmp);

        blocks.push(PerConfig {
            bit_width: bw,
            n_corpus,
            n_queries,
            dim: raw_dim,
            r_at_1: sum_r1 / nf,
            r_at_10: sum_r10 / nf,
            r_at_100: sum_r100 / nf,
            p50_us: percentile(&mut lat.clone(), 0.50),
            p95_us: percentile(&mut lat.clone(), 0.95),
            p99_us: percentile(&mut lat.clone(), 0.99),
            mean_us: mean,
            limit_k,
            index_bytes,
            bytes_per_row: index_bytes as f64 / n_corpus as f64,
            build_secs,
        });
    }

    let report = Report {
        fixture_dir: fixture_dir.display().to_string(),
        fixture_dim: raw_dim,
        fixture_corpus_n: n_corpus,
        fixture_queries_n: n_queries,
        ground_truth_k: gt_k,
        pg_turbovec: blocks,
        exact_brute_force: baseline,
        timestamp,
        host: host_string(),
    };

    let json = serde_json::to_string_pretty(&report).expect("serde");
    if let Some(parent) = Path::new(&results_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&results_path, &json).expect("write results");
    println!("{}", json);
    eprintln!("wrote {}", results_path);
}

/// RFC-3339-ish timestamp without an extra dependency. Uses
/// SystemTime so we match what the SQL driver writes.
fn chrono_like_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Convert to UTC y-m-dTh:m:sZ via a tiny date-conversion routine.
    // We don't need leap-second accuracy; this is a tag, not a clock.
    let (y, mo, d, h, mi, se) = ymdhms_from_unix(secs as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{se:02}Z")
}

#[allow(clippy::many_single_char_names)]
fn ymdhms_from_unix(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    // Adapted from "Fliegel & Van Flandern" / common-knowledge POSIX.
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400) as u32;
    let h = rem / 3600;
    let mi = (rem % 3600) / 60;
    let se = rem % 60;
    // 1970-01-01 == day 0
    let mut z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let mut y = (yoe + era * 400) as i32;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let mo = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    if mo <= 2 {
        y += 1;
    }
    z = y as i64; // touch z to keep clippy quiet
    let _ = z;
    (y, mo, d, h, mi, se)
}

fn host_string() -> String {
    // Best-effort; we don't pull in the `hostname` crate.
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".to_string())
}
