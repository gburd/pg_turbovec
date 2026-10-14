# v2.12.1 qualification (2026-10-14)

What this release changes for users: the Nix build works again (flake hash key
plus turbovec fork carry #5 for LLVM 21), turbovec 1.1.1 → 1.1.2 (an upstream
recall fix in the staged "planes" search), pgrx 0.19.1 → 0.19.3, and a
`cargo update`. No wire, SQL or GUC change. This directory is the evidence.

Hosts (EC2 `kiwi`, us-east-2, Fedora 44, rustup 1.98.0 / LLVM 22.1.8 unless
noted; `raw/host_*.txt`):

- `c7i.4xlarge`: Sapphire Rapids, AVX-512 VBMI + VNNI. This is the host
  where carry #5's `vpdpbusd` runs.
- `c8g.4xlarge`: Graviton4 (Neoverse-V2, dotprod), a planes host for
  turbovec 1.1.2's change.
- Local x86 laptop (Lunar Lake, AVX2 only): nix rustc 1.97.0 / LLVM 21.1.8,
  plus the flake on nixpkgs `8edc0c7` (rustc 1.98.1 / LLVM 21.1.8).

## 1. Builds

| Build | Before | After |
|---|---|---|
| `nix build .#pg_turbovec_16` / `_18` (nixpkgs rustc 1.98.1, LLVM 21.1.8) | fails at vendoring (`A hash was specified for turbovec-0.9.0, but there is no corresponding git dependency`); with the key fixed, fails compiling turbovec (`intrinsic signature mismatch for llvm.x86.avx512.vpdpbusd.512`) | builds; the `.so` contains 305 `vpdpbusd` and 152 `vpermb` (`raw/nixbuild_summary.txt`) |
| turbovec with nix rustc 1.97.0 / LLVM 21.1.8 | same intrinsic error | builds; `cargo test --release --lib --tests` 524 passed / 0 failed (`raw/turbovec_test_x86_local_llvm21.txt`, an AVX2 host, so the asm path compiles here but the VNNI kernels don't run) |

## 2. Tests

| What | Host | Result |
|---|---|---|
| turbovec `cargo test --release --all-targets`, `TURBOVEC_DPBUSD_ASM=1` (asm path) | c7i (VNNI) | 525 passed / 0 failed, including `dpbusd_matches_the_scalar_definition` (`raw/turbovec_test_x86_forced_asm.txt`) |
| same, intrinsic path | c7i (VNNI) | 525 / 0, same test passes (`raw/turbovec_test_x86_intrinsic.txt`) |
| same | c8g (aarch64) | 522 / 0 (`raw/turbovec_test_aarch64.txt`; the x86-only test isn't compiled) |
| same, rustup 1.98.0 / LLVM 22 | local | 525 / 0 |
| `cargo pgrx test pg16` | c7i | **472 passed / 0 failed / 9 ignored** (`raw/pgtest_pg16_x86_c7i.txt`) |
| `cargo pgrx test pg16` | c8g | **472 / 0 / 9** (`raw/pgtest_pg16_aarch64_c8g.txt`), run with `RUSTFLAGS="-C target-feature=+fp16"`. That flag is pre-existing and was documented in the 2.11.0 build note: `gemm-common` 0.18.2's fp16 inline asm doesn't assemble at opt-level 0 on aarch64 (upstream sarah-quinones/gemm#31). Release builds (what ships) don't need it. |
| `cargo pgrx test pg16` | local | 472 / 0 / 9 (`raw/pgtest_pg16_local.txt`) |

The dpbusd test, built normally on LLVM 22, exercises the intrinsic. Under
`TURBOVEC_DPBUSD_ASM=1` (or on LLVM < 22) it exercises the asm. Either way it
compares 50,000 random 64-byte lane sets, plus u8-max × i8-min and
i32-wraparound cases, against the scalar definition of `vpdpbusd`. It skips
on a host without AVX-512 VNNI and says so.

## 3. turbovec 1.1.2 on a planes host: Graviton4 A/B

`v2.12.0` (old, turbovec 1.1.1, `.so` md5 `bb30cfbd`) against this release
(new, turbovec 1.1.2, md5 `eadece95`):
- **Setup:** one flat 4-bit index, 534 MB, 1M × 1024-d Cohere
  wikipedia-2023-11 (multilingual-v3), 200 held-out queries, exact numpy
  ground truth. Non-cassert PG 16.15 `-O2`, `shared_buffers = 12GB`, heap
  and index prewarmed after every restart, glibc malloc tunables pinned.
  Scripts: `raw/tvq_*`, adapted from `tv111_arm_20261005`.
- **Rounds:** arms alternated old, new for 4 rounds. Round 1 failed for both
  arms (the first `pg_ctl stop` had not finished; `raw/ab_run.txt`), so
  rounds 2–4 are reported.

p50 ms, median of rounds 2–4 (range); whole query including the exact
recheck:

| search_k | v2.12.0 | 2.12.1 | new/old | R@10 old / new |
|---|---|---|---|---|
| 32 | 1.73 (1.73–1.78) | 1.74 (1.73–1.75) | 1.005 | 1.000 / 1.000 |
| 100 | 2.67 (2.63–2.70) | 2.69 (2.66–2.69) | 1.008 | 1.000 / 1.000 |
| 256 | 4.49 (4.41–4.49) | 4.50 (4.48–4.58) | 1.003 | 1.000 / 1.000 |
| 1024 | 13.84 (13.73–13.87) | 13.89 (13.76–14.26) | 1.004 | 1.000 / 1.000 |

The two arms' round ranges overlap in every row: **no latency change is
resolved at this precision** (≤ 0.8% either way). Returned ids: for all 200
queries, at both `search_k = 32` and `1024`, the ordered top-10 is
**identical** between the arms (`raw/ids_{old,new}.json`).

What this does and doesn't show. It shows that 1.1.2 does not regress latency
or results on an OpenAI-class corpus (Cohere's has no strong shared
direction), which is what upstream predicts: "OpenAI-class corpora return the
same ids as before". It does **not** reproduce upstream's MedCPT recall gain;
we have no bunched corpus in this run. That gain is upstream's measurement,
not ours.

## Not measured

- x86 AVX-512 VNNI latency for the planes change. The c7i host ran tests only.
- 2-bit indexes, where upstream reports 2–12% slower at k = 10 from wider
  stage widths.
- IVF, ColBERT, 1-bit.
- Any PostgreSQL other than 16 in this run. CI covers 13–19 for correctness.
