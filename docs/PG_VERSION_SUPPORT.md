# PostgreSQL version support matrix

`pg_turbovec` supports PostgreSQL **13, 14, 15, 16, 17, 18** as of v1.0.1.

| Version | Tested patch | Status | Tests | Notes |
|---|---|---|---|---|
| 13.23 | ✅ | Supported | 261/262 (1 ignored) | `aminsert` 7-arg shape; no `amsummarizing` / `amadjustmembers` fields. |
| 14.23 | ✅ | Supported | 261/262 (1 ignored) | `aminsert` gained `indexUnchanged`; no `amsummarizing` field. |
| 15.18 | ✅ | Supported | 261/262 (1 ignored) | Same shape as 14. |
| 16.14 | ✅ | Supported | 261/262 (1 ignored) | Reference platform during development. |
| 17.10 | ✅ | Supported | 261/262 (1 ignored) | Benchmark platform (`arnold`). |
| 18.4  | ✅ | Supported | 261/262 (1 ignored) | `relopt_parse_elt` gained `isset_offset`. |

> Test counts and patch versions above are the exact numbers CI
> installs and reports as of the most recent green run
> (`gh run view` on `.github/workflows/test.yml`'s `test` job, one
> leg per `pg<N>` matrix entry). `cargo pgrx init --pgN download`
> always fetches the latest point release for major `N` at run
> time, so these patch versions drift upward on their own — re-run
> `bash scripts/drift-check.sh` and check the latest CI log rather
> than trusting this table blindly. The one ignored test
> (`src/index/ivf.rs`, `ivf_batch_speedup`) is a perf-only timing
> comparison, not a correctness gate; it's `#[ignore]`d deliberately
> on every PG version, not a skip specific to any one of them.

> The out-of-core IVF build (v1.12.0+) uses PG's `BufFile` temp-file
> API, whose signatures differ across majors (`BufFileReadExact` is
> PG16+; `BufFileWrite`'s pointer type changed). v1.15.1 added
> `scripts/compile-matrix.sh` (a `cargo check` across every `pgNN`
> feature, wired into the pre-push hook) so version-specific C-API
> breaks are caught before tagging — a v1.12.0–v1.15.0 regression
> that broke the pg13/14/15/18 build legs slipped through because
> local dev was pg16-only.

## How tests are run

As of v1.3.0 (Phase Q), the `experimental_index_am` and
`relfile_storage` Cargo features are gone; the only build knob
is `pg<N>`:

```bash
cargo pgrx test pg<N> --no-default-features \
    --features "pg<N> pg_test"
```

(Or simply `cargo pgrx test pg<N>` if you're happy with the
default feature set, which already enables `pg16`.)

For each supported version, the test suite drives every type
(`vector`, `halfvec`, `sparsevec`, `bitvec`), every distance
operator (`<->`, `<#>`, `<=>`, `<+>`, `<~>`, `<%>`), the index
access method opclasses, and aminsert / ambulkdelete via VACUUM.

## Why each gate exists

### `(*routine).amsummarizing` — `cfg = pg16+`

The `IndexAmRoutine` struct gained an `amsummarizing: bool` field
in PG 16 to drive BRIN's summarising-index codepath. Earlier
versions don't know about it.

```rust
#[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18"))]
{
    (*routine).amsummarizing = false;
}
```

### `(*routine).amadjustmembers` — `cfg = pg14+`

`amadjustmembers` is the op-family-adjust-members callback added
in PG 14 (`be08e10b41fd`). pg13 doesn't have the field.

```rust
#[cfg(not(feature = "pg13"))]
{
    (*routine).amadjustmembers = None;
}
```

### `aminsert` callback — split for `pg13`

PG 14 added an `indexUnchanged: bool` parameter to `aminsert` for
HOT-chain elision (`9dc718bdf2b1`). pg13's signature is one
argument shorter. We expose two thin C-ABI wrappers selecting on
the feature flag and a shared `aminsert_impl` Rust function:

```rust
#[cfg(not(feature = "pg13"))]
#[pgrx::pg_guard]
pub(crate) unsafe extern "C-unwind" fn aminsert(
    index_relation: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap_relation: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,                  // <-- added in PG 14
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    aminsert_impl(index_relation, values, isnull, heap_tid)
}
```

### `relopt_parse_elt::isset_offset` — `cfg = pg18`

PG 18 added `isset_offset: i32` to `relopt_parse_elt` so callers
can distinguish "explicitly set to default" from "never set". We
don't track that distinction; `-1` ("unused") works on every
field.

```rust
pg_sys::relopt_parse_elt {
    optname: c"bit_width".as_ptr(),
    opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
    offset: std::mem::offset_of!(TurbovecRelopts, bit_width) as i32,
    #[cfg(feature = "pg18")]
    isset_offset: -1,
},
```

## Gotcha: `pgrx::pg_guard` reserves `<fn>_inner`

The `#[pgrx::pg_guard]` macro expands to a wrapper plus a private
helper named `<original_name>_inner`. If you split a callback
into a public C-ABI wrapper and an inner Rust impl, **don't name
the inner helper `<fn>_inner`** — it collides with the macro's
generated symbol. We use `<fn>_impl` instead. Surface that any
new callback you split follows the same convention.

## Adding a future PG version

1. Add `pgN = ["pgrx/pgN", "pgrx-tests/pgN"]` to `[features]` in
   `Cargo.toml`.
2. Run `cargo pgrx test pgN`.
3. Compile errors will point at any new fields in
   `IndexAmRoutine`, callback shape changes, or relopt struct
   drift. Add `#[cfg(feature = "pgN")]` (or `#[cfg(not(feature =
   "pgM"))]` for "all versions <= M") gates to `src/index/`.
4. Re-run the full matrix to make sure no previous version was
   broken.
5. Update this file's table and the `CHANGELOG.md` entry.

## PostgreSQL 19

Not supported yet — PG19 is still in upstream beta (`REL_19_BETA1`)
and the pinned `pgrx = "=0.17.0"` dependency has no `pg19` feature.
 for the blocker detail, the C-API delta
found so far, and the recommended timeline (wait for PG19 RC1+, then
treat the pgrx upgrade + port as its own dedicated piece of work).
