# PostgreSQL version support matrix

`pg_turbovec` supports PostgreSQL **13, 14, 15, 16, 17, 18** as of v1.0.1.

| Version | Tested patch | Status | Tests | Notes |
|---|---|---|---|---|
| 13.23 | ✅ | Supported | 92/92 | `aminsert` 7-arg shape; no `amsummarizing` / `amadjustmembers` fields. |
| 14.22 | ✅ | Supported | 92/92 | `aminsert` gained `indexUnchanged`; no `amsummarizing` field. |
| 15.17 | ✅ | Supported | 92/92 | Same shape as 14. |
| 16.13 | ✅ | Supported | 92/92 | Reference platform during development. |
| 17.9  | ✅ | Supported | 92/92 | Benchmark platform (`arnold`). |
| 18.3  | ✅ | Supported | 92/92 | `relopt_parse_elt` gained `isset_offset`. |

## How tests are run

```bash
cargo pgrx test pg<N> --no-default-features \
    --features "pg<N> experimental_index_am pg_test"
```

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
