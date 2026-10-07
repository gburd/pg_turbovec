# Raw `float4` varlena for `turbovec.vector` (replace serde-CBOR): design

**Status:** design only. No code in this change. Step 6 of the recheck-cost
plan (`benches/results/perf_abc_20261006/TEAM_BRIEF.md`). Branch
`design/raw-vector-varlena`, written 2026-10-06 against v2.11.0 + the
recheck-analysis commits (`3551a90`).

Release numbers updated 2026-10-07: 2.12.0 shipped the recheck-speed work;
the read-both phase is now 2.13.0.

**Decision:** a second datum format for `turbovec.vector`, laid out as
`[varlena hdr][0xFE][0x01][u16 dim LE][f32 LE × dim]`. It is chosen by its
first payload byte, so it can never be confused with the CBOR every released
binary has written (that always starts with `0xA1`). The CBOR reader stays
forever and mixed tables are the permanent normal state. Rollout: minor
2.13.0 reads both formats and keeps writing CBOR by default; operators opt in
to `raw` per database or cluster. **No minor ever flips the write default.**
The default becomes `raw` only at the next major (3.0.0). No index changes,
no REINDEX, no forced rewrite.

Evidence labels used below: **measured** = produced in this session or in
a cited FINDINGS file; **computed** = arithmetic from PostgreSQL constants;
**estimate** = extrapolation, not measured; **unverified** = believed true,
not checked. Probe details for every "measured here" claim are in the
Appendix.

---

## 0. Decisions on one screen

| question | decision |
|---|---|
| Layout | `vl_len_` + `magic u8 = 0xFE` + `version u8 = 0x01` + `dim u16 LE` + `f32 LE × dim`. VARSIZE = `8 + 4·dim`, the same size as pgvector's. |
| Discriminator | Byte 0 of the payload: `0xA1` → legacy CBOR, `0xFE` → raw v1, anything else → `ERROR`. Proof in §2.3. |
| Why not the pgvector layout | An old binary reading a pgvector-layout payload can **silently mis-decode** it (measured: a crafted all-finite 447-d payload decodes to `[1]` in the v2.11.0 binary; 2.12.0's decode path is unchanged). With `0xFE` the old binary always fails at the first byte. §2.4. |
| Endianness | Fixed little-endian. It is native on every host we build or test (x86_64, aarch64, riscv64). Big-endian builds are refused with `compile_error!`; the index relfile is already native-endian and doesn't support BE either. §2.5. |
| Alignment | `typalign` stays `'i'`; it cannot be changed (`typecmds.c:4282`). `f32` data sits at varlena offset 8. The reader still checks pointer alignment and copies if the pointer is unaligned (1-byte-header datums, dim ≤ 30). §3.1. |
| pgrx surface | Keep `#[derive(PostgresType)]` + `#[inoutfuncs]`, add `#[bikeshed_postgres_type_manually_impl_from_into_datum]`, and hand-write `FromDatum`/`IntoDatum`/`UnboxDatum`/`ArgAbi`/`BoxRet`. Generated SQL and C symbol names (`vector_in_wrapper`, `vector_out_wrapper`) are unchanged, so no catalog change. §3.2. |
| Zero-copy | Fix B already gives the distance functions a raw-datum argument type, `VectorArg` (shipped in 2.12.0, `src/distance.rs`). This design makes its `Slot::get` bypass the cache and borrow raw payloads directly when `payload[0] == 0xFE`. No separate `VectorRef` type. §3.3. |
| Write path | `IntoDatum` is the only write point. GUC `turbovec.vector_write_format = cbor | raw` (`Userset`), set once per database or cluster, never per session or pool (§4.2). Existing rows are never rewritten; `UPDATE t SET v = v` and `VACUUM FULL` do **not** re-encode (measured). A same-value rewrite across formats is not HOT (§4.1). §4. |
| Release type | **Minor, then major.** 2.13.0 (minor): reads both, writes CBOR by default, operators opt in to `raw`, adds `turbovec.vector_format(vector)`. The default flips to `raw` only at **3.0.0** (major); no minor flips it. The write format is decided by the **loaded binary**, not by `ALTER EXTENSION` (§5.1). §5, §8. |
| Index | Untouched. The AM never persists the varlena; index wire format stays v8. §6. |
| Scope | `vector` only. `halfvec`, `sparsevec`, `bitvec` and the `*Accum` states are deferred (§8.4). |
| Expected gain | About 3.7–4 µs per rechecked candidate after Fix B (estimate), for new-format rows only. The 20% byte saving does **not** cut heap pages or TOAST chunks at 1024-d (measured). §9. |

---

## 1. Goal, non-goals, motivation

### 1.1 Goal

Remove the serde-CBOR decode from every read of a `turbovec.vector` datum
for rows written by new binaries, without rewriting or invalidating any
existing row, index, catalog constant, dump, or replica, and without any
path that can silently return a wrong vector.

### 1.2 Non-goals

- **Byte compatibility with pgvector.** `src/vec.rs:3-8` and
  `docs/ARCHITECTURE.md` §3.2 planned it. This design gives it up on
  purpose (§2.4): the payload differs from pgvector's only in the 4-byte
  inner header, so a later `pgvector.vector ↔ turbovec.vector` cast is a
  palloc plus memcpy, not `WITHOUT FUNCTION`.
- Rewriting legacy rows. Mixed tables are permanent.
- Changing the type's default `STORAGE` (still `extended`). That is Fix C's
  territory and an SQL change of its own (`TEAM_BRIEF.md`).
- A typmod (`vector(N)`), a binary `send`/`recv`, or equality/btree
  operators. All are orthogonal; §5.3 covers `send`/`recv`.
- `halfvec`, `sparsevec`, `bitvec` (§8.4).

### 1.3 Measured motivation

From `benches/results/recheck_20261006/FINDINGS.md` (floki, PG 16.15
non-assert, 200k × 1024-d Cohere, 4-bit flat, `search_k = 1024`):

- CBOR decode of both distance-function arguments: **8.13 µs/candidate**
  (default storage), 7.80 µs (`PLAIN`). That is the largest single
  component of the ~18 µs backend cost per candidate.
- `cbor_bench.rs`: decoding one 1024-d vector costs **3.73 µs**, a raw LE
  copy 0.08 µs, a zero-copy view ~0.
- The CBOR payload is 5,125 B against 4,096 B of floats (the fixture's
  value mix). With random non-f16-exact values it is 5,129 B (measured here,
  `pg_column_size`).

Re-run here on the same laptop (floki, Intel Core Ultra 7 258V, rustc 1.97
`--release`, 200k iterations, 1024-d), **measured**: serde_cbor decode
3.02–3.09 µs, `memcpy` of 4,096 B 0.10–0.12 µs. Consistent with FINDINGS.

The two decodes per candidate existed because on v2.11.0 every distance
function took `(a: Vector, b: Vector)` by value (`src/distance.rs:41, 57,
72, 95, 117, 132`), so the constant query was decoded again for each
candidate. Fix B, shipped in 2.12.0, changed those six functions to take
`VectorArg`, the raw `Datum`, which decodes through a per-`FmgrInfo` cache
(`Slot::get`); that removed the query-side decode. This design removes the
candidate-side decode. Line numbers in this doc are v2.11.0's: 2.12.0
changed `src/distance.rs` and `src/kernels.rs` (and tests in `src/lib.rs`),
so `src/distance.rs` line numbers have moved; every other cited file is
unchanged.

The index build pays the same decode once per heap row
(`src/index/build.rs:2004`). The v2.10.1 build-memory bug
(`src/index/build.rs:312-323`) was the CBOR-decoded buffers piling up.

---

## 2. On-disk layout

### 2.1 Bytes

```
offset (from varlena start, 4-byte-header form)
 0  vl_len_   4 B   PostgreSQL varlena header (SET_VARSIZE). On disk it may be
                    1-byte (short) when VARSIZE <= 127, i.e. dim <= 30, or a
                    TOAST pointer / compressed; the payload is identical after
                    pg_detoast_datum_packed.
 4  magic     1 B   0xFE
 5  version   1 B   0x01   (raw-v1). Future layouts bump this byte.
 6  dim       2 B   u16 little-endian, 1..=MAX_DIM (16,000; src/vec.rs:23)
 8  x[dim]    4·dim IEEE-754 binary32, little-endian, every value finite
VARSIZE = 8 + 4·dim     (payload after the header = 4 + 4·dim)
```

Short-header bound, **computed** from `VARATT_CAN_MAKE_SHORT`
(`varatt.h:258-260`): `(8 + 4d) − 4 + 1 ≤ 127` ⇒ `d ≤ 30`. Such values are
stored with a 1-byte header and **no alignment** (`heaptuple.c:347-352`).

The reader rejects:

- a payload length other than `4 + 4·dim`
- `dim = 0` or `dim > MAX_DIM`
- an unknown version: `ERROR … written by a newer pg_turbovec`, `HINT`
  upgrade

It does **not** re-check finiteness on read. The writer guarantees it
(`Vector::from_vec`, `src/vec.rs:43-56`), as it does for CBOR today, and
pgvector doesn't check on read either. A non-finite value can only come in
through page corruption or a superuser `WITHOUT FUNCTION` cast. It gives a
NaN distance, never memory unsafety.

`dim` is a u16 rather than pgvector's i16. It costs the same and leaves no
sign to validate. The writer checks `dim <= MAX_DIM` (16,000 < 65,535) and
errors **before** the `as u16` cast, so a truncating cast can never write a
wrong `dim`.

### 2.2 What a legacy CBOR datum looks like

**Measured** with pageinspect on the v2.11.0 binary, row
`'[1,2,3,4,5,6,7,8]'`:

```
t_data = 01000000 | 41 | a1 64 64617461 88 f93c00 f94000 f94200 …
          id int4   1B    map(1) "data"  array(8) f16 1.0  f16 2.0 …
                    hdr
```

That is `a1` map(1), `64 'd' 'a' 't' 'a'` text(4), `8x/98 xx/99 xx xx`
array(dim), then `f9 hhhh` (f16, when exact) or `fa ffffffff` (f32) per
element.

### 2.3 Proof the discriminator cannot collide

Let *O* be every payload any released binary can have stored as a
`turbovec.vector` datum, on the heap, in TOAST, in an array element, or as a
catalog `Const` (views and defaults store datum bytes too; **measured**
`pg_attrdef.adbin` `constvalue 17 [68 0 0 0 -95 100 100 97 116 97 …]` on
x86). `outDatum` prints each byte as `(int)` of a plain `char`
(`outfuncs.c:356-363`, 16.14), so `0xA1` prints `-95` (signed `char`, x86)
or `161` (unsigned `char`, aarch64).

**Claim 1.** For every `p` in *O*, `p[0] = 0xA1`.

1. The only producer is pgrx's generated `IntoDatum` →
   `pgrx::datum::cbor_encode` (`pgrx-0.19.1/src/datum/varlena.rs:377-393`;
   identical in 0.17.0). It writes 4 zero bytes, then
   `serde_cbor::to_writer(&Vector)`, then `SET_VARSIZE`. There is no other
   route in:
   - **No receive function.** `#[pg_binary_protocol]` was never set
     (`git log -S`, all history), so `pg_type.typreceive = '-'` (measured).
   - **No binary-coercible cast.** All 10 casts touching `vector` have
     `castmethod = 'f'` (measured), and `WITHOUT FUNCTION` / `WITH INOUT`
     never appear in history.
2. serde_cbor is locked at **0.11.2** in `Cargo.lock` at every one of the 89
   v1.x/v2.x tags (checked tag by tag). The lockfile first appeared in commit
   `94dbd87` (v0.6.0, untagged), and no tag predates it. The encoder uses the
   default non-packed `Serializer` (`ser.rs:54, 69`).
3. `Vector` has been one field, `data: Vec<f32>`, with no serde attributes,
   at every tag (90 refs checked, including `Tvector` at `v1.0.0-rc.1`).
   The derived `Serialize` calls `serialize_struct(_, 1)` →
   `write_u64(5, 1)` → one byte `(5 << 5) | 1 = 0xA1` (`ser.rs:141-149,
   495-498`). The key is then emitted by name (`ser.rs:601-605`) as
   `64 64 61 74 61`.
4. Empirically, for dim 1..=16,000 and four value patterns, every encoding
   begins `A1 64 64 61 74 61` (measured here).

**Claim 2.** Every new-format payload has `p[0] = 0xFE ≠ 0xA1`, by
construction.

So dispatching on `p[0]` is a total, unambiguous function on *O* ∪ *New*.
Bytes outside both sets (e.g. injected by a superuser `CREATE CAST (bytea AS
turbovec.vector) WITHOUT FUNCTION`) get an `ERROR`, never a guess.

**Claim 3 (downgrade fails closed).** An old binary given a raw payload
always errors and never returns a value. The old reader is
`cbor_decode` → `serde_cbor::from_slice::<Vector>` (`varlena.rs:396-405`).
`deserialize_struct` forwards to `deserialize_any` (`de.rs:881-885`), which
is `parse_value` (`de.rs:784-789`). That reads byte 0 first (`de.rs:582-585`)
and maps `0xfc..=0xfe` to `Err(UnassignedCode)` before consuming anything
else (`de.rs:769`). So no type-directed path ever reads byte 1.

- **Measured:** 2,000,000 random `0xFE`-prefixed payloads (1–300 B) were all
  rejected with exactly `unassigned type at offset 1`.
- **Measured** in-database, v2.11.0 `.so` (verified on 2.11.0; 2.12.0
  decode path unchanged), raw payload injected through a
  test-only `WITHOUT FUNCTION` cast: `ERROR: failed to decode CBOR:
  ErrorImpl { code: UnassignedCode, offset: 1 }` from both `::text` and
  `vector_dims()`.

Independently, initial byte `0xFE` is major type 7 with additional-info 30.
RFC 8949 §3 reserves additional-info 28–30 and makes an item using them not
well-formed (31 is the break/indefinite marker), so no conforming CBOR
decoder accepts it either (verified against RFC 8949 §3, additional
information 28–30). The proof
above doesn't depend on it; it depends only on the one decoder that has
ever read these datums.

### 2.4 The rejected alternative: pgvector's exact layout

`[i16 dim][i16 unused = 0][f32 × dim]` (`pgvector src/vector.h:18-24`,
v0.8.7 `f37c13f`). The **new** reader could tell it apart from CBOR: bytes
`A1 64` read as an i16 dim are 25,761 (LE) or negative (BE), both invalid.
The problem is the **old** reader in the downgrade direction:

- Natural data: 0 of 128,000 payloads were accepted (dim 1..16,000 × {zero,
  one, random, sin} × {LE, BE}; measured). Most fail on byte 0 or on the
  zero `unused` field.
- **Crafted data: accepted.** Take dim 447 (LE `BF 01`), so byte 0 opens an
  indefinite map. Field key 1 is ignored, key 0 = `data` is read from the
  first float's bytes `81 F9 3C 00` = `[1.0]`, a byte-string swallows the
  rest, and the last float's top byte `FF` closes the map. Every float is
  finite. serde_cbor returns `Ok([1.0])`, and the v2.11.0 binary in-database
  prints `[1]` with `vector_dims = 1` (measured on 2.11.0; 2.12.0 decode
  path unchanged).

Pathological, but "a wrong vector, no error" is the corruption class the
HARD MANDATE forbids. A one-byte magic removes it at no size cost: total
size is identical to pgvector's.

### 2.5 Endianness

PostgreSQL data files are host-endian and not portable across byte order
(`xlog.c:4026` "mismatched byte ordering … initdb"). So host order would be
legal, and it is what pgvector does. We specify LE anyway, so committed
golden fixtures (§7) are arch-independent bytes, and every supported host
(x86_64, aarch64, riscv64) is LE, so zero-copy costs nothing.

The extension does not support big-endian today, and this design doesn't
pretend to. `src/index/page.rs` uses `to_le_bytes` (91 sites), but the
relfile writes `u64` id and `f32` centroid/mean slices as **native-endian**
bytes via `from_raw_parts` (`src/index/relfile.rs:906-909, 921-922,
935-937, 1769-1770, 1781-1782`). So rather than an untested BE code path,
the type refuses to build there:

```rust
#[cfg(target_endian = "big")]
compile_error!("pg_turbovec supports little-endian targets only (vector datum and index relfile are LE / native-LE)");
```

The copying path (`f32::from_le_bytes`) stays: on LE it is what unaligned
1-byte-header datums (dim ≤ 30) take (§3.1).

### 2.6 `typalign` `'i'` vs `'d'`

The type was created without `ALIGNMENT`, so `pg_type.typalign = 'i'`
(measured). `ALTER TYPE … SET (ALIGNMENT …)` is rejected as "cannot be
changed" (`typecmds.c:4278-4292`), so `'i'` is permanent.

It is also sufficient. A 4-byte-header datum starts 4-aligned in a tuple
(`att_align_nominal`, `heaptuple.c:362-366`) and in an array
(`ArrayCastAndSet`, `arrayfuncs.c:4857-4882`). palloc'd copies (detoast,
`datumCopy`) are MAXALIGNed. So `x[]` at offset 8 is 4-aligned, which is
all `f32` needs. `'d'` would add up to 4 B of padding per tuple and buy
nothing for f32; the SIMD kernels use unaligned loads.

---

## 3. Read path and pgrx type surface

### 3.1 Decoding primitive

One function in `src/vec.rs`, roughly
`unsafe fn decode<'a>(datum) -> Cow<'a, [f32]>`:

1. `p = pg_detoast_datum_packed(datum)`. This is what pgrx's `cbor_decode`
   uses today (`varlena.rs:400`). It expands external and compressed values
   into a fresh MAXALIGNed palloc and leaves 1-byte-header datums in place
   (`fmgr.c:1835-1841`). `pg_detoast_datum` (`fmgr.c:1803-1809`, pgvector's
   `PG_DETOAST_DATUM`, `vector.h:14`) would also expand short headers, at
   the cost of a palloc for every dim ≤ 30 value. Either works. `_packed`
   plus the alignment check below is cheaper and never wrong.
2. `payload = VARDATA_ANY(p)`, `len = VARSIZE_ANY_EXHDR(p)`.
3. Dispatch on `payload[0]`:
   - **`0xFE`**: validate (§2.1). If `payload + 4` is 4-aligned →
     `Cow::Borrowed(from_raw_parts(...))`. Otherwise → `Cow::Owned` via
     `chunks_exact(4).map(f32::from_le_bytes)`. (BE builds are refused,
     §2.5.) The alignment check is required for Rust soundness, not only
     for strict-alignment CPUs: an unaligned `&[f32]` is UB even on x86.
   - **`0xA1`**: `serde_cbor::from_slice::<Vector>` exactly as today →
     `Cow::Owned`. Same decoder and same derived impl, so the result is
     bit-identical to the old binary's by construction. **Pin
     `serde_cbor = "=0.11.2"` directly in our `Cargo.toml` and keep `half
     1.8.3` locked.** Today serde_cbor only arrives through pgrx
     (`pgrx-0.19.1/Cargo.toml:96-97`), and a pgrx bump must never be able to
     move the legacy decoder. Its f16 → f32 widening (`0xF9` elements,
     `de.rs:567-569`) comes from `half` **1.8.3**, pulled in by serde_cbor's
     `half = "1.2.0"`; that is a different crate version from the `half`
     2.7.1 our `halfvec` uses, and it is part of the legacy decoder too. The
     fixture test (§7.1) fails if either moves.
   - **anything else**: `ERROR` with SQLSTATE `XX001` (`data_corrupted`):
     `unrecognized turbovec.vector datum format (first byte 0x..)`. An
     unknown raw **version** byte is `0A000` (`feature_not_supported`):
     `… written by a newer pg_turbovec`, `HINT` upgrade.

   Steps 2–3 live in a **pure** function, `fn decode_payload(&[u8]) ->
   Result<Cow<'_, [f32]>, DecodeError>`, with no `pg_sys` in it, so Miri
   and hegel exercise exactly the shipping code. The `pg_sys` wrapper does
   only the detoast and maps `DecodeError` to the SQLSTATEs above.

Borrow lifetime: the slice points into either the tuple (valid while the
slot/fcinfo holds it) or the detoast palloc (current memory context; the
per-tuple context for recheck). Nobody stores it past the call: every AM
consumer already copies into a `Vec<f32>` (`src/index/insert.rs:131-136`,
`src/index/scan.rs:441-445`, `src/index/build.rs` staging).

### 3.2 Keeping `Vector` and its SQL surface

What `#[derive(PostgresType)] #[inoutfuncs]` generates
(`pgrx-macros-0.19.1/src/lib.rs`):

| item | where | change |
|---|---|---|
| `IntoDatum`, `BoxRet`, `FromDatum` (incl. `from_datum_in_memory_context` → `pg_detoast_datum_copy`), `UnboxDatum`, `ArgAbi` | `lib.rs:908-979`; **skipped entirely** when the struct carries `#[bikeshed_postgres_type_manually_impl_from_into_datum]` (`lib.rs:908, 1248`; present in both 0.17.0 and 0.19.1) | hand-written. Copy the generated bodies and swap `cbor_encode`/`cbor_decode` for §3.1 / §4. |
| `vector_in` / `vector_out` `#[pg_extern]` wrappers calling `InOutFuncs` | `lib.rs:1001-1023`; names from `lib.rs:848-849` | unchanged; they reach the datum only through the impls above |
| `vector_recv` / `vector_send` | `lib.rs:1049-1094`, only under `#[pg_binary_protocol]` | not generated today, not added here |
| `CREATE TYPE Vector (INTERNALLENGTH = variable, INPUT = vector_in, OUTPUT = vector_out, STORAGE = extended)` | `pgrx-sql-entity-graph-0.19.1/src/postgres_type/entity.rs:339-344` | unchanged (driven by the derive, which stays) |

The installed v2.11.0 script (generated `pg_turbovec--2.11.0.sql`, lines
802-829) binds `vector_in`/`vector_out` to C symbols `vector_in_wrapper`
and `vector_out_wrapper` in `$libdir/pg_turbovec`. `pg_proc.prosrc` matches
(measured).

**How `ALTER EXTENSION UPDATE` "swaps" I/O behaviour:** it doesn't need to.
The catalog points at symbol names, and the new `.so` exports the same
names with new behaviour. A restart (or a new backend) loads it. That is
fortunate, because `ALTER TYPE … SET` can change `RECEIVE`, `SEND`,
`TYPMOD_IN/OUT`, `ANALYZE`, `SUBSCRIPT` and `STORAGE`
(`typecmds.c:4156-4272`), but **not** `INPUT`, `OUTPUT`, `INTERNALLENGTH`,
`ALIGNMENT` and others ("cannot be changed", `typecmds.c:4278-4292`), and
`CREATE TYPE` can't be redefined in place. Replacing an I/O function also
requires superuser (`typecmds.c:4213` …). None of this applies here: no
function signature, OID, or `pg_type` row changes. Every other SQL function
over `vector` (operators `src/distance.rs:220-280`, opclasses
`src/index/mod.rs:189-260`, casts `src/cast.rs:136-167`) likewise keeps its
symbol and only changes behaviour through `FromDatum`/`IntoDatum`.

The `Serialize`/`Deserialize` derives stay; the legacy decoder in §3.1
needs `Deserialize`. Field `data` keeps its name. The direct `.data` /
`Vector { .. }` uses (`src/cast.rs:62, 114`, `src/vec.rs:62, 69, 103`,
`src/hybrid.rs:169`) compile unchanged.

Phase 1 gate: the order-independent `cargo pgrx schema` diff against
2.12.0 contains only `vector_format`. That catches any SQL the hand-written
impls or `VectorArg` change by accident, not only `VectorArg`'s.

Fallback if pgrx renames the `bikeshed_` attribute: compilation fails
(safe). We would then drop the derive and hand-write the `CREATE TYPE` with
`extension_sql!`, which must reproduce the same SQL. That costs more; see
§10.

### 3.3 Zero-copy argument: Fix B's `VectorArg`

`Vector` owns a `Vec<f32>`, so the raw path through `FromDatum` still costs
one `memcpy`: 0.10 µs at 1024-d (measured), about 40× less than CBOR.

Fix B (shipped in 2.12.0, `src/distance.rs`) changed the six hot
distance functions to take `VectorArg(pg_sys::Datum)`, with
`SqlTranslatable` consts delegated to `Vector` so the generated SQL is
unchanged (pinned there by `distance_cache_null_and_markings_unchanged`).
Its `Slot::get` detoasts with `pg_detoast_datum_packed`, compares the
payload bytes with the cached ones, and on a miss calls
`Vector::from_datum` and copies the payload into the slot. So there is no
separate `VectorRef` type in this design; `VectorArg` is it:

- In `Slot::get`, when `payload[0] == 0xFE`, **bypass the cache** and
  return the borrowed `Cow` from `decode_payload` directly. No memcmp, no
  payload copy, no `Vector` allocation.
- When `payload[0] == 0xA1`, keep Fix B's cache unchanged. It is what saves
  the CBOR query-side decode.
- `with_operands`' closure takes `&[f32]` (or a `Cow`) instead of
  `&Vector`, which the kernels already consume.

The other read-only hot functions (`src/distance.rs:144, 158` on v2.11.0)
can move to `VectorArg` the same way. The index paths
(`src/index/build.rs:2004`, `src/index/insert.rs:73`,
`src/index/scan.rs:435`) call `decode_payload` directly, since they copy
into a `Vec<f32>` anyway. Arrays (`Vec<Vector>`: `src/index/build.rs:1868`,
`src/colbert.rs:69`, `src/hybrid.rs:105, 123`) stay owned; they are rare.

Gate: an A/B at `search_k = 1024`, and an order-independent `cargo pgrx
schema` diff that is empty (same gate Fix B uses). It changes no format and
no SQL, so it is patch-eligible.

---

## 4. Write path

- **Single write point.** Every new `vector` datum goes through
  `Vector::into_datum`: `vector_in` (text input, COPY, dump restore), every
  `Vector`-returning function (`src/cast.rs`, `src/distance.rs:164-213`,
  `src/extras.rs:33, 88, 144, 169`, `src/normalize.rs:18, 25, 77`,
  `src/aggregate.rs:123, 135`, `src/halfvec_ops.rs:163`,
  `src/sparsevec_ops.rs:152`), and SPI parameters
  (`src/partition.rs:285-289`). It reads GUC
  `turbovec.vector_write_format` (enum `cbor | raw`; `GucContext::Userset`,
  as `scripts/drift-check.sh` §11b requires) and emits that format.
  Parallel workers inherit the leader's value.
- **What re-encodes:** only a value that passes through a type function.
  `INSERT`, `COPY FROM`, or `UPDATE … SET v = <expression producing a new
  vector>`.
- **What does not re-encode (measured):**
  - `UPDATE t SET v = v` keeps the stored bytes; the TOAST value is even
    reused (same `chunk_id` before and after, per `toast_helper.c:75-79`).
  - `VACUUM FULL` / `CLUSTER` copy tuples byte-wise (`rewriteheap.c`; zero
    type-function calls observed).
  - `ALTER TABLE … SET STORAGE` doesn't rewrite.
  - `pg_upgrade` carries files as-is.
- **Explicit migration (optional, never required):**
  `UPDATE t SET v = v::real[]::turbovec.vector WHERE
  turbovec.vector_format(v) = 'cbor'` in batches, with `VACUUM` between
  batches (measured: the cast round-trip writes a new TOAST value). Cost:
  - a new heap tuple, new TOAST value, and WAL for every row; roughly the
    table's size again
  - the new bytes differ from the old, so `heap_attr_equals`
    (`heapam.c:4148-4184`) rules out HOT (§4.1), and **every index gets an
    `aminsert` per row**
  - an IVF turbovec index appends those rows and degrades to a flat scan
    (AGENTS.md "Degradation must be OBSERVABLE"). So on an IVF-indexed table
    this migration effectively forces the REINDEX the HARD MANDATE says a
    migration must never need. **Do not recommend it; mixed is the
    supported steady state.**
  - `ALTER COLUMN … TYPE … USING` also re-encodes but rewrites the table and
    rebuilds every index (measured: both relfilenodes change). Same
    objection.
- **Observability:** `turbovec.vector_format(vector) RETURNS text`
  (`'cbor'` / `'raw'`, IMMUTABLE, additive). An operator needs it to answer
  "is a downgrade safe?" and "how much legacy data is left?". It needs only
  byte 0, so it can use `pg_detoast_datum_slice(…, 0, 1)`. For an
  uncompressed external value `detoast_attr_slice` fetches only the chunks
  covering the slice, here one of three (`detoast.c:226-234`, PG 16.14;
  verified by review). For a compressed value it fetches a prefix bounded
  by `pglz_maximum_compressed_size` and decompresses that.

### 4.1 Byte equality is not value equality

PostgreSQL compares some datums byte-wise:

- `heap_attr_equals` for HOT eligibility. It is a plain `datumIsEqual`, a
  size check plus `memcmp` (`heapam.c:4148-4184`, `datum.c:248-254`, PG
  16.14), called from `HeapDetermineColumnsInfo` (`heapam.c:3224`) before
  the new tuple is toasted (`heapam.c:3612-3620`).
- `record_image_eq` (`*=`) for `REFRESH MATERIALIZED VIEW CONCURRENTLY`
  (`matview.c:636, 812`).
- `suppress_redundant_updates_trigger` (`memcmp`, `trigfuncs.c:76`).
- `Const` equality in the planner (`equal()`, `equalfuncs.c:112`), e.g.
  partial-index predicate matching. A planning difference only, never a
  wrong answer.

A CBOR datum and a raw datum of the same vector compare unequal.
Consequences:

- **(a)** For inline-stored vectors (≤ 397-d under `EXTENDED`, or
  `MAIN`/`PLAIN` where the page has room; Fix C recommends `MAIN`, and
  384-d models are common), an UPDATE that rewrites the column with an
  unchanged value through a type function (an ORM full-row save such as
  Django `save()` or Hibernate's default, `SET emb = $1`) is HOT-eligible
  when the stored and new formats match. When they differ it is a non-HOT
  update: every index gets an insert, and IVF turbovec indexes append to
  the delta and can cross `turbovec.ivf_max_delta_pct` (default 10) and
  degrade. Under one consistent write format this happens once per legacy
  row. If sessions disagree on `turbovec.vector_write_format`, it repeats
  on every such rewrite, because the value flips encoding each time and is
  never HOT.
- **(b)** External-TOAST values (1024-d under default storage) are
  unaffected: `heap_attr_equals` compares an 18-byte toast pointer against
  the ~5 KB inline new value, so those rewrites are already non-HOT today.
- **(c)** Operators must set the write format once per database (§4.2),
  never per application pool. `REFRESH … CONCURRENTLY` over computed vector
  columns and `suppress_redundant_updates_trigger` each churn once per row
  after an operator switches the format.

Not affected: there is no `=`, hash or btree opclass on `vector`
(`src/distance.rs:220-280`, `src/index/mod.rs:189-260`), so DISTINCT, GROUP
BY, UNIQUE, ON CONFLICT and Memoize already fail with "could not identify an
equality operator". Logical-replication `tuples_equal` uses the type's `=`
and already errors under `REPLICA IDENTITY FULL` (`execReplication.c:307-311`);
see §5.2.

**Release gate for any default flip (3.0.0, §8.1):** a measured A/B of
`pg_stat_user_tables.n_tup_hot_upd` and `turbovec.index_degradation()` under
a 384-d full-row-save workload, CBOR-stored rows rewritten under `raw` vs
`cbor`.

### 4.2 Pinning the write format

Set the format once per database: `ALTER DATABASE … SET
turbovec.vector_write_format = …`, `ALTER SYSTEM`, or the provider's
parameter group. Do not set it per session or per pool (§4.1).

The GUC stays `GucContext::Userset`. `scripts/drift-check.sh` §11b
(lines 295-304) fails any GUC whose context is not `Userset`, and `Userset`
is the only context a managed-PG user can set without superuser, so it is
also what gives operators the `cbor` escape hatch the HARD MANDATE needs.
`PGC_SUSET` would be worse: it needs a §11b allowlist entry and locks
managed-PG users out of choosing `cbor`.

The cost: any role that can INSERT can close the ≤ 2.12 downgrade path in
2.13.0 by setting `raw` in its own session. This is accepted because it is
visible through `turbovec.vector_format()` and never corrupts data.

---

## 5. Compatibility matrix

### 5.1 Version combinations

| reader \ data | legacy CBOR rows | raw rows |
|---|---|---|
| ≤ 2.12.x binary | yes (today) | **ERROR**, `UnassignedCode, offset 1`, always (§2.3 claim 3); never a wrong value |
| 2.13.x+ binary | yes, bit-identical, forever | yes |

**Upgrade:** any 2.x → 2.13.0: in place, zero rewrite, no REINDEX (HARD
MANDATE #2(b)). From 1.x, the existing v2.0.0 REINDEX still applies; this
change adds nothing to it.

**The write format is decided by the loaded binary, not by `ALTER
EXTENSION`.** The GUC default lives in the `.so`, and every backend writes
whatever its loaded copy says. Consequences:

- Restart the whole cluster after installing the new binary. Without
  `shared_preload_libraries` (managed PG, many self-managed installs), new
  backends `dlopen` the new file while long-lived pooled backends keep
  running the old code. A backend still on ≤ 2.12 code will ERROR on raw
  rows written by a new backend: fail-closed, but an outage.
- Failing over to a physical standby whose binary is still ≤ 2.12 is a
  **downgrade**. Upgrade every standby's binary before any session on the
  primary writes `raw`. Nothing enforces this; the docs must say it.
- Because 2.13.0 keeps the `cbor` default, none of this bites until an
  operator opts in to `raw`. That is the reason no minor flips the default
  (§8.1): a binary swap alone must never start writing a format the
  previous binary cannot read.

**Downgrade**, stated honestly: an old binary can't read raw rows. A
downgrade loses no data, because reinstalling the newer binary reads
everything again, but rows go unreadable until then.

- 2.13.x with the default `cbor`: downgrade to ≤ 2.12 is safe if no session
  ever wrote `raw`. Check with the census below.
- 2.13.x after an operator opted in to `raw`: downgrade to ≤ 2.12 is unsafe
  until every raw value is re-encoded. That can be done in place with the
  2.13 binary still installed: the batched re-encode in §4 under `SET
  turbovec.vector_write_format = cbor`. It costs what §4 says.
- 3.0.0 (default `raw`): downgrade to any 2.13.x is always safe; 2.13.x is
  the downgrade floor of the 3.x line.

**Catalog side of a downgrade.** The GUC is registered at `_PG_init`, so a
session could set `raw` after the binary swap but before `ALTER EXTENSION
UPDATE`, when `turbovec.vector_format()` (which the census needs) doesn't
exist yet. So: **opt in to `raw` only after `ALTER EXTENSION pg_turbovec
UPDATE`.** Going back, a 2.12 `.so` under the 2.13.0 catalog leaves
`turbovec.vector_format` bound to a missing C symbol. Recommended: ship
`sql/pg_turbovec--2.13.0--2.12.N.sql` (N = the last 2.12 patch, since phases
0 and 0b ship as 2.12.x patches), which drops `vector_format`, so the
downgrade is `ALTER EXTENSION pg_turbovec UPDATE TO '2.12.N'` after the
census passes and before the binary swap. (The alternative, documenting
that `vector_format()` ERRORs harmlessly under a downgraded binary and that
`extversion` stays at `2.13.0` until the binary is upgraded again, leaves a
catalog that disagrees with the loaded code; not recommended.)

**Downgrade census.** "No raw values" must be checked everywhere a
`turbovec.vector` datum can be stored, not only in user columns. Run the
census in every database where the extension is installed.

The census is a point-in-time check, and the GUC is `Userset`, so a session
can `SET turbovec.vector_write_format = raw` after the census passes (such a
row would fail closed on the downgraded binary, not corrupt, but it is an
outage). So fence first: `ALTER SYSTEM SET turbovec.vector_write_format =
cbor`, remove every stored `raw` setting from `ALTER DATABASE` / `ALTER ROLE
… SET` (check `pg_db_role_setting`) and from `CREATE/ALTER FUNCTION … SET`
(check `pg_proc.proconfig`), restart, and keep out any writer that
could `SET raw`. Only then run census → `ALTER EXTENSION … UPDATE TO
'2.12.N'` → binary swap. The census cannot be re-run after that `UPDATE TO`,
because it drops `vector_format`.

1. Every user column whose type contains `vector` anywhere: `vector`,
   `vector[]`, domains over either (`pg_type.typbasetype`), and composite
   types with such an attribute, checked recursively. Array columns are
   checked element-wise with `EXISTS (SELECT 1 FROM unnest(col) e WHERE
   turbovec.vector_format(e) = 'raw')`. The column query also covers
   `pg_attribute.attmissingval` (fast-default `ADD COLUMN … DEFAULT`),
   because a read returns the missing value.
2. **Extension-owned data:** `turbovec.partition_summary.centroid`
   (`src/partition.rs:376-380`), written through SPI in the writing
   session's format (`src/partition.rs:285-289`).
3. **Node-tree catalogs** that can hold a vector `Const`:
   `pg_attrdef.adbin`, `pg_rewrite.ev_action` / `ev_qual`,
   `pg_index.indexprs` / `indpred`, `pg_constraint.conbin`,
   `pg_trigger.tgqual`, `pg_policy.polqual` / `polwithcheck`,
   `pg_proc.prosqlbody` (PG14+ SQL-standard bodies), `pg_proc.proargdefaults`
   (function argument `DEFAULT '[…]'::vector`), `pg_type.typdefaultbin`
   (domain default), `pg_statistic_ext.stxexprs` (expression statistics),
   `pg_partitioned_table.partexprs` (expression partition keys such as
   `(emb <-> '[…]')`). A CBOR `Const` prints as `constvalue N [ b0 b1 b2 b3
   -95 100 …` on x86 (measured, §2.3) or `… 161 100 …` on aarch64; a raw one
   prints `… -2 1 …` (x86) or `… 254 1 …` (aarch64) (`0xFE 0x01`). Array
   `Const`s (`consttype` = the `vector[]` OID) must be matched too: either
   match that OID and treat any `(-2|254) 1` byte pair inside it as a hit
   (false positives accepted), or run the census through a C/Rust helper
   that walks the node tree.

Sketch (to be finalised and shipped as a function or documented query in
phase 1):

```sql
-- 1 + 2: data columns, including turbovec.partition_summary. Sketch covers
--    vector and vector[] directly; the phase-1 version also resolves domains
--    (typbasetype) and composites (typrelid, recursively).
SELECT format(CASE WHEN a.atttypid = 'turbovec.vector'::regtype
  THEN 'SELECT %L, count(*) FROM %s WHERE turbovec.vector_format(%I) = ''raw'''
  ELSE 'SELECT %L, count(*) FROM %s WHERE EXISTS (SELECT 1 FROM unnest(%I) e WHERE turbovec.vector_format(e) = ''raw'')' END,
              a.attrelid::regclass || '.' || a.attname, a.attrelid::regclass, a.attname)
FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
WHERE a.atttypid IN ('turbovec.vector'::regtype, 'turbovec.vector[]'::regtype)
  AND a.attnum > 0 AND NOT a.attisdropped AND c.relkind IN ('r', 'p', 'm');
-- run each generated statement (\gexec), in every database with the extension.

-- 3: node trees. A raw vector Const's magic + version bytes print as "-2 1"
--    (x86, signed char) or "254 1" (aarch64, unsigned char) after the header
--    bytes. :R is the regex below with the vector type OID substituted, so
--    other types' Consts never match; for the vector[] OID use R_arr (an
--    array Const has the ArrayType header after the varlena header, so the
--    magic can appear at any offset; false positives are accepted).
--    R     = ':consttype <oid> [^}]*:constvalue \d+ \[ ?(-?\d+ ){4}(-2|254) 1 '
--    R_arr = ':consttype <vector[]_oid> [^}]*:constvalue \d+ \[[^]]* (-2|254) 1 '
--    (PostgreSQL `~` is ARE, so \d works; but psql's \set strips single
--    backslashes, so set :R / :R_arr with doubled backslashes, or write
--    [0-9]+ for \d+ and [[] for \[.)
SELECT 'pg_attrdef', oid FROM pg_attrdef WHERE adbin::text ~ :'R'
UNION ALL SELECT 'pg_rewrite', oid FROM pg_rewrite
  WHERE ev_action::text ~ :'R' OR ev_qual::text ~ :'R'
-- …same pattern for pg_index.indexprs/indpred, pg_constraint.conbin,
--    pg_trigger.tgqual, pg_policy.polqual/polwithcheck, pg_proc.prosqlbody,
--    pg_proc.proargdefaults, pg_type.typdefaultbin,
--    pg_statistic_ext.stxexprs, pg_partitioned_table.partexprs
;
```

The node-tree regex is a heuristic over `nodeToString` output (a 1-byte
short-header `Const` has 1 header byte, not 4: the same regex with `{1}`
in place of `{4}`, keeping `(-2|254)`); the phase-1 version must be tested
against both header forms and on both x86 and aarch64 (§7 test 6).

**Recommended release type: a minor (2.13.0) for the reader, a major
(3.0.0) for the default flip.** 2.13.0 removes no SQL, makes no format
unreadable, needs no upgrade action, and changes nothing a binary swap
writes. Flipping the default is deferred to a major because it is the first
change in the project that closes a downgrade path by default (§10 item 13).

### 5.2 Paths that never see the datum format

- **pg_dump / pg_restore** (plain, `-Fc`, `-Fd`) use text `COPY` through
  `vector_out` / `vector_in`. The text form stays byte-identical
  (`src/vec.rs:98-112` is unchanged; a fixture test pins it). A dump taken
  on any version restores on any version. Restoring writes whatever format
  the restoring session uses (`raw` only if the operator opted in, or from
  3.0.0 by default).
- **Logical replication** sends text unless `binary = true` *and* the type
  has `typsend` (`proto.c:836`). We have no `typsend`, so it is always text,
  so publisher and subscriber formats are independent in both directions.
  Existing limits, unchanged by this design: `tuples_equal` uses the type's
  `=` operator, so `REPLICA IDENTITY FULL` on a table with a `vector` column
  already errors (`execReplication.c:307-311`); and on PG16+ a `binary =
  true` subscription's initial sync uses `COPY (FORMAT binary)`
  (`tablesync.c:1228-1235`), which already fails for `vector` (§7 test 13
  asserts both stay as they are).
- **COPY BINARY:** impossible today. **Measured:** `COPY t TO STDOUT
  (FORMAT binary)` → `ERROR: no binary output function available for type
  vector`. So this design changes no wire protocol.
- **Physical replicas** replay the primary's bytes and need the same `.so`.
  **Upgrade every standby's binary first** (a 2.13+ standby reads both
  formats), then the primary, and only then let any session write `raw`. A
  ≤ 2.12 standby behind a raw-writing primary replays fine but errors on
  reads of those rows (fails closed); promoting it is a downgrade (§5.1).
- **pg_upgrade:** heap and TOAST files are copied or linked, so the result
  is a mixed (all-CBOR) cluster, which the new reader handles.
- **Catalog constants** (view/rule `Const`, column `DEFAULT`, the other
  node-tree catalogs in §5.1's census): CBOR until the DDL is re-run. Read
  fine; they count toward the downgrade census.
- **pg_statistic:** `vector` has no `=`/`<` operator, so ANALYZE uses
  `compute_trivial_stats` (`analyze.c:1937-1949`) and stores no
  `stavalues` (verified by review present on PG 13.23 and 19beta1, the
  ends of the CI range).
- **Arrays (`vector[]`, ColBERT):** `array_agg` over a mixed table copies
  element bytes, so **one array can contain both formats**. Decoding is per
  element (`FromDatum for Vec<T>`, `pgrx-0.19.1/src/datum/array.rs:896-925`
  → our `UnboxDatum`), so this works, but it needs its own test (§7).

### 5.3 `send` / `recv` (separate, optional)

Adding binary I/O is additive: `ALTER TYPE turbovec.vector SET (RECEIVE =
vector_recv, SEND = vector_send)` in an upgrade script, which is pgvector's
own precedent (`sql/vector--0.1.0--0.1.1.sql:10`; supported on PG13,
`typecmds.c` 13.23). It needs superuser at `ALTER EXTENSION UPDATE` time,
the same privilege class as the original `CREATE TYPE` (`typecmds.c:215-218`).
If we add it, the wire format should be pgvector's (`int16 dim, int16 0,
float4 × dim`, network order; `pgvector src/vector.c:374-422`) so existing
pgvector client codecs work. A binary wire format is a protocol and freezes
on first release. It doesn't depend on the storage format and shouldn't
ship in the same release.

---

## 6. Index interaction

The AM stores codes, not vectors. Checked:

- **Build** decodes `Vector` and stages `f32`s
  (`src/index/build.rs:2004`; the spill record is `8 + dim·4`, the id plus
  floats, `build.rs:64`). ColBERT does the same per token
  (`build.rs:1868`).
- **Insert** decodes, then `normalise_to_vec` / `to_vec`
  (`src/index/insert.rs:73, 131-136, 511-513, 627-629`).
- **Scan** decodes the ORDER BY datum into `opaque.query: Vec<f32>`
  (`src/index/scan.rs:435-445`).
- **No chain holds a varlena.** Chains: codes, scales, ids, blocked,
  rotation, coarse, cell_dir, tombstone, graph, bq_mean
  (`src/index/page.rs:295-410`). There is no
  `FromDatum`/`IntoDatum`/varlena use in `relfile.rs`, `page.rs`,
  `cache.rs`, or `xact.rs` (grep).
- SPI readers fetch `v::turbovec.vector::real[]` through `vec_to_array`
  (`src/knn.rs:367, 473`, `src/colbert.rs:508, 832, 903`,
  `src/partition.rs:183`), so the dual-format `FromDatum` covers them.

So `MetaPageData::version` stays 8 (`wire_format_version_is_stable`,
`src/lib.rs:2201-2210`), there is no REINDEX, and drift-check §7 is
unaffected. Both formats decode to bit-identical `f32`s, so an index built
over a CBOR heap and one built over the same values stored raw must be
**byte-identical**; §7 tests exactly that. Build and insert each get one
decode cheaper per row: ~3.7 s of CPU per 1M × 1024-d build (estimate,
3.73 µs × 1M), small next to the build total.

---

## 7. Test plan (HARD MANDATE #1)

All in-database tests run under `cargo pgrx test` on every CI leg (pg13–19).
Mixed-format tables are created in tests with `SET
turbovec.vector_write_format`. The GUC doubles as the test harness: with it,
one binary can write both formats.

**Which tests fail before and pass after.** HARD MANDATE #1 asks for a
reproduction that fails before the change. Here the tests split in two:

- **Regression guards, green on both sides:** 0b, 1, 2, the CBOR halves of
  3 and 4, the 2.12-only parts of 9, and the `binary = true` / `REPLICA
  IDENTITY FULL` assertions of 13. They pin what 2.12.x already does (CBOR
  bytes, decode bits, fail-closed reads, dump text, replication limits) so
  the change can't move it. 0b, 1 and 2 land first, in phases 0 and 0b, and
  must be green on the unmodified 2.12.x tree.
- **Fail before, pass after:** every test that reads or writes a raw datum
  through the new binary (the raw halves of 3 and 4, 5–8, 10–12, the raw
  parts of 13, 14, 15). On 2.12.x they fail because the raw value ERRORs or
  the GUC doesn't exist. They are the new behaviour's reproduction.

0. **(0b) Old-binary fail-closed invariant, shipped in the 2.12.x line before
   any format code (phase 0b).** A test-only `CREATE CAST (bytea AS
   turbovec.vector) WITHOUT FUNCTION` (created and dropped inside the test)
   injects a `0xFE 0x01 <dim LE> <f32 LE …>` payload, and the test asserts an
   ERROR, never a value, from each entry point:
   - `vector_out` (`::text`)
   - every distance function and operator, including 2.12.0's operand-cache
     path (Fix B: `Slot::get`, `src/distance.rs`), called both
     through an operator (cached) and via a `DirectFunctionCall`-style path
     (no `flinfo`)
   - casts to `real[]`, `jsonb`, `halfvec`, `sparsevec`
   - `ambuild`: `CREATE INDEX` and `REINDEX` over a table holding one raw row
   - `aminsert`, via a non-HOT update of another column (the raw value is
     re-indexed without passing through a type function)
   - `amrescan`, via a raw `Const` in a view used as the ORDER BY query
   - `max_sim` and the ColBERT build over a `vector[]` holding one raw
     element
   - `knn()` and `colbert_search` (SPI)

   **Recommended:** replace the hand-maintained
   function list with a catalog-driven sweep: call every `pg_proc` whose
   `proargtypes` include `vector` or `vector[]` with the injected raw value
   and assert an ERROR, keeping the AM (`ambuild`, `aminsert`, `amrescan`)
   and SPI (`knn()`, `colbert_search`) items as explicit cases. New read
   paths are then covered automatically; the test is what makes "an old
   binary fails closed" an invariant rather than an audit result.
1. **Golden CBOR fixtures, committed before any format code.** Phase 0 is a
   separate test-only commit on the current tree. Fixtures
   `tests/fixtures/vector_cbor_v2/…` hold hex payload plus expected `f32`
   bit patterns for:
   - dims {1, 8, 23, 24, 29, 30, 31, 255, 256, 397, 398, 1024, 16000}
   - value classes {f16-exact (0, ±1, 0.5), −0.0, random f32, subnormals,
     ±f32::MAX, ±f32::MIN_POSITIVE, mixed}
   - f16 edge classes, because serde_cbor emits `0xF9` (f16) whenever a
     value round-trips exactly (`ser.rs:312-330`) and widens it through
     `half` 1.8.3: f16-subnormal-exact values (2⁻²⁴, 2⁻¹⁵), 65504 (f16
     max), and −0.0 encoded as f16 `0x8000`
   - each fixture's `vector_out` text

   Fixtures are captured from real `.so`s, not only from the current tree:
   one from a 2.x build (pgrx 0.19.1) and one from a v1.x build on pgrx
   0.17.0 (51 of the 89 tags used it; its `cbor_encode` is identical, and
   the fixture proves it rather than asserting it).

   A `#[test]` asserts `serde_cbor::to_vec` reproduces every fixture
   byte-for-byte, and decodes each back to the expected bits. It pins the
   encoder and the decoder, including `serde_cbor =0.11.2` and `half
   1.8.3`, against dependency drift: it fails if either crate moves. A
   `#[pg_test]` asserts the in-database datum bytes equal it (pageinspect,
   or a test-only `vector → bytea WITHOUT FUNCTION` cast). Those two prove
   the fixtures are what real installs hold. Add a drift-check rule:
   `tests/fixtures/vector_cbor_v2/` must be byte-identical to the last tag
   (append-only).
2. **Legacy decode is bit-identical.** Every fixture decodes through the new
   reader to the expected bits (`to_bits()`, not `==`, so −0.0 and
   subnormals are checked). `vector_out` text equals the fixture text.
3. **Discriminator properties** (hegel, the repo's PBT crate,
   `Cargo.toml:87`; pattern in `src/index/graph.rs:3086-3110`):
   - any finite `Vec<f32>` (len 1..=16000) → CBOR starts `A1 64 64 61 74
     61`
   - any raw payload → `serde_cbor::from_slice::<Vector>` is `Err` at
     offset 1 (pins claim 3)
   - **reader fuzz:** arbitrary bytes → `decode_payload` (the pure
     function, §3.1) returns `Ok` or `Err`, never panics or hits UB, at
     every buffer offset mod 4. Because it has no `pg_sys`, the same test
     runs under Miri (whether the serde_cbor arm builds under Miri is
     unverified; the raw arm has no dependency)
4. **Round-trips:** raw encode → decode is bit-identical (property);
   CBOR fixture → decode → raw encode → decode is bit-identical; unknown
   version → `0A000`, unknown first byte → `XX001`, `dim = 0`, short or
   long length → `ERROR`; a writer given `dim > MAX_DIM` errors before the
   `u16` cast (test 65,536 + 1, which would wrap to 1).
5. **Every operator and function on old/new/mixed arguments:** all four
   format pairs for `<->`, `<#>`, `<=>`, `<+>`, `+`, `-`, `*`, `||`, and the
   named distance functions. Also:
   - `vector_dims`, `vector_norm`
   - casts to and from `real[]`, `float8[]`, `int4[]`, `jsonb`, `halfvec`,
     `sparsevec`; `binary_quantize`
   - `subvector`, `vec_check_dim`, normalize functions, `avg`/`sum`
     (including parallel combine)
   - `max_sim` over a `vector[]` holding **both formats in one array**
   - `knn()`, `colbert_search`, `refresh_partition_summary` /
     `nearest_partitions`
   - ORDER BY scans with the query literal in each format

   Results must be bit-identical across format variants.
6. **Storage forms:**
   - external (1024-d default)
   - inline compressed: a 1024-d all-zero vector under EXTENDED; assert
     `pg_column_compression` is non-NULL, then decode
   - `lz4` where the build has it
   - `MAIN`, `PLAIN`
   - 1-byte header (dim ≤ 30; assert the short header via pageinspect)
   - views/defaults with CBOR `Const`s
   - the §5.1 downgrade census (both header forms, scalar and array
     `Const`s, every listed catalog) run on **both x86_64 and aarch64**
     (Graviton), because `outDatum` prints bytes signed on one and unsigned
     on the other
7. **Alignment fuzz:** tables with leading `bool`/`text`/`smallint` columns
   of varying length before a dim 1..30 vector, so the short payload lands
   at every offset mod 4. Read through seq scan, index recheck, `vector[]`,
   and `datumCopy` (reorder queue). Plus a `#[test]` of the primitive at
   buffer offsets 0–3.
8. **Index is format-blind:** load the same rows in the same order into two
   tables, one per format, and into a third with the formats **interleaved
   row by row** (alternate `SET turbovec.vector_write_format` per insert).
   Build flat / IVF / BQ indexes, serially and as a parallel build, and
   again with `REINDEX CONCURRENTLY`, and sha256 the relfile chains. All
   must be identical.
9. **Dump/restore:** dump a mixed table from 2.12.0 (CBOR) and from 2.13+
   (mixed), restore into both, and compare `vector_out` text and decoded
   bits. **COPY BINARY:** assert the current `ERROR` is unchanged (or
   round-trip, if §5.3 ships).
10. **Downgrade fail-closed, end to end** (EC2 or local, three builds:
    2.13+, 2.12.0, 2.12.N): write raw rows with 2.13+, then swap in the
    2.12.0 `.so` and the 2.12.N `.so`
    in turn, and assert every read is an `ERROR`, never a value, from every
    entry point in test 0b, under each. Exercise
    the supported path too: fence (§5.1, asserting no `raw` remains in
    `pg_db_role_setting` or `pg_proc.proconfig`), census, `ALTER EXTENSION
    pg_turbovec UPDATE TO '2.12.N'` via the downgrade script, binary swap,
    restart, and assert the catalog matches the loaded code. Also assert the
    operator-visible facts:
    - neither 2.12.x binary can `pg_dump` a table holding raw rows (the dump
      fails; it never writes a partial or wrong dump)
    - a `REINDEX CONCURRENTLY` that fails on a raw row leaves an INVALID
      index, which the docs must tell the operator to drop

    Swap back and assert all rows read correctly.
11. **Sustained-load soak** (pattern:
    `benches/results/tv111_arm_20261005/raw/soak2.py`), ≥ 90 min:
    - seed 60k rows half CBOR, half raw
    - writers randomly `SET` the write format per session, COPY-INSERT,
      `UPDATE SET tv = tv` (no re-encode), `UPDATE SET tv =
      tv::real[]::turbovec.vector` (re-encode), and DELETE
    - an ORM-style writer that saves the whole row with an unchanged vector
      (`UPDATE … SET tv = $1, other = $2` with `$1` the row's own text
      form), and a writer whose session flips the format on every
      transaction (the §4.1 worst case)
    - `VACUUM` every ~5 min; `pg_terminate_backend` mid-transaction
    - verify: every live row's decoded bits equal the deterministic
      generator's `vec(id)` (a content check that doesn't care about
      format); `turbovec_check` stays clean; soak2's byte-level
      index-vs-fresh-REINDEX comparison; **backend RSS flat** (AGENTS.md:
      watch RSS, not just correctness; borrowed views change palloc
      ownership); `turbovec.index_degradation()` sampled every minute, so
      the flip-flop writer's delta growth is visible
    - run on arnold (AVX2) and Graviton `c8g` (planes path; AGENTS.md)
12. **pg_upgrade:** PG N with 2.12 (all CBOR, including a view `Const` and a
    column `DEFAULT`) → PG N+1 with 2.13, `--link` and `--copy`. Assert every
    value decodes bit-identically, indexes need no REINDEX, and new writes
    under `raw` mix in correctly.
13. **Logical replication:** 2.12 publisher → 2.13 subscriber writing `raw`,
    and 2.13 (raw) publisher → 2.12 subscriber; values arrive as text and
    decode identically. Include a `binary = true` subscription: assert that
    on PG16+ its initial sync still fails the way it does today (`COPY
    (FORMAT binary)`, `tablesync.c:1228-1235`) and that `REPLICA IDENTITY
    FULL` still errors (`execReplication.c:307-311`). Both are regression
    guards on existing behaviour, not new features.
14. **Physical standby on 2.12** behind a 2.13 primary writing `raw`: WAL
    replay succeeds, reads of raw rows on the standby ERROR (never a value),
    and swapping the standby's `.so` to 2.13 makes them read correctly
    without any rewrite.
15. **HOT / degradation (pins §4.1):** a 384-d table under `MAIN` with an
    IVF index and `fillfactor = 50` (so HOT can't fail for page-full
    reasons), rows written CBOR. Same-value full-row UPDATEs under `cbor`
    must be HOT (`n_tup_hot_upd` grows, delta doesn't); under `raw` the
    first rewrite of each row is non-HOT and grows the delta, and a second
    same-value rewrite under `raw` is HOT again. Assert
    `turbovec.index_is_degraded()` / `index_degradation()` move exactly as
    §4.1 predicts. This is also the measured A/B the 3.0.0 default flip is
    gated on.

Any failure blocks the release. Code reasoning alone is not evidence here
(HARD MANDATE #1; v1.28.4).

---

## 8. Rollout

### 8.1 Phases

| phase | release | contents | SQL change | user action | Mandate |
|---|---|---|---|---|---|
| 0 | next patch (2.12.x), before any format code | golden CBOR fixtures + encoder-pin tests (§7.1); `serde_cbor = "=0.11.2"` declared directly, `half 1.8.3` kept locked | none | none | #2(a): zero format change, test-only |
| 0b | next patch (2.12.x), before any format code | fail-closed invariant test (§7 test 0b): a `0xFE 0x01 …` payload must ERROR, never yield a value, from every read entry point of the 2.12.x binary, including 2.12.0's `Slot::get` operand cache (catalog-driven sweep recommended) | none (test-only cast created and dropped inside the test) | none | #2(a): zero format change, test-only |
| 1 | **minor 2.13.0** | dual-format reader; hand-written datum impls; GUC `turbovec.vector_write_format` (`Userset`) default **`cbor`**; `turbovec.vector_format(vector)`; downgrade census (§5.1) | +1 function (`sql/pg_turbovec--2.12.x--2.13.0.sql` via `cargo pgrx schema`, `migrations/087_…`) + downgrade script `sql/pg_turbovec--2.13.0--2.12.N.sql` (drops `vector_format`). Gate: the order-independent `cargo pgrx schema` diff against 2.12.0 contains only `vector_format` | `ALTER EXTENSION … UPDATE` + **cluster restart**. Opting in to `raw` is a per-database / `ALTER SYSTEM` decision (§4.2), made only after every standby runs 2.13+ and after `ALTER EXTENSION … UPDATE` | #2(b): old datums read transparently, no REINDEX, no rewrite. Gated by #1: §7 tests 1–15, including the soak (test 11) |
| 1b | patch or minor, after an A/B | zero-copy borrowed `VectorArg` path (§3.3) | none (gate: A/B + empty schema diff + §7 tests 3, 5, 7 and 11 (soak with RSS) re-run, because 1b changes palloc ownership on the read path, where the v2.11.0 RSS leak came from) | none | #2(a): zero format change |
| 2 | **major 3.0.0** | GUC default → **`raw`**, gated on the §4.1 HOT/degradation A/B and the §10.7 compression measurement | none for the type (empty upgrade script + migration file); other 3.0 changes ride along | restart; upgrade standbys first; `ALTER DATABASE … SET turbovec.vector_write_format = cbor` keeps a ≤ 2.12 floor | major, but no format break: every datum stays readable, so #3's offline converter is N/A |
| 3 | ≥ two minors after 3.0.0 | optionally retire the `cbor` *writer* value (AGENTS.md two-release deprecation window); the CBOR *reader* is never removed | none | none | no break: the reader is never removed |

There is deliberately no "2.14.0 flips the default" phase. A minor whose
only effect is that a binary swap starts writing a format the previous
binary can't read would make every 2.12 → 2.14 skip, every unrestarted
pooled backend and every lagging standby a fail-closed outage (§5.1). In a
minor the operator opts in; at the major the default changes.

### 8.2 `docs/UPGRADING.md` rows (draft text)

| From | To | Required action | Notes |
|---|---|---|---|
| any 1.x | 2.13.0 | `REINDEX INDEX` (unchanged, v2.0.0 wire v7→v8) | datum change adds no action |
| 2.0.0–2.12.x | 2.13.0 | _none_ (no REINDEX, no rewrite); restart the cluster | MINOR. `turbovec.vector` gains a second on-disk datum format (raw `float4`, tagged `0xFE`). This release **reads** it but writes the old CBOR format by default; `ALTER DATABASE … SET turbovec.vector_write_format = raw` (or `ALTER SYSTEM`) opts in. Set it per database, not per session or pool: mixing formats on rewrite defeats HOT. New `turbovec.vector_format(vector)` reports `cbor`/`raw` per value. Index wire format unchanged (v8). The write format is decided by the loaded binary, not `ALTER EXTENSION`: restart after installing it, and upgrade every standby's binary before any session writes `raw` (failing over to a ≤ 2.12 standby is a downgrade). **Downgrade floor: 2.12.x while no `raw` value exists anywhere** (census in the design doc §5.1); ≤ 2.12.x reading a raw value raises `failed to decode CBOR … UnassignedCode, offset 1`, never a wrong result. `ALTER EXTENSION pg_turbovec UPDATE TO '2.13.0';` + restart. To downgrade: fence and run the census (§5.1), then `ALTER EXTENSION pg_turbovec UPDATE TO '2.12.N'`, then install the 2.12 binary and restart. |
| 2.13.x | 3.0.0 | per the 3.0.0 row; for this type: none, restart | MAJOR. New and updated values are written raw by default. Existing rows stay CBOR, are read forever, and are never rewritten. **Downgrade floor: 2.13.0.** Upgrade standbys before the primary. `ALTER DATABASE … SET turbovec.vector_write_format = cbor` keeps writing the old format and keeps the ≤ 2.12 floor open. From ≤ 2.12.x directly to 3.0.0: supported only with a full cluster restart and every standby upgraded first, because a surviving ≤ 2.12 backend ERRORs on raw rows. |

### 8.3 Constants and guards to add

- `VECTOR_DATUM_MAGIC = 0xFE`, `VECTOR_DATUM_VERSION = 1` in `src/vec.rs`
- a `vector_datum_format_is_stable` test mirroring
  `wire_format_version_is_stable`
- the fixture-immutability drift-check rule (§7.1)

### 8.4 Explicitly deferred

- **`halfvec`** (`src/halfvec.rs:27-32`): `half`'s serde writes each f16 as
  a CBOR u16 (measured: `a1 64 data 82 19 3c00 19 4000`), first byte
  `0xA1`. The same `0xFE` scheme applies; payload f16 × dim.
- **`sparsevec`** (`src/sparsevec.rs:24-34`): map(3), first byte `0xA3`
  (measured); `dim, nnz, i32[nnz], f32[nnz]`.
- **`bitvec`** (`src/bitvec.rs:25-33`): map(2), first byte `0xA2`
  (measured).
- **`VecAccum`, `HalfvecAccum`, `SparsevecAccum`:** in-memory aggregate
  states, never stored, so no compatibility burden and no benefit worth
  the churn.

Each later type goes through the same reader-in-a-minor, default-at-a-major
sequence with its own fixtures. `vector` goes first because it is the indexed hot
path and the only one with a measured cost.

---

## 9. Expected gain

- **Decode.** FINDINGS measured two decodes per candidate at 8.1 µs. Fix B
  leaves one (~4 µs, estimate: half of 8.1). This design takes the
  remaining candidate decode to ~0.1 µs (owned copy) or ~0 (borrowed through `VectorArg`, §3.3).
  Estimate: **−3.7 to −4 µs per rechecked candidate, ~4 ms/query backend
  CPU at `search_k = 1024`, 1024-d.** Basis: 3.73 µs CBOR vs 0.08 µs copy
  (`cbor_bench.rs`). Composed with A+B+C, FINDINGS projects ~6 → ~2 µs per
  candidate (estimate; not measured end to end).
- **Raw rows only.** A table loaded before `raw` was enabled and never
  updated keeps paying CBOR on every legacy row. An optional mitigation, deferred: a
  hand-written decoder for exactly the canonical shape, falling back to
  serde for anything else. A prototype measured **0.89–0.91 µs vs serde
  3.02–3.09 µs**, bit-exact on two test vectors. It would need the §7
  fixtures plus a differential property test against serde_cbor before it
  could ship.
- **Size, honestly.** Raw is 4,104 B vs CBOR 5,133 B per 1024-d datum with
  random values (20% smaller). That does **not** become fewer pages at
  1024-d:

  | 1024-d, 20k rows | CBOR | raw (bytea emulation, same payload size) |
  |---|---|---|
  | main heap | 1,212,416 B | 1,212,416 B |
  | TOAST relation | 117,030,912 B | 109,232,128 B (−6.7%) |
  | TOAST chunks per value | 3 | 3 |
  | `PLAIN`, 1,000 rows | 1,000 pages | 1,000 pages |

  All measured. The TOAST fetch (~5 µs) is therefore roughly unchanged at
  1024-d (estimate); Fix C (`SET STORAGE MAIN`) is still the lever there.
- **Inline thresholds** for a `(bigint, vector)` row: `TOAST_TUPLE_THRESHOLD
  = 2032` (computed, `heaptoast.h:46-48`), and the `MAIN`/`PLAIN` cap is
  8,160 (`heaptoast.h:59-61`). Raw sizes don't depend on the data; CBOR
  depends on how many values are f16-exact.

  | | CBOR (random values) | raw |
  |---|---|---|
  | inline under default storage | ≤ **397-d** (measured) | ≤ **498-d** (measured via emulation; computed `(2000 − 8)/4`) |
  | fits in-line under `MAIN`/`PLAIN` | ≤ **1,623-d** (measured) | ≤ **2,030-d** (measured via emulation) |
  | heap tuples/page at 384-d (inline both ways) | 4 | 5 (computed) |
  | heap tuples/page at 512 / 768 / 1024 / 1536-d under `MAIN` | 3 / 2 / 1 / 1 | 3 / 2 / 1 / 1 (computed; 1024 measured) |

  None of the common embedding sizes (384, 512, 768, 1024, 1536, 2048)
  crosses a threshold. 384-d gets ~20% fewer heap pages (computed). The
  case for this change is CPU, not storage. Raw sizes are identical to
  pgvector's (`VECTOR_SIZE`, `vector.h`), so pgvector sizing guidance
  carries over.
- **Writes and loads.** Encoding is a memcpy instead of serde (expected
  faster; unmeasured). Build: ~3.7 s CPU per 1M × 1024-d (estimate).

---

## 10. Open questions and risks

1. **pgvector byte compatibility is given up** (§2.4). It contradicts
   `src/vec.rs:3-8` and `docs/ARCHITECTURE.md` §3.2, and needs sign-off. The
   alternative, pgvector's layout, is unambiguous for the new reader but
   lets an old binary silently mis-decode crafted data.
2. **`bikeshed_postgres_type_manually_impl_from_into_datum`** is pgrx's
   deliberately unstable name. If it is renamed, the build fails (safe), and
   the fallback is a hand-written `CREATE TYPE` that must match the current
   SQL exactly.
3. **Schema identity.** Fix B's `VectorArg` already delegates
   `SqlTranslatable` to `Vector` and pins the generated SQL with a test.
   Phase 1's gate is broader: the order-independent `cargo pgrx schema`
   diff against 2.12.0 must contain only the new `vector_format`
   function (§8.1). Phase 1b's diff must be empty.
4. **serde_cbor 0.11.2 is unmaintained** (RUSTSEC-2021-0127,
   informational) and we need it forever. Pin it directly. Vendoring it, or
   replacing it with a fixture-validated hand decoder, is a later option.
5. **Downgrade depends on operators:** physical standbys must be upgraded
   first, every backend must be restarted onto the new binary (§5.1), and
   after 3.0.0 a per-database `cbor` setting is the only way to keep a
   ≤ 2.12 floor. Docs must say this prominently.
6. **A `Userset` GUC means any session can write raw** in 2.13.0 and close
   the ≤ 2.12 downgrade for those rows, and per-session values defeat HOT
   on same-value rewrites (§4.1). Accepted, with §4.2's guidance: §11b
   requires `Userset`, correctness on 2.13+ is unaffected, and
   `vector_format()` makes it visible.
7. **Compression must be measured before any default flip (3.0.0 gate).**
   Structured raw floats (many zeros, quantized embeddings) may now
   pglz-compress under the default `EXTENDED` storage. A compressed raw
   datum costs a decompress on every read and loses zero-copy, which could
   erase the gain. Unmeasured. pgvector uses `STORAGE external`, but
   changing ours affects only new columns and is a separate SQL decision.
8. **Big-endian is refused at build time** (`compile_error!`, §2.5). The
   index relfile is already native-endian (`src/index/relfile.rs`), so BE
   was never supported; the guard makes that explicit instead of leaving
   an untested code path.
9. **`ANALYZE` stores no vector `stavalues`:** `compute_trivial_stats`
   is present on PG 13.23 and 19beta1 and there is no `=` operator, so no
   `stavalues` are stored (verified by review, §5.2).
10. **A future `=`, hash or btree opclass for `vector` must compare and
    hash decoded floats, never bytes.** With two formats, byte equality is
    no longer value equality (§4.1). A byte-wise `=` would make the same
    vector unequal to itself across formats and break hash joins and UNIQUE.
11. **The `0xFE` RFC 8949 "not well-formed" status** (additional-info 30,
    reserved; §2.3) is verified against RFC 8949 §3, additional
    information 28–30. The proof still rests on serde_cbor 0.11.2 code
    plus measurement, not on the RFC.
12. **The 2.12.0 per-`FmgrInfo` operand cache (Fix B) is only worth it for
    CBOR.** Its own
    `ponytail:` note says a slot whose argument changes every call (the
    candidate side, every row) misses and pays a memcmp-to-first-difference
    plus a payload memcpy (~5 KB at 1024-d) on top of the decode. For a raw
    payload that miss cost is about the whole raw decode cost (memcpy of
    4 KB ≈ 0.1 µs, measured), so caching raw payloads would roughly double
    their cost and save nothing. Hence the `0xFE` bypass in §3.3; the cache
    stays for CBOR payloads, where a hit saves ~3.7 µs.
13. **This is the project's first downgrade floor, and policy doesn't cover
    it.** AGENTS.md has no downgrade rule, and `docs/UPGRADING.md` mentions
    downgrades only for 1.6–1.7.3. Proposed text, **not applied in this
    change** (both files are policy and need the maintainer's sign-off):
    - `docs/UPGRADING.md`: every migration-matrix row gains a **Downgrade
      floor** line: the oldest binary that can still read everything the
      new release may have written, and the condition under which that
      holds (e.g. 2.13.0: "2.12.x while no `raw` value exists; census
      §5.1"; 3.0.0: "2.13.0").
    - `AGENTS.md`, under the versioning policy: "Any change to what a
      datum's type functions **write** (`IntoDatum`, `*_in`, `*_recv`, any
      new on-disk datum format) changes the downgrade floor and needs
      explicit maintainer sign-off. A minor release must never change the
      default write format; the new format may ship in a minor only as an
      operator opt-in, with the default flip at the next major."

---

## Appendix: probes behind "measured here" (2026-10-06)

Host floki (Intel Core Ultra 7 258V, AVX2), PG 16.15 from
`~/.pgrx/16.15/pgrx-install` with the installed pg_turbovec **2.11.0**
`.so` (all probes verified on 2.11.0 only; 2.12.0 changed `src/distance.rs`
and `src/kernels.rs`, not `src/vec.rs` or the CBOR decode), on a **private throwaway cluster** (own `initdb`, port 54999, socket
`/tmp`; not the shared pgrx test cluster; stopped with `pg_ctl -m fast`
afterwards). Rust probes in a scratch crate against `serde_cbor = "=0.11.2"`
and the same derive shape as `src/vec.rs:31-36`, rustc 1.97.0.

- **CBOR prefix:** `serde_cbor::to_vec(&Vector{data})` for dims {1, 8, 23,
  24, 255, 256, 1024, 16000} × value patterns. All start `a1 64 64 61 74
  61`. `pageinspect` `heap_page_items` on a 2.11.0 table shows the same
  bytes after the varlena header.
- **Downgrade, pgvector layout:** 16,000 dims × {zero, one, random, sin} ×
  {LE, BE} raw payloads into `serde_cbor::from_slice::<Vector>`: 0 accepted.
  The crafted 447-d payload (§2.4) was accepted as `[1.0]`. In-database,
  through a test-only superuser `CREATE CAST (bytea AS turbovec.vector)
  WITHOUT FUNCTION` (dropped afterwards), 2.11.0 printed `[1]` and
  `vector_dims = 1`.
- **Downgrade, `0xFE` layout:** 2,000,000 random `0xFE`-prefixed payloads:
  0 accepted, all `unassigned type at offset 1`. In-database: `ERROR: failed
  to decode CBOR: ErrorImpl { code: UnassignedCode, offset: 1 }`. A
  first-byte scan found 49 initial bytes the old decoder always rejects at
  offset 1 (`0x1c–1f, 3c–3f, 5c–5e, 7c–7e, 9c–9e, bc–be, dc–df, e0–f3, f8,
  fc–ff`). `0xFE` was picked from those; `0xFF` was avoided because it is
  CBOR "break".
- **Sizes:** `pg_column_size` of 1024-d random CBOR = 5,129 (payload). The
  inline boundary is 397/398-d (`lp_len` 2030 vs 50). `PLAIN` accepts 1,623
  and rejects 1,624 (`row is too big: size 8168, maximum size 8160`). The
  raw layout was emulated with `bytea` of exactly `4 + 4·dim` bytes under
  `STORAGE EXTERNAL`/`PLAIN`: inline ≤ 498, `PLAIN` ≤ 2,030. The 20k-row
  TOAST and 1,000-row `PLAIN` page counts are in §9.
- **Rewrite behaviour:** a table's TOAST `chunk_id` is unchanged by `UPDATE
  SET v = v` and changes under `SET v = v::real[]::turbovec.vector`. `VACUUM
  FULL` and `ALTER COLUMN TYPE … USING` change relfilenodes. The latter
  also rebuilds the turbovec index.
- **Catalog facts:** `pg_type` for `vector` is `typlen -1, typalign i,
  typstorage x, typreceive -, typsend -`. All 10 `pg_cast` rows are
  `castmethod f`. `COPY … (FORMAT binary)` fails with "no binary output
  function". `pg_proc.prosrc` is `vector_in_wrapper` / `vector_out_wrapper`.
- **Decode microbench** (200k iterations, 1024-d, `--release`): serde 3.02–3.09
  µs, canonical fast-path prototype 0.89–0.91 µs, memcpy 0.10–0.12 µs.
