# Raw `float4` varlena for `turbovec.vector` (replace serde-CBOR): design

**Status:** design only. No code in this change. Step 6 of the recheck-cost
plan (`benches/results/perf_abc_20261006/TEAM_BRIEF.md`). Branch
`design/raw-vector-varlena`, written 2026-10-06 against v2.11.0 + the
recheck-analysis commits (`3551a90`).

**Decision:** a second datum format for `turbovec.vector`, laid out as
`[varlena hdr][0xFE][0x01][u16 dim LE][f32 LE × dim]`. It is chosen by its
first payload byte, so it can never be confused with the CBOR every released
binary has written (that always starts with `0xA1`). The CBOR reader stays
forever and mixed tables are the permanent normal state. Rollout is two minor
releases: the first can read the new format, the second writes it by default.
No index changes, no REINDEX, no forced rewrite, and no major release needed.

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
| Why not the pgvector layout | An old binary reading a pgvector-layout payload can **silently mis-decode** it (measured: a crafted all-finite 447-d payload decodes to `[1]` in the v2.11.0 binary). With `0xFE` the old binary always fails at the first byte. §2.4. |
| Endianness | Fixed little-endian. It is native on every host we build or test (x86_64, aarch64, riscv64), so the zero-copy path is free there. A big-endian host falls back to the copying decoder, the same code that handles unaligned short-header datums. §2.5. |
| Alignment | `typalign` stays `'i'`; it cannot be changed (`typecmds.c:4282`). `f32` data sits at varlena offset 8. The reader still checks pointer alignment and copies if the pointer is unaligned (1-byte-header datums, dim ≤ 30). §3.1. |
| pgrx surface | Keep `#[derive(PostgresType)]` + `#[inoutfuncs]`, add `#[bikeshed_postgres_type_manually_impl_from_into_datum]`, and hand-write `FromDatum`/`IntoDatum`/`UnboxDatum`/`ArgAbi`/`BoxRet`. Generated SQL and C symbol names (`vector_in_wrapper`, `vector_out_wrapper`) are unchanged, so no catalog change. §3.2. |
| Zero-copy | Optional borrowed `VectorRef<'fcx>` argument type for the 8 read-only hot functions, delegating `SqlTranslatable` to `Vector` (the pgrx `PgVarlena<T>` pattern). Ship it after an A/B. §3.3. |
| Write path | `IntoDatum` is the only write point. GUC `turbovec.vector_write_format = cbor | raw`. Existing rows are never rewritten; `UPDATE t SET v = v` and `VACUUM FULL` do **not** re-encode (measured). §4. |
| Release type | **Minor ×2.** 2.12.0: reads both, writes CBOR by default (opt-in `raw`), adds `turbovec.vector_format(vector)`. 2.13.0: default flips to `raw`; 2.12.x is the safe downgrade floor. §5, §8. |
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

The two decodes per candidate exist because every distance function takes
`(a: Vector, b: Vector)` by value (`src/distance.rs:41, 57, 72, 95, 117,
132`), so the constant query is decoded again for each candidate. Fix B
removes the query-side decode; this design removes the candidate-side
decode.

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
sign to validate.

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
`pg_attrdef.adbin` `constvalue 17 [68 0 0 0 -95 100 100 97 116 97 …]`, where
−95 is `0xA1`).

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
- **Measured** in-database, v2.11.0 `.so`, raw payload injected through a
  test-only `WITHOUT FUNCTION` cast: `ERROR: failed to decode CBOR:
  ErrorImpl { code: UnassignedCode, offset: 1 }` from both `::text` and
  `vector_dims()`.

Independently, initial byte `0xFE` is major type 7 with additional-info 30,
which RFC 8949 reserves as not well-formed (**unverified** against the RFC
text). The proof above doesn't depend on it; it depends only on the one
decoder that has ever read these datums.

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
  prints `[1]` with `vector_dims = 1` (measured).

Pathological, but "a wrong vector, no error" is the corruption class the
HARD MANDATE forbids. A one-byte magic removes it at no size cost: total
size is identical to pgvector's.

### 2.5 Endianness

PostgreSQL data files are host-endian and not portable across byte order
(`xlog.c:4026` "mismatched byte ordering … initdb"). So host order would be
legal, and it is what pgvector does. We fix LE anyway:

- the index relfile already uses explicit LE everywhere (`to_le_bytes`
  ×130 in `src/index/page.rs`, no `to_ne_bytes`)
- committed golden fixtures (§7) become arch-independent bytes
- every supported host is LE, so zero-copy costs nothing there

On a big-endian target the reader always takes the copying path, which
decodes with `f32::from_le_bytes`. LE hosts exercise the same path for
unaligned short datums, so there is no BE-only code. Tests can't run on BE,
though (no BE CI); see §10.

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
   - **`0xFE`**: validate (§2.1). If `cfg!(target_endian = "little")` and
     `payload + 4` is 4-aligned → `Cow::Borrowed(from_raw_parts(...))`.
     Otherwise → `Cow::Owned` via `chunks_exact(4).map(f32::from_le_bytes)`.
     The alignment check is required for Rust soundness, not only for
     strict-alignment CPUs: an unaligned `&[f32]` is UB even on x86.
   - **`0xA1`**: `serde_cbor::from_slice::<Vector>` exactly as today →
     `Cow::Owned`. Same decoder and same derived impl, so the result is
     bit-identical to the old binary's by construction. Declare
     `serde_cbor = "=0.11.2"` **directly** in our `Cargo.toml`. Today it
     only arrives through pgrx (`pgrx-0.19.1/Cargo.toml:96-97`), and a pgrx
     bump must never be able to move the legacy decoder.
   - **anything else**: `ERROR: unrecognized turbovec.vector datum format
     (first byte 0x..)`.

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

Fallback if pgrx renames the `bikeshed_` attribute: compilation fails
(safe). We would then drop the derive and hand-write the `CREATE TYPE` with
`extension_sql!`, which must reproduce the same SQL. That costs more; see
§10.

### 3.3 Zero-copy argument type (optional, after an A/B)

`Vector` owns a `Vec<f32>`, so the raw path still costs one `memcpy`:
0.10 µs at 1024-d (measured), about 40× less than CBOR. For the 8 read-only
hot functions (`src/distance.rs:41, 57, 72, 95, 117, 132, 144, 158`), a
borrowed `VectorRef<'fcx>` holding the `Cow` from §3.1 also removes the
memcpy and a 4 KB palloc:

- `impl FromDatum` + `ArgAbi<'fcx>`, the same way pgrx borrows `&'fcx str`
  / `&'fcx [u8]` (`pgrx-0.19.1/src/callconv.rs:278`).
- `unsafe impl SqlTranslatable for VectorRef<'_>` delegating `TYPE_IDENT`,
  `ARGUMENT_SQL` and `RETURN_SQL` to `Vector`. This is the pattern pgrx
  itself uses for `PgVarlena<T>` (`varlena.rs:423-431`). The generated SQL
  should then be byte-identical. **Unverified;** acceptance gate: `cargo
  pgrx schema` output diffs empty against the previous release.

The index paths (`src/index/build.rs:2004`, `src/index/insert.rs:73`,
`src/index/scan.rs:435`) can call the primitive directly, since they copy
into a `Vec<f32>` anyway. Arrays (`Vec<Vector>`: `src/index/build.rs:1868`,
`src/colbert.rs:69`, `src/hybrid.rs:105, 123`) stay owned; they are rare.

Ship `VectorRef` only if an A/B at `search_k = 1024` shows a gain (the
expected saving is ~0.1–0.2 µs/candidate, estimate). It changes no format
and no SQL, so it is patch-eligible if the schema diff is empty.

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
    (`heapam.c:4148-4184`) rules out HOT, and **every index gets an
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
  byte 0, so it can use `pg_detoast_datum_slice(…, 0, 1)` and fetch one
  TOAST chunk instead of three (**unverified** that the slice path avoids
  the other chunks for uncompressed external values).

---

## 5. Compatibility matrix

### 5.1 Version combinations

| reader \ data | legacy CBOR rows | raw rows |
|---|---|---|
| ≤ 2.11.x binary | yes (today) | **ERROR**, `UnassignedCode, offset 1`, always (§2.3 claim 3); never a wrong value |
| 2.12.x binary (reader release) | yes, bit-identical | yes |
| 2.13.x+ binary | yes, forever | yes |

**Upgrade** (any 1.x/2.x → 2.12.0 / 2.13.0): in place, zero rewrite, no
REINDEX. This satisfies HARD MANDATE #2(b).

**Downgrade**, stated honestly: an old binary can't read raw rows. A
downgrade loses no data, because reinstalling the newer binary reads
everything again, but rows go unreadable until then. The two-step rollout
keeps one safe downgrade target at every point:

- 2.12.0 with default `cbor`: downgrade to ≤ 2.11 is safe if no session set
  `raw`. Check with `SELECT count(*) FROM t WHERE turbovec.vector_format(v)
  = 'raw'` = 0 for every vector column, plus views/defaults created by
  sessions writing `raw` (catalog `Const`s, §2.3).
- 2.13.0 with default `raw`: downgrade to 2.12.x is always safe.
  Downgrading to ≤ 2.11.x is unsafe once any row has been written. Setting
  `turbovec.vector_write_format = cbor` cluster-wide **before** writing
  keeps the 2.11 floor open.
- A 2.12.x → 2.11.x downgrade with raw rows present can be undone in place
  with the 2.12 binary still installed: the batched re-encode above under
  `SET turbovec.vector_write_format = cbor`. It costs what §4 says.

**Recommended release type: two minors, not a major.** No SQL is removed,
no format becomes unreadable, the upgrade needs no action, and a safe
downgrade target exists by construction. A major would buy nothing.
Collapsing both steps into one minor would make the first raw write after
upgrade a one-way door; that is the case against a single release.

### 5.2 Paths that never see the datum format

- **pg_dump / pg_restore** (plain, `-Fc`, `-Fd`) use text `COPY` through
  `vector_out` / `vector_in`. The text form stays byte-identical
  (`src/vec.rs:98-112` is unchanged; a fixture test pins it). A dump taken
  on any version restores on any version. Restoring into 2.13+ writes raw.
- **Logical replication** sends text unless `binary = true` *and* the type
  has `typsend` (`proto.c:836`). We have no `typsend`, so it is always text,
  so publisher and subscriber formats are independent in both directions.
- **COPY BINARY:** impossible today. **Measured:** `COPY t TO STDOUT
  (FORMAT binary)` → `ERROR: no binary output function available for type
  vector`. So this design changes no wire protocol.
- **Physical replicas** replay the primary's bytes and need the same `.so`.
  **Upgrade standbys first** (a 2.12+ standby reads both formats), then the
  primary. A ≤ 2.11 standby behind a raw-writing primary errors on those
  rows (fails closed).
- **pg_upgrade:** heap and TOAST files are copied or linked, so the result
  is a mixed (all-CBOR) cluster, which the new reader handles.
- **Catalog constants** (view/rule `Const`, column `DEFAULT`): CBOR until
  the DDL is re-run. Read fine; they count toward the downgrade check.
- **pg_statistic:** `vector` has no `=`/`<` operator, so ANALYZE uses
  `compute_trivial_stats` (`analyze.c:1937-1949`) and stores no
  `stavalues` (**unverified** for every PG 13–19 leg).
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

1. **Golden CBOR fixtures, committed before any format code.** Phase 0 is a
   separate test-only commit on the current tree. Fixtures
   `tests/fixtures/vector_cbor_v2/…` hold hex payload plus expected `f32`
   bit patterns for:
   - dims {1, 8, 23, 24, 29, 30, 31, 255, 256, 397, 398, 1024, 16000}
   - value classes {f16-exact (0, ±1, 0.5), −0.0, random f32, subnormals,
     ±f32::MAX, ±f32::MIN_POSITIVE, mixed}
   - each fixture's `vector_out` text

   A `#[test]` asserts `serde_cbor::to_vec` reproduces every fixture
   byte-for-byte (this pins the encoder against dependency drift). A
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
   - **reader fuzz:** arbitrary bytes behind a varlena header → `Ok` or
     `ERROR`, never a panic-abort or UB. Run the unit-test form under Miri
     if it builds (unverified)
4. **Round-trips:** raw encode → decode is bit-identical (property);
   CBOR fixture → decode → raw encode → decode is bit-identical; unknown
   version, `dim = 0`, `dim > MAX_DIM`, short or long length → `ERROR`.
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
7. **Alignment fuzz:** tables with leading `bool`/`text`/`smallint` columns
   of varying length before a dim 1..30 vector, so the short payload lands
   at every offset mod 4. Read through seq scan, index recheck, `vector[]`,
   and `datumCopy` (reorder queue). Plus a `#[test]` of the primitive at
   buffer offsets 0–3.
8. **Index is format-blind:** load the same rows in the same order into two
   tables, one per format, build flat / IVF / BQ indexes, and sha256 the
   relfile chains. Must be identical.
9. **Dump/restore:** dump a mixed table from 2.11.0 (CBOR) and from 2.12+
   (mixed), restore into both, and compare `vector_out` text and decoded
   bits. **COPY BINARY:** assert the current `ERROR` is unchanged (or
   round-trip, if §5.3 ships).
10. **Downgrade fail-closed, end to end** (EC2 or local, two builds): write
    raw rows with 2.12+, swap in the 2.11.0 `.so`, and assert every read is
    an `ERROR`, never a value. Swap back and assert all rows read correctly.
11. **Sustained-load soak** (pattern:
    `benches/results/tv111_arm_20261005/raw/soak2.py`), ≥ 90 min:
    - seed 60k rows half CBOR, half raw
    - writers randomly `SET` the write format per session, COPY-INSERT,
      `UPDATE SET tv = tv` (no re-encode), `UPDATE SET tv =
      tv::real[]::turbovec.vector` (re-encode), and DELETE
    - `VACUUM` every ~5 min; `pg_terminate_backend` mid-transaction
    - verify: every live row's decoded bits equal the deterministic
      generator's `vec(id)` (a content check that doesn't care about
      format); `turbovec_check` stays clean; soak2's byte-level
      index-vs-fresh-REINDEX comparison; **backend RSS flat** (AGENTS.md:
      watch RSS, not just correctness; borrowed views change palloc
      ownership)
    - run on arnold (AVX2) and Graviton `c8g` (planes path; AGENTS.md)

Any failure blocks the release. Code reasoning alone is not evidence here
(HARD MANDATE #1; v1.28.4).

---

## 8. Rollout

### 8.1 Phases

| phase | release | contents | SQL change | user action |
|---|---|---|---|---|
| 0 | next patch (2.11.x) or the first commit of phase 1 | golden CBOR fixtures + encoder-pin tests (§7.1); `serde_cbor = "=0.11.2"` declared directly | none | none |
| 1 | **minor 2.12.0** | dual-format reader; hand-written datum impls; GUC `turbovec.vector_write_format` default **`cbor`**; `turbovec.vector_format(vector)` | +1 function (`sql/pg_turbovec--2.11.x--2.12.0.sql` via `cargo pgrx schema`, `migrations/086_…`) | `ALTER EXTENSION … UPDATE` + restart |
| 1b | patch or minor, after an A/B | `VectorRef` zero-copy args (§3.3) | none (gate: empty schema diff) | none |
| 2 | **minor 2.13.0** | GUC default → **`raw`** | none (empty upgrade script + migration file) | upgrade standbys before the primary |
| 3 | ≥ two minors after 2.13.0 | optionally retire the `cbor` *writer* value (AGENTS.md two-release deprecation window); the CBOR *reader* is never removed | none | none |

### 8.2 `docs/UPGRADING.md` rows (draft text)

| From | To | Required action | Notes |
|---|---|---|---|
| 2.11.x | 2.12.0 | _none_ (no REINDEX, no rewrite) | MINOR. `turbovec.vector` gains a second on-disk datum format (raw `float4`, tagged `0xFE`). This release **reads** it but writes the old CBOR format by default; `SET turbovec.vector_write_format = raw` opts in. New `turbovec.vector_format(vector)` reports `cbor`/`raw` per value. Index wire format unchanged (v8). **Downgrade to 2.11.x is safe only while no `raw` values exist** (`SELECT count(*) … WHERE turbovec.vector_format(col) = 'raw'` = 0 for every vector column); 2.11.x reading a raw value raises `failed to decode CBOR … UnassignedCode, offset 1`, never a wrong result. `ALTER EXTENSION pg_turbovec UPDATE TO '2.12.0';` + restart. |
| 2.12.x | 2.13.0 | _none_ | MINOR. New and updated values are written raw by default (about 4 µs less CPU per rechecked candidate at 1024-d for raw rows; estimate). Existing rows stay CBOR, are read forever, and are never rewritten (`UPDATE t SET v = v` does not convert them). **Downgrade floor is 2.12.0.** On physical replication, upgrade standbys before the primary. `SET turbovec.vector_write_format = cbor` keeps writing the old format. `ALTER EXTENSION pg_turbovec UPDATE TO '2.13.0';` + restart. |

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

Each later type goes through the same two-minor reader-then-writer sequence
with its own fixtures. `vector` goes first because it is the indexed hot
path and the only one with a measured cost.

---

## 9. Expected gain

- **Decode.** FINDINGS measured two decodes per candidate at 8.1 µs. Fix B
  leaves one (~4 µs, estimate: half of 8.1). This design takes the
  remaining candidate decode to ~0.1 µs (owned copy) or ~0 (`VectorRef`).
  Estimate: **−3.7 to −4 µs per rechecked candidate, ~4 ms/query backend
  CPU at `search_k = 1024`, 1024-d.** Basis: 3.73 µs CBOR vs 0.08 µs copy
  (`cbor_bench.rs`). Composed with A+B+C, FINDINGS projects ~6 → ~2 µs per
  candidate (estimate; not measured end to end).
- **Raw rows only.** A table loaded before the flip and never updated keeps
  paying CBOR on every legacy row. An optional mitigation, deferred: a
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
3. **`VectorRef` schema identity** (§3.3) is unverified; gate it on an
   empty `cargo pgrx schema` diff.
4. **serde_cbor 0.11.2 is unmaintained** (RUSTSEC-2021-0127,
   informational) and we need it forever. Pin it directly. Vendoring it, or
   replacing it with a fixture-validated hand decoder, is a later option.
5. **Downgrade depends on operators:** physical standbys must be upgraded
   first, and a cluster-wide `cbor` setting is the only way to keep a ≤ 2.11
   floor after 2.13.0. Docs must say this prominently.
6. **A `Userset` GUC means any session can write raw** in 2.12.0 and close
   the ≤ 2.11 downgrade for those rows. Accepted: §11b requires `Userset`,
   and correctness on 2.12+ is unaffected. `vector_format()` makes it
   visible.
7. **Compression:** structured raw floats (many zeros, quantized embeddings)
   may now pglz-compress under the default `EXTENDED` storage, making every
   read a decompress. Unmeasured. pgvector uses `STORAGE external`, but
   changing ours affects only new columns and is a separate SQL decision.
8. **Big-endian** is untested (no BE CI). The copying path is shared with LE
   short datums, but it has never run on BE. Option: refuse BE builds with
   `compile_error!`.
9. **`ANALYZE` stores no vector `stavalues`:** verified by reading
   `analyze.c` (PG16) only, not on every PG 13–19 leg.
10. **A future `=`/btree/hash opclass for `vector`** must compare decoded
    values. With two formats, byte equality is no longer value equality.
11. **The `0xFE` RFC 8949 "not well-formed" status** is unverified (§2.3).
    The proof rests on serde_cbor 0.11.2 code plus measurement only.
12. **Fix B's per-scan query cache stays useful** for CBOR queries. With raw
    queries it saves only a memcpy. Keep it; it costs nothing.

---

## Appendix: probes behind "measured here" (2026-10-06)

Host floki (Intel Core Ultra 7 258V, AVX2), PG 16.15 from
`~/.pgrx/16.15/pgrx-install` with the installed pg_turbovec **2.11.0**
`.so`, on a **private throwaway cluster** (own `initdb`, port 54999, socket
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
