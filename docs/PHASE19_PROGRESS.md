# Phase 19 progress - binary-compatible varlena layout

**Branch base:** 65a3f15 (Phase 18 forced-index-scan fix landed; main test count 40/40).
**Status at handoff:** Investigation/planning only. **No code changes committed.**

## Why nothing is committed

The agent session that picked up this task hit a tool-harness failure
that prevented it from issuing write calls with the full multi-line
src/vec.rs payload (small smoke writes succeeded; the full file
did not). After the user requested an early commit, the only safe
action was to record the investigation here and exit. Re-running the
task starts from a clean main; this file is the only delta.

## Target on-disk layout (confirmed pgvector-compatible)

    typedef struct Vector {
        int32 vl_len_;       /* varlena header (SET_VARSIZE_4B) */
        int16 dim;           /* 1..16000                        */
        int16 unused;        /* always 0                        */
        float x[FLEXIBLE_ARRAY_MEMBER];
    } Vector;

Total = 8 + 4*dim bytes. alignment = double, storage = extended.
With typalign='d' Postgres will not emit a 1-byte short varlena
header (short headers violate double alignment), so a 4-byte header
is guaranteed on disk. Defensive code should still call
pg_detoast_datum (not _packed) on the way in.

## pgrx 0.17 trait surface for a hand-rolled type

#[derive(PostgresType)] cannot be kept once storage stops being CBOR.
The replacement set of impls -- cribbed from
pgrx-macros-0.17.0/src/lib.rs around line 845 (the
PostgresTypeAttribute::ManualFromIntoDatum branch) and from
pgrx-0.17.0/src/datum/varlena.rs (PgVarlena<T>) -- is:

1. impl PostgresType for Vec {} -- marker trait
   (pgrx::datum::PostgresType).
2. impl IntoDatum for Vec -- into_datum(self) returns
   Some(Datum::from(self.0)); type_oid() calls
   pgrx::wrappers::rust_regtypein::<Self>(). That helper strips
   everything before the last :: in std::any::type_name::<T>(), so
   the SQL type MUST be named exactly Vec (case-insensitive in
   unquoted SQL -- matches the existing CREATE TYPE Vec; in the
   generated pgrx schema).
3. impl FromDatum for Vec --
   from_polymorphic_datum: detoast (pg_sys::pg_detoast_datum), read
   dim from offset 4, validate 1..=MAX_DIM, wrap.
   from_datum_in_memory_context: switch context, then
   pg_detoast_datum_copy, then from_datum.
4. unsafe impl UnboxDatum for Vec with type As<'dat> = Self where
   Self: 'dat -- delegate to FromDatum.
5. unsafe impl<'fcx> ArgAbi<'fcx> for Vec -- call
   arg.unbox_arg_using_from_datum().unwrap_or_else(|| panic!(...)).
6. unsafe impl BoxRet for Vec -- match IntoDatum::into_datum,
   return null or fcinfo.return_raw_datum(datum).
7. unsafe impl SqlTranslatable for Vec --
   argument_sql() = SqlMapping::As("Vec".into()),
   return_sql() = Returns::One(SqlMapping::As("Vec".into())).

## CREATE TYPE wiring

Two extension_sql! blocks, mirroring the docstring example near
pgrx-macros-0.17.0/src/lib.rs:400 for Complex:

    extension_sql!(
        "CREATE TYPE Vec;",
        name = "concrete_tvector_shell",
        creates = [Type(Vec)],
    );

    // ...pg_extern fns tvector_in/_out/_recv/_send...

    extension_sql!(
        r#"CREATE TYPE Vec (
            INPUT          = tvector_in,
            OUTPUT         = tvector_out,
            RECEIVE        = tvector_recv,
            SEND           = tvector_send,
            INTERNALLENGTH = variable,
            ALIGNMENT      = double,
            STORAGE        = extended
        );"#,
        name = "concrete_tvector",
        requires = ["concrete_tvector_shell",
                    tvector_in, tvector_out, tvector_recv, tvector_send],
    );

Existing call sites in cast.rs / distance.rs / aggregate.rs that have
requires = [Vec, ...] keep working because the shell entity is
registered under the Rust path Vec via creates = [Type(Vec)].

## Wire format for tvector_send / tvector_recv

Matches pgvector exactly:

  - pq_sendint(&buf, dim, 2)  -- i16 dim NBO
  - pq_sendint(&buf, 0,   2)  -- i16 unused NBO
  - dim * pq_sendfloat4(&buf, x[i])  -- f32 NBO

Easiest pgrx-side spelling for send:

    #[pg_extern(immutable, parallel_safe)]
    fn tvector_send(v: Vec) -> Vec<u8> {
        let n = v.dim();
        let mut out = Vec::with_capacity(4 + 4 * n);
        out.extend_from_slice(&(n as i16).to_be_bytes());
        out.extend_from_slice(&0i16.to_be_bytes());
        for f in v.as_slice() { out.extend_from_slice(&f.to_be_bytes()); }
        out
    }

tvector_recv takes pgrx::Internal (the SQL internal cookie that holds
a StringInfo), reaches the inner pg_sys::Datum, casts to
pg_sys::StringInfo, then drives pq_getmsgint(buf, 2) twice and
pq_getmsgfloat4 dim times.

## Files that touch Vec::data or Vec::from_vec directly

(grep over the clean 65a3f15 tree)

  src/vec.rs                         -- full rewrite (the work)
  src/cast.rs                            -- vec_to_array reads v.data;
                                            switch to v.as_slice().to_vec()
  src/distance.rs                        -- Vec::from_vec only
  src/aggregate.rs                       -- Vec::from_vec only
  src/normalize.rs                       -- Vec::from_vec only
  src/extras.rs                          -- Vec::from_vec only
  src/knn.rs                             -- query.as_slice() only
  src/index/{build,insert,scan}.rs       -- pgrx::FromDatum::from_datum
                                            only; works with new impl

Net: the only required ripple is src/cast.rs::vec_to_array's
v.data access. Every other module talks to the type through
as_slice(), dim(), from_vec(); keep that surface stable.

## Risks the next agent should keep in mind

1. VecAccum is also #[derive(PostgresType)] (CBOR). It is
   intentionally NOT being migrated -- internal aggregate state, never
   appears in user tables, CBOR is fine. Do not let "expand all
   PostgresType derives" temptation lead you to rewrite it.
2. Do NOT impl Drop for Vec. The wrapped pointer is palloc'd in a
   Postgres memory context; freeing on Rust drop will corrupt the
   backend.
3. Drop the derive(Clone, PartialEq) on Vec. A grep
   (\.clone\(\)|Vec\b.*==|PartialEq) shows zero call sites outside
   the derive itself.
4. Test cluster on-disk format break: first cargo pgrx test pg16 after
   this lands must be preceded by cargo pgrx stop pg16, then DROP
   DATABASE pgrx_tests (or wipe ~/.pgrx/data-16/pgrx_tests via your
   trash tool). Cached CBOR varlenas will be misread and crash the
   backend on first SELECT.
5. vl_len_ is encoded. Never read it directly. Use varsize_any /
   varsize_any_exhdr / vardata_any. The struct field is only for the
   SET_VARSIZE_4B write path.
6. The forced-index-scan test (index_am_forced_index_scan) was un-
   #[ignore]d in 65a3f15 and is the 40th test. Make sure the new
   vector handling does not regress it -- amrescan copies ScanKey
   datums and invokes the new FromDatum impl on the order-by argument.

## Suggested order of operations next pass

1. Replace src/vec.rs with the new layout + traits + in/out/recv/send.
2. Patch src/cast.rs::vec_to_array (v.data -> v.as_slice().to_vec()).
3. cargo pgrx stop pg16 ; clear out the pgrx_tests DB once.
4. cargo pgrx test pg16 -- fix compile errors module by module.
5. Add a binary_compatible_with_pgvector_wire_format #[pg_test] that
   round-trips '[1,2,3]'::vector through tvector_send / tvector_recv
   and asserts the sent bytes are
   00 03 00 00 3f 80 00 00 40 00 00 00 40 40 00 00. Skip the actual
   pgvector.vector interop unless the test cluster has vector
   installed -- leave a #[pg_test] #[ignore = "needs pgvector"] stub.
6. Bump Cargo.toml / pg_turbovec.control to 1.1.0-binary-layout and
   add the matching CHANGELOG entry calling out
   **Breaking on-disk format change. Dump and restore via text I/O.**

## Why no partial commit

Anything less than steps 1-4 leaves cargo pgrx test pg16 red, which is
worse than the clean tree on 65a3f15. This document is the deliverable
for this aborted session; commit it and let the next agent start from
a known-good base.
