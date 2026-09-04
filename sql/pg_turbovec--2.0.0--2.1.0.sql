-- pg_turbovec upgrade 2.0.0 -> 2.1.0 (MINOR)
--
-- The graph-kind correctness fixes (C1 concurrent-insert corruption + the
-- silent lost update, BUG#2 short returns, FINDING#2 un-cancellable build)
-- and the BUG#6 upstream-limitation documentation are all BINARY-side and
-- need no SQL.
--
-- The ONE SQL-surface change: turbovec_check() gains a trailing
-- `reason text` OUT column, which CHANGES the function's return type.
-- `CREATE OR REPLACE FUNCTION` cannot change a return type, so the old
-- function is DROPped and re-CREATEd here. That signature change is also
-- why 2.1.0 is a MINOR rather than a patch.
--
-- No wire-format change (stays v8). NO REINDEX required.

DROP FUNCTION IF EXISTS turbovec."turbovec_check"(oid);

CREATE FUNCTION turbovec."turbovec_check"(
	"index" oid
) RETURNS TABLE (
	"wire_version" INT,
	"kind" TEXT,
	"n_vectors" bigint,
	"slot_count" bigint,
	"count_matches" bool,
	"duplicate_id" bigint,
	"is_corrupt" bool,
	"tombstone_density" double precision,
	"reason" TEXT
)
STRICT STABLE PARALLEL SAFE
LANGUAGE c
AS 'MODULE_PATHNAME', 'turbovec_check_wrapper';
