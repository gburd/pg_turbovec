-- pg_turbovec upgrade 1.28.3 -> 1.28.4
-- Adds the read-only, ownership-checked index integrity function.
-- Run automatically by `ALTER EXTENSION pg_turbovec UPDATE TO '1.28.4';`.
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
	"tombstone_density" double precision
)
STRICT STABLE PARALLEL SAFE
LANGUAGE c
AS 'MODULE_PATHNAME', 'turbovec_check_wrapper';
