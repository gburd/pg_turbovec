-- pg_turbovec upgrade 2.8.4 -> 2.9.0
-- Run automatically by `ALTER EXTENSION pg_turbovec UPDATE TO '2.9.0';`.
--
-- Phase Z5 (reporting): adds ONE function. Strictly additive -- no existing
-- object changes, no index wire-format change (stays v8), so existing
-- indexes decode byte-identically and NO REINDEX is required.
--
-- Quantifies an IVF degradation instead of merely flagging it. Phase Z1
-- made it observable (`index_is_degraded`) and Phase Z4 made the planner
-- cost it, but neither told an operator the SIZE of the problem -- which is
-- what decides whether to act. A degraded 10k-row index is a non-event; a
-- degraded 10M-row index is an outage.
--
-- Reads only the meta page (one buffer hit), no chain scan, so it is safe
-- to poll from monitoring.

CREATE FUNCTION turbovec."index_degradation"(
	"index" oid
) RETURNS TABLE (
	"degraded" bool,
	"lists" INT,
	"n_vectors" bigint,
	"scan_fraction" double precision,
	"est_slowdown" double precision,
	"recovery" TEXT
)
STRICT STABLE PARALLEL SAFE
LANGUAGE c
AS 'MODULE_PATHNAME', 'index_degradation_wrapper';
