-- pg_turbovec upgrade 2.4.0 -> 2.5.0
-- Run automatically by `ALTER EXTENSION pg_turbovec UPDATE TO '2.5.0';`.
--
-- Phase S-1: partition-level coarse quantizer for partition pruning.
-- Adds a derived catalog table plus two functions. This is ADDITIVE: no
-- existing object changes, no index wire-format change (stays v8), no
-- REINDEX. The graph-kind deprecation in this release is a WARNING emitted
-- by the reloption validator and needs no SQL.

-- Per-partition summary catalog: partition means in the ORIGINAL
-- (un-rotated) vector space, so they are comparable ACROSS partitions.
-- Derived data, not wire format -- safe to TRUNCATE and rebuild at any
-- time via turbovec.refresh_partition_summary().
CREATE TABLE turbovec.partition_summary (
	parent    regclass NOT NULL,
	partition regclass NOT NULL,
	centroid  turbovec.vector NOT NULL,
	PRIMARY KEY (parent, partition)
);

-- (Re)compute the summary for every non-empty partition of `parent`.
-- VOLATILE: it writes turbovec.partition_summary.
CREATE FUNCTION turbovec."refresh_partition_summary"(
	"parent" oid,
	"vec_col" TEXT
) RETURNS bigint
STRICT VOLATILE
LANGUAGE c
AS 'MODULE_PATHNAME', 'refresh_partition_summary_wrapper';

-- Return the `k_partitions` partitions whose summaries are nearest the
-- query, nearest first. Fan a kNN query out to ONLY these partitions
-- instead of all N. `metric` must match the operator the kNN uses.
CREATE FUNCTION turbovec."nearest_partitions"(
	"parent" oid,
	"query" turbovec.vector,
	"k_partitions" INT,
	"metric" TEXT DEFAULT '<=>'
) RETURNS SETOF oid
STRICT STABLE PARALLEL SAFE
LANGUAGE c
AS 'MODULE_PATHNAME', 'nearest_partitions_wrapper';
