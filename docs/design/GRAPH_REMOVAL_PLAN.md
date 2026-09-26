# Graph kind removal — shape (a), MINOR 2.11.0 (from Plan subagent 2026-09-26)

DECISION: remove BUILD path only. Keep decode/scan/insert/vacuum so existing
KIND_GRAPH indexes stay readable+writable (NO REINDEX). Wire stays v8. MINOR bump.
Rationale: graph indexes have no in-place converter (Vamana adjacency has no
flat/IVF target), so a minor cannot strand them per the HARD MANDATE.

PRESERVE (never touch): turbovec.coarse_graph + ivf::CentroidGraph/build_centroid_graph/
graph_probe/GRAPH_MIN_LISTS/GRAPH_EF_* (IVF centroid nav); pack::repack;
turbovec.graph_ef (surviving graph scan uses it); all wire/running-sum code
(graph_count sums at page.rs 827/886/2158/3212, relfile.rs 886 — DO NOT touch).

REMOVE at shape (a): build.rs graph path (graph_build_and_write + state.graph
branches), turbovec.graph_build_partitions GUC, options.rs WARNING->ERROR on
graph=true, build-side graph.rs symbols cargo flags dead, build-path tests.

Build-rejection ERROR text names 2.11.0 + points at flat/IVF + "existing graph
indexes remain readable; REINDEX at your convenience".

Test-only graph writer needed: keep graph::build_vamana test-gated + call
relfile::write_full_with_prepared_graph directly so existing-graph-index
read/insert/vacuum tests survive (the MANDATE's fail-before/pass-after read guard).

Full checklist in the subagent transcript; version touch-list standard + migration
085 + sql/pg_turbovec--2.10.3--2.11.0.sql. graph_ef "retained technique" framing
in AGENTS.md is misleading — it's retained because the graph READ path is, not as
a reusable technique; correct that wording.
