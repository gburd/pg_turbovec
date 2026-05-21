//! `pg_turbovec` — a PostgreSQL extension providing a vector type and an
//! approximate nearest-neighbour index access method backed by the
//! [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm via the
//! [`turbovec`](https://crates.io/crates/turbovec) crate.
//!
//! The public SQL surface mirrors `pgvector` so that existing
//! applications and ORMs work with minimal changes:
//!
//! - The `tvector` type (variable dimension `f32` vectors).
//! - Distance operators: `<->` (L2), `<#>` (negative inner product),
//!   `<=>` (cosine distance), `<+>` (L1).
//! - Helper functions: `l2_distance`, `inner_product`, `cosine_distance`,
//!   `l1_distance`, `vector_dims`, `vector_norm`.
//! - Aggregates: `avg(tvector)`, `sum(tvector)`.
//! - Index access method `turbovec` with operator classes
//!   `tvector_ip_ops` and `tvector_cosine_ops`.
//!
//! See `docs/ARCHITECTURE.md` for the full design.

use pgrx::prelude::*;

mod aggregate;
mod distance;
mod guc;
mod tvector;

pgrx::pg_module_magic!();

/// Extension initialization — called when the shared library is loaded.
#[allow(non_snake_case)]
#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    guc::register_gucs();
}

/// Returns the version string for the extension.
#[pg_extern]
fn turbovec_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_turbovec_version() {
        let v: Option<String> = Spi::get_one("SELECT turbovec.turbovec_version()").unwrap();
        assert_eq!(v.as_deref(), Some(env!("CARGO_PKG_VERSION")));
    }
}

/// PGRX test runner harness.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
