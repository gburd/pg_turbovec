{
  description = "pg_turbovec — PostgreSQL vector index backed by the TurboQuant quantizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;

        # nixpkgs' pinned cargo-pgrx set stops at 0.18.x; pg_turbovec
        # pins pgrx 0.19.3 (pg19 support; the 0.19 line), so we
        # build the matching cargo-pgrx here with the same `generic`
        # shape nixpkgs uses (pkgs/development/tools/rust/cargo-pgrx).
        cargo-pgrx_0_19 = pkgs.rustPlatform.buildRustPackage rec {
          pname = "cargo-pgrx";
          version = "0.19.3";
          src = pkgs.fetchCrate {
            inherit pname version;
            hash = "sha256-TyViMU+ZLUHNK5CRi8CN3tThFNUJQnuWTeCxT2JTYG0=";
          };
          cargoHash = "sha256-43B3zl/mJXuBD/w8x+lek7Wtg/lZ/bZWuJdv5v9ImSE=";
          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = [ pkgs.openssl ];
          # cargo-pgrx's test suite wants a pgrx-init'd home; skip.
          doCheck = false;
          meta.mainProgram = "cargo-pgrx";
        };

        # The turbovec kernel is pulled from a git rev (Cargo.lock
        # [[package]] source = "git+https://github.com/gburd/turbovec.git?rev=..."),
        # so cargo vendoring needs its output hash pinned. Nix keys these by
        # <crate>-<version> from Cargo.lock: when turbovec's version changes,
        # the key must change with it (a stale key fails vendoring with "A hash
        # was specified for turbovec-X, but there is no corresponding git
        # dependency"). nixpkgs' rustc links LLVM 21; turbovec's fork carry #5
        # makes that build (inline-asm vpdpbusd), so no toolchain override.
        cargoLock = {
          lockFile = ./Cargo.lock;
          outputHashes = {
            "turbovec-1.1.2" = "sha256-6zde/GqNtf328IP6iqKfXSlQ4mrGo3KvmVfK4clxsCQ=";
          };
        };

        # One buildPgrxExtension invocation per supported PG major.
        # `pg_test` stays off; the #[pg_test] suite needs a live
        # cluster and is CI's job, not the package build's.
        mkTurbovec =
          postgresql:
          pkgs.buildPgrxExtension {
            pname = "pg_turbovec";
            version = "2.14.1";
            src = self;
            inherit postgresql cargoLock;
            cargo-pgrx = cargo-pgrx_0_19;
            buildFeatures = [ "pg${lib.versions.major postgresql.version}" ];
            buildNoDefaultFeatures = true;
            # turbovec links BLAS for the k-means GEMM path.
            buildInputs = [ pkgs.openblas ];
            doCheck = false;
            meta = {
              description = "PostgreSQL vector index backed by the TurboQuant quantizer";
              homepage = "https://codeberg.org/gregburd/pg_turbovec";
              license = lib.licenses.asl20;
            };
          };
      in
      {
        packages = {
          pg_turbovec_13 = mkTurbovec pkgs.postgresql_13;
          pg_turbovec_14 = mkTurbovec pkgs.postgresql_14;
          pg_turbovec_15 = mkTurbovec pkgs.postgresql_15;
          pg_turbovec_16 = mkTurbovec pkgs.postgresql_16;
          pg_turbovec_17 = mkTurbovec pkgs.postgresql_17;
          pg_turbovec_18 = mkTurbovec pkgs.postgresql_18;
          # PG19 is upstream beta; experimental, mirrors the Cargo pg19 feature.
          pg_turbovec_19 = mkTurbovec pkgs.postgresql_19;
          # `nix build` default: the current GA-latest PostgreSQL.
          default = mkTurbovec pkgs.postgresql_18;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustc
            cargo
            rustfmt
            clippy
            cargo-pgrx_0_19
            pkg-config
            openssl
            openblas
            libclang.lib
            postgresql_18
            bison
            flex
            readline
            zlib
            icu
          ];
          LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
        };
      }
    );
}
