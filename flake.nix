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
        # pins pgrx 0.19.1 (the first line with a pg19 feature), so we
        # build the matching cargo-pgrx here with the same `generic`
        # shape nixpkgs uses (pkgs/development/tools/rust/cargo-pgrx).
        cargo-pgrx_0_19_1 = pkgs.rustPlatform.buildRustPackage rec {
          pname = "cargo-pgrx";
          version = "0.19.1";
          src = pkgs.fetchCrate {
            inherit pname version;
            hash = "sha256-D4rkD+Koetd5dc91RxO+R1v2km/DVb4HMhowK2hdyNY=";
          };
          cargoHash = "sha256-qUCQRbzA4me1lkNNO2kQtW4DOjiHTTrhmUnLkJ91QrY=";
          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = [ pkgs.openssl ];
          # cargo-pgrx's test suite wants a pgrx-init'd home; skip.
          doCheck = false;
          meta.mainProgram = "cargo-pgrx";
        };

        # The turbovec kernel is pulled from a git rev (Cargo.lock
        # [[package]] source = "git+https://github.com/gburd/turbovec.git?rev=..."),
        # so cargo vendoring needs its output hash pinned.
        cargoLock = {
          lockFile = ./Cargo.lock;
          outputHashes = {
            "turbovec-0.9.0" = "sha256-Karmjgfv4rops9TiJAjOgJkPkgGI7wyWHvRyWtVws7k=";
          };
        };

        # One buildPgrxExtension invocation per supported PG major.
        # `pg_test` stays off; the #[pg_test] suite needs a live
        # cluster and is CI's job, not the package build's.
        mkTurbovec =
          postgresql:
          pkgs.buildPgrxExtension {
            pname = "pg_turbovec";
            version = "1.29.0";
            src = self;
            inherit postgresql cargoLock;
            cargo-pgrx = cargo-pgrx_0_19_1;
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
            cargo-pgrx_0_19_1
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
