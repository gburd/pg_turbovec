# Building pg_turbovec from source on NixOS / nix-using systems

This is what worked on the dev machine; adapt to your environment as
necessary.

## 1. Prerequisites

- A Rust toolchain >= 1.85.
- Nix-installed PostgreSQL 16 (we'll copy it to a writable
  location).
- libclang (for `pgrx-pg-sys` bindgen).
- libopenblas + openblas-dev headers (for `turbovec`).
- pkg-config + libssl + libssl-dev (for `cargo install
  cargo-pgrx`).

## 2. Build cargo-pgrx

```bash
SSLDIR=$(find /nix/store -maxdepth 3 -name 'openssl-3*-dev' -type d | head -1)
SSLDIR=${SSLDIR%-dev}
SSLDEV=${SSLDIR}-dev
OPENSSL_LIB_DIR=$SSLDIR/lib OPENSSL_INCLUDE_DIR=$SSLDEV/include OPENSSL_NO_VENDOR=1 \
    cargo install cargo-pgrx --version 0.17.0
```

## 3. Set up a writable PG16 install

The system pg_config lives in a Nix store path that's read-only;
`cargo pgrx install` would fail trying to copy the .control file
into it. We copy the install tree to a writable location and wrap
`pg_config` to rewrite its hard-coded paths:

```bash
PG_RO=$(find /nix/store -maxdepth 3 -name 'postgresql-16.9' -type d | head -1)
PG_RO_DEV=$(find /nix/store -maxdepth 3 -name 'postgresql-16.9-dev' -type d | head -1)
PG_RW=$HOME/.pgrx/install-pg16

mkdir -p $PG_RW
cp -rfL $PG_RO/. $PG_RW/.        && chmod -R u+w $PG_RW
cp -rfL $PG_RO_DEV/bin/. $PG_RW/bin/.        && chmod -R u+w $PG_RW/bin
mkdir -p $PG_RW/include
cp -rfL $PG_RO_DEV/include/. $PG_RW/include/. && chmod -R u+w $PG_RW/include
cp -rfL $PG_RO_DEV/lib/. $PG_RW/lib/.        && chmod -R u+w $PG_RW/lib

# Wrap pg_config so its --pkglibdir / --sharedir / --bindir all
# point at the writable copy.
mv $PG_RW/bin/pg_config $PG_RW/bin/pg_config.real
cat > $PG_RW/bin/pg_config <<WRAP
#!/bin/sh
"$PG_RW/bin/pg_config.real" "\$@" \
    | sed "s|$PG_RO_DEV|$PG_RW|g; s|$PG_RO|$PG_RW|g"
WRAP
chmod +x $PG_RW/bin/pg_config
```

Then point pgrx at it:

```bash
mkdir -p $HOME/.pgrx
cat > $HOME/.pgrx/config.toml <<EOF
[configs]
pg16 = "$PG_RW/bin/pg_config"
EOF
```

## 4. Initialise the per-version data directory

```bash
cargo pgrx init --pg16 $PG_RW/bin/pg_config
```

## 5. Build & test

```bash
cd ~/ws/pg_turbovec
export LIBCLANG_PATH=$(find /nix/store -maxdepth 3 -name 'clang-*-lib' -type d | head -1)/lib
GLIBC_INC=$(find /nix/store -maxdepth 3 -name 'glibc-*-dev' -type d | head -1)/include
CLANG_INC=$LIBCLANG_PATH/clang/$(ls $LIBCLANG_PATH/clang | head -1)/include
export BINDGEN_EXTRA_CLANG_ARGS="-isystem $GLIBC_INC -isystem $CLANG_INC"
export RUSTFLAGS="-L $(find /nix/store -maxdepth 3 -name 'openblas-0.3.30' -type d | head -1)/lib"

# Default-feature build (type, ops, knn(), aggregates, casts).
cargo build --no-default-features --features pg16
cargo pgrx test pg16

# Including the experimental index access method.
cargo build --no-default-features --features pg16,experimental_index_am
cargo pgrx test pg16 --features experimental_index_am
```

## 6. Verified outcomes (this session)

```
default features:           28 passed; 0 failed
experimental_index_am:      28 passed; 0 failed   (includes
                                                   index_am_create_and_query)
```

## 7. Why so much manual setup?

Because `cargo pgrx init download` requires ICU + a writable
extension dir + ~10 minutes to compile Postgres from source, and
the local machine already has a working PG16 install. The wrapper
around `pg_config` is the cleanest way to redirect the read-only
Nix store paths to a writable copy without rebuilding anything.

For a non-Nix system (Ubuntu, Fedora, macOS Homebrew) none of this
gymnastics is needed:

```bash
cargo install cargo-pgrx --version 0.17.0
cargo pgrx init                      # uses system pg_config
cargo pgrx test pg16
```
