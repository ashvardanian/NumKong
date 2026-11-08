#!/usr/bin/env bash
# Writes the symbol bundle unum.cloud renders: each binding's own tool output, or its sources where no tool emits one.
#
#     .github/scripts/build-docs.sh        # → build/numkong_docs_<version>.tar.gz
set -euo pipefail
version=$(cat VERSION)
name=numkong_docs_$version out=build/numkong_docs_$version
# Rustdoc's JSON is unstable, and `rust-version` needs 1.100, which reaches stable on 2026-11-12.
toolchain=${NUMKONG_RUST_TOOLCHAIN:-beta-2026-10-18}
rm -rf "$out" && mkdir -p "$out"/{doxygen,headers,python,rust,swift,javascript,go} build/docs-scratch/library

rm -rf build/xml && (cd docs && doxygen conf.dox) && cp -r build/xml/. "$out/doxygen"
git ls-files ':(glob)c/numkong/**/*.h' ':(glob)cpp/numkong/**/*.hpp' ':(glob)include/numkong/*.h' \
    ':(glob)include/numkong/*.hpp' | xargs cp --parents -t "$out/headers"
griffe dump numkong --search python --no-inspection --docstyle google --full --output "$out/python/numkong.json"
cp python/numkong/__init__.pyi "$out/python/"
NUMKONG_LIBRARY_DIR=$PWD/build/docs-scratch/library RUSTC_BOOTSTRAP=1 \
    cargo +"$toolchain" rustdoc --lib -- -Z unstable-options --output-format json
cp "${CARGO_TARGET_DIR:-target}/doc/numkong.json" "$out/rust/"
swiftc -emit-module -emit-module-path build/docs-scratch/NumKong.swiftmodule -module-name NumKong \
    -package-name NumKong -parse-as-library -I c -I include \
    -emit-symbol-graph -emit-symbol-graph-dir "$out/swift" swift/*.swift
cp javascript/*.ts "$out/javascript/"
cp golang/*.go go.mod "$out/go/" && rm -f "$out"/go/*_test.go
# griffe, rustdoc and Doxygen's body locations hold absolute paths, which would leak the build machine's directories.
sed -i "s|$PWD/||g; s|$PWD|.|g; s|$HOME|~|g" "$out/python/numkong.json" "$out/rust/numkong.json" "$out"/doxygen/*.xml
printf '{"project":"numkong","version":"%s","commit":"%s"}\n' "$version" "$(git rev-parse HEAD)" > "$out/manifest.json"
tar -czf "build/$name.tar.gz" -C build "$name"
