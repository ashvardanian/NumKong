# NumKong Precision Tests

Custom test framework comparing NumKong kernels against high-precision `f118_t` double-double references.
Every kernel family has its own precision model with ULP-based error analysis.

## C++

No GTest dependency — the framework is self-contained in `harness.hpp`.

### Building

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D NUMKONG_BUILD_TEST=1
cmake --build build_release --config Release --parallel
build_release/numkong_cpu_test
```

To compile with BLAS cross-validation:

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D NUMKONG_BUILD_TEST=1 -D NUMKONG_COMPARE_TO_BLAS=1
```

Compiler requirements vary by ISA target — see [CONTRIBUTING.md](../CONTRIBUTING.md#compiler-requirements) for the full table.

The GPU suites build with their backend, and the `cuda`, `rocm`, and `metal` presets run them through `cmake --workflow --preset <name>`.
The CUDA suite includes tensor transfers and exhaustive FP6/FP8 conversion checks against CUDA intrinsics.
Building it needs `nvcc` on `PATH`:

```sh
cmake -B build_cuda -D NUMKONG_BUILD_TEST=ON -D NUMKONG_CUDA_ARCH=native
cmake --build build_cuda
ctest --test-dir build_cuda -R '^numkong_cpu(_shared)?_test$' --output-on-failure
```

### Running

```sh
NUMKONG_FILTER=dot build_release/numkong_cpu_test             # run only the cases whose labels match "dot"
NUMKONG_FILTER="dot|spatial" build_release/numkong_cpu_test   # regex filter
NUMKONG_ASSERT=0 build_release/numkong_cpu_test               # exit 0 even when a kernel fails its accuracy check
build_release/numkong_cpu_test --time-limit 5s                # 5 seconds per case, as a flag
```

Each variable has a `--name` flag of the same name, which overrides it.
The linked runner uses device zero of each available compiled GPU backend by default.
An explicit device list selects all listed devices and rejects unavailable backends or ordinals.

### Environment Variables

[CONTRIBUTING.md](../CONTRIBUTING.md#environment) lists every variable the suites read, with its flag, default and meaning.
[Running](../CONTRIBUTING.md#running) there shows the verdict, `rerun:` and summary lines a run prints, its exit codes, and what a crash prints.

### Precision Families

Each kernel is assigned a __comparison family__ that determines which error metrics are reported and what constitutes failure.
Families are defined in `harness.hpp` as `comparison_family_t`.
All floating-point families report `max_abs` and `max_rel`; the rest of each row is some of `mean_ulp`, `max_ulp`, `exact` match counts, `mean_abs`, `mean_rel` and `max_bound`.

- __`exact_k`__ — integer and binary metrics: Hamming, Jaccard, set intersections, integer min/max.
  Reports `max_dist`, `mean_dist`, `max_abs`, `mismatch`, `exact`.
  Fails on any `max_dist > 0`.
- __`approximate_k`__ — results computed and stored in one type, measured against a ULP budget: trigonometry and elementwise sums.
  Fails on `max_ulp > NUMKONG_ULP_THRESHOLD_{F32,F16,BF16}`, picked by the type of the results.
- __`bounded_k`__ — results held to the `nk_*_error_bound` of their family, like `nk_dot_error_bound`, on every backend: dot products, batched dots and distances, bilinear and Mahalanobis forms, sparse dots, moments, probability divergences, single-vector distances, MaxSim, geographic distances, and elementwise scale, blend and FMA.
  Shape distances have no `nk_*_error_bound`, so `test/shapes.cpp` derives each bound from the cloud sizes and radii, the unit roundoff of the distance type, and the transport tolerance.
  A result may land (roundings + 1) · bound · Σ|terms| from exact, plus the rounding into its output type, where `tracked` references carry the roundings and Σ|terms| through the same arithmetic.
  Roundings of products, quotients and functions count against the term's bound, and those of sums against the reduction's, which differ when a family sums in a wider type than it computes its terms in.
  Reports `max_bound`, the largest error as a share of its own bound, and fails when it exceeds 1.
- __`normalized_reduction_k`__ — attention outputs, which pass through zero where ULP distances explode.
  Fails when `max_abs` exceeds `NUMKONG_SCALE_THRESHOLD` times the largest reference.

### Reference Baselines

C++ tests compare SIMD kernels against high-precision serial references.
The baseline type depends on the input dtype, selected by the `reference_for<input>` template in `harness.hpp`.

- __f32 and f64 inputs__ use `f118_t` — a double-double type with ~103-bit mantissa, defined in `types.hpp`.
  Two `double` values track a high and low component, capturing rounding errors that a single `double` would lose.
  This is critical because f32 kernels use f64 accumulators internally — testing against plain f64 would not catch accumulation drift.
- __Complex f32c and f64c inputs__ use `f118c_t` — a pair of `f118_t` for real and imaginary parts.
- __Half-precision, mini-floats, and integers__ — f16, bf16, e4m3, e5m2, i8, u8, etc. — use plain `f64_t`.
  These types have at most 10-bit mantissas, so f64's 52-bit mantissa already provides >40 bits of headroom.
- __Complex halfs__ — f16c, bf16c — use `f64c_t` for the same reason.

### WASM

A WebAssembly module carries one SIMD capability, `serial`, `v128` or `v128relaxed`, which `NUMKONG_MARCH` selects under every toolchain; Emscripten names each module after it.

__Emscripten__

```sh
source ~/emsdk/emsdk_env.sh
cmake --preset wasm32_emscripten -D NUMKONG_BUILD_TEST=ON
cmake --build --preset wasm32_emscripten
```

For the relaxed capability of the same 32-bit module, and for wasm64 — Memory64:

```sh
cmake --preset wasm32_emscripten -D NUMKONG_BUILD_TEST=ON -D NUMKONG_MARCH=v128relaxed
cmake --build --preset wasm32_emscripten
cmake --preset wasm64_emscripten -D NUMKONG_BUILD_TEST=ON
cmake --build --preset wasm64_emscripten
```

__WASI__

```sh
export WASI_SDK_PATH=~/wasi-sdk
cmake --preset wasm32_wasi
cmake --build --preset wasm32_wasi
```

For a module over an imported shared memory, which hosts with WASI threads run:

```sh
cmake --preset wasm32_wasi_threads
cmake --build --preset wasm32_wasi_threads
```

__Running WASM Tests__

`ctest --preset wasm32_wasi` and `ctest --preset wasm32_wasi_threads` run the WASI binaries through `CMAKE_CROSSCOMPILING_EMULATOR`: `wasmtime` by default, or `wasmer` or `node` with `test/wasi.mjs` when the build names them, with the flags each module needs.
The same runs by hand:

```sh
wasmtime run -W relaxed-simd=y -S inherit-env=y ./build_wasm32_wasi/numkong_cpu_test.wasm
wasmer run --enable-simd --enable-relaxed-simd ./build_wasm32_wasi/numkong_cpu_test.wasm
wasmtime run -W relaxed-simd=y,threads=y -S threads=y,inherit-env=y ./build_wasm32_wasi_threads/numkong_cpu_test.wasm
node ./build_wasm32_emscripten/numkong_cpu_test.js
```

The JavaScript wrappers run over the Emscripten modules, and over the WASI tests, through `test/wasm.mjs`:

```sh
npm run build-js
npm run test:wasm:emscripten                                     # build_wasm32_emscripten
npm run test:wasm:emscripten64                                   # build_wasm64_emscripten
npm run test:wasm:wasi                                           # build_wasm32_wasi
```

__Memory Model__

The toolchain files configure memory limits appropriate for each target:

| Target       | Initial | Maximum | Pointers | Memory           | Emscripten |
| ------------ | ------: | ------: | -------: | ---------------- | ---------: |
| wasm32       |   64 MB |    2 GB |   32-bit | imported         |    3.1.27+ |
| wasm64       |  256 MB |   16 GB |   64-bit | imported         |    3.1.35+ |
| WASI         |       … |       … |   32-bit | self-contained   |          … |
| WASI threads |       … |    2 GB |   32-bit | imported, shared |          … |

Stack size is 5 MB across all Emscripten targets.
Only the WASI threads build uses `wasm32-wasip1-threads` with `-pthread`; the kernels are thread-free, so every other module is single-threaded and loads in any page.

__SIMD and Relaxed SIMD Support__

The SIMD builds require fixed-width SIMD — 128-bit `v128`.
The `serial` build runs on engines that validate neither probe.
The `v128relaxed` capability adds Relaxed SIMD instructions like `f32x4.relaxed_madd` and the fused `i8` dot product; an engine without them refuses the whole module at instantiation, which is why the capability is a build choice rather than a runtime one.
Inside a module, `nk_capabilities_detected_cpu` reports the capabilities it was compiled with, since an engine validated it whole, so `detected`, `compiled` and `enabled` agree.

| Engine   | SIMD128 | Relaxed SIMD | Threads | Memory64 |
| :------- | ------: | -----------: | ------: | -------: |
| Chrome   |      91 |          114 |      74 |      133 |
| Firefox  |      89 |          145 |      79 |      134 |
| Safari   |    16.4 |         flag |    14.1 |        … |
| Node.js  |    16.4 |         21.0 |    16.4 |     24.0 |
| Wasmtime |    0.33 |           15 |      15 |       30 |
| Wasmer   |     2.0 |          7.1 |     4.0 |        … |

For up-to-date engine support, see [WebAssembly Roadmap](https://webassembly.org/features/) and [caniuse Relaxed SIMD](https://caniuse.com/wasm-relaxed-simd).

### Cross-Compilation

Tests run under QEMU through `CMAKE_CROSSCOMPILING_EMULATOR`, so `ctest --test-dir <build>` runs a cross build like a native one.
Set `NUMKONG_IN_QEMU=1` to shrink test shapes under emulation, and in Python the default `NUMKONG_SCALE` too.
[CONTRIBUTING.md](../CONTRIBUTING.md#cross-compilation) lists the toolchain files, their emulators and prerequisites, and a recipe per target, Android's device run included.

## Rust

```sh
cargo test -p numkong
cargo test -p numkong -- --nocapture    # with output
cargo test -p numkong --all-features    # all optional features
cargo check -p numkong --no-default-features  # no-std compatibility
```

## Python

```sh
pip install -e . --group test --group test-oracles
python -X faulthandler -m pytest -x
```

The `test` group holds pytest, pytest-repeat and NumPy.
The `test-oracles` group adds the references the suite compares against when they install:

| Package     | What it unlocks                                         |
| :---------- | :------------------------------------------------------ |
| `scipy`     | Cross-validation against `scipy.spatial.distance`       |
| `ml_dtypes` | `__array_interface__` fallback for bfloat16 / fp8 / fp6 |

Tests that require a missing optional dependency are skipped automatically.

```sh
python -m pytest -k dot                           # only the tests whose names mention "dot"
python -m pytest -k "dot or spatial"
```

### Environment Variables

| Variable                  |       Default | Description                                                        |
| :------------------------ | ------------: | :----------------------------------------------------------------- |
| `NUMKONG_SEED`            |          `42` | Any unsigned 64-bit seed, or `random` to draw one                  |
| `NUMKONG_SCALE`           |           `1` | Randomized tests repeat `round(10·scale)` times, `0.3` under QEMU  |
| `NUMKONG_DIMS`            | `1,2,...,128` | Comma-separated vector dimensions, 19 sizes around powers of two   |
| `NUMKONG_CURVED_DIMS`     |     5 sampled | Dimensions for curved-space tests, drawn from the dense ones       |
| `NUMKONG_DIMS_HEIGHT`     |     6 sampled | GEMM M dimensions, drawn from the dense ones                       |
| `NUMKONG_DIMS_WIDTH`      |     6 sampled | GEMM N dimensions, drawn from the dense ones                       |
| `NUMKONG_DIMS_DEPTH`      |     6 sampled | GEMM K dimensions, drawn from the dense ones                       |
| `NUMKONG_IN_QEMU`         |           `0` | Shrink dimensions and the default scale; unset or `0` is off       |
| `NUMKONG_SPARSE_DIMS`     |         `256` | Universe size for sparse tests                                     |
| `NUMKONG_SHAPE_POINTS`    |          `32` | Point count for shape distance tests                               |
| `NUMKONG_MAX_COORD_ANGLE` |         `180` | Maximum angle in degrees for geospatial                            |
| `NUMKONG_EXPECT_SIMD`     |           `1` | `0` skips the check that a SIMD-capable machine dispatches to SIMD |

The pytest header prints the seed, scale and filter as the C++ runner does, then every other setting as `- Name: value`, so a `random` seed draw replays.
A value that does not parse stops the session with `NUMKONG_SEED="x" does not parse, expected an unsigned integer or random`.

The `pytest-repeat` plugin re-runs each randomized test `round(10·NUMKONG_SCALE)` times, and the costlier matrix and shape sweeps `round(2·NUMKONG_SCALE)` times, never fewer than once.
Each step draws from `stream_key(NUMKONG_SEED, name, step)`, where `name` is the test and its parameters but the CPU capability, so repeats cover more inputs while every capability sees the same ones and a run stays reproducible.

### Reference Baselines

Python tests use `decimal.Decimal` at 120-digit precision as ground truth for assertions.
Functions like `precise_inner()`, `precise_sqeuclidean()`, and `precise_angular()` convert each element to `Decimal` before accumulation, exceeding even `f118_t` accuracy.
A secondary NumPy baseline at native precision — f64 for floats, i64 for integers — is used for error statistics collection.

## JavaScript

```sh
npm test                                # Node.js native addon
```

### WASM Runtimes

JavaScript tests support multiple WASM runtimes via the `NUMKONG_RUNTIME` environment variable.

```sh
NUMKONG_RUNTIME=emscripten node --test test/wasm.mjs      # Emscripten 32-bit
NUMKONG_RUNTIME=emscripten64 node --test test/wasm.mjs    # Emscripten 64-bit, Memory64
NUMKONG_RUNTIME=wasi-node node --test test/wasm.mjs       # WASI via Node.js
npx playwright test --config test/playwright.config.ts    # Browser via Playwright
```

### Environment Variables

| Variable          |         Default | Description                                                   |
| :---------------- | --------------: | :------------------------------------------------------------ |
| `NUMKONG_RUNTIME` |        `native` | Runtime: `emscripten`, `emscripten64`, `wasi-node`            |
| `NUMKONG_SEED`    |            `42` | Seed for test data, or `random` to draw one; printed at start |
| `NUMKONG_DIMS`    | `3,16,128,1536` | Comma-separated vector dimensions                             |

## Swift

```sh
swift build && swift test -v
```

For iOS simulator testing:

```sh
xcodebuild test -scheme NumKong -destination 'platform=iOS Simulator,name=iPhone 16'
```

On Linux without a native Swift installation, use the official Docker image:

```sh
sudo docker run --rm -v "$PWD:/workspace" -w /workspace swift:6.4 \
  /bin/bash -cl "swift build -c release --static-swift-stdlib && swift test -c release"
```
