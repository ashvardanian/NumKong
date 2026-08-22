# NumKong Precision Tests

Custom test framework comparing NumKong kernels against high-precision `f118_t` double-double references.
Every kernel family has its own precision model with ULP-based error analysis.

## C++

No GTest dependency — the framework is self-contained in `test.hpp`.

### Building

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D NK_BUILD_TEST=1
cmake --build build_release --config Release --parallel
build_release/nk_test
```

To compile with BLAS cross-validation:

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D NK_BUILD_TEST=1 -D NK_COMPARE_TO_BLAS=1
```

Compiler requirements vary by ISA target — see [CONTRIBUTING.md](../CONTRIBUTING.md#compiler-requirements) for the full table.

### Running

```sh
build_release/nk_test --filter=dot           # run only tests matching "dot"
build_release/nk_test --filter="dot|spatial"  # regex filter
build_release/nk_test --assert               # abort on first accuracy failure
build_release/nk_test --verbose              # per-dimension ULP breakdown
build_release/nk_test --time-budget=5000     # 5 seconds per kernel (milliseconds)
```

Foreign flag mapping for muscle-memory compatibility:

| Foreign Flag                 | Maps To                                     |
| ---------------------------- | ------------------------------------------- |
| `--gtest_filter=<regex>`     | forwards to `--filter=<regex>` with warning |
| `--benchmark_filter=<regex>` | forwards to `--filter=<regex>` with warning |
| `--benchmark_min_time=<N>s`  | converts to ms and maps to `--time-budget`  |

### Environment Variables

| Variable                 | Default       | Description                                          |
| ------------------------ | ------------- | ---------------------------------------------------- |
| `NK_FILTER`              | `.*`          | Regex to filter tests by name                        |
| `NK_SEED`                | `42`          | RNG seed for reproducible inputs                     |
| `NK_BUDGET_SECS`         | `1`           | Time budget per kernel in seconds                    |
| `NK_DENSE_DIMENSIONS`    | `1536`        | Vector dimension for dot/spatial tests               |
| `NK_CURVED_DIMENSIONS`   | `64`          | Vector dimension for curved tests                    |
| `NK_SPARSE_DIMENSIONS`   | `256`         | Vector dimension for sparse tests                    |
| `NK_MESH_POINTS`         | `1000`        | Point count for mesh tests                           |
| `NK_MATRIX_HEIGHT`       | `1024`        | GEMM M dimension                                     |
| `NK_MATRIX_WIDTH`        | `128`         | GEMM N dimension                                     |
| `NK_MATRIX_DEPTH`        | `1536`        | GEMM K dimension                                     |
| `NK_MAX_COORD_ANGLE`     | `180`         | Maximum angle in degrees for geospatial tests        |
| `NK_IN_QEMU`             | unset         | Relax accuracy thresholds for QEMU emulation         |
| `NK_TEST_ASSERT`         | `0`           | Assert (abort) on failed accuracy checks             |
| `NK_TEST_VERBOSE`        | `0`           | Show per-dimension ULP breakdown                     |
| `NK_ULP_THRESHOLD_F32`   | `4`           | Max allowed ULP distance for f32                     |
| `NK_ULP_THRESHOLD_F16`   | `32`          | Max allowed ULP distance for f16                     |
| `NK_ULP_THRESHOLD_BF16`  | `256`         | Max allowed ULP distance for bf16                    |
| `NK_RANDOM_DISTRIBUTION` | `lognormal_k` | Distribution: `uniform_k`, `lognormal_k`, `cauchy_k` |
| `NO_COLOR`               | unset         | Disable colored output                               |
| `FORCE_COLOR`            | unset         | Force colored output even without TTY                |

### Precision Families

Each kernel is assigned a __comparison family__ that determines which error metrics are reported and what constitutes failure.
Families are defined in `test.hpp` as `comparison_family_t`.
All floating-point families report `max_abs`, `max_rel`, and `mean_ulp`; most also report `max_ulp` and `exact` match counts, and some substitute `mean_abs` or `mean_rel`.

- __`exact_k`__ — integer and binary metrics: Hamming, Jaccard, set intersections, integer min/max.
  Reports `max_dist`, `mean_dist`, `max_abs`, `mismatch`, `exact`.
  Fails on any `max_dist > 0`.
- __`narrow_arithmetic_k`__ — elementwise float ops: sum, scale, blend, fma, sin, cos, atan, cast.
  Fails on `max_ulp > NK_ULP_THRESHOLD_{F32,F16,BF16}`.
- __`mixed_precision_reduction_k`__ — reductions and dot products: dot, angular, euclidean, sqeuclidean, reduce_moments, reduce_minmax.
  Wider tolerance than `narrow_arithmetic_k` due to accumulation error.
- __`probability_k`__ — probability divergences: KL, Jensen-Shannon.
  Also reports `mean_abs` and `mean_rel`.
- __`geospatial_k`__ — geographic distances: Haversine, Vincenty.
  Also reports `mean_abs`.
- __`external_baseline_k`__ — cross-backend comparison against system BLAS — OpenBLAS, MKL, Accelerate.

### Reference Baselines

C++ tests compare SIMD kernels against high-precision serial references.
The baseline type depends on the input dtype, selected by the `reference_for<input, result>` template in `test.hpp`.

- __f32 and f64 inputs__ use `f118_t` — a double-double type with ~103-bit mantissa, defined in `types.hpp`.
  Two `double` values track a high and low component, capturing rounding errors that a single `double` would lose.
  This is critical because f32 kernels use f64 accumulators internally — testing against plain f64 would not catch accumulation drift.
- __Complex f32c and f64c inputs__ use `f118c_t` — a pair of `f118_t` for real and imaginary parts.
- __Half-precision, mini-floats, and integers__ — f16, bf16, e4m3, e5m2, i8, u8, etc. — use plain `f64_t`.
  These types have at most 10-bit mantissas, so f64's 52-bit mantissa already provides >40 bits of headroom.
- __Complex halfs__ — f16c, bf16c — use `f64c_t` for the same reason.

### WASM

__Emscripten__

```sh
source ~/emsdk/emsdk_env.sh
cmake -B build-wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm.cmake -DNK_BUILD_TEST=1
cmake --build build-wasm --parallel
```

For wasm64 — Memory64:

```sh
cmake -B build-wasm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm64.cmake -DNK_BUILD_TEST=1
cmake --build build-wasm64 --parallel
```

__WASI__

```sh
export WASI_SDK_PATH=~/wasi-sdk-24.0-x86_64-linux
cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasi.cmake -DNK_BUILD_TEST=1
cmake --build build-wasi --parallel
```

__Running WASM Tests__

```sh
wasmtime run -W simd=y,relaxed-simd=y,threads=y,shared-memory=y -S threads=y,inherit-env=y ./build-wasi/nk_test.wasm
wasmer run --enable-simd --enable-relaxed-simd ./build-wasi/nk_test.wasm
node ./build-wasm/nk_test.js
```

__Memory Model__

The toolchain files configure memory limits appropriate for each target:

| Target | Initial Memory | Max Memory | Pointer Width | Emscripten Version |
| ------ | -------------- | ---------- | ------------- | ------------------ |
| wasm32 | 64 MB          | 2 GB       | 32-bit        | 3.1.27+            |
| wasm64 | 256 MB         | 16 GB      | 64-bit        | 3.1.35+            |
| WASI   | —              | 2 GB       | 32-bit        | —                  |

All three targets enable `-msimd128` and `-mrelaxed-simd` automatically.
Stack size is 5 MB across all Emscripten targets.
WASI builds use `wasm32-wasip1-threads` with shared memory and `-pthread` for threading support.

__SIMD and Relaxed SIMD Support__

All WASM builds require fixed-width SIMD — 128-bit `v128`.
Relaxed SIMD instructions like `f32x4.relaxed_madd` and fused dot-product are used when available — the toolchain enables them unconditionally and the WASI test runner probes for support at startup via `WebAssembly.validate()`.

Known-good minimum versions: Chrome 114+, Firefox 120+, Node.js 20+, Wasmtime 14+.
Safari supports `v128` SIMD from 16.4 but has incomplete Relaxed SIMD coverage; Memory64 is not yet available in Safari or Wasmer.
For up-to-date engine support, see [WebAssembly Roadmap](https://webassembly.org/features/) and [caniuse Relaxed SIMD](https://caniuse.com/wasm-relaxed-simd).

### Cross-Compilation

NumKong ships 11 toolchain files in `cmake/` for cross-compiling to non-native targets.
Tests run transparently under QEMU via `CMAKE_CROSSCOMPILING_EMULATOR`.
Set `NK_IN_QEMU=1` to relax half-precision accuracy thresholds under emulation.

__ARM64 Linux__

```sh
cmake -B build_arm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-gnu.cmake \
      -DNK_BUILD_TEST=1
cmake --build build_arm64 --parallel
NK_IN_QEMU=1 ctest --test-dir build_arm64    # runs under qemu-aarch64 -cpu max
```

Default arch: `armv9-a+sve2+fp16+bf16+i8mm+dotprod+fp16fml`.

__RISC-V 64 with GCC__

```sh
export RISCV_TOOLCHAIN_PATH=/path/to/riscv-gnu-toolchain    # optional
# export RISCV_SYSROOT=/path/to/riscv-sysroot               # optional override
cmake -B build_riscv -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-gnu.cmake \
      -DNK_BUILD_TEST=1
cmake --build build_riscv --parallel
NK_IN_QEMU=1 ctest --test-dir build_riscv    # runs under qemu-riscv64 -cpu max
```

Default arch: `rv64gcv_zvfh_zvfbfwma_zvbb`.

__RISC-V 64 with LLVM__

```sh
export RISCV_SYSROOT=/path/to/riscv-sysroot
export LLVM_ROOT=/path/to/llvm                           # optional
cmake -B build_riscv_llvm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-llvm.cmake \
      -DNK_BUILD_TEST=1
cmake --build build_riscv_llvm --parallel
NK_IN_QEMU=1 ctest --test-dir build_riscv_llvm
```

__Android ARM64__

```sh
cmake -B build_android -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-android-arm64.cmake \
      -DNK_BUILD_TEST=1
cmake --build build_android --parallel
adb push build_android/nk_test /data/local/tmp/
adb shell /data/local/tmp/nk_test
```

## Rust

```sh
cargo test -p numkong
cargo test -p numkong -- --nocapture    # with output
cargo test -p numkong --all-features    # all optional features
cargo check -p numkong --no-default-features  # no-std compatibility
```

### WASM via Wasmtime

The `wasm-runtime` feature embeds a Wasmtime runtime to test WASM modules from within `cargo test`.

```sh
cargo test -p numkong --features wasm-runtime -- wasm_runtime
```

## Python

```sh
pip install -e .
pip install pytest pytest-repeat pytest-randomly numpy scipy ml_dtypes tabulate
pytest test/ -s -x -Wd
```

Optional dependencies for extended test coverage:

| Package     | What it unlocks                                         |
| ----------- | ------------------------------------------------------- |
| `numpy`     | Array interop, cdist, custom dtype registration         |
| `scipy`     | Cross-validation against `scipy.spatial.distance`       |
| `ml_dtypes` | `__array_interface__` fallback for bfloat16 / fp8 / fp6 |
| `tabulate`  | Formatted precision report tables                       |

Tests that require a missing optional dependency are skipped automatically.

```sh
pytest test/ -s -x -Wd -k dot         # filter by name
pytest test/ -s -x -Wd -k "dot or spatial"
```

### Environment Variables

| Variable               | Default          | Description                             |
| ---------------------- | ---------------- | --------------------------------------- |
| `NK_DENSE_DIMENSIONS`  | `1,2,3,...,1536` | Comma-separated vector dimensions       |
| `NK_CURVED_DIMENSIONS` | `11,97`          | Dimensions for curved-space tests       |
| `NK_MATRIX_HEIGHT`     | `1024`           | GEMM M dimension                        |
| `NK_MATRIX_WIDTH`      | `128`            | GEMM N dimension                        |
| `NK_MATRIX_DEPTH`      | `1536`           | GEMM K dimension                        |
| `NK_SEED`              | OS entropy       | Deterministic seed for `np.random`      |
| `NK_REPETITIONS`       | `10`             | Randomized test repeat count            |
| `NK_IN_QEMU`           | unset            | Relax accuracy thresholds               |
| `NK_SPARSE_DIMENSIONS` | `256`            | Universe size for sparse tests          |
| `NK_MESH_POINTS`       | `100`            | Point count for mesh alignment tests    |
| `NK_MAX_COORD_ANGLE`   | `180`            | Maximum angle in degrees for geospatial |

The `pytest-repeat` plugin re-runs each test `NK_REPETITIONS` times with auto-seeding — each iteration gets a unique seed derived from the base `NK_SEED`, ensuring broader input coverage without sacrificing reproducibility.

### Reference Baselines

Python tests use `decimal.Decimal` at 120-digit precision as ground truth for assertions.
Functions like `precise_inner()`, `precise_sqeuclidean()`, and `precise_angular()` convert each element to `Decimal` before accumulation, exceeding even `f118_t` accuracy.
A secondary NumPy baseline at native precision — f64 for floats, i64 for integers — is used for error statistics collection.

## JavaScript

```sh
npm test                                # Node.js native addon
```

### WASM Runtimes

JavaScript tests support multiple WASM runtimes via the `NK_RUNTIME` environment variable.

```sh
NK_RUNTIME=emscripten node --test test/test-wasm.mjs      # Emscripten 32-bit
NK_RUNTIME=emscripten64 node --test test/test-wasm.mjs    # Emscripten 64-bit, Memory64
NK_RUNTIME=wasi-node node --test test/test-wasm.mjs       # WASI via Node.js
npx playwright test --config test/playwright.config.ts    # Browser via Playwright
```

### Environment Variables

| Variable              | Default         | Description                                        |
| --------------------- | --------------- | -------------------------------------------------- |
| `NK_RUNTIME`          | `native`        | Runtime: `emscripten`, `emscripten64`, `wasi-node` |
| `NK_SEED`             | `42`            | Random seed for reproducible test data             |
| `NK_DENSE_DIMENSIONS` | `3,16,128,1536` | Comma-separated vector dimensions                  |

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
sudo docker run --rm -v "$PWD:/workspace" -w /workspace swift:5.9 \
  /bin/bash -cl "swift build -c release --static-swift-stdlib && swift test -c release --enable-test-discovery"
```
