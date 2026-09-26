# NumKong Precision Tests

Custom test framework comparing NumKong kernels against high-precision `f118_t` double-double references.
Every kernel family has its own precision model with ULP-based error analysis.

## C++

No GTest dependency — the framework is self-contained in `harness.hpp`.

### Building

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D NUMKONG_BUILD_TEST=1
cmake --build build_release --config Release --parallel
build_release/numkong_test
```

To compile with BLAS cross-validation:

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D NUMKONG_BUILD_TEST=1 -D NUMKONG_COMPARE_TO_BLAS=1
```

Compiler requirements vary by ISA target — see [CONTRIBUTING.md](../CONTRIBUTING.md#compiler-requirements) for the full table.

### Running

```sh
build_release/numkong_test --filter=dot           # run only tests matching "dot"
build_release/numkong_test --filter="dot|spatial"  # regex filter
build_release/numkong_test --assert               # exit 1 when any kernel fails its accuracy check
build_release/numkong_test --verbose              # per-dimension ULP breakdown
build_release/numkong_test --budget-secs=5        # 5 seconds per kernel
```

Foreign flag mapping for muscle-memory compatibility:

| Foreign Flag                 | Maps To                                     |
| :--------------------------- | :------------------------------------------ |
| `--gtest_filter=<regex>`     | forwards to `--filter=<regex>` with warning |
| `--benchmark_filter=<regex>` | forwards to `--filter=<regex>` with warning |
| `--benchmark_min_time=<N>s`  | maps to `--budget-secs=<N>`                 |

### Environment Variables

| Variable                      |       Default | Description                                                           |
| :---------------------------- | ------------: | :-------------------------------------------------------------------- |
| `NUMKONG_FILTER`              |          `.*` | Regex searched in kernel names; an invalid one matches as a substring |
| `NUMKONG_SEED`                |          `42` | RNG seed, or `random` to draw one; the `- Seed:` line prints it       |
| `NUMKONG_BUDGET_SECS`         |           `1` | Time budget per kernel in seconds; `0` or less keeps the default      |
| `NUMKONG_DENSE_DIMENSIONS`    |        `1536` | Vector dimension for dot/spatial tests                                |
| `NUMKONG_CURVED_DIMENSIONS`   |          `64` | Vector dimension for curved tests                                     |
| `NUMKONG_SPARSE_DIMENSIONS`   |         `256` | Vector dimension for sparse tests                                     |
| `NUMKONG_MESH_POINTS`         |        `1000` | Point count for mesh tests                                            |
| `NUMKONG_MATRIX_HEIGHT`       |        `1024` | GEMM M dimension                                                      |
| `NUMKONG_MATRIX_WIDTH`        |         `128` | GEMM N dimension                                                      |
| `NUMKONG_MATRIX_DEPTH`        |        `1536` | GEMM K dimension                                                      |
| `NUMKONG_MAX_COORD_ANGLE`     |         `180` | Maximum angle in degrees for geospatial tests                         |
| `NUMKONG_IN_QEMU`             |         unset | Shrink shapes for emulation; `0` or `false` also means off            |
| `NUMKONG_ASSERT`              |           `0` | Exit 1 when any kernel fails its check                                |
| `NUMKONG_VERBOSE`             |           `0` | Show per-dimension ULP breakdown                                      |
| `NUMKONG_ULP_THRESHOLD_F32`   |           `4` | Max allowed ULP distance for f32                                      |
| `NUMKONG_ULP_THRESHOLD_F16`   |          `32` | Max allowed ULP distance for f16                                      |
| `NUMKONG_ULP_THRESHOLD_BF16`  |         `256` | Max allowed ULP distance for bf16                                     |
| `NUMKONG_SCALE_THRESHOLD`     |        `0.02` | Max error over the largest reference, for attention                   |
| `NUMKONG_RANDOM_DISTRIBUTION` | `lognormal_k` | Distribution: `uniform_k`, `lognormal_k`, `cauchy_k`                  |

The dimension and GEMM variables also take the comma-separated lists the Python suite reads, and the C++ suite runs the first entry.
A first entry that is not a positive count aborts the run, naming the variable.
A filter that matches no kernel runs nothing and passes.

Every kernel's row lands in the table whether it passes or not, and the summary counts the failures.
The run exits 1 on a failure only under `NUMKONG_ASSERT=1` or `--assert`.
A failing row is followed by a line that replays that kernel alone, with the seed of this run:

```
  rerun: NUMKONG_SEED=42 NUMKONG_FILTER='^dot_f32_serial$'
```

A kernel that crashes prints the signal it died of and its name to stderr, and the run stops.
Put that name into the template the run opens with:

```
- Seed: 42
- Rerun one test: NUMKONG_SEED=42 NUMKONG_FILTER='^<name>$' build_release/numkong_test
```

### Precision Families

Each kernel is assigned a __comparison family__ that determines which error metrics are reported and what constitutes failure.
Families are defined in `harness.hpp` as `comparison_family_t`.
All floating-point families report `max_abs`, `max_rel`, and `mean_ulp`; most also report `max_ulp` and `exact` match counts, and some substitute `mean_abs` or `mean_rel`.

- __`exact_k`__ — integer and binary metrics: Hamming, Jaccard, set intersections, integer min/max.
  Reports `max_dist`, `mean_dist`, `max_abs`, `mismatch`, `exact`.
  Fails on any `max_dist > 0`.
- __`approximate_k`__ — everything measured against a ULP budget: elementwise float ops, reductions and dot products.
  Fails on `max_ulp > NUMKONG_ULP_THRESHOLD_{F32,F16,BF16}`.
- __`probability_k`__ — probability divergences: KL, Jensen-Shannon.
  Also reports `mean_abs` and `mean_rel`.
- __`geospatial_k`__ — geographic distances: Haversine, Vincenty.
  Also reports `mean_abs`.

### Reference Baselines

C++ tests compare SIMD kernels against high-precision serial references.
The baseline type depends on the input dtype, selected by the `reference_for<input, result>` template in `harness.hpp`.

- __f32 and f64 inputs__ use `f118_t` — a double-double type with ~103-bit mantissa, defined in `types.hpp`.
  Two `double` values track a high and low component, capturing rounding errors that a single `double` would lose.
  This is critical because f32 kernels use f64 accumulators internally — testing against plain f64 would not catch accumulation drift.
- __Complex f32c and f64c inputs__ use `f118c_t` — a pair of `f118_t` for real and imaginary parts.
- __Half-precision, mini-floats, and integers__ — f16, bf16, e4m3, e5m2, i8, u8, etc. — use plain `f64_t`.
  These types have at most 10-bit mantissas, so f64's 52-bit mantissa already provides >40 bits of headroom.
- __Complex halfs__ — f16c, bf16c — use `f64c_t` for the same reason.

### WASM

A WebAssembly module carries one SIMD tier, so every wasm toolchain fixes it through `NUMKONG_WASM_SIMD` — `v128` or `v128relaxed` — and one build directory holds one tier.

__Emscripten__

```sh
source ~/emsdk/emsdk_env.sh
cmake -B build-wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-emscripten.cmake -DNUMKONG_BUILD_TEST=1
cmake --build build-wasm --parallel
```

For the relaxed tier of the same 32-bit module, and for wasm64 — Memory64:

```sh
cmake -B build-wasm-relaxed -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-emscripten.cmake -DNUMKONG_WASM_SIMD=v128relaxed -DNUMKONG_BUILD_TEST=1
cmake --build build-wasm-relaxed --parallel
cmake -B build-wasm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm64-emscripten.cmake -DNUMKONG_BUILD_TEST=1
cmake --build build-wasm64 --parallel
```

__WASI__

```sh
export WASI_SDK_PATH=~/wasi-sdk-24.0-x86_64-linux
cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi.cmake -DNUMKONG_BUILD_TEST=1
cmake --build build-wasi --parallel
```

For a module over an imported shared memory, which hosts with WASI threads run:

```sh
cmake -B build-wasi-threads -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi-threads.cmake -DNUMKONG_BUILD_TEST=1
cmake --build build-wasi-threads --parallel
```

__Running WASM Tests__

`ctest` runs the WASI binary through the runtime `NUMKONG_WASM_RUNTIME` names — `wasmtime` by default, or `wasmer` or `node` — with the flags each module needs.
The same runs by hand:

```sh
wasmtime run -W relaxed-simd=y ./build-wasi/numkong_test.wasm
wasmer run --enable-simd --enable-relaxed-simd ./build-wasi/numkong_test.wasm
wasmtime run -W relaxed-simd=y,threads=y -S threads=y,inherit-env=y ./build-wasi-threads/numkong_test.wasm
node ./build-wasm/numkong_test.js
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

All WASM builds require fixed-width SIMD — 128-bit `v128`.
The `v128relaxed` tier adds Relaxed SIMD instructions like `f32x4.relaxed_madd` and the fused `i8` dot product; an engine without them refuses the whole module at instantiation, which is why the tier is a build choice rather than a runtime one.
Inside a module, `nk_cpu_capabilities_detected` still reports what the host validates — through `EM_JS` probes under Emscripten and through the `env.nk_has_*` imports a Node host supplies under `NUMKONG_WASI_HOSTED` — and `nk_cpu_capabilities_enabled` intersects that with what was compiled.

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

NumKong ships 12 toolchain files in `cmake/` for cross-compiling to non-native targets.
Tests run transparently under QEMU via `CMAKE_CROSSCOMPILING_EMULATOR`.
Set `NUMKONG_IN_QEMU=1` to shrink test shapes under emulation, and repetitions too in Python.

__ARM64 Linux__

```sh
cmake -B build_arm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-gnu.cmake \
      -DNUMKONG_BUILD_TEST=1
cmake --build build_arm64 --parallel
NUMKONG_IN_QEMU=1 ctest --test-dir build_arm64 # runs under qemu-aarch64 -cpu max
```

The ISA floor is `armv8-a`; individual kernels are gated by the compile probes in `cmake/`.

__RISC-V 64 with GCC__

```sh
cmake -B build_riscv -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-gnu.cmake \
      -DNUMKONG_BUILD_TEST=1
cmake --build build_riscv --parallel
NUMKONG_IN_QEMU=1 ctest --test-dir build_riscv    # runs under qemu-riscv64 -cpu max
```

Default arch: `rv64gcv_zvfh_zvfbfwma_zvbb`.
Needs GCC 16 or newer for the RVV kernels.

__RISC-V 64 with LLVM__

```sh
export LLVM_ROOT=/path/to/llvm # optional
cmake -B build_riscv_llvm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-llvm.cmake \
      -DNUMKONG_BUILD_TEST=1
cmake --build build_riscv_llvm --parallel
NUMKONG_IN_QEMU=1 ctest --test-dir build_riscv_llvm
```

Set `RISCV_SYSROOT` only for a self-contained toolchain; distribution cross packages need none.

__Android ARM64__

```sh
cmake -B build_android -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-android-arm64.cmake \
      -DNUMKONG_BUILD_TEST=1
cmake --build build_android --parallel
adb push build_android/numkong_test /data/local/tmp/
adb shell /data/local/tmp/numkong_test
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
| :---------- | :------------------------------------------------------ |
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

| Variable                    |          Default | Description                                              |
| :-------------------------- | ---------------: | :------------------------------------------------------- |
| `NUMKONG_DENSE_DIMENSIONS`  | `1,2,3,...,1536` | Comma-separated vector dimensions                        |
| `NUMKONG_CURVED_DIMENSIONS` |          `11,97` | Dimensions for curved-space tests                        |
| `NUMKONG_MATRIX_HEIGHT`     |           `1024` | GEMM M dimension                                         |
| `NUMKONG_MATRIX_WIDTH`      |            `128` | GEMM N dimension                                         |
| `NUMKONG_MATRIX_DEPTH`      |           `1536` | GEMM K dimension                                         |
| `NUMKONG_SEED`              |             `42` | Seed for `np.random`, or `random` to draw one            |
| `NUMKONG_REPETITIONS`       |             `10` | Randomized test repeat count                             |
| `NUMKONG_IN_QEMU`           |            unset | Shrink dimensions and repetitions; `0` or `false` is off |
| `NUMKONG_SPARSE_DIMENSIONS` |            `256` | Universe size for sparse tests                           |
| `NUMKONG_MESH_POINTS`       |            `100` | Point count for mesh alignment tests                     |
| `NUMKONG_MAX_COORD_ANGLE`   |            `180` | Maximum angle in degrees for geospatial                  |

The pytest header names the seed as `seed: <n>, pin with NUMKONG_SEED`, so a `random` draw replays.
A seed that is neither a number nor `random` stops the session, naming the variable.

The `pytest-repeat` plugin re-runs each test `NUMKONG_REPETITIONS` times with auto-seeding — each iteration gets a unique seed derived from the base `NUMKONG_SEED`, ensuring broader input coverage without sacrificing reproducibility.

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

| Variable                   |         Default | Description                                                   |
| :------------------------- | --------------: | :------------------------------------------------------------ |
| `NUMKONG_RUNTIME`          |        `native` | Runtime: `emscripten`, `emscripten64`, `wasi-node`            |
| `NUMKONG_SEED`             |            `42` | Seed for test data, or `random` to draw one; printed at start |
| `NUMKONG_DENSE_DIMENSIONS` | `3,16,128,1536` | Comma-separated vector dimensions                             |

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
