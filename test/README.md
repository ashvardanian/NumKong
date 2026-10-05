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
cmake -B build_cuda -D NUMKONG_BUILD_TEST=ON -D NUMKONG_BUILD_CUDA=ON
cmake --build build_cuda
ctest --test-dir build_cuda -R '^numkong_cpu(_shared)?_test$' --output-on-failure
```

### Running

```sh
NUMKONG_FILTER=dot build_release/numkong_cpu_test             # run only tests matching "dot"
NUMKONG_FILTER="dot|spatial" build_release/numkong_cpu_test   # regex filter
NUMKONG_ASSERT=0 build_release/numkong_cpu_test               # exit 0 even when a kernel fails its accuracy check
NUMKONG_TIME_LIMIT=5s build_release/numkong_cpu_test          # 5 seconds per kernel
```

The binary takes no arguments; the environment configures it.
The linked runner uses device zero of each available compiled GPU backend by default.
An explicit device list selects all listed devices and rejects unavailable backends or ordinals.

### Environment Variables

| Variable                      |     Default | Description                                                               |
| :---------------------------- | ----------: | :------------------------------------------------------------------------ |
| `NUMKONG_DEVICES`             |        auto | Comma-separated vendor-local devices, for example `cuda:0,rocm:1` |
| `NUMKONG_FILTER`              |       unset | ECMAScript regex searched in kernel names; an invalid one is a substring  |
| `NUMKONG_SEED`                |        `42` | RNG seed, or `random` to draw one; the `- Seed:` line prints it           |
| `NUMKONG_TIME_LIMIT`          |        `1s` | Time limit per kernel, `<int>ms` or `<int>s`; zero or a bare number exits |
| `NUMKONG_DIMS`                |      `1536` | Vector dimension for dot/spatial tests                                    |
| `NUMKONG_CURVED_DIMS`         |        `64` | Vector dimension for curved tests                                         |
| `NUMKONG_SPARSE_DIMS`         |       `256` | Vector dimension for sparse tests                                         |
| `NUMKONG_MESH_POINTS`         |      `1000` | Point count for mesh tests                                                |
| `NUMKONG_DIMS_HEIGHT`         |      `1024` | GEMM M dimension                                                          |
| `NUMKONG_DIMS_WIDTH`          |       `128` | GEMM N dimension                                                          |
| `NUMKONG_DIMS_DEPTH`          |      `1536` | GEMM K dimension                                                          |
| `NUMKONG_MAX_COORD_ANGLE`     |       `180` | Maximum angle in degrees for geospatial tests                             |
| `NUMKONG_IN_QEMU`             |         `0` | Shrink shapes for emulation: `0`, `1`, `true` or `false`                  |
| `NUMKONG_ASSERT`              |         `1` | Exit 1 when any kernel fails its check; `0` only reports it               |
| `NUMKONG_ULP_THRESHOLD_F32`   |         `4` | Max allowed ULP distance for f32                                          |
| `NUMKONG_ULP_THRESHOLD_F16`   |        `32` | Max allowed ULP distance for f16                                          |
| `NUMKONG_ULP_THRESHOLD_BF16`  |       `256` | Max allowed ULP distance for bf16                                         |
| `NUMKONG_SCALE_THRESHOLD`     |      `0.02` | Max error over the largest reference, for attention                       |
| `NUMKONG_RANDOM_DISTRIBUTION` | `lognormal` | Distribution: `uniform`, `lognormal`, `cauchy`                            |

The dimension and GEMM variables also take the comma-separated lists the Python suite reads, and the C++ suite runs the first entry.
A first entry that is not a positive count exits 1, naming the variable, as does any value that does not parse.
A filter that matches no kernel runs nothing and passes.

Every kernel's row lands in the table whether it passes or not, and the summary counts the failures.
The run exits 1 on a failure unless `NUMKONG_ASSERT=0`.
A failing row is followed by a line that replays that kernel alone, with the seed of this run:

```
  rerun: NUMKONG_SEED=42 NUMKONG_FILTER='^dot_f32_serial$'
```

A kernel that crashes prints the signal it died of and its name to stderr, and the run stops.
Put that name into the template the run opens with:

```
- Seed: 42
- Rerun one test: NUMKONG_SEED=42 NUMKONG_FILTER='^<name>$' build_release/numkong_cpu_test
```

### Precision Families

Each kernel is assigned a __comparison family__ that determines which error metrics are reported and what constitutes failure.
Families are defined in `harness.hpp` as `comparison_family_t`.
All floating-point families report `max_abs` and `max_rel`; the rest of each row is some of `mean_ulp`, `max_ulp`, `exact` match counts, `mean_abs`, `mean_rel` and `max_bound`.

- __`exact_k`__ — integer and binary metrics: Hamming, Jaccard, set intersections, integer min/max.
  Reports `max_dist`, `mean_dist`, `max_abs`, `mismatch`, `exact`.
  Fails on any `max_dist > 0`.
- __`approximate_k`__ — results computed and stored in one type, measured against a ULP budget: trigonometry, mesh alignment and elementwise sums.
  Fails on `max_ulp > NUMKONG_ULP_THRESHOLD_{F32,F16,BF16}`, picked by the type of the results.
- __`bounded_k`__ — results held to the `nk_*_error_bound` of their family, like `nk_dot_error_bound`, on every backend: dot products, batched dots and distances, bilinear and Mahalanobis forms, sparse dots, moments, probability divergences, single-vector distances, MaxSim, geographic distances, and elementwise scale, blend and FMA.
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

A WebAssembly module carries one SIMD capability, `serial`, `v128` or `v128relaxed`, which `NUMKONG_TARGET_ARCH` selects under every toolchain; Emscripten names each module after it.

__Emscripten__

```sh
source ~/emsdk/emsdk_env.sh
cmake --preset wasm32_emscripten -D NUMKONG_BUILD_TEST=ON
cmake --build --preset wasm32_emscripten
```

For the relaxed capability of the same 32-bit module, and for wasm64 — Memory64:

```sh
cmake --preset wasm32_emscripten -D NUMKONG_BUILD_TEST=ON -D NUMKONG_TARGET_ARCH=v128relaxed
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

The JavaScript wrappers run over the Emscripten modules, and over the WASI tests with the host probes, through `test/wasm.mjs`:

```sh
npm run build-js
npm run test:wasm:emscripten                                     # build_wasm32_emscripten
npm run test:wasm:emscripten64                                   # build_wasm64_emscripten
npm run test:wasm:wasi                                           # build_wasm32_wasi, with -D NUMKONG_WITH_HOST_PROBES=ON
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
The `v128relaxed` capability adds Relaxed SIMD instructions like `f32x4.relaxed_madd` and the fused `i8` dot product; an engine without them refuses the whole module at instantiation, which is why the capability is a build choice rather than a runtime one.
Inside a module, `nk_cpu_capabilities_detected` still reports what the host validates — through the `env.nk_has_*` imports a Node host supplies under `NUMKONG_WITH_HOST_PROBES`, and as the compiled capabilities otherwise — and `nk_cpu_capabilities_enabled` intersects that with what was compiled.

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
Set `NUMKONG_IN_QEMU=1` to shrink test shapes under emulation, and repetitions too in Python.
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
NUMKONG_FILTER=dot python -m pytest               # node ids matching a regex, or else a substring
NUMKONG_FILTER="dot|spatial" python -m pytest
python -m pytest -k "dot or spatial"              # pytest's own filter still applies on top
```

### Environment Variables

| Variable                  |       Default | Description                                                        |
| :------------------------ | ------------: | :----------------------------------------------------------------- |
| `NUMKONG_DIMS`            | `1,2,...,128` | Comma-separated vector dimensions, 19 sizes around powers of two   |
| `NUMKONG_CURVED_DIMS`     |     5 sampled | Dimensions for curved-space tests, drawn from the dense ones       |
| `NUMKONG_DIMS_HEIGHT`     |     6 sampled | GEMM M dimensions, drawn from the dense ones                       |
| `NUMKONG_DIMS_WIDTH`      |     6 sampled | GEMM N dimensions, drawn from the dense ones                       |
| `NUMKONG_DIMS_DEPTH`      |     6 sampled | GEMM K dimensions, drawn from the dense ones                       |
| `NUMKONG_SEED`            |          `42` | Seed for `np.random`, or `random` to draw one                      |
| `NUMKONG_FILTER`          |         unset | Regex searched in test node ids; an invalid one is a substring     |
| `NUMKONG_REPETITIONS`     |          `10` | Randomized test repeat count, `3` under `NUMKONG_IN_QEMU`          |
| `NUMKONG_IN_QEMU`         |           `0` | Shrink dimensions and repetitions: `0`, `1`, `true` or `false`     |
| `NUMKONG_SPARSE_DIMS`     |         `256` | Universe size for sparse tests                                     |
| `NUMKONG_MESH_POINTS`     |        `1000` | Point count for mesh alignment tests                               |
| `NUMKONG_MAX_COORD_ANGLE` |         `180` | Maximum angle in degrees for geospatial                            |
| `NUMKONG_EXPECT_SIMD`     |           `1` | `0` skips the check that a SIMD-capable machine dispatches to SIMD |

The pytest header prints every setting as `- Name: value`, so a `random` seed draw replays.
A value that does not parse stops the session with `NUMKONG_SEED="x" does not parse, expected an unsigned integer or random`.

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
