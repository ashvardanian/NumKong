# NumKong Built-in Benchmarks

Internal profiling suite comparing NumKong's SIMD backends against each other and optionally against BLAS libraries.
For broader comparisons — Rust, Python, etc. — see [NumWars](https://github.com/ashvardanian/NumWars).

- On x86 it compares serial code to manually-vectorized Haswell, Skylake, Ice Lake, Genoa, Sapphire Rapids and newer-generation SIMD kernels.
- On Arm it compares serial code to manually-vectorized NEON, SVE, SVE2, SME, SME2 with various extensions for BF16 and mixed-precision dot-products.
- On RISC-V it compares serial code to manually-vectorized RVV 1.0 kernels with and without BB, BF16, and F16 extensions.
- In WASM environments it compares serial code to manually-vectorized V128 kernels with Relaxed SIMD extensions.

## C++

### Building

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D NK_BUILD_BENCH=1
cmake --build build_release --config Release --parallel
```

With BLAS or MKL cross-validation:

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release \
      -D NK_BUILD_BENCH=1 \
      -D NK_COMPARE_TO_BLAS=1 \
      -D NK_COMPARE_TO_MKL=1
```

On macOS with Homebrew Clang and OpenBLAS — see [CONTRIBUTING.md](../CONTRIBUTING.md#macos) for the full recipe, adding `-DNK_BUILD_BENCH=1` to the cmake flags.

Compiler requirements vary by ISA target — see [CONTRIBUTING.md](../CONTRIBUTING.md#compiler-requirements) for the full table.

### Running

```sh
build_release/numkong_bench                                    # run all benchmarks
build_release/numkong_bench --benchmark_filter=dot             # filter by name
build_release/numkong_bench --benchmark_min_time=10s           # longer runs for stable results
build_release/numkong_bench --filter=dot                       # shorthand for --benchmark_filter
```

### Environment Variables

| Variable                  | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `NK_FILTER`               | `.*`    | Regex to filter benchmarks by name                     |
| `NK_SEED`                 | `42`    | RNG seed for reproducible inputs                       |
| `NK_BUDGET_SECS`          | `10`    | Minimum time per benchmark in seconds                  |
| `NK_BUDGET_MB`            | `1024`  | Memory budget for pre-allocated inputs                 |
| `NK_DENSE_DIMENSIONS`     | `1536`  | Vector dimension for dot/spatial benchmarks            |
| `NK_CURVED_DIMENSIONS`    | `64`    | Vector dimension for curved / bilinear form benchmarks |
| `NK_MESH_POINTS`          | `1000`  | Point count for mesh / RMSD / Kabsch benchmarks        |
| `NK_MATRIX_HEIGHT`        | `1024`  | GEMM M dimension, dataset size in kNN                  |
| `NK_MATRIX_WIDTH`         | `128`   | GEMM N dimension, query count in kNN                   |
| `NK_MATRIX_DEPTH`         | `1536`  | GEMM K dimension, vector dimension in kNN              |
| `NK_SPARSE_FIRST_LENGTH`  | `1024`  | First set size for sparse benchmarks                   |
| `NK_SPARSE_SECOND_LENGTH` | `8192`  | Second set size for sparse benchmarks                  |
| `NK_SPARSE_INTERSECTION`  | `0.5`   | Intersection share [0.0, 1.0] for sparse benchmarks    |
| `NK_MAX_COORD_ANGLE`      | `180`   | Maximum angle in degrees for geospatial benchmarks     |

Disable multi-threading in BLAS libraries to avoid interference:

```sh
export OPENBLAS_NUM_THREADS=1    # for OpenBLAS
export MKL_NUM_THREADS=1         # for Intel MKL
export VECLIB_MAXIMUM_THREADS=1  # for Apple Accelerate
export BLIS_NUM_THREADS=1        # for BLIS
```

### Reported Units

| Benchmark Type                          | Counter        | Meaning                                                         |
| --------------------------------------- | -------------- | --------------------------------------------------------------- |
| Vector kernels — dot, spatial, set, ... | `bytes/s`      | Bytes of input consumed per second, both input vectors combined |
| GEMM, symmetric, batch                  | `scalar-ops/s` | Scalar multiply-accumulate operations per second / FLOPS        |
| Reductions, casts, trigonometry         | `bytes/s`      | Bytes of input consumed per second, single input vector         |

__bytes__: total bytes across all input vectors read per call.
For a pair of 1536-dimensional `f32` vectors: `2 * 1536 * 4 = 12288` bytes per call.

__scalar-ops__: number of scalar arithmetic operations.
For dense GEMM: `2 * M * N * K` per call.
For symmetric GEMM: `N * (N + 1) * K` per call.

## JavaScript

### Running

```sh
npm run bench:native                            # Node.js native addon
npm run bench:emscripten                        # Emscripten WASM with SIMD
npm run bench:wasi                              # WASI portable execution
npm run bench:browser                           # Chromium via Playwright
npm run bench:all                               # all runtimes
```

```sh
NK_DIMENSIONS=768 NK_FILTER="dot" npm run bench:native    # custom config
```

| Variable        | Default  | Description                             |
| --------------- | -------- | --------------------------------------- |
| `NK_DIMENSIONS` | `1536`   | Vector dimensionality                   |
| `NK_ITERATIONS` | `1000`   | Number of benchmark iterations          |
| `NK_FILTER`     | `.*`     | Regex to filter benchmarks              |
| `NK_RUNTIME`    | `native` | Runtime: `native`, `emscripten`, `wasi` |
| `NK_SEED`       | `42`     | Random seed for reproducible data       |

### Output

JSON results are written to `bench/results/`.
Generate a Markdown comparison report:

```sh
npm run bench:report
cat bench/results/report.md
```

## WASM

__Emscripten — wasm32 and wasm64__

```sh
source ~/emsdk/emsdk_env.sh
cmake -B build-wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm.cmake -DNK_BUILD_BENCH=1
cmake --build build-wasm --parallel
```

For wasm64:

```sh
cmake -B build-wasm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm64.cmake -DNK_BUILD_BENCH=1
cmake --build build-wasm64 --parallel
```

The toolchain files enable `-msimd128` and `-mrelaxed-simd` automatically.

__WASI__

```sh
export WASI_SDK_PATH=~/wasi-sdk-24.0-x86_64-linux
cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasi.cmake -DNK_BUILD_BENCH=1
cmake --build build-wasi --parallel
```

__Running__

```sh
wasmtime run -W simd=y,relaxed-simd=y,threads=y,shared-memory=y -S threads=y,inherit-env=y ./build-wasi/numkong_bench.wasm
wasmer run --enable-simd --enable-relaxed-simd ./build-wasi/numkong_bench.wasm
node ./build-wasm/numkong_bench.js
```

Browser benchmarks via Playwright:

```sh
npm run bench:browser
```

__Interpreting WASM Results__

WASM benchmarks run slower than native due to JIT compilation overhead and memory indirection.
Expected performance relative to native:

| Runtime              | Typical Throughput vs Native |
| -------------------- | ---------------------------- |
| Emscripten / Node.js | 60–80%                       |
| WASI / Wasmtime      | 50–70%                       |
| Browser / Chromium   | 40–60%                       |

wasm64 / Memory64 adds ~5–10% overhead vs wasm32 due to 64-bit pointer arithmetic.
Relaxed SIMD provides measurable gains for fused multiply-add patterns — compare with and without `--wasm-features relaxed-simd` to quantify.

## Frequency Scaling on AMX and SME

Intel AMX tiles on Sapphire Rapids and later cause P-state throttling: the CPU reduces its frequency when AMX instructions execute, similar to heavy AVX-512 workloads.
Arm SME streaming mode on Graviton4 and Apple M4 has analogous frequency effects when entering and exiting streaming SVE mode.

This means AMX/SME benchmarks that interleave with non-AMX/SME work will show misleading throughput numbers as the CPU oscillates between frequency states.

Mitigations:

- Use `--benchmark_min_time=10s` or higher to amortize warm-up over a longer measurement window.
- Disable turbo boost with `echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo` on Linux.
- Run AMX/SME benchmarks in isolation — do not mix them with non-AMX/SME benchmarks in the same invocation.
- Filter with `--benchmark_filter=amx` or `--benchmark_filter=sme` for dedicated runs.

## Pinning to Performance Cores

### Linux

```sh
taskset -c 0-3 ./build_release/numkong_bench
numactl --physcpubind=0-3 ./build_release/numkong_bench
```

For dedicated benchmarking machines, add `isolcpus=4-7` to the kernel command line and pin benchmarks to isolated cores.

### macOS

No direct core-pinning API exists on macOS.
Use QoS to avoid efficiency cores:

```sh
taskpolicy -b ./build_release/numkong_bench
```

On Apple Silicon there is no public API for P/E core pinning.
Run with minimal background load for reproducible results.

### Windows

```sh
start /affinity 0xF numkong_bench.exe
```

The hex mask `0xF` pins to cores 0-3.
