# Contributing

To keep the quality of the code high, we follow the [coding style and conventions](https://github.com/ashvardanian/ashvardanian/blob/master/CONTRIBUTING.md) shared across multiple projects — covering Git history, C++ and Python formatting, dependency management, and documentation.

## Directory Tree

```
include/numkong/          C and C++ headers — one .h per kernel family, one .hpp per C++ API
include/numkong/*/        Per-ISA kernel implementations — serial, haswell, neon, rvv, sme, etc.
c/                        Runtime dispatch layer — one dispatch_*.c per dtype
test/                     C++ precision tests — see test/README.md
bench/                    C++ Google Benchmark suite and JS bench runner — see bench/README.md
python/                   CPython extension, no SWIG or PyBind11
javascript/               Node.js native addon + Emscripten WASM + TypeScript API
rust/                     Rust FFI bindings
swift/                    Swift Package Manager bindings
golang/                   Go cgo bindings
cmake/                    Toolchain files for cross-compilation — WASM, WASI, RISC-V, AArch64
```

## C and C++

### Building

The presets in `CMakePresets.json` are the configurations CI builds, and `cmake --list-presets` shows the ones this machine can run.

```sh
cmake --preset release -D NUMKONG_COMPARE_TO_BLAS=1
cmake --build --preset release
ctest --preset release                  # the static and shared-library suites
build_release/numkong_bench
```

`debug`, `cuda`, the `linux_<arch>` cross builds under QEMU, Android and WASM follow the same three commands.
Machine-specific settings, like a compiler path, belong in an untracked `CMakeUserPresets.json`.

| CMake Flag                | Default            | Description                                                                   |
| :------------------------ | :----------------- | :---------------------------------------------------------------------------- |
| `NUMKONG_BUILD_TEST`      | `OFF`              | Compile precision tests, and with the shared library also tests against it    |
| `NUMKONG_BUILD_BENCH`     | `OFF`              | Compile micro-benchmarks                                                      |
| `NUMKONG_BUILD_SHARED`    | `ON`, if top-level | Compile dynamic library                                                       |
| `NUMKONG_BUILD_CUDA`      | `OFF`              | Add CUDA to whichever of the test and bench builds is on                      |
| `NUMKONG_COMPARE_TO_BLAS` | `AUTO`             | Include OpenBLAS, or Apple's Accelerate on macOS, into test/bench comparisons |
| `NUMKONG_COMPARE_TO_MKL`  | `AUTO`             | Include Intel' MKL into test/bench comparisons                                |
| `NUMKONG_TARGET_ARCH`     | empty              | Tune for a CPU, like the host with `native`                                   |

The test suites seed from 42, or from a fresh draw under `NUMKONG_SEED=random`, and print the seed they use.
`NUMKONG_FILTER` is a regex over kernel names, and each failing kernel prints a `rerun:` line with its seed and a filter selecting it alone.
Failures are always counted, and `NUMKONG_ASSERT=1` turns them into exit code 1.
The [test README](test/README.md#environment-variables) lists every variable.

### Target Baseline Policy

`CMakeLists.txt`, `build.rs`, `setup.py`, and `binding.gyp` pin the TU-level baseline to each architecture's ABI floor so distributable artifacts run on any CPU matching the ABI, not just the build host.
SIMD kernels live inside `#pragma GCC target(...)` regions and are only called after runtime probing — see the README's [Compile-Time and Run-Time Dispatch](README.md#compile-time-and-run-time-dispatch) section.

| Target arch   | GCC/Clang baseline          | MSVC baseline   | Notes                                                       |
| :------------ | :-------------------------- | :-------------- | :---------------------------------------------------------- |
| `x86_64`      | `-march=x86-64`             | `/arch:SSE2`    | System V psABI / Microsoft x64 ABI floor; SSE2 is mandatory |
| `aarch64`     | `-march=armv8-a`            | `/arch:armv8.0` | ARMv8-A ABI floor; NEON is mandatory                        |
| `riscv64`     | `-march=rv64gc`             | …               | V extension is runtime-probed and dispatched                |
| `powerpc64le` | `-mcpu=power8`              | …               | ELFv2 ABI floor (VSX is mandatory)                          |
| `loongarch64` | `-march=loongarch64 -mlasx` | …               | LASX baked into the baseline — see LoongArch note below     |

GCC/Clang builds also pass `-fno-tree-vectorize -fno-tree-slp-vectorize` so the auto-vectorizer cannot promote serial fallbacks to baseline SIMD (NEON, SSE2, VSX, …).
That keeps the tiered dispatch design intact: "serial" kernels stay actually serial, and the per-pragma SIMD kernels — which use explicit intrinsics, not vectorized scalar code — are the sole source of SIMD emission.
MSVC has no per-function target pragma and no command-line vectorizer toggle, so the explicit `/arch:` flags above match defaults and document intent only; NumKong's MSVC strategy is compile-time gating via `_MSC_VER` version checks (see `include/numkong/types.h`).
LoongArch is the one arch that can't honor the per-function-pragma model: `__attribute__((target("lasx")))` and `#pragma GCC target("lasx")` only landed in GCC 15.1 (Feb 2025) and Clang 22.1 (May 2025), and the bundled `lasxintrin.h` gates every wrapper on the `__loongarch_asx` macro that those older toolchains only set via TU-level `-mlasx`.
Until NumKong's minimum supported toolchain catches up, LoongArch artifacts require LASX-capable hardware (LA464+, c. 2021).
`Package.swift` and `golang/numkong.go` do not pin baselines: SPM forbids `.unsafeFlags()` on remotely consumed targets, and the cgo bindings rely on the surrounding compiler default.

For host-tuned local builds, set `NUMKONG_TARGET_ARCH=native` (env var honored by `build.rs`, `setup.py` and `binding.gyp`; CMake option `-DNUMKONG_TARGET_ARCH=native`).
The resulting artifact bakes host-specific instructions into scaffolding code and is __not__ portable.

### Compiler Requirements

| ISA Family                              |  GCC | Clang | AppleClang     |        MSVC |
| :-------------------------------------- | ---: | ----: | :------------- | ----------: |
| Base — serial, NEON, AVX2               |   9+ |   10+ | Any            |       2019+ |
| Float16 — NEONHalf, Sapphire FP16, Zvfh |  12+ |   16+ | Any            | 2022 17.14+ |
| AVX-512 — Skylake, Ice Lake             |   9+ |   10+ | …              |       2019+ |
| AVX-512BF16 — Genoa                     |  12+ |   16+ | …              | 2022 17.14+ |
| Intel AMX — Sapphire, Granite           |  14+ |   18+ | …              | 2022 17.14+ |
| Arm SME/SME2                            |  14+ |   18+ | 16+ / Xcode 16 |           … |
| RISC-V Vector — RVV 1.0                 |  13+ |   17+ | …              |           … |
| RVV + Zvfh/Zvfbfwma/Zvbb                |  14+ |   18+ | …              |           … |

To install on Ubuntu 22.04:

```sh
sudo apt install gcc-12 g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100
```

### Cross-Compilation

NumKong ships 12 toolchain files in `cmake/`, each named `toolchain-<name>.cmake`.
Tests and benchmarks run transparently under QEMU via `CMAKE_CROSSCOMPILING_EMULATOR`.
Targets with a `qemu-*` emulator additionally require `qemu-user`.

| Target                  | Toolchain             | Emulator                    | Prerequisites                                     |
| :---------------------- | :-------------------- | :-------------------------- | :------------------------------------------------ |
| ARM64 Linux             | `aarch64-gnu`         | `qemu-aarch64 -cpu max`     | `gcc-aarch64-linux-gnu`                           |
| RISC-V 64 LLVM          | `riscv64-llvm`        | `qemu-riscv64 -cpu max`     | Clang 17+, `gcc-riscv64-linux-gnu`                |
| RISC-V 64 GCC           | `riscv64-gnu`         | `qemu-riscv64 -cpu max`     | GCC 16+, `gcc-riscv64-linux-gnu`                  |
| ppc64le Linux           | `ppc64le-gnu`         | `qemu-ppc64le -cpu power10` | `gcc-powerpc64le-linux-gnu`                       |
| LoongArch 64            | `loongarch64-gnu`     | `qemu-loongarch64 -cpu max` | `gcc-loongarch64-linux-gnu`                       |
| Android ARM64           | `android-arm64`       | …                           | `ANDROID_NDK_ROOT`                                |
| Android ARMv7           | `android-armv7`       | …                           | `ANDROID_NDK_ROOT`                                |
| x86_64 on Apple Silicon | `x86_64-llvm`         | `arch -x86_64`              | Homebrew LLVM                                     |
| WASM32 Emscripten       | `wasm32-emscripten`   | Node.js                     | Emscripten 3.1.27+, tier `v128` by default        |
| WASM64 Emscripten       | `wasm64-emscripten`   | Node.js 24+                 | Emscripten 3.1.35+, tier `v128relaxed` by default |
| WASI                    | `wasm32-wasi`         | Wasmtime / Wasmer           | WASI SDK 24+, tier `v128` by default              |
| WASI threads            | `wasm32-wasi-threads` | Wasmtime with threads       | WASI SDK 24+, tier `v128relaxed` by default       |

A WebAssembly module carries one SIMD tier, so each wasm toolchain fixes it through `NUMKONG_WASM_SIMD` — `v128` or `v128relaxed` — and one build directory holds one tier.

Set `NUMKONG_IN_QEMU=1` to shrink test shapes under emulation, and repetitions too in Python.

__ARM64 Linux__

```sh
cmake -B build_arm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-gnu.cmake
cmake --build build_arm64 --parallel
```

To build and run tests under emulation, see [test/README.md](test/README.md#cross-compilation).

The ISA floor is `armv8-a`; individual kernels are gated by the compile probes in `cmake/`.
Use Clang: GCC 14 compiles the SME kernels but its `libgcc` has no `__arm_tpidr2_save`, so the link fails.

__RISC-V 64 with GCC__

```sh
cmake -B build_riscv -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-gnu.cmake
cmake --build build_riscv --parallel
```

To build and run tests under emulation, see [test/README.md](test/README.md#cross-compilation).

Default arch: `rv64gcv_zvfh_zvfbfwma_zvbb`.
Needs GCC 16 or newer: the RVV kernels gate on `#pragma GCC target("arch=+v")`, which GCC implements for RISC-V only from 16, and 14 and 15 ignore it and then fail on the intrinsics.
GCC 16 is not in Debian stable yet, so this currently needs it from `sid`.

__RISC-V 64 with LLVM__

```sh
cmake -B build_riscv_llvm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-llvm.cmake
cmake --build build_riscv_llvm --parallel
```

To build and run tests under emulation, see [test/README.md](test/README.md#cross-compilation).

Set `RISCV_SYSROOT` only for a self-contained toolchain; distribution cross packages need none.

__Android ARM64__

```sh
cmake -B build_android -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-android-arm64.cmake
cmake --build build_android --parallel
```

To build and run tests under emulation, see [test/README.md](test/README.md#cross-compilation).

__WASM via Emscripten__

```sh
source ~/emsdk/emsdk_env.sh
cmake -B build-wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-emscripten.cmake
cmake --build build-wasm --parallel
```

For the relaxed tier of the same 32-bit module, and for wasm64 — Memory64:

```sh
cmake -B build-wasm-relaxed -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-emscripten.cmake -DNUMKONG_WASM_SIMD=v128relaxed
cmake --build build-wasm-relaxed --parallel
cmake -B build-wasm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm64-emscripten.cmake
cmake --build build-wasm64 --parallel
```

__WASI__

```sh
export WASI_SDK_PATH=~/wasi-sdk-24.0-x86_64-linux
cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi.cmake
cmake --build build-wasi --parallel
```

For a module over an imported shared memory, which hosts with WASI threads run:

```sh
cmake -B build-wasi-threads -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi-threads.cmake
cmake --build build-wasi-threads --parallel
```

__iOS Simulator via Xcode__

```sh
xcodebuild test -scheme NumKong -destination 'platform=iOS Simulator,name=iPhone 16'
```

__x86_64 from Apple Silicon__

```sh
cmake -B build_x86 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-x86_64-llvm.cmake
cmake --build build_x86 --parallel
```

### macOS

With Apple Clang and Homebrew OpenBLAS:

```sh
brew install openblas
cmake -B build_release -D CMAKE_BUILD_TYPE=Release \
      -D NUMKONG_BUILD_TEST=1 \
      -D NUMKONG_BUILD_BENCH=1 \
      -D NUMKONG_COMPARE_TO_BLAS=1 \
      -D CMAKE_PREFIX_PATH="$(brew --prefix openblas)" \
      -D CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES="$(brew --prefix openblas)/include"
cmake --build build_release --config Release --parallel
```

With Homebrew Clang — recommended for full ISA support:

```sh
brew install llvm openblas
unset DEVELOPER_DIR
cmake -B build_release -D CMAKE_BUILD_TYPE=Release \
      -D NUMKONG_BUILD_TEST=1 \
      -D NUMKONG_BUILD_BENCH=1 \
      -D NUMKONG_COMPARE_TO_BLAS=1 \
      -D CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES="$(brew --prefix openblas)/include" \
      -D CMAKE_C_LINK_FLAGS="-L$(xcrun --sdk macosx --show-sdk-path)/usr/lib" \
      -D CMAKE_EXE_LINKER_FLAGS="-L$(xcrun --sdk macosx --show-sdk-path)/usr/lib" \
      -D CMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
      -D CMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
      -D CMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)" \
      -D CMAKE_OSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion)
cmake --build build_release --config Release --parallel
```

### BLAS Threading

When benchmarking with BLAS cross-validation, disable multi-threading in BLAS libraries to avoid interference — see [bench/README.md](bench/README.md#environment-variables) for the `*_NUM_THREADS` variables.

### Debugger Breakpoints

Useful breakpoints for debugging:

- `__asan::ReportGenericError` — illegal memory accesses.
- `__GI_exit` — exit points at end of any executable.
- `__builtin_unreachable` — unexpected code paths.
- `abort` — failed `nk_assert_` invariant checks in `NUMKONG_DEBUG` builds.

See [test/README.md](test/README.md) for test framework details and [bench/README.md](bench/README.md) for benchmark configuration.

### Static Analysis & Formatting

Once done editing the code, please run analyzers and formatters:

```bash
git ls-files '*.h' '*.c' '*.hpp' '*.cpp' | xargs clang-format -i # Use Clang Format 22 or newer
```

## Python

Python bindings are implemented using pure CPython, so you wouldn't need to install SWIG, PyBind11, or any other third-party library.
Still, you need a virtual environment.
If you already have one:

```sh
pip install -e .                                    # build locally from source
pip install pytest pytest-repeat pytest-randomly    # testing dependencies
pip install numpy scipy ml_dtypes tabulate          # optional reference libraries
pytest test/ -s -x -Wd                              # to run tests

# to check supported SIMD instructions:
python -c "import numkong; print(numkong.get_capabilities_available())"
```

Alternatively, use `uv` to create the virtual environment.

```sh
uv venv --python 3.13t          # or your preferred version
source .venv/bin/activate       # activate the environment
uv pip install -e .             # build locally from source

# to run GIL-related tests in a free-threaded environment:
uv pip install pytest pytest-repeat pytest-randomly numpy scipy ml_dtypes tabulate
PYTHON_GIL=0 python -m pytest test/ -s -x -Wd -k gil
```

Here, `-s` will output the logs.
The `-x` will stop on the first failure.
The `-Wd` will silence overflows and runtime warnings.

When building on macOS, same as with C/C++, use non-Apple Clang version:

```sh
brew install llvm libomp
CC=$(brew --prefix llvm)/bin/clang CXX=$(brew --prefix llvm)/bin/clang++ pip install -e .
```

Wheels pin a portable per-arch baseline by default — see [Target Baseline Policy](#target-baseline-policy).
For host-tuned local installs, set `NUMKONG_TARGET_ARCH=native pip install -e .` (the resulting build is not redistributable).

Before merging your changes you may want to test your changes against the entire matrix of Python versions NumKong supports.
For that you need the `cibuildwheel`, which is tricky to use on macOS and Windows, as it would target just the local environment.
Still, if you have Docker running on any desktop OS, you can use it to build and test the Python bindings for all Python versions for Linux:

```sh
pip install cibuildwheel
cibuildwheel
cibuildwheel --platform linux                   # works on any OS and builds all Linux backends
cibuildwheel --platform linux --archs x86_64    # 64-bit x86, the most common on desktop and servers
cibuildwheel --platform linux --archs aarch64   # 64-bit Arm for mobile devices, Apple M-series, and AWS Graviton
cibuildwheel --platform linux --archs i686      # 32-bit Linux
cibuildwheel --platform macos                   # works only on macOS
cibuildwheel --platform windows                 # works only on Windows
```

You may need root privileges for multi-architecture builds:

```sh
sudo $(which cibuildwheel) --platform linux
```

On Windows and macOS, to avoid frequent path resolution issues, you may want to use:

```sh
python -m cibuildwheel --platform windows
```

### Static Analysis & Formatting

Once done editing the code, please run analyzers and formatters:

```bash
ruff check .   # lint (docstrings, imports, bugbears — see pyproject [tool.ruff])
ruff format .  # format (replaces Black; same 120-column width)
```

Configuring the CMake build (`cmake -B build ...`) arms the repo's Git hooks (`core.hooksPath -> .githooks`), which re-run these formatters plus `clang-format` / `cmake-format` on staged changes and enforce the `<Verb>: <Summary>` commit message. Bypass knowingly with `git commit --no-verify`.

## Rust

```sh
cargo test -p numkong
cargo test -p numkong -- --nocapture      # to see the output
NUMKONG_TARGET_ARCH=native cargo build --release   # for host-tuned local builds
```

The crate pins a portable per-arch baseline by default — see [Target Baseline Policy](#target-baseline-policy).

To automatically detect the Minimum Supported Rust Version — MSRV:

```sh
cargo +stable install cargo-msrv
cargo msrv find --ignore-lockfile
```

Please avoid the temptation of using macros in this Rust code.

## JavaScript

See [javascript/README.md](javascript/README.md) for JavaScript/TypeScript development, WASM support, and API documentation.

Quick reference:

```sh
npm run build-js        # Build TypeScript
npm test                # Run tests
npm run bench           # Run benchmarks
```

## Swift

```sh
swift build && swift test -v
```

Running Swift on Linux requires a couple of extra steps, as the Swift compiler is not available in the default repositories.
Please get the most recent Swift tarball from the [official website](https://www.swift.org/install/).
At the time of writing, for 64-bit Arm CPU running Ubuntu 22.04, the following commands would work:

```bash
wget https://download.swift.org/swift-5.9.2-release/ubuntu2204-aarch64/swift-5.9.2-RELEASE/swift-5.9.2-RELEASE-ubuntu22.04-aarch64.tar.gz
tar xzf swift-5.9.2-RELEASE-ubuntu22.04-aarch64.tar.gz
sudo mv swift-5.9.2-RELEASE-ubuntu22.04-aarch64 /usr/share/swift
echo "export PATH=/usr/share/swift/usr/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

You can check the available images on [`swift.org/download` page](https://www.swift.org/download/#releases).
For x86 CPUs, the following commands would work:

```bash
wget https://download.swift.org/swift-5.9.2-release/ubuntu2204/swift-5.9.2-RELEASE/swift-5.9.2-RELEASE-ubuntu22.04.tar.gz
tar xzf swift-5.9.2-RELEASE-ubuntu22.04.tar.gz
sudo mv swift-5.9.2-RELEASE-ubuntu22.04 /usr/share/swift
echo "export PATH=/usr/share/swift/usr/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

Alternatively, on Linux, the official Swift Docker image can be used for builds and tests:

```bash
sudo docker run --rm -v "$PWD:/workspace" -w /workspace swift:5.9 /bin/bash -cl "swift build -c release --static-swift-stdlib && swift test -c release --enable-test-discovery"
```

## GoLang

```sh
go test ./golang/ # To test
go test -run=^$ -bench=. -benchmem ./bench/golang/ # To benchmark
```

## Adding a New Kernel Family

To add a new operation family, for example `foo`:

1. __C header__: create `include/numkong/foo.h` with serial implementation and dispatch function signatures.
2. __ISA implementations__: add `include/numkong/foo/serial.h`, `foo/neon.h`, `foo/haswell.h`, etc.
3. __Dispatch layer__: add entries to the appropriate `c/dispatch_*.c` files for each dtype the kernel supports.
4. __C++ wrapper__: create `include/numkong/foo.hpp` with the typed C++ API.
5. __Test__: create `test/foo.cpp` with precision validation against `f118_t` references.
6. __Benchmark__: create `bench/foo.cpp` with Google Benchmark harness.
7. __Cross-platform tests__: add a scenario to `test/cross.cuh`, then register it in the relevant `test/cross_*.cpp` files and, for CUDA kernels, in `test/main.cu`.
8. __CMakeLists.txt__: wire the new source files into the `numkong_test` and `numkong_bench` targets.
9. __Language bindings__: update `python/numkong.c`, `javascript/numkong.c`, `rust/numkong.rs`, etc. as needed.

## Adding a Backend Kernel to an Existing Family

For primary kernels, every backend implementation should be wired in five places beyond the backend header itself:

1. __Forward declaration__: add the `NUMKONG_API_COMPTIME` declaration with the matching `@copydoc` in the first half of `include/numkong/<family>.h`.
2. __Compile-time dispatch__: add the `#if !NUMKONG_RUNTIME_DISPATCH` branch in the second half of `include/numkong/<family>.h`.
3. __Run-time dispatch__: add the dtype-specific entry to the relevant `c/dispatch_*.c` table.
4. __Precision tests__: register the kernel in `numkong_test`, usually in the existing `test/<family>.cpp` suite.
5. __Benchmarks__: register the kernel in `numkong_bench`, usually in the existing `bench/<family>.cpp` suite.

Use the existing family suite unless the kernel introduces a genuinely new test shape.
The rule is about coverage and reachability, not about creating a brand new source file for every symbol.

There are two intentional exceptions:

- `cast`: the family-level `nk_cast_*` kernels follow the same header/dispatch/test/bench rule, but scalar conversion helpers are wired through `c/dispatch_other.c` and are covered through `test/cast.cpp` and `bench/cast.cpp`.
- `scalar`: scalar helpers are centrally declared in `include/numkong/scalar.h`, wired through `c/dispatch_other.c`, and currently do not follow the per-helper `numkong_test` and `numkong_bench` registration pattern.

## Wording & Styling

A lot of effort goes into keeping the wording and styling of the code consistent.
Variable names must reflect the __semantic operation__, not just the intrinsic name.

### Variable Names & Type Suffixes

Reading mixed-precision kernels can be very confusing when different wide registers encode numbers differently.
So most of the kernel code encodes the inner register representation into the symbol name:

- Fixed-width ISAs (NEON, x86, WASM) use `<name>_<dtype>x<count>` variable naming convention — e.g. `sum_f32x4`, `a_f64x2`, `query_f64x8`.
- SVE uses `<name>_<dtype>x` with no count, since VL is runtime — e.g. `a_f32x`, `accumulator_f64x`.
- RVV uses `<name>_<dtype>m<lmul>` for the LMUL register-group multiplier — e.g. `a_f32m1`, `sum_f64m2`.

For the `<name>` part, prefer full words over abbreviations: `accumulator` instead of `acc`, `sum` instead of `s`, `low` & `high` instead of `lo` & `hi`.
Regardless of the intrinsic name used to produce a value, the variable name should reflect its relation to surrounding code.

> A good example is naming upcasted register halves.
> With `svunpklo` & `svunpkhi` in SVE, `vget_low` & `vget_high` in NEON, or `_mm256_extractf128` in x86, the values are contiguous halves of the register — so we call them `low` and `high`:
>
> ```c
> svfloat32_t values_low_f32x  = svreinterpret_f32_u32(svlsl_n_u32_x(p, svunpklo_u32(raw), 16));
> svfloat32_t values_high_f32x = svreinterpret_f32_u32(svlsl_n_u32_x(p, svunpkhi_u32(raw), 16));
> ```
>
> But `svcvt` & `svcvtlt` in SVE select interleaved even/odd elements, not contiguous halves.
> Using `low` & `high` here would mislead reviewers into assuming a different control flow.
> So we compensate the non-expressive intrinsic name with a more accurate variable name:
>
> ```c
> svfloat32_t values_even_f32x = svcvt_f32_f16_x(pred_even_b32x, values_f16x);   // elements 0,2,4,...
> svfloat32_t values_odd_f32x  = svcvtlt_f32_f16_x(pred_odd_b32x, values_f16x);  // elements 1,3,5,...
> ```
>
> Similarly, in AMX tile-based GEMMs, the A matrix is split into a top half and a bottom half, while B tiles cover left and right halfs.
> Using `high` & `low` would suggest register halves; `top` & `bottom` reflects the spatial role in the matrix multiplication:
>
> ```c
> _tile_loadd(0, a_tile_top, a_stride_bytes);         // A top rows
> _tile_loadd(1, a_tile_bottom, a_stride_bytes);      // A bottom rows
> _tile_loadd(2, b_tile_left, 64);                    // B left columns
> _tile_loadd(3, b_tile_right, 64);                   // B right columns
> ```

For the `<dtype>` part, values like `u8`, `bf16`, `f64c`, `i4`, and `e3m2` are used — except where the type doesn't matter, such as predicate masks, loads, and stores.
Those use `b32` or `b8`, reflecting the number of bits in each mask element.

---

For scalar variables, similar preferences for cleaner and longer variable names apply:

- Loop variables use `i` for simple loops; `row_tile_index`, `column_tile_index`, `depth_step` for nested tile loops.
- Matrix / GEMM dimensions use `rows`, `columns`, `depth` — never single-letter `m`, `n`, `k`.
- Tile terminology is descriptive: `tile_dimension`, `row_in_tile`, `column_within_tile`.
- Element counts are explicit about what's counted: `count_scalars`, `count_pairs`.
- Strides explicitly mention the units: `a_stride_in_bytes`, `a_stride_elements = a_stride_in_bytes / sizeof(nk_f16_t)`.

### Intrinsic Style

Prefer explicit named intrinsics over implicit syntax or manual bit manipulation.
Power VSX uses `vec_xl()`, `vec_xst()` — never implicit Altivec vector operators.
x86 AVX-512 uses `_mm512_mask_*` K-mask intrinsics — never manual bitwise ops on `__mmask16`.
When hardware has no intrinsic, wrap raw assembly in a `NUMKONG_HELPER_INLINE` helper and document the instruction mnemonic:

```c
NUMKONG_HELPER_INLINE void nk_sme_start_streaming_(void) {
    __asm__ __volatile__("smstart sm" ::: "memory");
}
```

### Function Naming

Public API: `nk_<operation>_<dtype>_<isa>` — e.g. `nk_dot_f32_sve`, `nk_angular_f16_sme`.
Internal helpers use a trailing underscore: `nk_reduce_add_f32x16_skylake_`.
Conversions: `nk_<src>x<count>_to_<dst>x<count>_<isa>_` — e.g. `nk_e4m3x8_to_f32x8_haswell_`.

### Macro Naming

Every all-caps name starts with the full project name, `NUMKONG_`.
A trailing `_` marks a name as internal: it may change in any release, and nothing outside this repository may define or test it.
A name without it is a public contract, either a switch you may set or a value you may read.

| Family                       | Form                       | Example                                   |
| :--------------------------- | :------------------------- | :---------------------------------------- |
| ISA tier, backend, GPU layer | `NUMKONG_TARGET_<TIER>`    | `NUMKONG_TARGET_HASWELL`                  |
| Dispatch mode                | `NUMKONG_RUNTIME_DISPATCH` |                                           |
| Permission for a liberty     | `NUMKONG_ALLOW_<LIBERTY>`  | `NUMKONG_ALLOW_ISA_REDIRECT`              |
| Architecture fact            | `NUMKONG_ARCH_<ARCH>_`     | `NUMKONG_ARCH_X86_64_`                    |
| Operating-system fact        | `NUMKONG_OS_<OS>_`         | `NUMKONG_OS_LINUX_`                       |
| Toolchain fact               | `NUMKONG_HAS_<FEATURE>_`   | `NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_` |

Architectures are spelled `X86_64`, `X86_32`, `ARM64`, `RISCV64`, `PPC64`, `LOONGARCH64`, `S390X` and `WASM`.
Every name in these families is always defined, as 0 or 1, and tested with `#if`, never with `defined(...)`.
