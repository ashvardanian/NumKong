# Contributing

To keep the quality of the code high, we follow the [coding style and conventions](https://github.com/ashvardanian/ashvardanian/blob/master/CONTRIBUTING.md) shared across multiple projects — covering Git history, C++ and Python formatting, dependency management, and documentation.

## Directory Tree

```
include/numkong/          C and C++ headers — one .h per kernel family with its dispatch points, one .hpp per C++ API
include/numkong/*/        Kernels, one file per CPU or GPU capability — serial, haswell, neon, sme, rvv, ampere, etc.
c/                        Library units — per-capability kernels, per-family dispatch points, binding thread pools
probes/                   One compile probe per kit, `<kit>.c`, which CMake runs for every binding
test/                     C++ precision tests — see test/README.md
bench/                    C++ benchmark suite and JS bench runner — see bench/README.md
python/                   CPython extension, no SWIG or PyBind11
javascript/               Node.js native addon + Emscripten WASM + TypeScript API
rust/                     Rust FFI bindings
swift/                    Swift Package Manager bindings
golang/                   Go cgo bindings
cmake/                    The ISA probe module, the package config, and cross-compilation toolchain files
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

`debug`, `cuda`, `rocm`, `metal`, the `linux_<arch>` cross builds under QEMU, Android and WASM follow the same three commands.
Machine-specific settings, like a compiler path, belong in an untracked `CMakeUserPresets.json`.

| CMake Flag                    | Default                         | Description                                                                   |
| :---------------------------- | :------------------------------ | :---------------------------------------------------------------------------- |
| `NUMKONG_BUILD_TEST`          | `OFF`                           | Compile precision tests, and with the shared library also tests against it    |
| `NUMKONG_BUILD_BENCH`         | `OFF`                           | Compile micro-benchmarks                                                      |
| `NUMKONG_BUILD_PYTHON`        | `OFF`                           | Compile the CPython extension, as `pip install .` does through scikit-build   |
| `NUMKONG_BUILD_NODE`          | `OFF`                           | Compile the Node addon, as `npm run build-native` does through `cmake-js`     |
| `NUMKONG_BUILD_SWIFT`         | `OFF`                           | Package the static library for SwiftPM, as the `swift` preset does            |
| `NUMKONG_BUILD_SHARED`        | `ON`, if top-level              | Compile dynamic library                                                       |
| `NUMKONG_INSTALL`             | `ON`, if top-level              | Install headers, libraries, and the CMake package files                       |
| `NUMKONG_ENABLE_ASAN`         | `ON`, if top-level              | Enable AddressSanitizer in Debug builds                                       |
| `NUMKONG_BUILD_CUDA`          | `OFF`                           | Add CUDA to the libraries and to whichever of the test and bench builds is on |
| `NUMKONG_BUILD_ROCM`          | `OFF`                           | Add ROCm, compiled through HIP, to the libraries and to the test build        |
| `NUMKONG_BUILD_METAL`         | `OFF`                           | Add Metal to the libraries and to the test and bench builds that are on       |
| `NUMKONG_CUDA_ARCHITECTURES`  | Turing to Blackwell             | CUDA codes to compile, like `100f-real` for one family                        |
| `NUMKONG_ROCM_ARCHITECTURES`  | `gfx942;gfx950;gfx1250;gfx1251` | AMD codes to compile, like `gfx942` for one                                   |
| `NUMKONG_METAL_ARCHITECTURES` | `apple7;apple9;apple10`         | Apple GPU families whose capabilities to embed, like `apple7` without Apple10 |
| `NUMKONG_COMPARE_TO_BLAS`     | `AUTO`                          | Include OpenBLAS, or Apple's Accelerate on macOS, into test/bench comparisons |
| `NUMKONG_COMPARE_TO_MKL`      | `AUTO`                          | Include Intel MKL into test/bench comparisons                                 |
| `NUMKONG_COMPARE_TO_CUBLAS`   | `ON`                            | Include cuBLASLt and cuBLAS into CUDA benchmarks                              |
| `NUMKONG_COMPARE_TO_CUDNN`    | `OFF`                           | Include cuDNN attention into CUDA benchmarks, from `NUMKONG_CUDNN_ROOT`       |
| `NUMKONG_COMPARE_TO_CUVS`     | `OFF`                           | Include cuVS distances into CUDA benchmarks, from `NUMKONG_CUVS_ROOTS`        |
| `NUMKONG_TARGET_ARCH`         | `$NUMKONG_TARGET_ARCH`          | Tune for a CPU, like the host with `native`, or pick a WASM `v128relaxed`     |
| `NUMKONG_TARGET_<KIT>`        | whether `probes/<kit>.c` builds | `0` drops a kit, and `1` keeps one only where its probe compiles              |

Every binding builds the static library through this `CMakeLists.txt`, or links one it built, so CMake is the one place that probes which kits the toolchain builds.
Each kit's probe, `probes/<kit>.c`, calls one of its kernels, compiled header-only at the baseline flags as the library compiles it.
Kits are scoped per function by target pragmas on every platform, except LASX and POWER9, which compile file-wide with `-mlasx` and `-mcpu=power9` in their own unit, probe, and header-only test alone.
The library compiles every kit the toolchain builds and dispatches between them by runtime detection, and `-D NUMKONG_TARGET_<KIT>=0` or `=1` overrides a probe.
Each binding hands CMake its options its own way:

| Binding | Builds                                            | Passing another option                                                  |
| :------ | :------------------------------------------------ | :---------------------------------------------------------------------- |
| Python  | `NUMKONG_BUILD_PYTHON`, from `pyproject.toml`     | `pip install . -C cmake.define.NUMKONG_BUILD_CUDA=ON`                   |
| Rust    | `numkong_static`, from `build.rs`                 | `cargo build --features cuda`, or `rocm` for `NUMKONG_BUILD_ROCM`       |
| Node    | `NUMKONG_BUILD_NODE`, from `package.json`         | `npm run build-native -- --CDNUMKONG_TARGET_ARCH=native`                |
| Go      | `numkong_static`, built by hand for `cgo`         | `cmake -D NUMKONG_TARGET_ARCH=native`, then `--target numkong_static`   |
| Swift   | `NUMKONG_BUILD_SWIFT`, from the `swift` preset    | `cmake --preset swift -D NUMKONG_TARGET_ARCH=native`                    |
| WASM    | the `wasm*_emscripten` and `wasm32_wasi*` presets | `cmake --preset wasm32_wasi -D NUMKONG_TARGET_ARCH=v128relaxed`         |

Rust links an archive CMake already built when `NUMKONG_LIBRARY_DIR` names its directory, and Go always does.
On WebAssembly, `build.rs` hands CMake the toolchain the Rust target needs and the SIMD capability its target features declare, as [rust/README.md](rust/README.md#webassembly) lists.
`numkong_static` needs nothing beyond the C library, so its consumers add only what their own sources need: `libm` for Python's, and OpenMP where `c/parallel.c` joins the Python extension or the Node addon.
A GPU build adds the runtime it compiled against, `cudart` or `amdhip64`, which the CMake targets carry and `build.rs` names.

The test suites seed from 42, or from a fresh draw under `NUMKONG_SEED=random`, and print the seed they use.
`NUMKONG_FILTER` is a regex over kernel names, and each failing kernel prints a `rerun:` line with its seed and a filter selecting it alone.
Failures are always counted and exit 1, unless `NUMKONG_ASSERT=0` turns them into a report only.
The [test README](test/README.md#environment-variables) lists every variable.

### Target Baseline Policy

`CMakeLists.txt`, which every binding builds through, pins the TU-level baseline to each architecture's ABI floor so distributable artifacts run on any CPU matching the ABI, not just the build host.
SIMD kernels live inside `#pragma GCC target(...)` regions and run only when the capability mask holds their capability — see the README's [Dispatch Points & Capability Masks](README.md#dispatch-points--capability-masks) section.

| Target arch   | GCC/Clang baseline   | MSVC baseline   | Notes                                                       |
| :------------ | :------------------- | :-------------- | :---------------------------------------------------------- |
| `x86_64`      | `-march=x86-64`      | `/arch:SSE2`    | System V psABI / Microsoft x64 ABI floor; SSE2 is mandatory |
| `aarch64`     | `-march=armv8-a`     | `/arch:armv8.0` | ARMv8-A ABI floor; NEON is mandatory                        |
| `riscv64`     | `-march=rv64gc`      | …               | V extension is runtime-probed and dispatched                |
| `powerpc64le` | `-mcpu=power8`       | …               | ELFv2 ABI floor (VSX is mandatory); POWER9 is dispatched    |
| `loongarch64` | `-march=loongarch64` | …               | LASX is runtime-probed and dispatched                       |

GCC/Clang builds also pass `-fno-tree-vectorize -fno-tree-slp-vectorize` so the auto-vectorizer cannot promote serial fallbacks to baseline SIMD (NEON, SSE2, VSX, …).
That keeps the capability dispatch design intact: "serial" kernels stay actually serial, and the per-pragma SIMD kernels — which use explicit intrinsics, not vectorized scalar code — are the sole source of SIMD emission.
MSVC has no per-function target pragma and no command-line vectorizer toggle, so the explicit `/arch:` flags above match defaults and document intent only; NumKong's MSVC strategy is compile-time gating via `_MSC_VER` version checks (see `include/numkong/types.h`).
LASX and POWER9 are the two kits compiled file-wide, because Clang's `lasxintrin.h` and `altivec.h` hide their contents without `-mlasx` or `-mcpu=power9`.
CMake gives that flag to `c/cpu/loongsonasx.c` or `c/cpu/powervsx.c`, to the kit's probe, and to `test/cross_loongarch64.cpp` or `test/cross_ppc64.cpp` in the header-only test alone, so every other unit stays at the baseline and runs on LASX-less and POWER8 hosts.

For host-tuned local builds, set `NUMKONG_TARGET_ARCH=native`, either as the CMake option `-DNUMKONG_TARGET_ARCH=native` or as an environment variable.
`pip install` reads the variable on every build, while `cargo build` and `npm run build-native` read it when they first configure their build directory.
The resulting artifact bakes host-specific instructions into scaffolding code and is __not__ portable.

### Compiler Requirements

| ISA Family                              |  GCC | Clang | AppleClang     |        MSVC |
| :-------------------------------------- | ---: | ----: | :------------- | ----------: |
| Base — serial, NEON, AVX2               |   9+ |   10+ | Any            |       2019+ |
| Float16 — NEONHalf, Sapphire FP16, Zvfh |  12+ |   16+ | Any            | 2022 17.14+ |
| AVX-512 — Skylake, Ice Lake             |   9+ |   10+ | …              |       2019+ |
| AVX-512BF16 — Genoa                     |  12+ |   16+ | …              | 2022 17.14+ |
| Intel AMX — Sapphire, Granite           |  14+ |   18+ | …              | 2022 17.14+ |
| Arm SME                                 |  14+ |   18+ | 16+ / Xcode 16 |           … |
| RISC-V Vector — RVV 1.0                 |  16+ |   17+ | …              |           … |
| RVV + Zvfh/Zvfbfwma/Zvbb                |  16+ |   18+ | …              |           … |

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

| Target                  | Toolchain             | Emulator                      | Prerequisites                                   |
| :---------------------- | :-------------------- | :---------------------------- | :---------------------------------------------- |
| ARM64 Linux             | `aarch64-gnu`         | `qemu-aarch64 -cpu max`       | `gcc-aarch64-linux-gnu`                         |
| RISC-V 64 LLVM          | `riscv64-llvm`        | `qemu-riscv64 -cpu max`       | Clang 21+, `gcc-riscv64-linux-gnu`              |
| RISC-V 64 GCC           | `riscv64-gnu`         | `qemu-riscv64 -cpu max`       | GCC 16+, `gcc-riscv64-linux-gnu`                |
| ppc64le Linux           | `ppc64le-gnu`         | `qemu-ppc64le -cpu power10`   | `gcc-powerpc64le-linux-gnu`                     |
| LoongArch 64            | `loongarch64-gnu`     | `qemu-loongarch64 -cpu la464` | `gcc-loongarch64-linux-gnu`                     |
| Android ARM64           | `android-arm64`       | …                             | `ANDROID_NDK_ROOT`                              |
| Android ARMv7           | `android-armv7`       | …                             | `ANDROID_NDK_ROOT`                              |
| x86_64 on Apple Silicon | `x86_64-llvm`         | `arch -x86_64`                | Homebrew LLVM                                   |
| WASM32 Emscripten       | `wasm32-emscripten`   | Node.js                       | Emscripten 3.1.27+, `v128` by default           |
| WASM64 Emscripten       | `wasm64-emscripten`   | Node.js 24+                   | Emscripten 3.1.35+, `v128relaxed` by default    |
| WASI                    | `wasm32-wasi`         | Wasmtime / Wasmer             | WASI SDK 24+, `v128` by default                 |
| WASI threads            | `wasm32-wasi-threads` | Wasmtime with threads         | WASI SDK 24+, `v128relaxed` by default          |

Each QEMU toolchain names its emulated core in `<ARCH>_QEMU_CPU`, the richest by default so the tests run every compiled kit.
A baseline leg passes the baseline core instead, which runs the library without any kit: `-D AARCH64_QEMU_CPU=cortex-a53`, `-D RISCV_QEMU_CPU=rv64`, `-D PPC_QEMU_CPU=power8`, or `-D LOONGARCH_QEMU_CPU=la464,lsx=off,lasx=off`.

A WebAssembly module carries one SIMD capability, `serial`, `v128` or `v128relaxed`, and `NUMKONG_TARGET_ARCH` selects it for every toolchain and binding.
Each toolchain file sets its default, and `CMakeLists.txt` turns the choice into `-msimd128` and `-mrelaxed-simd` for every unit, whatever flags a binding passes.
Emscripten names each module after its capability, so both wasm32 modules can share the preset's build directory.

The Linux and Android recipes below build the tests too, and `ctest --test-dir <build>` runs them under the emulator the table names; [test/README.md](test/README.md#wasm) covers the WASM runtimes.
Set `NUMKONG_IN_QEMU=1` to shrink test shapes under emulation, and repetitions too in Python.

__ARM64 Linux__

```sh
cmake -B build_arm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-gnu.cmake -DNUMKONG_BUILD_TEST=1
cmake --build build_arm64 --parallel
NUMKONG_IN_QEMU=1 ctest --test-dir build_arm64 # runs under qemu-aarch64 -cpu max
```

The ISA floor is `armv8-a`; individual kernels are gated by the compile probes in `probes/`.
GCC 14 builds and links the SME kernels, since `dots/sme.h` carries weak `__arm_tpidr2_save` and `__arm_tpidr2_restore` stubs.

__RISC-V 64 with GCC__

```sh
cmake -B build_riscv -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-gnu.cmake -DNUMKONG_BUILD_TEST=1
cmake --build build_riscv --parallel
NUMKONG_IN_QEMU=1 ctest --test-dir build_riscv # runs under qemu-riscv64 -cpu max
```

The ISA floor is `rv64gc`, and the RVV kits dispatch at runtime.
Needs GCC 16 or newer: the RVV kernels gate on `#pragma GCC target("arch=+v")`, which GCC implements for RISC-V only from 16, and 14 and 15 ignore it and then fail on the intrinsics.
GCC 16 is not in Debian stable yet, so this currently needs it from `sid`.

__RISC-V 64 with LLVM__

```sh
export LLVM_ROOT=/path/to/llvm # optional
cmake -B build_riscv_llvm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-llvm.cmake -DNUMKONG_BUILD_TEST=1
cmake --build build_riscv_llvm --parallel
NUMKONG_IN_QEMU=1 ctest --test-dir build_riscv_llvm
```

Set `RISCV_SYSROOT` only for a self-contained toolchain; distribution cross packages need none.

__Android ARM64__

Android has no emulator in the table, so the tests run on a device:

```sh
cmake -B build_android -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-android-arm64.cmake -DNUMKONG_BUILD_TEST=1
cmake --build build_android --parallel
adb push build_android/numkong_cpu_test /data/local/tmp/
adb shell /data/local/tmp/numkong_cpu_test
```

__WASM via Emscripten__

```sh
source ~/emsdk/emsdk_env.sh
cmake --preset wasm32_emscripten                                     # numkong-wasm32-v128
cmake --build --preset wasm32_emscripten
```

For the `v128relaxed` module beside it, and for wasm64 — Memory64:

```sh
cmake --preset wasm32_emscripten -D NUMKONG_TARGET_ARCH=v128relaxed  # numkong-wasm32-v128relaxed
cmake --build --preset wasm32_emscripten
cmake --preset wasm64_emscripten                                     # numkong-wasm64-v128relaxed
cmake --build --preset wasm64_emscripten
```

__WASI__

```sh
export WASI_SDK_PATH=~/wasi-sdk                                      # the default, when unset
cmake --preset wasm32_wasi
cmake --build --preset wasm32_wasi
ctest --preset wasm32_wasi
```

For a module over an imported shared memory, which hosts with WASI threads run:

```sh
cmake --preset wasm32_wasi_threads
cmake --build --preset wasm32_wasi_threads
ctest --preset wasm32_wasi_threads
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

### C and C++ Formatting

Once done editing the code, please run analyzers and formatters:

```bash
git ls-files '*.h' '*.c' '*.hpp' '*.cpp' | xargs clang-format -i # Use Clang Format 22 or newer
```

## Python

Python bindings are implemented using pure CPython, so you wouldn't need to install SWIG, PyBind11, or any other third-party library.
Still, you need a virtual environment.
If you already have one:

```sh
pip install -e . --group test                       # build locally from source, with the test dependencies
pip install --group test-oracles                    # optional reference libraries
python -X faulthandler -m pytest -x                 # to run tests

# to check supported SIMD instructions:
python -c "import numkong; print(repr(numkong.cpu_capabilities_enabled()))"
```

Alternatively, use `uv` to create the virtual environment.

```sh
uv venv --python 3.13t          # or your preferred version
source .venv/bin/activate       # activate the environment
uv pip install -e .             # build locally from source

# to run GIL-related tests in a free-threaded environment:
uv pip install --group test --group test-oracles
PYTHON_GIL=0 python -m pytest -x -k gil
```

Here, `-x` will stop on the first failure.
Warnings are errors, as `pyproject.toml` configures, and the header prints every `NUMKONG_*` setting the run uses.

When building on macOS, same as with C/C++, use non-Apple Clang version:

```sh
brew install llvm
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

### Python Linting & Formatting

Once done editing the code, please run analyzers and formatters:

```bash
ruff check .   # lint (docstrings, imports, bugbears — see pyproject [tool.ruff])
ruff format .  # format (replaces Black; same 120-column width)
```

Configuring the CMake build (`cmake -B build ...`) arms the repo's Git hooks (`core.hooksPath` → `.githooks`), which re-run these formatters plus `clang-format` / `cmake-format` on staged changes and enforce the `<Verb>: <Summary>` commit message.
Bypass knowingly with `git commit --no-verify`.

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

SwiftPM links the static library CMake builds: an XCFramework on Apple platforms and an [SE-0482](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0482-swiftpm-static-library-binary-target-non-apple-platforms.md) artifact bundle elsewhere.
A checkout builds its own and points `NUMKONG_SWIFT_ARTIFACT` at it, relative to the package root, while every other consumer downloads the release's:

```sh
cmake --preset swift && cmake --build --preset swift
export NUMKONG_SWIFT_ARTIFACT=build_swift/CNumKong.xcframework # or build_swift/CNumKong.artifactbundle off Apple platforms
swift build && swift test -v
```

A package depending on this one by path, as USearch does, reads the same variable.
The manifest reads it from the environment of whatever evaluates it, `swift`, `xcodebuild` or Xcode, so an Xcode started from the Dock does not see a shell's `export`.
Each build adds its target to the artifact in `NUMKONG_SWIFT_DIRECTORY`, which defaults to the build directory, so a simulator slice joins the macOS one:

```sh
cmake --preset swift -B build_swift_iossim -D NUMKONG_SWIFT_DIRECTORY=$PWD/build_swift \
    -D CMAKE_SYSTEM_NAME=iOS -D CMAKE_OSX_SYSROOT=iphonesimulator -D CMAKE_OSX_ARCHITECTURES=arm64 \
    -D CMAKE_OSX_DEPLOYMENT_TARGET=15.0 -D NUMKONG_TARGET_SME=0 -D NUMKONG_TARGET_SMEF64=0 \
    -D NUMKONG_TARGET_SMEBI32=0
cmake --build build_swift_iossim
```

The simulators lack the `__sme_memset` routine streaming SME code calls, hence the SME verdicts turned off.
`.github/workflows/_swift.yml` builds every slice and triple the release ships, and `release.yml` writes their checksums into `Package.swift` before tagging.
On Linux, `swift package experimental-audit-binary-artifact build_swift/CNumKong.artifactbundle` checks that the archive needs nothing beyond the C library.

`Package.swift` declares `swift-tools-version:6.4`, so the package needs Swift 6.4 or newer.
Running Swift on Linux requires a couple of extra steps, as the Swift compiler is not available in the default repositories.
Please get the most recent Swift tarball from the [official website](https://www.swift.org/install/).
At the time of writing, for 64-bit Arm CPU running Ubuntu 24.04, the following commands would work:

```bash
wget https://download.swift.org/swift-6.4.0-release/ubuntu2404-aarch64/swift-6.4.0-RELEASE/swift-6.4.0-RELEASE-ubuntu24.04-aarch64.tar.gz
tar xzf swift-6.4.0-RELEASE-ubuntu24.04-aarch64.tar.gz
sudo mv swift-6.4.0-RELEASE-ubuntu24.04-aarch64 /usr/share/swift
echo "export PATH=/usr/share/swift/usr/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

You can check the available images on [`swift.org/download` page](https://www.swift.org/download/#releases).
For x86 CPUs, the following commands would work:

```bash
wget https://download.swift.org/swift-6.4.0-release/ubuntu2404/swift-6.4.0-RELEASE/swift-6.4.0-RELEASE-ubuntu24.04.tar.gz
tar xzf swift-6.4.0-RELEASE-ubuntu24.04.tar.gz
sudo mv swift-6.4.0-RELEASE-ubuntu24.04 /usr/share/swift
echo "export PATH=/usr/share/swift/usr/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

Alternatively, on Linux, the official Swift Docker image can be used for builds and tests:

```bash
sudo docker run --rm -v "$PWD:/workspace" -w /workspace swift:6.4 /bin/bash -cl "
    apt-get update && apt-get install -y cmake &&
    cmake --preset swift -B build_swift_linux -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ &&
    cmake --build build_swift_linux &&
    NUMKONG_SWIFT_ARTIFACT=build_swift_linux/CNumKong.artifactbundle swift test -c release"
```

## GoLang

```sh
go test ./golang/ # To test
go test -run=^$ -bench=. -benchmem ./bench/golang/ # To benchmark
```

## Adding a New Kernel Family

To add a new operation family, for example `foo`:

1. __C header__: create `include/numkong/foo.h`.
   It declares, with `NUMKONG_API`, the dispatch points like `nk_foo_f32_best`, every capability's kernel under its `NUMKONG_TARGET_*` guard, and the finder `nk_foo_find_kernel`.
   It ends with the header-only tail: the capability headers, then a stub per dispatch point and for the finder, each returning `nk_missing_library_k`.
2. __Kernels per capability__: add `include/numkong/foo/serial.h`, `foo/neon.h`, `foo/haswell.h`, etc.
   Helpers compile wherever `NUMKONG_ARCH_<ARCH>_<CAPABILITY>_` holds, and kernels only under their own `NUMKONG_TARGET_<CAPABILITY>`.
   A kernel never calls another public kernel: logic two kernels share lives in a helper named for what it computes.
3. __Library__: create `c/dispatch/foo.c` with a `static` capability list per dispatch point, like `nk_foo_f32_capabilities`, the `_best` body that picks from it, and `nk_foo_find_kernel`.
   Include each capability header from its unit, like `c/cpu/haswell.c`, and route the family's kernel kinds to `nk_foo_find_kernel` in `c/numkong.c`.
4. __C++ wrapper__: create `include/numkong/foo.hpp` with the typed C++ API, ending in the dispatch point's mask and stream.
5. __Test__: create `test/foo.cpp` with precision validation against `f118_t` references.
6. __Benchmark__: create `bench/foo.cpp` over the `bench/harness.hpp` timing loop.
7. __Cross-platform tests__: add a scenario to `test/cross.hpp`, then register it in the relevant `test/cross_*` files, `test/cross_cuda.cu` and `test/cross_rocm.hip` included.
8. __CMakeLists.txt__: wire the new source files into the `numkong_cpu_test` and `numkong_bench` targets.
9. __Language bindings__: update `python/numkong.c`, `javascript/numkong.c`, `rust/numkong.rs`, etc. as needed.

## Adding a Capability to an Existing Family

A capability's kernel shares its dispatch point's signature short of the mask: it returns an `nk_status_t` and takes a trailing `void *stream`, which a CPU kernel asserts is null.
Every such kernel is wired in four places beyond its capability's header:

1. __Declaration__: add the `NUMKONG_API` declaration with the matching `@copydoc` under its `NUMKONG_TARGET_*` guard in the first half of `include/numkong/<family>.h`.
2. __Capability kernels__: add the kernel under its `NUMKONG_TARGET_*` guard to its capability group's array in `nk_<operation>_<dtype>_capabilities` in `c/dispatch/<family>.c`, in the order of the capability bits, and its bit to that group's mask.
   Its unit, like `c/cpu/haswell.c` or `c/cuda/hopper.cu`, defines it; a new capability gets a new unit.
3. __Precision tests__: register the kernel in `numkong_cpu_test`, usually in the existing `test/<family>.cpp` suite.
4. __Benchmarks__: register the kernel in `numkong_bench`, usually in the existing `bench/<family>.cpp` suite.

Use the existing family suite unless the kernel introduces a genuinely new test shape.
The rule is about coverage and reachability, not about creating a brand new source file for every symbol.

There are two intentional exceptions:

- `cast`: the family-level `nk_cast_*` kernels follow the same header, list, test, and bench rule, and the scalar conversions are dispatched in `c/dispatch/cast.c` and covered through `test/cast.cpp` and `bench/cast.cpp`.
- `scalar`: scalar helpers are centrally declared in `include/numkong/scalar.h`, dispatched in `c/dispatch/scalar.c`, and currently do not follow the per-helper `numkong_cpu_test` and `numkong_bench` registration pattern.

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
> _tile_loadd(0, a_tile_top, a_stride);         // A top rows
> _tile_loadd(1, a_tile_bottom, a_stride);      // A bottom rows
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
- Strides are always in bytes and named `a_stride`; element strides say so: `a_stride_elements = a_stride / sizeof(nk_f16_t)`.

### Intrinsic Style

Prefer explicit named intrinsics over implicit syntax or manual bit manipulation.
Power VSX uses `vec_xl()`, `vec_xst()` — never implicit Altivec vector operators.
x86 AVX-512 uses `_mm512_mask_*` K-mask intrinsics — never manual bitwise ops on `__mmask16`.
When hardware has no intrinsic, wrap raw assembly in a `NUMKONG_INLINE` helper and document the instruction mnemonic:

```c
NUMKONG_INLINE void nk_sme_start_streaming_(void) {
    __asm__ __volatile__("smstart sm" ::: "memory");
}
```

### Function Naming

Kernels: `nk_<operation>_<dtype>_<capability>` — e.g. `nk_dot_f32_sve`, `nk_dots_packed_bf16_ampere`.
Dispatch points replace the capability with `best` — e.g. `nk_dot_f32_best` — and the static lists of their kernels in every capability group add `capabilities` — e.g. `nk_dot_f32_capabilities`.
Internal helpers use a trailing underscore: `nk_reduce_add_f32x16_skylake_`.
Conversions: `nk_<src>x<count>_to_<dst>x<count>_<isa>_` — e.g. `nk_e4m3x8_to_f32x8_haswell_`.
Helpers, types and constants put the capability last as well, and only a role suffix may follow it: `_t` for a type, `_k` for a constant, `_kernel_` for a GPU entry point, or the trailing `_` of an internal name, like `nk_attention_pack_directory_simt_kernel_`.

Code that several capabilities share belongs to one of three layers, named in the capability's place:

- `serial`, portable and capability-neutral, in each family's `serial.h`.
- `simt`, the single C source that CUDA and HIP both compile, in each family's `simt.cuh`, like `nk_cast_bit_simt_`.
- `metal`, what every Metal tier shares, in a family's `metal.h` and the `metal.metal` shaders it embeds, like `nk_cross_encode_metal_`.

A constant identical across the GPU layers is defined once in `serial.h` with the adjective `gpu`, and one whose value differs per layer takes its layer instead.

### GPU Naming

The GPU capability groups are `cuda`, `rocm` and `metal`, beside the CPU's `cpu`.
Each word names its group's baseline bit, like `nk_cap_cuda_k`, its functions and its library units, like `c/cuda/hopper.cu`, and no symbol or source path names a vendor, like `nvidia`, `amd` or `apple`.
A function that touches a device is either a producer or a consumer:

- A __producer__ reports a device's capabilities or opens a stream on it.
  It starts with its group, and it is the only kind of function that takes a device's `ordinal`: `nk_cuda_count_devices(&count)`, `nk_cuda_capabilities_enabled(ordinal, &capabilities)` and `nk_cuda_stream_init(ordinal, &stream)`.
- A __consumer__ takes the `capabilities` it picks from and a trailing `void *stream`, but never an ordinal, since the stream names its device: `nk_memory_allocate_unified_best(bytes, &pointer, capabilities, stream)`.
  It ends in `best` like any dispatch point, and its twins end in their capability, like `nk_memory_allocate_unified_cuda(bytes, &pointer, stream)`.

A null stream is the default stream of the default device: the calling thread's current device on CUDA and ROCm, and the system default device on Metal.
On the CPU the stream must be null.
Each of these words has one meaning across the library:

| Word                                             | Meaning                                                                          | Appears as                                                                        |
| :----------------------------------------------- | :------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| `cpu`, `cuda`, `rocm`, `metal`                   | A capability group and its baseline bit                                          | Producer prefix, twin and kernel suffix, `c/<group>/`                             |
| `simt`                                           | The single C source CUDA and HIP both compile                                    | `<family>/simt.cuh`, the suffixes `_simt_`, `_simt_t`, `_simt_k`                  |
| `metal`, as a layer                              | What every Metal tier shares                                                     | `<family>/metal.h`, `<family>/metal.metal`, the suffix `_metal_`                  |
| `gpu`                                            | Adjective for every GPU group                                                    | `nk_cap_gpus_k`, `nk_missing_gpu_k`                                               |
| `device`                                         | A processor kernels run on, which a stream belongs to                            | `nk_cuda_count_devices`, `nk_device_memory_mismatch_k`, `nk_device_current_simt_` |
| `ordinal`                                        | A device's index within its group, as its runtime numbers it                     | Producer parameters only                                                          |
| `stream`                                         | A `cudaStream_t`, `hipStream_t` or `id<MTLCommandQueue>`, which names its device | The trailing `void *stream` of every consumer                                     |
| `unified`                                        | Memory both the host and the stream's device address                             | `nk_memory_allocate_unified_best`                                                 |
| A capability, like `hopper`, `cdna4` or `apple9` | One bit of a mask                                                                | The last token before the role suffix                                             |
| `kernel`                                         | A GPU entry point                                                                | `_kernel_`, right after the capability                                            |

### Macro Naming

Every all-caps name starts with the full project name, `NUMKONG_`.
A trailing `_` marks a name as internal: it may change in any release, and nothing outside this repository may define or test it.
A name without it is a public contract, either a switch you may set or a value you may read.

| Family                           | Form                          | Example                                   |
| :------------------------------- | :---------------------------- | :---------------------------------------- |
| Capability whose kernels compile | `NUMKONG_TARGET_<NAME>`       | `NUMKONG_TARGET_HASWELL`                  |
| Capability whose helpers compile | `NUMKONG_ARCH_<ARCH>_<NAME>_` | `NUMKONG_ARCH_X8664_HASWELL_`             |
| Header-only build                | `NUMKONG_HEADER_ONLY`         |                                           |
| GPU runtime the build links      | `NUMKONG_WITH_<RUNTIME>`      | `NUMKONG_WITH_METAL`                      |
| Permission for a liberty         | `NUMKONG_ALLOW_<LIBERTY>`     | `NUMKONG_ALLOW_ISA_REDIRECT`              |
| Architecture fact                | `NUMKONG_ARCH_<ARCH>_`        | `NUMKONG_ARCH_X8664_`                     |
| Operating-system fact            | `NUMKONG_OS_<OS>_`            | `NUMKONG_OS_LINUX_`                       |
| Toolchain fact                   | `NUMKONG_HAS_<FEATURE>_`      | `NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_` |

Architectures are spelled as one token each, `X8664`, `X8632`, `ARM64`, `RISCV64`, `PPC64`, `LOONGARCH64`, `S390X` and `WASM`, and GPU architectures `CUDA` and `ROCM`.
A GPU architecture holds beside the host's architecture in both compiler passes, so it never follows a CPU architecture in an `#elif` chain.
Metal has no compiler macro on the host side, so its host API is a switch the build sets where it links Metal and Foundation.
The library also sets `NUMKONG_ARCH_CUDA_` or `NUMKONG_ARCH_ROCM_` for its host-only units, so `c/dispatch/*.c` list the kernels that the `c/cuda/*.cu` and `c/rocm/*.hip` units compile.
The build passes every unit the same `NUMKONG_TARGET_*` verdicts, so a unit turns off each capability its headers include besides its own, like `c/cpu/genoa.c` turning off Haswell, Skylake and Icelake, and each kernel is defined in exactly one unit.
A new cross-capability include needs the same line in the unit, or the link reports the kernels defined twice.
A capability's helpers follow `NUMKONG_ARCH_<ARCH>_<NAME>_`: its own target, or any capability whose headers include its headers.
Each GPU unit compiles only the codes its generation runs.
Every name in these families is always defined, as 0 or 1, and tested with `#if`, never with `defined(...)`.
