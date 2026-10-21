# CMake is the preferred way to compile the C shared libraries, tests, and benchmarks.
# Use like:
#
#   cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D NUMKONG_BUILD_TEST=ON
#   cmake --build build_release --config Release
#   build_release/numkong_cpu_test
#
# CMake options:
#
#   -D CMAKE_BUILD_TYPE=<Release|Debug|RelWithDebInfo>
#   -D NUMKONG_BUILD_TEST=<ON|OFF>           (default: OFF, with NUMKONG_BUILD_SHARED also the shared-library test)
#   -D NUMKONG_BUILD_BENCH=<ON|OFF>          (default: OFF)
#   -D NUMKONG_BUILD_PYTHON=<ON|OFF>         (default: OFF, the CPython extension scikit-build-core packages)
#   -D NUMKONG_BUILD_NODE=<ON|OFF>           (default: OFF, the Node addon cmake-js builds)
#   -D NUMKONG_BUILD_SWIFT=<ON|OFF>          (default: OFF, the XCFramework or artifact bundle SwiftPM links)
#   -D NUMKONG_SWIFT_TRIPLES=<list>          (default: the target's, like aarch64-unknown-linux-gnu, off Apple platforms)
#   -D NUMKONG_BUILD_CUDA=<ON|OFF>           (default: OFF, adds CUDA to the libraries, tests and benchmarks built)
#   -D NUMKONG_CUDA_ARCHITECTURES=<list>     (default: 75-real;80-real;89-real;90a-real;100f-real;120f-real;80-virtual)
#   -D NUMKONG_BUILD_ROCM=<ON|OFF>           (default: OFF, adds HIP to the libraries, tests and benchmarks built)
#   -D NUMKONG_ROCM_ARCHITECTURES=<list>     (default: gfx942;gfx950;gfx1250;gfx1251)
#   -D NUMKONG_BUILD_METAL=<ON|OFF>          (default: OFF, adds Metal to the libraries, tests and benchmarks built)
#   -D NUMKONG_METAL_ARCHITECTURES=<list>    (default: apple7;apple9;apple10)
#   -D NUMKONG_COMPARE_TO_BLAS=<AUTO|ON|OFF> (default: AUTO)
#   -D NUMKONG_COMPARE_TO_MKL=<AUTO|ON|OFF>  (default: AUTO)
#   -D NUMKONG_BUILD_SHARED=<ON|OFF>         (default: ON when top-level, OFF as a subproject)
#   -D NUMKONG_WITH_HOST_PROBES=<ON|OFF>     (default: OFF, import capability probes from host)
#   -D NUMKONG_TARGET_ARCH=<arch>            (default: the NUMKONG_TARGET_ARCH environment variable, empty for the
#                                            portable ABI floor, `native`, or under the WebAssembly toolchains the
#                                            module's one SIMD capability, `serial`, `v128` or `v128relaxed`)
#   -D NUMKONG_TARGET_<CAPABILITY>=<0|1>     (default: whether `probes/<capability>.c` compiles, and 1 only where
#                                            it does)
#
# Different SIMD ISA extensions require different compiler versions:
#
#   - Basic functionality:  GCC 9+, Clang 10+, MSVC 2019+
#   - Float16 support:      GCC 12+, Clang 16+
#   - Intel AMX/Arm SME:    GCC 14+, Clang 18+
#
# To run on macOS with Homebrew LLVM:
#
#     brew install llvm openblas ccache
#     cmake -B build_release -D CMAKE_BUILD_TYPE=Release \
#           -D NUMKONG_BUILD_TEST=ON \
#           -D CMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
#           -D CMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
#           -D CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES="$(brew --prefix openblas)/include"
#     cmake --build build_release --config Release
#
# To cross-compile for ARM64:
#
#     cmake -B build_arm64 -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-gnu.cmake \
#           -D NUMKONG_BUILD_TEST=ON
#
# To cross-compile for RISC-V:
#
#     cmake -B build_riscv -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-llvm.cmake \
#           -D NUMKONG_BUILD_TEST=ON
#
# To cross-compile for Android ARM64 (requires ANDROID_NDK_ROOT):
#
#     cmake -B build_android -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-android-arm64.cmake \
#           -D NUMKONG_BUILD_SHARED=ON
#
# To build with Intel ICX (Intel oneAPI DPC++/C++ Compiler):
#
#     sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp
#     source /opt/intel/oneapi/setvars.sh
#     cmake -B build_icx -D CMAKE_BUILD_TYPE=Release \
#           -D CMAKE_C_COMPILER=icx -D CMAKE_CXX_COMPILER=icpx \
#           -D NUMKONG_BUILD_TEST=ON
#     cmake --build build_icx --config Release
#
# If your system CMake is older than 3.20, install a recent release from PyPI or the Kitware APT repo.
#
#     python -m pip install "cmake>=3.20"
#
# NumKong does not support NVHPC (nvc and nvc++): those compilers cannot apply per-function
# __attribute__((target(...))), which is how SIMD kernels are dispatched in this library.
#
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Leverage `ccache` for incremental compilation
find_program(NUMKONG_CCACHE_PROGRAM ccache)

if (NUMKONG_CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${NUMKONG_CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${NUMKONG_CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${NUMKONG_CCACHE_PROGRAM}")
    set(CMAKE_HIP_COMPILER_LAUNCHER "${NUMKONG_CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${NUMKONG_CCACHE_PROGRAM}")
    message(STATUS "IMPORTANT: Set environment variable CCACHE_SLOPPINESS=pch_defines,time_macros")
endif ()

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any
# cloned repo execute code), so point `core.hooksPath` at `.githooks` here — configuring the build arms
# the `pre-commit` / `commit-msg` checks. Skipped when built as a subproject — mutating a submodule's
# Git config is not a consumer's business — or without `.git` (tarball / export builds).
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE numkong_hooks_status_ OUTPUT_QUIET ERROR_QUIET
    )
    if (numkong_hooks_status_ EQUAL 0)
        message(STATUS "numkong: Git hooks enabled (core.hooksPath -> .githooks)")
    endif ()
endif ()

# Embedded debug information avoids depending on a running MSVC PDB server during probes.
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    if (NOT DEFINED CMAKE_MSVC_DEBUG_INFORMATION_FORMAT)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
    endif ()
endif ()

# Every binding reads its version from `VERSION`, which the release bumps.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" numkong_version_line_ LIMIT_COUNT 1)
# CMake versions are numeric, so the pre-release suffix of `VERSION` stays out of `PROJECT_VERSION`.
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" numkong_version_ "${numkong_version_line_}")
project(
    numkong
    VERSION ${numkong_version_}
    LANGUAGES C CXX
    DESCRIPTION "Portable mixed-precision BLAS-like vector math library for x86 and ARM"
    HOMEPAGE_URL "https://github.com/ashvardanian/NumKong"
)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED YES)

set(CMAKE_C_EXTENSIONS NO)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Determine if NumKong is built as a sub-project (using `add_subdirectory`) or if it is the main project
set(numkong_is_main_project_ ${PROJECT_IS_TOP_LEVEL})
set(numkong_os_wasi_ OFF)

if (CMAKE_SYSTEM_NAME STREQUAL "WASI")
    set(numkong_os_wasi_ ON)
endif ()

option(NUMKONG_BUILD_SHARED "Compile a dynamic library" ${numkong_is_main_project_})
option(NUMKONG_INSTALL "Install headers, libraries & the CMake package files" ${numkong_is_main_project_})
option(NUMKONG_BUILD_TEST "Compile precision tests, and shared-library tests with NUMKONG_BUILD_SHARED" OFF)
option(NUMKONG_BUILD_BENCH "Compile micro-benchmarks for current ISA" OFF)
option(NUMKONG_BUILD_PYTHON "Compile the CPython extension over the static library" OFF)
option(NUMKONG_BUILD_NODE "Compile the Node addon over the static library; cmake-js switches it on" OFF)
option(NUMKONG_BUILD_SWIFT "Package the static library for SwiftPM, as an XCFramework or an artifact bundle" OFF)
option(NUMKONG_BUILD_CUDA "Add CUDA to the libraries and to whichever of the test and bench builds is on" OFF)
option(NUMKONG_BUILD_ROCM
       "Add ROCm, compiled through HIP, to the libraries and to whichever of the test and bench builds is on" OFF
)
option(NUMKONG_BUILD_METAL "Add Metal to the libraries and to whichever of the test and bench builds is on" OFF)
option(NUMKONG_WITH_HOST_PROBES "Import the capability probes from the WASI host" OFF)

# `nk_cpu_capabilities_detected` lives in the library, so the imports reach every unit, not just the tests.
if (numkong_os_wasi_ AND NUMKONG_WITH_HOST_PROBES)
    add_compile_definitions(NUMKONG_WITH_HOST_PROBES=1)
endif ()

# Seeded from the environment at the first configure, so `cargo` and `cmake-js` honour `NUMKONG_TARGET_ARCH=native`.
set(NUMKONG_TARGET_ARCH "$ENV{NUMKONG_TARGET_ARCH}"
    CACHE STRING
          "Host-tuned `-march`, like native, or a WebAssembly module's SIMD capability, serial, v128 or v128relaxed"
)
option(NUMKONG_ENABLE_ASAN "Enable AddressSanitizer in Debug builds" ${numkong_is_main_project_})

# `add_test()` entries are only generated when testing is enabled.
if (NUMKONG_BUILD_TEST)
    enable_testing()
endif ()

# BLAS/MKL options: ON (require), OFF (disable), AUTO (detect if available)
set(NUMKONG_COMPARE_TO_BLAS "AUTO" CACHE STRING "Include BLAS (OpenBLAS/Accelerate) in tests and benchmarks")
set(NUMKONG_COMPARE_TO_MKL "AUTO" CACHE STRING "Include Intel MKL in tests and benchmarks")
set_property(CACHE NUMKONG_COMPARE_TO_BLAS PROPERTY STRINGS "AUTO" "ON" "OFF")
set_property(CACHE NUMKONG_COMPARE_TO_MKL PROPERTY STRINGS "AUTO" "ON" "OFF")

# CUDA baselines for the CUDA rows of the benchmarks
option(NUMKONG_COMPARE_TO_CUBLAS "Include cuBLASLt and cuBLAS in CUDA benchmarks" ON)
option(NUMKONG_COMPARE_TO_CUDNN "Include cuDNN attention in CUDA benchmarks, found under NUMKONG_CUDNN_ROOT" OFF)
option(NUMKONG_COMPARE_TO_CUVS "Include cuVS distances in CUDA benchmarks, found under NUMKONG_CUVS_ROOTS" OFF)
set(NUMKONG_CUDNN_ROOT "" CACHE PATH "cuDNN 9.26 or newer, holding include/cudnn.h and lib/libcudnn.so*")
set(NUMKONG_CUVS_ROOTS "" CACHE STRING "Prefixes holding libcuvs_c, librmm, librapids_logger and dlpack")

# Default to Release build type if not set
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Generate compile_commands.json for clang-tidy and other tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MSVC does not set `CMAKE_SYSTEM_PROCESSOR` correctly
# Use `CMAKE_C_COMPILER_ARCHITECTURE_ID` instead
if (MSVC)
    if (CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "x64")
        set(CMAKE_SYSTEM_PROCESSOR "AMD64")
    elseif (CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "X86")
        set(CMAKE_SYSTEM_PROCESSOR "X86")
    elseif (CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "ARM64")
        set(CMAKE_SYSTEM_PROCESSOR "ARM64")
    else ()
        message(WARNING "Unknown CMAKE_C_COMPILER_ARCHITECTURE_ID=${CMAKE_C_COMPILER_ARCHITECTURE_ID}")
    endif ()
endif ()

# On macOS, CMAKE_SYSTEM_PROCESSOR reflects the host, not CMAKE_OSX_ARCHITECTURES (the target slice
# Apple Clang compiles via `-arch`). Patch it here so the detection below matches the target. We do
# not build universal2, so CMAKE_OSX_ARCHITECTURES is always single-valued when set.
if (APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES numkong_osx_architectures_count_)
    if (numkong_osx_architectures_count_ EQUAL 1)
        set(CMAKE_SYSTEM_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
    else ()
        message(
            FATAL_ERROR "Multi-arch CMAKE_OSX_ARCHITECTURES='${CMAKE_OSX_ARCHITECTURES}' not supported; build per-arch"
        )
    endif ()
    unset(numkong_osx_architectures_count_)
endif ()

# Detect target architecture from CMAKE_SYSTEM_PROCESSOR (safe for cross-compilation)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|X86_64|AMD64|amd64")
    set(numkong_arch_x8664_ TRUE)
    message(STATUS "NumKong Platform: x86_64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
    set(numkong_arch_arm64_ TRUE)
    message(STATUS "NumKong Platform: ARM64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
    set(numkong_arch_riscv64_ TRUE)
    message(STATUS "NumKong Platform: RISC-V 64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64|LOONGARCH64")
    set(numkong_arch_loongarch64_ TRUE)
    message(STATUS "NumKong Platform: LoongArch 64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|ppc64|powerpc64le|powerpc64")
    set(numkong_arch_ppc64_ TRUE)
    message(STATUS "NumKong Platform: Power 64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^arm|^ARM|armv7")
    set(numkong_arch_arm32_ TRUE)
    message(STATUS "NumKong Platform: ARM32 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^wasm(32|64)$")
    set(numkong_arch_wasm_ TRUE)
    message(STATUS "NumKong Platform: WebAssembly (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
else ()
    message(WARNING "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()

# Detect target OS for platform-specific optimizations
if (CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
    set(numkong_os_freebsd_ TRUE)
    message(STATUS "NumKong OS: FreeBSD")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(numkong_os_linux_ TRUE)
    message(STATUS "NumKong OS: Linux")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(numkong_os_android_ TRUE)
    message(STATUS "NumKong OS: Android (API level ${ANDROID_NATIVE_API_LEVEL})")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(numkong_os_macos_ TRUE)
    message(STATUS "NumKong OS: macOS")
elseif (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(numkong_os_ios_ TRUE)
    message(STATUS "NumKong OS: iOS")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows" OR WIN32)
    set(numkong_os_windows_ TRUE)
    message(STATUS "NumKong OS: Windows")
endif ()

# Global compiler flags for debug and release
if (NUMKONG_ENABLE_ASAN)
    if (MSVC)
        # `cl.exe` reports an unknown `-f...` as warning D9002 and still exits 0, so the probe
        # below would pass and then replace `/Zi /Ob0 /Od /RTC1` with flags it ignores.
        set(numkong_has_asan_ FALSE)
    else ()
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
        check_cxx_source_compiles("int main(){return 0;}" numkong_has_asan_)
        unset(CMAKE_REQUIRED_FLAGS)
    endif ()

    if (numkong_has_asan_)
        set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
        set(CMAKE_C_FLAGS_DEBUG "-g -fsanitize=address")
    else ()
        message(STATUS "AddressSanitizer not available, Debug builds without ASAN")
    endif ()
endif ()

if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Gw /DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /Gw /DNDEBUG")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
endif ()

# Debug builds check the library's invariants through `nk_assert_`, which release builds compile out.
add_compile_definitions($<$<CONFIG:Debug>:NUMKONG_DEBUG=1>)

# Probe link-time optimization once and reuse it: the shared library, the tests, and the
# benchmarks all get IPO for Release builds so the exact `-O3 + LTO` codegen that ships also
# runs in the test suite. LTO-only miscompiles are otherwise invisible — e.g. cross-TU
# unrolling of a runtime-bounded loop can speculatively divide by a tensor shape's zero-filled
# spare extents and trap (#DE / SIGFPE); see shape_storage_::normalize_spare_.
include(CheckIPOSupported)
set(numkong_ipo_configuration_ "${CMAKE_TRY_COMPILE_CONFIGURATION}")
set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
check_ipo_supported(RESULT numkong_ipo_supported_ OUTPUT numkong_ipo_support_output_)
set(CMAKE_TRY_COMPILE_CONFIGURATION "${numkong_ipo_configuration_}")
unset(numkong_ipo_configuration_)
if (NOT numkong_ipo_supported_)
    message(STATUS "Link-time optimization not available: ${numkong_ipo_support_output_}")
endif ()

# Portable baseline policy (commits 43e8324d, 453ed5ff, 23f51955): each translation unit gets a per-arch
# ABI floor, auto-vectorization is forbidden so serial fallbacks stay scalar, and SIMD kernels opt in via
# per-function __attribute__((target(...))) or #pragma GCC target. NUMKONG_TARGET_ARCH=native opts out for a
# host-tuned build.
if (NUMKONG_TARGET_ARCH AND NOT MSVC AND NOT CMAKE_CROSSCOMPILING)
    message(
        WARNING "NUMKONG_TARGET_ARCH=${NUMKONG_TARGET_ARCH}: baking specific instructions; result may not run elsewhere"
    )
    if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|IntelLLVM)$")
        # Apple Clang's `-march=native` only advertises a subset of host features
        # (no SME/SME2/FP16_FML); `-mcpu=native` is the complete knob on macOS.
        # GCC/Clang on Linux/FreeBSD use `-march=native` as usual.
        if (APPLE AND NUMKONG_TARGET_ARCH STREQUAL "native")
            add_compile_options(-mcpu=native)
        else ()
            add_compile_options(-march=${NUMKONG_TARGET_ARCH})
        endif ()
    endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|IntelLLVM)$")
    # Clang and IntelLLVM accept the GCC spellings as aliases, so one pair covers all three compilers.
    add_compile_options(-fno-tree-vectorize -fno-tree-slp-vectorize)
    if (APPLE)
        # `-arch` already pins the ABI floor per slice, and a universal build passes both
        # to one clang invocation, where a per-arch `-march=` conflicts with the other slice.
        # `CMAKE_SYSTEM_PROCESSOR` names the host, so it cannot pick the flag here.
    elseif (numkong_arch_arm64_)
        add_compile_options(-march=armv8-a)
    elseif (numkong_arch_x8664_)
        add_compile_options(-march=x86-64)
    elseif (numkong_arch_riscv64_)
        add_compile_options(-march=rv64gc)
    elseif (numkong_arch_ppc64_)
        add_compile_options(-mcpu=power8)
    elseif (numkong_arch_loongarch64_)
        add_compile_options(-march=loongarch64)
    elseif (numkong_arch_wasm_)
        # An engine validates a module whole, so every unit carries the module's one SIMD capability,
        # whatever `CMAKE_C_FLAGS` a binding passes; `types.h` reads it back from the flags.
        if (NUMKONG_TARGET_ARCH STREQUAL "v128relaxed")
            add_compile_options(-msimd128 -mrelaxed-simd)
        elseif (NUMKONG_TARGET_ARCH STREQUAL "v128")
            add_compile_options(-msimd128)
        elseif (NOT NUMKONG_TARGET_ARCH STREQUAL "serial")
            message(
                FATAL_ERROR
                    "NUMKONG_TARGET_ARCH on WebAssembly: serial, v128 or v128relaxed, not `${NUMKONG_TARGET_ARCH}`"
            )
        endif ()
    endif ()
elseif (MSVC)
    # MSVC: no per-function target pragma; these match defaults. NVCC would read them as input files.
    if (numkong_arch_x8664_)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/arch:SSE2>)
    elseif (numkong_arch_arm64_)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/arch:armv8.0>)
    endif ()
endif ()

# Not for CUDA: nvcc hands the host compiler line markers that `-pedantic` flags on every line.
if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|IntelLLVM)$")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-pedantic>)
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang|IntelLLVM)$")
    add_compile_options(-ferror-limit=1 -Wno-tautological-constant-compare)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fmax-errors=1 -Wno-tautological-constant-compare -Wno-psabi)
endif ()

# Define the header-only library. The include path & sources wear `BUILD_INTERFACE` genexes, so the
# target stays exportable: the installed package resolves headers under the install prefix instead.
include(GNUInstallDirs)
file(GLOB numkong_header_sources_ include/numkong/*.h include/numkong/*.hpp)
add_library(numkong_header INTERFACE)
add_library(numkong::header ALIAS numkong_header)
set_target_properties(numkong_header PROPERTIES EXPORT_NAME header)
target_sources(numkong_header INTERFACE "$<BUILD_INTERFACE:${numkong_header_sources_}>")
target_include_directories(
    numkong_header INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(numkong_header INTERFACE c_std_99)
# Consumers of the header target compile every kernel themselves, and link no dispatch points.
target_compile_definitions(numkong_header INTERFACE NUMKONG_HEADER_ONLY=1)
if (APPLE)
    set(numkong_embed_compiler_
        "$<AND:$<PLATFORM_ID:Darwin>,$<OR:$<COMPILE_LANG_AND_ID:C,Clang,AppleClang>,$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>>>"
    )
    if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
        set(numkong_embed_install_dir_ "${CMAKE_INSTALL_INCLUDEDIR}")
    else ()
        set(numkong_embed_install_dir_ "$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}")
    endif ()
    set(numkong_embed_options_
        "$<${numkong_embed_compiler_}:$<BUILD_INTERFACE:--embed-dir=${PROJECT_SOURCE_DIR}/include>>"
        "$<${numkong_embed_compiler_}:$<INSTALL_INTERFACE:--embed-dir=${numkong_embed_install_dir_}>>"
    )
    target_compile_options(numkong_header INTERFACE ${numkong_embed_options_})
endif ()

# MSVC's traditional preprocessor can't short-circuit `defined(__has_builtin) && __has_builtin(...)`.
# The conforming preprocessor (/Zc:preprocessor, available since MSVC 2019 16.6) handles this.
# Scoped via INTERFACE so it propagates only to targets that link against numkong.
if (MSVC)
    target_compile_options(numkong_header INTERFACE /Zc:preprocessor)
endif ()

# Enable Clang C modules for incremental compilation. Build-tree only: the flags reference this
# build's module cache, so they must never leak into the exported package.
if (CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(
        numkong_header
        INTERFACE $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C>:-fmodules>>
                  $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C>:-fimplicit-module-maps>>
                  $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C>:-fmodules-cache-path=${CMAKE_BINARY_DIR}/module-cache>>
    )

    # Install module map
    if (NUMKONG_INSTALL)
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/module.modulemap DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif ()

    message(STATUS "Clang C modules enabled: cache at ${CMAKE_BINARY_DIR}/module-cache")
endif ()

# One compile probe per capability under `probes/`, defining `NUMKONG_TARGET_<CAPABILITY>` as the toolchain compiles
# it through `numkong::cpu_capabilities_compiled`, and as 0 through `numkong_cpu_capabilities_disabled_`.
include(cmake/nk_isa_probe.cmake)

# CUDA and HIP come on before any target: `numkong_static` carries their units, so every target linking it, the
# bindings, tests and benchmarks included, needs the architecture list. One cubin per device family plus a PTX tail,
# needing CUDA 13 or newer and a driver from 580 on; Turing, CUDA 13's floor, runs the CUDA baseline only. Every
# code lands in one fat binary, and each kernel checks its own architecture in device code. To iterate on one GPU
# keep its family, like "120f-real": "native" drops the family suffix and with it the family's capabilities.
if (NUMKONG_BUILD_CUDA)
    enable_language(CUDA)
    set(NUMKONG_CUDA_ARCHITECTURES "75-real;80-real;89-real;90a-real;100f-real;120f-real;80-virtual"
        CACHE STRING "GPU architectures baked into the CUDA targets"
    )
    set(CMAKE_CUDA_ARCHITECTURES "${NUMKONG_CUDA_ARCHITECTURES}")
    # NVCC's default leaves Turing and Ampere cubins uncompressed, ten times their compressed size.
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=size>)
endif ()
if (NUMKONG_BUILD_ROCM)
    enable_language(HIP)
    set(NUMKONG_ROCM_ARCHITECTURES "gfx942;gfx950;gfx1250;gfx1251"
        CACHE STRING "AMD GPU architectures baked into the ROCm targets"
    )
    set(CMAKE_HIP_ARCHITECTURES "${NUMKONG_ROCM_ARCHITECTURES}")
endif ()

# BLAS/MKL detection (shared by tests and benchmarks)
# Options can be: ON (require), OFF (disable), AUTO (detect if available)
set(numkong_blas_found_ FALSE)
set(numkong_mkl_found_ FALSE)

# Skip BLAS/MKL/Accelerate detection when cross-compiling (host libraries won't work)
if (CMAKE_CROSSCOMPILING AND NUMKONG_COMPARE_TO_MKL STREQUAL "AUTO")
    message(STATUS "Cross-compiling: skipping MKL detection (host library)")
    set(numkong_compare_to_mkl_ "OFF")
else ()
    set(numkong_compare_to_mkl_ "${NUMKONG_COMPARE_TO_MKL}")
endif ()

if (CMAKE_CROSSCOMPILING AND NUMKONG_COMPARE_TO_BLAS STREQUAL "AUTO")
    message(STATUS "Cross-compiling: skipping BLAS detection (host library)")
    set(numkong_compare_to_blas_ "OFF")
else ()
    set(numkong_compare_to_blas_ "${NUMKONG_COMPARE_TO_BLAS}")
endif ()

# Probe whether C++ test executables need an explicit math library (`-lm`) on this toolchain.
set(numkong_cxx_math_library_ "")
if (NUMKONG_BUILD_TEST)
    include(cmake/nk_math_lib.cmake)
    numkong_detect_cxx_math_library(numkong_cxx_math_library_)
    message(STATUS "NumKong: Explicit libm for C++ tests: ${numkong_cxx_math_library_}")
endif ()

# Only attempt detection if building tests or benchmarks
if (NUMKONG_BUILD_TEST OR NUMKONG_BUILD_BENCH)
    # MKL detection
    if (NOT numkong_compare_to_mkl_ STREQUAL "OFF")
        # Use environment variable MKLROOT or standard MKL paths
        if (DEFINED ENV{MKLROOT})
            set(numkong_mkl_root_ $ENV{MKLROOT})
        else ()
            # Try common installation paths
            if (EXISTS "/opt/intel/oneapi/mkl/latest")
                set(numkong_mkl_root_ "/opt/intel/oneapi/mkl/latest")
            elseif (EXISTS "/opt/intel/mkl")
                set(numkong_mkl_root_ "/opt/intel/mkl")
            endif ()
        endif ()

        if (numkong_mkl_root_)
            # Verify the libraries actually exist (prefer shared libraries for simpler linking)
            if (EXISTS "${numkong_mkl_root_}/lib/libmkl_intel_lp64.so")
                message(STATUS "MKL found at: ${numkong_mkl_root_}")
                set(numkong_mkl_found_ TRUE)
                set(numkong_mkl_include_dirs_ ${numkong_mkl_root_}/include)
                set(numkong_mkl_libraries_
                    ${numkong_mkl_root_}/lib/libmkl_intel_lp64.so ${numkong_mkl_root_}/lib/libmkl_sequential.so
                    ${numkong_mkl_root_}/lib/libmkl_core.so pthread m dl
                )
            elseif (EXISTS "${numkong_mkl_root_}/lib/intel64/libmkl_intel_lp64.a")
                # Fallback to static libraries (Linux/FreeBSD)
                message(STATUS "MKL found at: ${numkong_mkl_root_} (static)")
                set(numkong_mkl_found_ TRUE)
                set(numkong_mkl_include_dirs_ ${numkong_mkl_root_}/include)
                set(numkong_mkl_libraries_
                    -Wl,--start-group ${numkong_mkl_root_}/lib/intel64/libmkl_intel_lp64.a
                    ${numkong_mkl_root_}/lib/intel64/libmkl_sequential.a ${numkong_mkl_root_}/lib/intel64/libmkl_core.a
                    -Wl,--end-group pthread m dl
                )
            elseif (EXISTS "${numkong_mkl_root_}/lib/mkl_intel_lp64_dll.lib")
                # Windows import libraries
                message(STATUS "MKL found at: ${numkong_mkl_root_} (Windows)")
                set(numkong_mkl_found_ TRUE)
                set(numkong_mkl_include_dirs_ ${numkong_mkl_root_}/include)
                set(numkong_mkl_libraries_
                    ${numkong_mkl_root_}/lib/mkl_intel_lp64_dll.lib ${numkong_mkl_root_}/lib/mkl_sequential_dll.lib
                    ${numkong_mkl_root_}/lib/mkl_core_dll.lib
                )
            elseif (numkong_compare_to_mkl_ STREQUAL "ON")
                message(FATAL_ERROR "MKL libraries not found at ${numkong_mkl_root_}/lib/")
            else ()
                message(STATUS "MKL directory found but libraries missing, skipping MKL")
            endif ()
        elseif (numkong_compare_to_mkl_ STREQUAL "ON")
            message(FATAL_ERROR "MKL not found. Set MKLROOT environment variable or install Intel oneAPI.")
        else ()
            message(STATUS "MKL not found (auto-detection), skipping MKL benchmarks")
        endif ()
    endif ()

    # BLAS detection (skip if MKL found in AUTO mode, since MKL provides BLAS)
    # We differentiate between:
    # - numkong_mkl_found_: Intel MKL
    # - numkong_accelerate_found_: Apple Accelerate framework
    # - numkong_blas_found_: Generic CBLAS (OpenBLAS, etc.)
    if (NOT numkong_compare_to_blas_ STREQUAL "OFF")
        if (numkong_mkl_found_ AND NOT numkong_compare_to_blas_ STREQUAL "ON")
            message(STATUS "BLAS: Using MKL (skipping separate BLAS detection)")
        elseif (APPLE)
            # Apple Accelerate is always available on macOS
            message(STATUS "Using Apple Accelerate framework")
            set(numkong_accelerate_found_ TRUE)
            set(numkong_accelerate_libraries_ "-framework Accelerate")
            set(numkong_accelerate_include_dirs_ "")
        else ()
            # Try to find BLAS without REQUIRED flag for auto-detection
            find_package(BLAS QUIET)

            if (BLAS_FOUND)
                message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
                set(numkong_blas_found_ TRUE)
                set(numkong_blas_libraries_ ${BLAS_LIBRARIES})
                set(numkong_blas_include_dirs_ ${BLAS_INCLUDE_DIRS})
            elseif (numkong_compare_to_blas_ STREQUAL "ON")
                message(FATAL_ERROR "BLAS not found. Install OpenBLAS or set NUMKONG_COMPARE_TO_BLAS=OFF")
            else ()
                message(STATUS "BLAS not found (auto-detection), skipping BLAS benchmarks")
            endif ()
        endif ()
    endif ()

    # Summary
    if (numkong_blas_found_ OR numkong_mkl_found_ OR numkong_accelerate_found_)
        message(
            STATUS
                "Comparison libraries: BLAS=${numkong_blas_found_}, MKL=${numkong_mkl_found_}, Accelerate=${numkong_accelerate_found_}"
        )
    endif ()
endif ()

# The test and bench harnesses print through {fmt}, pinned to the release StringZilla and USearch fetch, because in a
# nested build the first `FetchContent_Declare` wins.
if (NUMKONG_BUILD_TEST OR NUMKONG_BUILD_BENCH)
    include(FetchContent)
    FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG 12.2.0 FIND_PACKAGE_ARGS 12.2.0)
    FetchContent_MakeAvailable(fmt)
endif ()

# Build benchmarks
if (NUMKONG_BUILD_BENCH)
    add_executable(
        numkong_bench
        bench/main.cpp bench/dot.cpp bench/spatial.cpp bench/set.cpp bench/curved.cpp bench/probability.cpp
        bench/each.cpp bench/trigonometry.cpp bench/geospatial.cpp bench/shapes.cpp bench/sparse.cpp bench/cast.cpp
        bench/reduce.cpp
        # ISA-family cross/batch files for parallel compilation
        bench/cross_serial.cpp bench/cross_x8664.cpp bench/cross_arm64.cpp bench/cross_blas.cpp bench/cross_riscv64.cpp
        bench/cross_ppc64.cpp bench/cross_wasm.cpp bench/cross_loongarch64.cpp bench/cross_metal.cpp bench/maxsim.cpp
    )
    target_link_libraries(numkong_bench numkong_static fmt::fmt)

    # Mirror the shipped Release codegen.
    if (numkong_ipo_supported_)
        set_target_properties(numkong_bench PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif ()

    # WASI-specific configuration for benchmarks
    if (numkong_os_wasi_)
        set_target_properties(numkong_bench PROPERTIES OUTPUT_NAME "numkong_bench.wasm")
        message(STATUS "NumKong: Building benchmarks for WASI (output: numkong_bench.wasm)")
    endif ()

    if (numkong_blas_found_ AND NOT numkong_os_wasi_)
        target_compile_definitions(numkong_bench PRIVATE NUMKONG_COMPARE_TO_BLAS=1)
        target_include_directories(numkong_bench PRIVATE ${numkong_blas_include_dirs_})
        target_link_libraries(numkong_bench ${numkong_blas_libraries_})
    endif ()

    if (numkong_accelerate_found_ AND NOT numkong_os_wasi_)
        target_compile_definitions(numkong_bench PRIVATE NUMKONG_COMPARE_TO_ACCELERATE=1 ACCELERATE_NEW_LAPACK)
        target_include_directories(numkong_bench PRIVATE ${numkong_accelerate_include_dirs_})
        target_link_libraries(numkong_bench ${numkong_accelerate_libraries_})
    endif ()

    if (numkong_mkl_found_ AND NOT numkong_os_wasi_)
        target_compile_definitions(numkong_bench PRIVATE NUMKONG_COMPARE_TO_MKL=1)
        target_include_directories(numkong_bench PRIVATE ${numkong_mkl_include_dirs_})
        target_link_libraries(numkong_bench ${numkong_mkl_libraries_})
    endif ()

    # Declares the capability kernels the library defines, to call them directly.
    target_link_libraries(numkong_bench numkong_cpu_capabilities_compiled)
endif ()

# The library's units: one dispatch unit per family, holding its dispatch points and finder, and one
# unit per CPU capability, defining that capability's kernels of every family. Each unit names its own
# capability, so every file compiles with the same flags and the units build in parallel.
set(numkong_library_units_
    c/dispatch/dot.c
    c/dispatch/dots.c
    c/dispatch/spatial.c
    c/dispatch/spatials.c
    c/dispatch/set.c
    c/dispatch/sets.c
    c/dispatch/maxsim.c
    c/dispatch/attention.c
    c/dispatch/sparse.c
    c/dispatch/reduce.c
    c/dispatch/each.c
    c/dispatch/cast.c
    c/dispatch/scalar.c
    c/dispatch/trigonometry.c
    c/dispatch/curved.c
    c/dispatch/probability.c
    c/dispatch/geospatial.c
    c/dispatch/shapes.c
    c/target/serial.c
    c/target/haswell.c
    c/target/alder.c
    c/target/sierra.c
    c/target/skylake.c
    c/target/icelake.c
    c/target/genoa.c
    c/target/turin.c
    c/target/sapphire.c
    c/target/diamond.c
    c/target/sapphireamx.c
    c/target/graniteamx.c
    c/target/diamondamx.c
    c/target/neon.c
    c/target/neonhalf.c
    c/target/neonbfdot.c
    c/target/neonfhm.c
    c/target/neonsdot.c
    c/target/neonfp8.c
    c/target/sve.c
    c/target/svehalf.c
    c/target/svesdot.c
    c/target/svebfdot.c
    c/target/sve2.c
    c/target/sme.c
    c/target/smef64.c
    c/target/smebi32.c
    c/target/rvv.c
    c/target/rvvhalf.c
    c/target/rvvbf16.c
    c/target/rvvbb.c
    c/target/v128.c
    c/target/v128relaxed.c
    c/target/powervsx.c
    c/target/loongsonasx.c
)

# Build tests
set(numkong_test_sources_
    test/main_cpu.cpp test/dot.cpp test/spatial.cpp test/set.cpp test/curved.cpp test/probability.cpp test/each.cpp
    test/trigonometry.cpp test/geospatial.cpp test/shapes.cpp test/cast.cpp test/reduce.cpp test/sparse.cpp
    # ISA-family cross/batch files for parallel compilation
    test/cross_serial.cpp test/cross_x8664.cpp test/cross_arm64.cpp test/cross_blas.cpp test/cross_riscv64.cpp
    test/cross_ppc64.cpp test/cross_loongarch64.cpp test/cross_wasm.cpp test/maxsim.cpp test/tensor.cpp
)

if (NUMKONG_BUILD_TEST)
    # C++ precision test (ULP-based error analysis)
    add_executable(numkong_cpu_test ${numkong_test_sources_})
    target_link_libraries(numkong_cpu_test numkong_static fmt::fmt ${numkong_cxx_math_library_})

    # Mirror the shipped Release codegen.
    if (numkong_ipo_supported_)
        set_target_properties(numkong_cpu_test PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif ()

    # WASI-specific configuration
    if (numkong_os_wasi_)
        set_target_properties(numkong_cpu_test PROPERTIES OUTPUT_NAME "numkong_cpu_test.wasm")
        message(STATUS "NumKong: Building tests for WASI (output: numkong_cpu_test.wasm)")
    endif ()

    if (numkong_blas_found_ AND NOT numkong_os_wasi_)
        target_compile_definitions(numkong_cpu_test PRIVATE NUMKONG_COMPARE_TO_BLAS=1)
        target_include_directories(numkong_cpu_test PRIVATE ${numkong_blas_include_dirs_})
        target_link_libraries(numkong_cpu_test ${numkong_blas_libraries_})
    endif ()

    if (numkong_accelerate_found_ AND NOT numkong_os_wasi_)
        target_compile_definitions(numkong_cpu_test PRIVATE NUMKONG_COMPARE_TO_ACCELERATE=1 ACCELERATE_NEW_LAPACK)
        target_include_directories(numkong_cpu_test PRIVATE ${numkong_accelerate_include_dirs_})
        target_link_libraries(numkong_cpu_test ${numkong_accelerate_libraries_})
    endif ()

    if (numkong_mkl_found_ AND NOT numkong_os_wasi_)
        target_compile_definitions(numkong_cpu_test PRIVATE NUMKONG_COMPARE_TO_MKL=1)
        target_include_directories(numkong_cpu_test PRIVATE ${numkong_mkl_include_dirs_})
        target_link_libraries(numkong_cpu_test ${numkong_mkl_libraries_})
    endif ()

    # Declares the capability kernels the library defines, to call them directly.
    target_link_libraries(numkong_cpu_test numkong_cpu_capabilities_compiled)

    add_test(NAME numkong_cpu_test COMMAND numkong_cpu_test)

    # The same suites compiling every kernel inline, with stubbed dispatch points, skipping the capabilities this CPU
    # lacks.
    add_executable(numkong_cpu_header_test ${numkong_test_sources_})
    target_link_libraries(
        numkong_cpu_header_test numkong_header numkong_cpu_capabilities_compiled fmt::fmt ${numkong_cxx_math_library_}
    )
    add_test(NAME numkong_cpu_header_test COMMAND numkong_cpu_header_test)
endif ()

# The static library, which every binding but Swift links and a parent project reaches as `numkong::static`, and the
# shared library, or the Emscripten JS module
set(numkong_library_sources_ ${numkong_header_sources_} c/numkong.c ${numkong_library_units_})

# Emscripten requires building as executable to generate JS bindings
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NUMKONG_BUILD_SHARED)
    set(numkong_libraries_ numkong_shared numkong_static)
    # Detect wasm64 (memory64) mode
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "wasm64")
        set(numkong_wasm64_ ON)
    else ()
        set(numkong_wasm64_ OFF)
    endif ()

    # Create a minimal main() wrapper for Emscripten
    file(WRITE "${CMAKE_BINARY_DIR}/emscripten_stub.c"
         "// Stub main for Emscripten to generate JS bindings\nint main(void) { return 0; }\n"
    )

    add_executable(numkong_shared ${numkong_library_sources_} "${CMAKE_BINARY_DIR}/emscripten_stub.c")

    # Consumers that `add_subdirectory(numkong)` link a library, not the `.js` executable.
    add_library(numkong_static STATIC ${numkong_library_sources_})
    add_library(numkong::static ALIAS numkong_static)
    set_target_properties(numkong_static PROPERTIES EXPORT_NAME static)

    # One module carries one SIMD capability, the toolchain's NUMKONG_TARGET_ARCH, so its name says which.
    if (numkong_wasm64_)
        set(numkong_wasm_module_name_ "numkong-wasm64-${NUMKONG_TARGET_ARCH}")
    else ()
        set(numkong_wasm_module_name_ "numkong-wasm32-${NUMKONG_TARGET_ARCH}")
    endif ()
    # Generates `${numkong_wasm_module_name_}.js` + `.wasm`
    set_target_properties(numkong_shared PROPERTIES OUTPUT_NAME "${numkong_wasm_module_name_}" SUFFIX ".js")
    message(STATUS "NumKong: Building Emscripten module ${numkong_wasm_module_name_}")

    # `types.h` leaves every non-WebAssembly capability off, and `v128` and `v128relaxed` follow `-msimd128` and
    # `-mrelaxed-simd`.
    foreach (library_ ${numkong_libraries_})
        # The library, unlike the headers, may use C11: `c/numkong.c` caches detection in <stdatomic.h> atomics.
        set_target_properties(${library_} PROPERTIES C_STANDARD 11)
        target_include_directories(
            ${library_} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                               $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
        # The library's units reach their private `dispatch.h` by name.
        target_include_directories(${library_} PRIVATE "${PROJECT_SOURCE_DIR}/c")
    endforeach ()

    target_link_options(
        numkong_shared
        PRIVATE
        "SHELL:-sEXPORT_ES6=1" # Generate ES6 module
        "SHELL:-sMODULARIZE=1" # Wrap in module factory
        "SHELL:-sEXPORT_NAME=NumKongModule" # Export name
        "SHELL:-sEXPORTED_FUNCTIONS=['_malloc','_free','_nk_dot_f32_best','_nk_dot_f64_best','_nk_dot_f16_best','_nk_dot_bf16_best','_nk_dot_i8_best','_nk_dot_u8_best','_nk_angular_f32_best','_nk_angular_f64_best','_nk_angular_f16_best','_nk_angular_bf16_best','_nk_angular_i8_best','_nk_angular_u8_best','_nk_sqeuclidean_f32_best','_nk_sqeuclidean_f64_best','_nk_sqeuclidean_f16_best','_nk_sqeuclidean_bf16_best','_nk_sqeuclidean_i8_best','_nk_sqeuclidean_u8_best','_nk_euclidean_f32_best','_nk_euclidean_f64_best','_nk_euclidean_f16_best','_nk_euclidean_bf16_best','_nk_euclidean_i8_best','_nk_euclidean_u8_best','_nk_hamming_u1_best','_nk_hamming_u8_best','_nk_jaccard_u1_best','_nk_jaccard_u16_best','_nk_jaccard_u32_best','_nk_kld_f32_best','_nk_kld_f64_best','_nk_kld_f16_best','_nk_kld_bf16_best','_nk_jsd_f32_best','_nk_jsd_f64_best','_nk_jsd_f16_best','_nk_jsd_bf16_best','_nk_cpu_capabilities_detected','_nk_cpu_capabilities_compiled','_nk_cpu_capabilities_enabled','_nk_capabilities_name','_nk_status_name','_nk_dots_pack_size_f32_best','_nk_dots_pack_size_f64_best','_nk_dots_pack_size_f16_best','_nk_dots_pack_size_bf16_best','_nk_dots_pack_size_i8_best','_nk_dots_pack_size_u8_best','_nk_dots_pack_f32_best','_nk_dots_pack_f64_best','_nk_dots_pack_f16_best','_nk_dots_pack_bf16_best','_nk_dots_pack_i8_best','_nk_dots_pack_u8_best','_nk_dots_packed_f32_best','_nk_dots_packed_f64_best','_nk_dots_packed_f16_best','_nk_dots_packed_bf16_best','_nk_dots_packed_i8_best','_nk_dots_packed_u8_best','_nk_dots_packed_shape_f32_best','_nk_dots_packed_shape_f64_best','_nk_dots_packed_shape_f16_best','_nk_dots_packed_shape_bf16_best','_nk_dots_packed_shape_i8_best','_nk_dots_packed_shape_u8_best','_nk_dots_symmetric_f32_best','_nk_dots_symmetric_f64_best','_nk_dots_symmetric_f16_best','_nk_dots_symmetric_bf16_best','_nk_dots_symmetric_i8_best','_nk_dots_symmetric_u8_best','_nk_angulars_packed_f32_best','_nk_angulars_packed_f64_best','_nk_angulars_packed_f16_best','_nk_angulars_packed_bf16_best','_nk_angulars_symmetric_f32_best','_nk_angulars_symmetric_f64_best','_nk_angulars_symmetric_f16_best','_nk_angulars_symmetric_bf16_best','_nk_euclideans_packed_f32_best','_nk_euclideans_packed_f64_best','_nk_euclideans_packed_f16_best','_nk_euclideans_packed_bf16_best','_nk_euclideans_symmetric_f32_best','_nk_euclideans_symmetric_f64_best','_nk_euclideans_symmetric_f16_best','_nk_euclideans_symmetric_bf16_best']"
        "SHELL:-sEXPORTED_RUNTIME_METHODS=['wasmMemory']" # Export memory object
        "SHELL:-sALLOW_MEMORY_GROWTH=1" # Dynamic memory
        "SHELL:-sENVIRONMENT=web,node" # Both browser and Node.js
        "SHELL:-sSINGLE_FILE=0" # Separate .wasm file (recommended)
        "SHELL:-sEXPORT_ALL=0" # Security - don't export everything
        "SHELL:-sSTANDALONE_WASM=0" # Keep JS wrapper (required for browser)
        "SHELL:-sINITIAL_MEMORY=16MB" # Initial heap size
        "SHELL:-sSTACK_SIZE=5MB" # Stack size for recursion
    )

    # Memory64 link option for wasm64 builds
    if (numkong_wasm64_)
        target_link_options(numkong_shared PRIVATE "SHELL:-sMEMORY64=1")
    endif ()

    # Under Emscripten numkong_shared is an executable (.js/.wasm), not a shared
    # library. CMake refuses target_link_libraries against an imported
    # executable unless it exports its symbol table. Upstream projects that
    # add_subdirectory(numkong) and link to numkong_shared need this.
    set_target_properties(numkong_shared PROPERTIES ENABLE_EXPORTS ON)
else ()
    if (NUMKONG_BUILD_SHARED)
        set(numkong_libraries_ numkong_static numkong_shared)
    else ()
        set(numkong_libraries_ numkong_static)
    endif ()
    add_library(numkong_static STATIC ${numkong_library_sources_})
    add_library(numkong::static ALIAS numkong_static)
    # Position-independent, as the bindings link it into shared objects and position-independent executables.
    set_target_properties(numkong_static PROPERTIES EXPORT_NAME static POSITION_INDEPENDENT_CODE ON)
    if (NUMKONG_BUILD_SHARED)
        add_library(numkong_shared SHARED ${numkong_library_sources_})
        add_library(numkong::shared ALIAS numkong_shared)
        # Windows exports every symbol, as `NUMKONG_API` spells no `__declspec` for the DLL.
        set_target_properties(numkong_shared PROPERTIES EXPORT_NAME shared WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif ()
    foreach (library_ ${numkong_libraries_})
        # The library, unlike the headers, may use C11: `c/numkong.c` caches detection in <stdatomic.h> atomics,
        # which MSVC gates behind a flag.
        set_target_properties(${library_} PROPERTIES C_STANDARD 11)
        target_compile_options(${library_} PRIVATE $<$<C_COMPILER_ID:MSVC>:/experimental:c11atomics>)
        target_include_directories(
            ${library_} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                               $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
        # The library's units reach their private `dispatch.h` by name.
        target_include_directories(${library_} PRIVATE "${PROJECT_SOURCE_DIR}/c")

        # Hide all symbols by default; only NUMKONG_API-annotated API functions are exported.
        # Strip unwind tables from this pure-C library (saves ~78 KB).
        if (NOT MSVC)
            target_compile_options(
                ${library_} PRIVATE -fvisibility=hidden -fno-asynchronous-unwind-tables -fno-builtin-memset
                                    -fno-builtin-memcpy -fno-builtin-memmove
            )
            if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
                target_compile_options(${library_} PRIVATE -fno-tree-loop-distribute-patterns)
            endif ()
        endif ()

        # MSVC ARM64 cross-compilation: <intrin.h> pulls in <winnt.h> which
        # requires _ARM64_ to be defined. The Visual Studio generator doesn't
        # always propagate this for pure-C targets, so it reaches every binding that includes the headers too.
        if (MSVC AND numkong_arch_arm64_)
            target_compile_definitions(${library_} PUBLIC _ARM64_)
        endif ()

        # Link-time optimization for the shared library only: the static archive ships machine code any toolchain,
        # cGo included, can link. MSVC's `/GL` objects hide the symbols `WINDOWS_EXPORT_ALL_SYMBOLS` must list.
        if (numkong_ipo_supported_ AND library_ STREQUAL "numkong_shared" AND NOT MSVC)
            set_target_properties(${library_} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        endif ()

        # Every capability the toolchain builds; `nk_cpu_capabilities_enabled()` picks among them at load.
        target_link_libraries(${library_} PRIVATE $<BUILD_INTERFACE:numkong_cpu_capabilities_compiled>)
    endforeach ()
endif ()

if (NUMKONG_INSTALL)
    install(
        TARGETS ${numkong_libraries_}
        EXPORT numkongTargets
        ARCHIVE
        BUNDLE
        FRAMEWORK
        LIBRARY
        OBJECTS
        PRIVATE_HEADER
        PUBLIC_HEADER
        RESOURCE
        RUNTIME
    )
endif ()

# The `numkong._numkong` extension, which scikit-build-core builds and packages from `pyproject.toml`.
if (NUMKONG_BUILD_PYTHON)
    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(OpenMP COMPONENTS C)
    python_add_library(
        numkong_python MODULE WITH_SOABI python/numkong.c python/tensor.c python/matrix.c python/types.c
        python/distance.c python/each.c python/trigonometry.c python/geospatial.c python/shapes.c python/attention.c
        python/maxsim.c python/numpy_interop.c python/dlpack_interop.c c/parallel.c
    )
    set_target_properties(numkong_python PROPERTIES OUTPUT_NAME "_numkong" C_STANDARD 11 C_EXTENSIONS OFF)
    target_include_directories(numkong_python PRIVATE python c)
    target_link_libraries(
        numkong_python PRIVATE numkong_static $<$<PLATFORM_ID:Linux,FreeBSD>:m>
                               $<$<AND:$<BOOL:${OpenMP_C_FOUND}>,$<PLATFORM_ID:Linux,FreeBSD>>:OpenMP::OpenMP_C>
    )
    install(TARGETS numkong_python DESTINATION numkong COMPONENT numkong_python)
endif ()

# The `numkong.node` addon, which `cmake-js` builds from `package.json` and which links the C interface as an archive.
if (NUMKONG_BUILD_NODE)
    if (NOT CMAKE_JS_INC)
        message(FATAL_ERROR "The Node addon is built by cmake-js, which supplies the headers and import library")
    endif ()
    find_package(OpenMP COMPONENTS C)
    # `CMAKE_JS_SRC` is the Windows delay-load hook and empty everywhere else.
    add_library(numkong_node SHARED javascript/numkong.c c/parallel.c ${CMAKE_JS_SRC})
    set_target_properties(
        numkong_node
        PROPERTIES PREFIX ""
                   SUFFIX ".node"
                   OUTPUT_NAME "numkong"
                   C_STANDARD 11
                   C_EXTENSIONS OFF
    )
    target_include_directories(numkong_node PRIVATE ${CMAKE_JS_INC} c)
    target_compile_definitions(numkong_node PRIVATE NAPI_VERSION=8)
    target_compile_options(numkong_node PRIVATE $<$<C_COMPILER_ID:MSVC>:/Zc:preprocessor>)
    target_link_libraries(
        numkong_node PRIVATE numkong_static ${CMAKE_JS_LIB}
                             $<$<AND:$<BOOL:${OpenMP_C_FOUND}>,$<PLATFORM_ID:Linux,FreeBSD>>:OpenMP::OpenMP_C>
    )
endif ()

# The static library as SwiftPM links it, beside the headers and their module map: an XCFramework on Apple platforms,
# and elsewhere an SE-0482 artifact bundle. Each build adds its target to the one in NUMKONG_SWIFT_DIRECTORY, so a
# release fills one artifact from a build per target.
if (NUMKONG_BUILD_SWIFT)
    set(NUMKONG_SWIFT_DIRECTORY "${PROJECT_BINARY_DIR}" CACHE PATH "Where the Swift artifact lands")
    if (APPLE)
        # The SDK names the XCFramework library an archive joins, and Mac Catalyst builds with the macOS one.
        get_filename_component(numkong_swift_sysroot_name_ "${CMAKE_OSX_SYSROOT}" NAME)
        string(TOLOWER "${numkong_swift_sysroot_name_}" numkong_swift_sysroot_lowercase_)
        string(REGEX REPLACE "[0-9.]*\\.sdk$" "" numkong_swift_sdk_ "${numkong_swift_sysroot_lowercase_}")
        if (CMAKE_C_FLAGS MATCHES "-macabi")
            set(numkong_swift_sdk_ maccatalyst)
        elseif (NOT numkong_swift_sdk_)
            set(numkong_swift_sdk_ macosx)
        endif ()
        if (CMAKE_OSX_ARCHITECTURES)
            set(numkong_swift_architecture_ "${CMAKE_OSX_ARCHITECTURES}")
        else ()
            set(numkong_swift_architecture_ "${CMAKE_SYSTEM_PROCESSOR}")
        endif ()
        set(numkong_swift_target_ -D "NUMKONG_SWIFT_SDK=${numkong_swift_sdk_}" -D
                                  "NUMKONG_SWIFT_ARCHITECTURE=${numkong_swift_architecture_}"
        )
    else ()
        # SwiftPM matches a variant by the exact triple it builds for, spelled as Swift spells it.
        if (numkong_arch_x8664_)
            set(numkong_swift_arch_ x86_64)
        elseif (numkong_arch_arm64_)
            set(numkong_swift_arch_ aarch64)
        elseif (numkong_arch_arm32_)
            set(numkong_swift_arch_ armv7)
        else ()
            set(numkong_swift_arch_ "${CMAKE_SYSTEM_PROCESSOR}")
        endif ()
        set(numkong_swift_triple_)
        if (numkong_os_linux_)
            set(numkong_swift_triple_ "${numkong_swift_arch_}-unknown-linux-gnu")
        elseif (numkong_os_android_)
            set(numkong_swift_triple_ "${numkong_swift_arch_}-unknown-linux-android${ANDROID_NATIVE_API_LEVEL}")
        elseif (numkong_os_windows_)
            set(numkong_swift_triple_ "${numkong_swift_arch_}-unknown-windows-msvc")
        elseif (numkong_os_wasi_)
            set(numkong_swift_triple_ "wasm32-unknown-wasip1")
        endif ()
        set(NUMKONG_SWIFT_TRIPLES "${numkong_swift_triple_}" CACHE STRING
                                                                   "Swift target triples the artifact bundle serves"
        )
        if (NOT NUMKONG_SWIFT_TRIPLES)
            message(FATAL_ERROR "NUMKONG_BUILD_SWIFT needs NUMKONG_SWIFT_TRIPLES on ${CMAKE_SYSTEM_NAME}")
        endif ()
        set(numkong_swift_target_ -D "NUMKONG_SWIFT_TRIPLES=$<JOIN:${NUMKONG_SWIFT_TRIPLES},,>" -D
                                  "NUMKONG_SWIFT_VERSION=${PROJECT_VERSION}"
        )
    endif ()
    add_custom_target(
        numkong_swift ALL
        COMMAND
            "${CMAKE_COMMAND}" -D "NUMKONG_SWIFT_ARCHIVE=$<TARGET_FILE:numkong_static>" -D
            "NUMKONG_SWIFT_HEADERS=${PROJECT_SOURCE_DIR}/include" -D
            "NUMKONG_SWIFT_DIRECTORY=${NUMKONG_SWIFT_DIRECTORY}" ${numkong_swift_target_} -P
            "${PROJECT_SOURCE_DIR}/cmake/nk_swift_artifact.cmake"
        DEPENDS numkong_static
        VERBATIM
    )
endif ()

# Shared-library test (tests runtime dispatch path)
if (NUMKONG_BUILD_TEST AND NUMKONG_BUILD_SHARED)
    add_executable(numkong_cpu_shared_test ${numkong_test_sources_})
    target_include_directories(numkong_cpu_shared_test PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(numkong_cpu_shared_test numkong_shared fmt::fmt ${numkong_cxx_math_library_})

    # Mirror the shipped Release codegen.
    if (numkong_ipo_supported_)
        set_target_properties(numkong_cpu_shared_test PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif ()

    # The shared-library test must exercise only the DLL dispatch path, so compile the test binary
    # at the platform baseline, without SIMD extensions. This keeps types.h auto-detection, such as
    # MSVC `_MSC_VER >= 1920` → `NUMKONG_TARGET_SKYLAKE=1`, from calling capability kernels directly.
    # All ISA-specific kernel coverage comes from numkong_cpu_test and numkong_cpu_header_test.
    target_link_libraries(numkong_cpu_shared_test numkong_cpu_capabilities_disabled_)

    add_test(NAME numkong_cpu_shared_test COMMAND numkong_cpu_shared_test)
endif ()

if (NUMKONG_BUILD_CUDA)
    # "native", "all" and "all-major" stand for the lists CMake resolved them to.
    set(numkong_cuda_resolved_architectures_)
    foreach (entry_ IN LISTS NUMKONG_CUDA_ARCHITECTURES)
        if (entry_ STREQUAL "native")
            list(APPEND numkong_cuda_resolved_architectures_ ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
        elseif (entry_ STREQUAL "all")
            list(APPEND numkong_cuda_resolved_architectures_ ${CMAKE_CUDA_ARCHITECTURES_ALL})
        elseif (entry_ STREQUAL "all-major")
            list(APPEND numkong_cuda_resolved_architectures_ ${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR})
        else ()
            list(APPEND numkong_cuda_resolved_architectures_ ${entry_})
        endif ()
    endforeach ()
endif ()

if (NUMKONG_BUILD_METAL AND NOT APPLE)
    message(FATAL_ERROR "NUMKONG_BUILD_METAL needs an Apple platform")
elseif (NUMKONG_BUILD_METAL)
    # Metal compiles on the device, so the families only choose the capabilities embedded: the Metal baseline always,
    # Apple9 from M3, Apple10 from M5.
    set(NUMKONG_METAL_ARCHITECTURES "apple7;apple9;apple10"
        CACHE STRING "Apple GPU families whose capabilities the Metal targets embed"
    )
endif ()

# One `numkong_add_gpu_capability` row per GPU capability. Past Ampere, the tensor-core capabilities need family- or
# architecture-specific code, like "120f" rather than "120".
# `mma.sync`: A100, A30, A40, RTX 30, Jetson Orin, L40, RTX 40, and every GPU since.
numkong_add_gpu_capability(
    ampere NUMKONG_TARGET_AMPERE CUDA_ARCHITECTURES 80 86 87 88 89 90 90a 100 100a 100f 103 103a 103f 110 110a 110f 120
                                                    120a 120f 121 121a 121f
)
# FP8 `mma.sync`: L4, L40S, RTX 40, and every GPU since.
numkong_add_gpu_capability(
    ada NUMKONG_TARGET_ADA CUDA_ARCHITECTURES 89 90 90a 100 100a 100f 103 103a 103f 110 110a 110f 120 120a 120f 121
                                              121a 121f
)
# `wgmma`: H100, H200, GH200.
numkong_add_gpu_capability(hopper NUMKONG_TARGET_HOPPER CUDA_ARCHITECTURES 90a)
# `tcgen05`: B200, GB200, B300, GB300.
numkong_add_gpu_capability(blackwell NUMKONG_TARGET_BLACKWELL CUDA_ARCHITECTURES 100a 100f 103a 103f)
# FP6/FP4 `mma.sync`: RTX 50, RTX PRO 6000, DGX Spark.
numkong_add_gpu_capability(blackwellrtx NUMKONG_TARGET_BLACKWELLRTX CUDA_ARCHITECTURES 120a 120f 121a 121f)
# `tcgen05.ld.red`: B300, GB300.
numkong_add_gpu_capability(blackwellultra NUMKONG_TARGET_BLACKWELLULTRA CUDA_ARCHITECTURES 103a 103f)
# MFMA over BF16, F16 and I8: MI300X, MI300A, MI325X.
numkong_add_gpu_capability(cdna3 NUMKONG_TARGET_CDNA3 ROCM_ARCHITECTURES gfx942)
# MFMA over FP8, FP6 and FP4: MI350X, MI355X.
numkong_add_gpu_capability(cdna4 NUMKONG_TARGET_CDNA4 ROCM_ARCHITECTURES gfx950)
# WMMA over FP8, FP6 and FP4: MI450X and MI455X on gfx1250, MI430X on gfx1251.
numkong_add_gpu_capability(cdna5 NUMKONG_TARGET_CDNA5 ROCM_ARCHITECTURES gfx1250 gfx1251)
# Apple9 from M3, Apple10 from M5.
numkong_add_gpu_capability(apple9 NUMKONG_TARGET_APPLE9 METAL_ARCHITECTURES apple9 apple10)
numkong_add_gpu_capability(apple10 NUMKONG_TARGET_APPLE10 METAL_ARCHITECTURES apple10)

# Sets `output_` to the entries of `codes_` that `capability_codes_` names, keeping CMake's "-real" and "-virtual"
# suffixes.
function (numkong_select_gpu_codes output_ codes_ capability_codes_)
    set(kept_)
    foreach (code_ IN LISTS codes_)
        string(REGEX REPLACE "-(real|virtual)$" "" architecture_ "${code_}")
        if (architecture_ IN_LIST capability_codes_)
            list(APPEND kept_ "${code_}")
        endif ()
    endforeach ()
    set(${output_} "${kept_}" PARENT_SCOPE)
endfunction ()

# Adds one GPU generation's unit, like c/target/hopper.cu, to both libraries: compiled for `codes_` alone, in the
# language its extension names, without the CPU capabilities and with its vendor's `capabilities_` target. A unit
# without codes to compile is skipped. The rest are the runtime libraries its calls need.
function (numkong_add_gpu_unit source_ codes_ capabilities_)
    if (NOT codes_)
        return()
    endif ()
    get_filename_component(unit_ "${source_}" NAME_WE)
    add_library(numkong_${unit_}_ OBJECT ${source_})
    target_include_directories(numkong_${unit_}_ PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/c")
    # C++20 makes `NUMKONG_CONSTEXPR` constexpr, which kernels reach through relaxed constexpr.
    set_target_properties(
        numkong_${unit_}_
        PROPERTIES POSITION_INDEPENDENT_CODE ON
                   CUDA_STANDARD 20
                   CUDA_STANDARD_REQUIRED YES
                   CUDA_ARCHITECTURES "${codes_}"
                   HIP_STANDARD 20
                   HIP_STANDARD_REQUIRED YES
                   HIP_ARCHITECTURES "${codes_}"
    )
    # Every CPU capability off, as NVCC's front end cannot parse GCC's `arm_neon.h` and its kin, and every GPU
    # capability the listed architectures run on, as the dispatch units see them; the unit turns off the ones its
    # headers include.
    target_link_libraries(numkong_${unit_}_ PRIVATE numkong_cpu_capabilities_disabled_ ${capabilities_})
    if (MSVC AND numkong_arch_arm64_)
        target_compile_definitions(numkong_${unit_}_ PRIVATE _ARM64_)
    endif ()
    # NVCC compiles a unit's codes one after another unless told to spread them over threads.
    target_compile_options(
        numkong_${unit_}_ PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--threads=0>"
                                  "$<$<COMPILE_LANGUAGE:HIP>:-fvisibility=hidden>"
    )
    if (NOT MSVC)
        target_compile_options(numkong_${unit_}_ PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>")
    endif ()
    target_link_libraries(numkong_${unit_}_ PRIVATE ${ARGN})
    foreach (library_ ${numkong_libraries_})
        target_sources(${library_} PRIVATE $<TARGET_OBJECTS:numkong_${unit_}_>)
        target_link_libraries(${library_} PUBLIC ${ARGN})
    endforeach ()
endfunction ()

# One unit per GPU generation, as USearch builds them, so no cubin or code object carries another generation's
# kernels and the units compile in parallel. The dispatch units see every capability some listed code runs.
if (NUMKONG_BUILD_CUDA)
    find_package(CUDAToolkit 13.0 REQUIRED)
    numkong_select_gpu_codes(
        numkong_ampere_unit_codes_ "${numkong_cuda_resolved_architectures_}" "${nk_target_ampere_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_ada_unit_codes_ "${numkong_cuda_resolved_architectures_}" "${nk_target_ada_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_hopper_unit_codes_ "${numkong_cuda_resolved_architectures_}" "${nk_target_hopper_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_blackwell_unit_codes_ "${numkong_cuda_resolved_architectures_}" "${nk_target_blackwell_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_blackwellrtx_unit_codes_ "${numkong_cuda_resolved_architectures_}"
        "${nk_target_blackwellrtx_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_blackwellultra_unit_codes_ "${numkong_cuda_resolved_architectures_}"
        "${nk_target_blackwellultra_architectures}"
    )
    numkong_add_gpu_unit(
        c/target/cuda.cu "${numkong_cuda_resolved_architectures_}" numkong_cuda_capabilities_compiled_ CUDA::cudart
    )
    numkong_add_gpu_unit(
        c/target/ampere.cu "${numkong_ampere_unit_codes_}" numkong_cuda_capabilities_compiled_ CUDA::cudart
    )
    numkong_add_gpu_unit(c/target/ada.cu "${numkong_ada_unit_codes_}" numkong_cuda_capabilities_compiled_ CUDA::cudart)
    numkong_add_gpu_unit(
        c/target/hopper.cu "${numkong_hopper_unit_codes_}" numkong_cuda_capabilities_compiled_ CUDA::cudart
    )
    numkong_add_gpu_unit(
        c/target/blackwell.cu "${numkong_blackwell_unit_codes_}" numkong_cuda_capabilities_compiled_ CUDA::cudart
    )
    numkong_add_gpu_unit(
        c/target/blackwellrtx.cu "${numkong_blackwellrtx_unit_codes_}" numkong_cuda_capabilities_compiled_ CUDA::cudart
    )
    numkong_add_gpu_unit(
        c/target/blackwellultra.cu "${numkong_blackwellultra_unit_codes_}" numkong_cuda_capabilities_compiled_
        CUDA::cudart
    )
    foreach (library_ ${numkong_libraries_})
        target_compile_definitions(${library_} PRIVATE NUMKONG_ARCH_CUDA_=1)
        target_link_libraries(${library_} PRIVATE $<BUILD_INTERFACE:numkong_cuda_capabilities_compiled_>)
    endforeach ()
endif ()

# Both GPU vendors may share one library: their units are separate files, each vendor has its own capability group,
# and each numbers its own devices.
if (NUMKONG_BUILD_ROCM)
    find_package(hip REQUIRED)
    numkong_select_gpu_codes(
        numkong_cdna3_unit_codes_ "${NUMKONG_ROCM_ARCHITECTURES}" "${nk_target_cdna3_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_cdna4_unit_codes_ "${NUMKONG_ROCM_ARCHITECTURES}" "${nk_target_cdna4_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_cdna5_unit_codes_ "${NUMKONG_ROCM_ARCHITECTURES}" "${nk_target_cdna5_architectures}"
    )
    numkong_add_gpu_unit(
        c/target/rocm.hip "${NUMKONG_ROCM_ARCHITECTURES}" numkong_rocm_capabilities_compiled_ hip::host
    )
    numkong_add_gpu_unit(
        c/target/cdna3.hip "${numkong_cdna3_unit_codes_}" numkong_rocm_capabilities_compiled_ hip::host
    )
    numkong_add_gpu_unit(
        c/target/cdna4.hip "${numkong_cdna4_unit_codes_}" numkong_rocm_capabilities_compiled_ hip::host
    )
    numkong_add_gpu_unit(
        c/target/cdna5.hip "${numkong_cdna5_unit_codes_}" numkong_rocm_capabilities_compiled_ hip::host
    )
    foreach (library_ ${numkong_libraries_})
        target_compile_definitions(${library_} PRIVATE NUMKONG_ARCH_ROCM_=1)
        target_link_libraries(${library_} PRIVATE $<BUILD_INTERFACE:numkong_rocm_capabilities_compiled_>)
    endforeach ()
    set_target_properties(${numkong_libraries_} PROPERTIES HIP_ARCHITECTURES "${NUMKONG_ROCM_ARCHITECTURES}")
    # HIP-Clang cannot read the GCC link-time-optimized objects of the CPU sources, so the host linker links.
    set_target_properties(${numkong_libraries_} PROPERTIES LINKER_LANGUAGE C)
endif ()

if (NUMKONG_BUILD_METAL)
    # Metal compiles its kernels on the device, so the C units hold only the host side of each Apple capability.
    add_library(numkong_metal_ OBJECT c/target/metal.c c/target/apple9.c c/target/apple10.c)
    target_include_directories(numkong_metal_ PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/c")
    set_target_properties(numkong_metal_ PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(
        numkong_metal_ PRIVATE numkong_cpu_capabilities_disabled_ numkong_metal_capabilities_compiled_
    )
    target_compile_options(numkong_metal_ PRIVATE -fvisibility=hidden "--embed-dir=${PROJECT_SOURCE_DIR}/include")
    foreach (library_ ${numkong_libraries_})
        target_sources(${library_} PRIVATE $<TARGET_OBJECTS:numkong_metal_>)
        target_link_libraries(${library_} PRIVATE $<BUILD_INTERFACE:numkong_metal_capabilities_compiled_>)
        target_link_libraries(${library_} PUBLIC "-framework Metal" "-framework Foundation")
    endforeach ()
    # Consumers see the Metal entries and the queue they take.
    foreach (library_ ${numkong_libraries_})
        target_compile_definitions(${library_} INTERFACE NUMKONG_ARCH_METAL_=1)
        target_compile_options(${library_} INTERFACE ${numkong_embed_options_})
    endforeach ()
endif ()

# Configures CUDA suites for `architectures_`: C++20, `constexpr` host helpers
# callable from device code so the tensor views stay GPU-friendly, and the host baseline forwarded to NVCC.
function (numkong_configure_cuda_suite target_ architectures_)
    target_include_directories(${target_} PRIVATE "${PROJECT_SOURCE_DIR}/include")
    set_target_properties(
        ${target_} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES CUDA_EXTENSIONS NO CUDA_ARCHITECTURES
                                                                                             "${architectures_}"
    )
    target_compile_options(${target_} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if (MSVC)
        # {fmt} demands `/utf-8` for Unicode output, which NVCC's front end on Windows cannot honour.
        target_compile_definitions(${target_} PRIVATE FMT_UNICODE=0)
        if (numkong_arch_arm64_)
            target_compile_definitions(${target_} PRIVATE _ARM64_)
        endif ()
    elseif (numkong_arch_x8664_)
        target_compile_options(
            ${target_}
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=x86-64>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-tree-vectorize>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-tree-slp-vectorize>
        )
    elseif (numkong_arch_arm64_)
        target_compile_options(
            ${target_}
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=armv8-a>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-tree-vectorize>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-tree-slp-vectorize>
        )
    endif ()
    target_link_libraries(${target_} PRIVATE fmt::fmt)
endfunction ()

# Compiles a ROCm test target for `architectures_` the way every ROCm test expects: C++20 over the public headers.
function (numkong_configure_rocm_test target_ architectures_)
    target_include_directories(${target_} PRIVATE "${PROJECT_SOURCE_DIR}/include")
    set_target_properties(
        ${target_} PROPERTIES HIP_ARCHITECTURES "${architectures_}" HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES
                              HIP_EXTENSIONS NO
    )
    target_link_libraries(${target_} PRIVATE fmt::fmt)
endfunction ()

if (NUMKONG_BUILD_TEST)
    foreach (vendor_ cuda rocm metal)
        if (vendor_ STREQUAL "cuda")
            set(vendor_sources_ test/cross_cuda.cu test/each_cuda.cu test/cast_cuda.cu test/reduce_cuda.cu
                                test/tensor_cuda.cu test/shapes_cuda.cu
            )
            set(vendor_enabled_ ${NUMKONG_BUILD_CUDA})
        elseif (vendor_ STREQUAL "rocm")
            set(vendor_sources_ test/cross_rocm.hip test/each_rocm.hip test/cast_rocm.hip)
            set(vendor_enabled_ ${NUMKONG_BUILD_ROCM})
        else ()
            set(vendor_sources_ test/cross_metal.cpp test/each_metal.cpp test/cast_metal.cpp)
            set(vendor_enabled_ ${NUMKONG_BUILD_METAL})
        endif ()
        set(leaf_ numkong_test_${vendor_})
        add_library(${leaf_} OBJECT ${vendor_sources_})
        target_include_directories(${leaf_} PRIVATE "${PROJECT_SOURCE_DIR}/include")
        target_link_libraries(${leaf_} PRIVATE fmt::fmt)
        target_link_libraries(${leaf_} PRIVATE numkong_cpu_capabilities_disabled_)
        if (vendor_enabled_)
            target_link_libraries(${leaf_} PRIVATE numkong_${vendor_}_capabilities_compiled_)
        else ()
            set_source_files_properties(${vendor_sources_} PROPERTIES LANGUAGE CXX)
        endif ()
        foreach (test_ numkong_cpu_test numkong_cpu_shared_test)
            if (TARGET ${test_})
                target_sources(${test_} PRIVATE $<TARGET_OBJECTS:${leaf_}>)
            endif ()
        endforeach ()
    endforeach ()
    if (NUMKONG_BUILD_CUDA)
        numkong_configure_cuda_suite(numkong_test_cuda "${NUMKONG_CUDA_ARCHITECTURES}")
        target_link_libraries(numkong_test_cuda PRIVATE CUDA::cudart)
    endif ()
    if (NUMKONG_BUILD_ROCM)
        numkong_configure_rocm_test(numkong_test_rocm "${NUMKONG_ROCM_ARCHITECTURES}")
        target_link_libraries(numkong_test_rocm PRIVATE hip::host)
    endif ()
    if (NUMKONG_BUILD_METAL)
        target_compile_options(numkong_test_metal PRIVATE ${numkong_embed_options_})
    endif ()
endif ()

if (NUMKONG_BUILD_BENCH)
    add_library(
        numkong_bench_cuda OBJECT bench/cross_cuda.cu bench/each_cuda.cu bench/cast_cuda.cu bench/reduce_cuda.cu
    )
    add_library(numkong_bench_rocm OBJECT bench/cross_rocm.hip bench/each_rocm.hip)
    foreach (leaf_ numkong_bench_cuda numkong_bench_rocm)
        target_include_directories(${leaf_} PRIVATE "${PROJECT_SOURCE_DIR}/include")
        target_link_libraries(${leaf_} PRIVATE fmt::fmt)
        target_link_libraries(${leaf_} PRIVATE numkong_cpu_capabilities_disabled_)
    endforeach ()
    target_sources(numkong_bench PRIVATE $<TARGET_OBJECTS:numkong_bench_cuda> $<TARGET_OBJECTS:numkong_bench_rocm>)
endif ()

# CUDA rows in the same `numkong_bench`, from bench/*_cuda.cu, with optional cuBLAS, cuDNN and cuVS baselines.
if (NUMKONG_BUILD_BENCH AND NUMKONG_BUILD_CUDA)
    find_package(CUDAToolkit 13.0 REQUIRED)
    numkong_configure_cuda_suite(numkong_bench_cuda "${NUMKONG_CUDA_ARCHITECTURES}")
    target_compile_definitions(
        numkong_bench_cuda
        PRIVATE NUMKONG_COMPARE_TO_CUBLAS=$<BOOL:${NUMKONG_COMPARE_TO_CUBLAS}>
                NUMKONG_COMPARE_TO_CUDNN=$<BOOL:${NUMKONG_COMPARE_TO_CUDNN}>
                NUMKONG_COMPARE_TO_CUVS=$<BOOL:${NUMKONG_COMPARE_TO_CUVS}>
    )
    target_link_libraries(numkong_bench CUDA::cudart)
    target_link_libraries(numkong_bench_cuda PRIVATE numkong_cuda_capabilities_compiled_ CUDA::cudart)
    if (NUMKONG_COMPARE_TO_CUBLAS)
        target_link_libraries(numkong_bench CUDA::cublas CUDA::cublasLt)
        target_link_libraries(numkong_bench_cuda PRIVATE CUDA::cublas CUDA::cublasLt)
    endif ()
    if (NUMKONG_COMPARE_TO_CUDNN)
        include(cmake/nk_cudnn.cmake)
        target_link_libraries(numkong_bench nk_cudnn)
        target_link_libraries(numkong_bench_cuda PRIVATE nk_cudnn)
    endif ()
    if (NUMKONG_COMPARE_TO_CUVS)
        include(cmake/nk_cuvs.cmake)
        target_link_libraries(numkong_bench nk_cuvs)
        target_link_libraries(numkong_bench_cuda PRIVATE nk_cuvs)
    endif ()
endif ()

# Without CUDA its rows compile away as C++, as every `cross_*` family does off its platform.
if (NUMKONG_BUILD_BENCH AND NOT NUMKONG_BUILD_CUDA)
    set_source_files_properties(
        bench/cross_cuda.cu bench/each_cuda.cu bench/cast_cuda.cu bench/reduce_cuda.cu PROPERTIES LANGUAGE CXX
    )
endif ()

if (NUMKONG_BUILD_BENCH AND NUMKONG_BUILD_ROCM)
    target_link_libraries(numkong_bench hip::host)
    target_link_libraries(numkong_bench_rocm PRIVATE hip::host)
    target_link_libraries(numkong_bench_rocm PRIVATE numkong_rocm_capabilities_compiled_)
    set_target_properties(
        numkong_bench_rocm PROPERTIES HIP_ARCHITECTURES "${NUMKONG_ROCM_ARCHITECTURES}" HIP_STANDARD 20
                                      HIP_STANDARD_REQUIRED YES HIP_EXTENSIONS NO
    )
elseif (NUMKONG_BUILD_BENCH)
    set_source_files_properties(bench/cross_rocm.hip bench/each_rocm.hip PROPERTIES LANGUAGE CXX)
endif ()

# Metal rows in the same `numkong_bench`, from bench/cross_metal.cpp, over the queue's shared memory.
if (NUMKONG_BUILD_BENCH AND NUMKONG_BUILD_METAL)
    target_link_libraries(numkong_bench numkong_metal_capabilities_compiled_)
    target_link_libraries(numkong_bench "-framework Metal" "-framework Foundation")
endif ()

# Install & export — gated, so a parent project's `make install` ships NumKong only on request, and
# `find_package(numkong)` works after installation, mirroring ForkUnion's packaging.
if (NUMKONG_INSTALL)
    include(CMakePackageConfigHelpers)

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY c/ DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${PROJECT_NAME}/src)
    install(TARGETS numkong_header EXPORT numkongTargets)
    install(EXPORT numkongTargets FILE numkongTargets.cmake NAMESPACE numkong::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numkong
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/numkongConfigVersion.cmake" VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/numkongConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/numkongConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numkong
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/numkongConfig.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/numkongConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numkong
    )
endif ()
