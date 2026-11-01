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
#   -D NUMKONG_CUDA_ARCH=<code>              (default: empty for no CUDA, `all` for the release set, `native`, or one
#                                            code like `100f`, added to the libraries, tests and benchmarks built)
#   -D NUMKONG_ROCM_ARCH=<code>              (default: empty for no ROCm, `all` for gfx942;gfx950;gfx1250;gfx1251,
#                                            `native`, or one gfx target)
#   -D NUMKONG_METAL_ARCH=<family>           (default: empty for no Metal, `all` for apple7;apple9;apple10, or one
#                                            family)
#   -D NUMKONG_COMPARE_TO_BLAS=<AUTO|ON|OFF> (default: AUTO)
#   -D NUMKONG_COMPARE_TO_MKL=<AUTO|ON|OFF>  (default: AUTO)
#   -D NUMKONG_BUILD_SHARED=<ON|OFF>         (default: ON when top-level, OFF as a subproject)
#   -D NUMKONG_BUILD_SANITIZERS=<off|address|thread> (default: off, address also arms UndefinedBehaviorSanitizer)
#   -D NUMKONG_WERROR=<ON|OFF>               (default: OFF, the `check` preset switches it on)
#   -D NUMKONG_MARCH=<arch>                  (default: the NUMKONG_MARCH environment variable, empty for the
#                                            portable ABI floor, an ISA level, a CPU name, `native`, or under the
#                                            WebAssembly toolchains the module's one SIMD capability, `serial`,
#                                            `v128` or `v128relaxed`)
#   -D NUMKONG_MULTIVERSIONED=<ON|OFF>       (default: ON, every capability above the floor with runtime dispatch;
#                                            OFF keeps only what NUMKONG_MARCH implies, one kernel per operation)
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
# If your system CMake is older than 3.28, install a recent release from PyPI or the Kitware APT repo.
#
#     python -m pip install "cmake>=3.28"
#
# NumKong does not support NVHPC (nvc and nvc++): those compilers cannot apply per-function
# __attribute__((target(...))), which is how SIMD kernels are dispatched in this library.
#
cmake_minimum_required(VERSION 3.28)

# Leverage `ccache` for incremental compilation
find_program(NUMKONG_CCACHE_PROGRAM ccache)

if (NUMKONG_CCACHE_PROGRAM AND NOT CMAKE_C_COMPILER_LAUNCHER)
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

# Standards are floors each target declares through `target_compile_features`, so a parent project's units and the
# fetched dependencies build at the standard they pick for themselves.
set(CMAKE_C_EXTENSIONS NO)
set(CMAKE_CXX_EXTENSIONS NO)
# No target uses C++20 modules, and a Clang installed without `clang-scan-deps` would fail every C++ unit.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(numkong_os_wasi_ OFF)

if (CMAKE_SYSTEM_NAME STREQUAL "WASI")
    set(numkong_os_wasi_ ON)
endif ()

option(NUMKONG_BUILD_SHARED "Compile a dynamic library" ${PROJECT_IS_TOP_LEVEL})
option(NUMKONG_INSTALL "Install headers, libraries & the CMake package files" ${PROJECT_IS_TOP_LEVEL})
option(NUMKONG_BUILD_TEST "Compile precision tests, and shared-library tests with NUMKONG_BUILD_SHARED" OFF)
option(NUMKONG_BUILD_BENCH "Compile micro-benchmarks for current ISA" OFF)
option(NUMKONG_BUILD_PYTHON "Compile the CPython extension over the static library" OFF)
option(NUMKONG_BUILD_NODE "Compile the Node addon over the static library; cmake-js switches it on" OFF)
option(NUMKONG_BUILD_SWIFT "Package the static library for SwiftPM, as an XCFramework or an artifact bundle" OFF)
set(NUMKONG_CUDA_ARCH "" CACHE STRING "NVIDIA code the libraries, tests and benchmarks build for: all, native or one")
set(NUMKONG_ROCM_ARCH "" CACHE STRING "AMD target the libraries, tests and benchmarks build for: all, native or one")
set(NUMKONG_METAL_ARCH "" CACHE STRING "Apple GPU family whose capabilities the libraries embed: all or one")

# Seeded from the environment at the first configure, so `cargo` and `cmake-js` honour `NUMKONG_MARCH=native`.
set(NUMKONG_MARCH "$ENV{NUMKONG_MARCH}"
    CACHE STRING "The `-march` floor, like x86-64-v3 or native, or a WebAssembly module's SIMD capability, like v128"
)
option(NUMKONG_MULTIVERSIONED "Keep every capability above the floor, or only the best one NUMKONG_MARCH implies" ON)
option(NUMKONG_WERROR "Treat compiler warnings as errors" OFF)
set(NUMKONG_BUILD_SANITIZERS "off"
    CACHE STRING "Sanitizers for every target NumKong builds: off, address with undefined, or thread"
)
set_property(CACHE NUMKONG_BUILD_SANITIZERS PROPERTY STRINGS off address thread)

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

include(cmake/numkong_flags.cmake)

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

# Define the header-only library. The include path & sources wear `BUILD_INTERFACE` genexes, so the
# target stays exportable: the installed package resolves headers under the install prefix instead.
include(GNUInstallDirs)
file(GLOB numkong_header_sources_ c/numkong/*.h include/numkong/*.h include/numkong/*.hpp cpp/numkong/*.hpp)
add_library(numkong_header INTERFACE)
add_library(numkong::header ALIAS numkong_header)
set_target_properties(numkong_header PROPERTIES EXPORT_NAME header)
target_sources(numkong_header INTERFACE "$<BUILD_INTERFACE:${numkong_header_sources_}>")
target_include_directories(
    numkong_header
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/c/module.modulemap DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif ()

    message(STATUS "Clang C modules enabled: cache at ${CMAKE_BINARY_DIR}/module-cache")
endif ()

include(cmake/numkong_platform.cmake)

# One compile probe per capability under `probes/`, defining `NUMKONG_TARGET_<CAPABILITY>` as the toolchain compiles
# it through `numkong::cpu_capabilities_compiled`, and as 0 through `numkong_cpu_capabilities_disabled_`. Each probe
# compiles at the units' `numkong_unit_arch_flags_`, which `try_compile` would not take from `add_compile_options`.
include(cmake/numkong_probes.cmake)

include(cmake/numkong_gpu.cmake)

if (NUMKONG_MARCH)
    set(numkong_floor_ "${NUMKONG_MARCH}")
else ()
    set(numkong_floor_ "the portable floor")
endif ()
get_property(numkong_capabilities_enabled_ GLOBAL PROPERTY numkong_capabilities_enabled)
list(JOIN numkong_capabilities_enabled_ " " numkong_capabilities_summary_)
if (NUMKONG_MULTIVERSIONED)
    message(STATUS "NumKong: multiversioned above ${numkong_floor_}")
else ()
    message(STATUS "NumKong: monoversioned for ${numkong_floor_}: ${numkong_capabilities_summary_}")
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
    include(cmake/numkong_math_library.cmake)
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
    target_link_libraries(numkong_bench numkong_static numkong_warnings fmt::fmt)
    target_compile_features(numkong_bench PRIVATE cxx_std_20)

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
# unit per enabled CPU capability, defining that capability's kernels of every family. Each unit names its own
# capability, so every file compiles with the same flags and the units build in parallel. WebAssembly enables its
# capabilities through NUMKONG_MARCH, so its units follow the platform.
set(numkong_dispatch_units_
    src/dispatch/dot.c src/dispatch/dots.c src/dispatch/spatial.c src/dispatch/spatials.c src/dispatch/set.c
    src/dispatch/sets.c src/dispatch/maxsim.c src/dispatch/attention.c src/dispatch/sparse.c src/dispatch/reduce.c
    src/dispatch/each.c src/dispatch/cast.c src/dispatch/scalar.c src/dispatch/trigonometry.c src/dispatch/curved.c
    src/dispatch/probability.c src/dispatch/geospatial.c src/dispatch/shapes.c
)
set(numkong_library_units_
    ${numkong_dispatch_units_}
    src/capabilities/serial.c
    $<$<BOOL:${numkong_target_haswell_enabled}>:src/capabilities/haswell.c>
    $<$<BOOL:${numkong_target_alder_enabled}>:src/capabilities/alder.c>
    $<$<BOOL:${numkong_target_sierra_enabled}>:src/capabilities/sierra.c>
    $<$<BOOL:${numkong_target_skylake_enabled}>:src/capabilities/skylake.c>
    $<$<BOOL:${numkong_target_icelake_enabled}>:src/capabilities/icelake.c>
    $<$<BOOL:${numkong_target_genoa_enabled}>:src/capabilities/genoa.c>
    $<$<BOOL:${numkong_target_turin_enabled}>:src/capabilities/turin.c>
    $<$<BOOL:${numkong_target_sapphire_enabled}>:src/capabilities/sapphire.c>
    $<$<BOOL:${numkong_target_diamond_enabled}>:src/capabilities/diamond.c>
    $<$<BOOL:${numkong_target_sapphireamx_enabled}>:src/capabilities/sapphireamx.c>
    $<$<BOOL:${numkong_target_graniteamx_enabled}>:src/capabilities/graniteamx.c>
    $<$<BOOL:${numkong_target_diamondamx_enabled}>:src/capabilities/diamondamx.c>
    $<$<BOOL:${numkong_target_neon_enabled}>:src/capabilities/neon.c>
    $<$<BOOL:${numkong_target_neonhalf_enabled}>:src/capabilities/neonhalf.c>
    $<$<BOOL:${numkong_target_neonbfdot_enabled}>:src/capabilities/neonbfdot.c>
    $<$<BOOL:${numkong_target_neonfhm_enabled}>:src/capabilities/neonfhm.c>
    $<$<BOOL:${numkong_target_neonsdot_enabled}>:src/capabilities/neonsdot.c>
    $<$<BOOL:${numkong_target_neonfp8_enabled}>:src/capabilities/neonfp8.c>
    $<$<BOOL:${numkong_target_sve_enabled}>:src/capabilities/sve.c>
    $<$<BOOL:${numkong_target_svehalf_enabled}>:src/capabilities/svehalf.c>
    $<$<BOOL:${numkong_target_svesdot_enabled}>:src/capabilities/svesdot.c>
    $<$<BOOL:${numkong_target_svebfdot_enabled}>:src/capabilities/svebfdot.c>
    $<$<BOOL:${numkong_target_sve2_enabled}>:src/capabilities/sve2.c>
    $<$<BOOL:${numkong_target_sme_enabled}>:src/capabilities/sme.c>
    $<$<BOOL:${numkong_target_smef64_enabled}>:src/capabilities/smef64.c>
    $<$<BOOL:${numkong_target_smebi32_enabled}>:src/capabilities/smebi32.c>
    $<$<BOOL:${numkong_target_rvv_enabled}>:src/capabilities/rvv.c>
    $<$<BOOL:${numkong_target_rvvhalf_enabled}>:src/capabilities/rvvhalf.c>
    $<$<BOOL:${numkong_target_rvvbf16_enabled}>:src/capabilities/rvvbf16.c>
    $<$<BOOL:${numkong_target_rvvbb_enabled}>:src/capabilities/rvvbb.c>
    $<$<BOOL:${numkong_arch_wasm_}>:src/capabilities/v128.c>
    $<$<BOOL:${numkong_arch_wasm_}>:src/capabilities/v128relaxed.c>
    $<$<BOOL:${numkong_target_powervsx_enabled}>:src/capabilities/powervsx.c>
    $<$<BOOL:${numkong_target_loongsonasx_enabled}>:src/capabilities/loongsonasx.c>
)
# Dispatch calls kernels through `void *` signatures on purpose, which every supported ABI passes alike
set_source_files_properties(
    ${numkong_dispatch_units_}
    PROPERTIES COMPILE_OPTIONS
               "$<$<AND:$<C_COMPILER_ID:Clang,AppleClang>,$<BOOL:${NUMKONG_BUILD_SANITIZERS}>>:-fno-sanitize=function>"
)

# Build tests
set(numkong_test_sources_
    test/main_cpu.cpp test/dot.cpp test/spatial.cpp test/set.cpp test/curved.cpp test/probability.cpp test/each.cpp
    test/trigonometry.cpp test/geospatial.cpp test/shapes.cpp test/cast.cpp test/reduce.cpp test/sparse.cpp
    # ISA-family cross/batch files for parallel compilation
    test/cross_serial.cpp test/cross_runtime.cpp test/cross_x8664.cpp test/cross_x8664amx.cpp test/cross_arm64.cpp
    test/cross_blas.cpp test/cross_riscv64.cpp test/cross_ppc64.cpp test/cross_loongarch64.cpp test/cross_wasm.cpp
    test/maxsim.cpp test/tensor.cpp
)
# The standard and {fmt} headers of `test/harness.hpp`, parsed once per test binary rather than once per suite.
set(numkong_test_precompiled_ <chrono> <complex> <random> <regex> <string> <vector> <fmt/format.h>)

if (NUMKONG_BUILD_TEST)
    # C++ precision test (ULP-based error analysis)
    add_executable(numkong_cpu_test ${numkong_test_sources_})
    target_link_libraries(numkong_cpu_test numkong_static numkong_warnings fmt::fmt ${numkong_cxx_math_library_})
    target_compile_features(numkong_cpu_test PRIVATE cxx_std_20)
    target_precompile_headers(numkong_cpu_test PRIVATE ${numkong_test_precompiled_})

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
        numkong_cpu_header_test numkong_header numkong_cpu_capabilities_compiled numkong_warnings fmt::fmt
        ${numkong_cxx_math_library_}
    )
    target_compile_features(numkong_cpu_header_test PRIVATE cxx_std_20)
    target_precompile_headers(numkong_cpu_header_test PRIVATE ${numkong_test_precompiled_})
    add_test(NAME numkong_cpu_header_test COMMAND numkong_cpu_header_test)
endif ()

# The static library, which every binding but Swift links and a parent project reaches as `numkong::static`, and the
# shared library, or the Emscripten JS module
set(numkong_library_sources_ ${numkong_header_sources_} src/numkong.c ${numkong_library_units_})

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

    # One module carries one SIMD capability, the toolchain's NUMKONG_MARCH, so its name says which.
    if (numkong_wasm64_)
        set(numkong_wasm_module_name_ "numkong-wasm64-${numkong_wasm_simd_}")
    else ()
        set(numkong_wasm_module_name_ "numkong-wasm32-${numkong_wasm_simd_}")
    endif ()
    # Generates `${numkong_wasm_module_name_}.js` + `.wasm`
    set_target_properties(numkong_shared PROPERTIES OUTPUT_NAME "${numkong_wasm_module_name_}" SUFFIX ".js")
    message(STATUS "NumKong: Building Emscripten module ${numkong_wasm_module_name_}")

    # `types.h` leaves every non-WebAssembly capability off, and `v128` and `v128relaxed` follow `-msimd128` and
    # `-mrelaxed-simd`.
    foreach (library_ ${numkong_libraries_})
        # The library, unlike the headers, may use C11: `src/numkong.c` caches detection in <stdatomic.h> atomics.
        target_compile_features(${library_} PRIVATE c_std_11)
        target_link_libraries(${library_} PRIVATE $<BUILD_INTERFACE:numkong_warnings>)
        target_include_directories(
            ${library_}
            PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
        # The library's units reach their private `dispatch.h` by name.
        target_include_directories(${library_} PRIVATE "${PROJECT_SOURCE_DIR}/src")
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
        target_link_options(numkong_shared PRIVATE "SHELL:--target=wasm64")
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
        # The library, unlike the headers, may use C11: `src/numkong.c` caches detection in <stdatomic.h> atomics,
        # which MSVC gates behind a flag.
        target_compile_features(${library_} PRIVATE c_std_11)
        target_link_libraries(${library_} PRIVATE $<BUILD_INTERFACE:numkong_warnings>)
        target_compile_options(${library_} PRIVATE $<$<C_COMPILER_ID:MSVC>:/experimental:c11atomics>)
        target_include_directories(
            ${library_}
            PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
        # The library's units reach their private `dispatch.h` by name.
        target_include_directories(${library_} PRIVATE "${PROJECT_SOURCE_DIR}/src")

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

        # Link-time optimization for the shared library, and for a monoversioned static archive, whose folded
        # tables then leave the kernels they skip out of the products' links. Any other static archive ships machine
        # code any toolchain, cGo included, can link. MSVC's `/GL` objects hide the symbols `WINDOWS_EXPORT_ALL_SYMBOLS`
        # must list.
        if (numkong_ipo_supported_ AND NOT MSVC AND library_ STREQUAL "numkong_shared")
            set_target_properties(${library_} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        elseif (numkong_ipo_supported_ AND NOT MSVC AND NOT NUMKONG_MULTIVERSIONED)
            set_target_properties(${library_} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
            # Its objects are bitcode, so every link that takes them must optimize at link time too.
            target_link_options(${library_} INTERFACE ${CMAKE_C_COMPILE_OPTIONS_IPO} ${CMAKE_C_LINK_OPTIONS_IPO})
        endif ()

        # Every capability the toolchain builds; `nk_cpu_capabilities_enabled()` picks among them at load.
        target_link_libraries(${library_} PRIVATE $<BUILD_INTERFACE:numkong_cpu_capabilities_compiled>)
    endforeach ()
endif ()

# The `numkong._numkong` extension, which scikit-build-core builds and packages from `pyproject.toml`.
if (NUMKONG_BUILD_PYTHON)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(OpenMP COMPONENTS C)
    python_add_library(
        numkong_python MODULE WITH_SOABI python/numkong.c python/tensor.c python/matrix.c python/types.c
        python/distance.c python/each.c python/trigonometry.c python/geospatial.c python/shapes.c python/attention.c
        python/maxsim.c python/numpy_interop.c python/dlpack_interop.c src/parallel.c
    )
    set_target_properties(numkong_python PROPERTIES OUTPUT_NAME "_numkong")
    target_compile_features(numkong_python PRIVATE c_std_11)
    target_include_directories(numkong_python PRIVATE python src)
    target_link_libraries(
        numkong_python PRIVATE numkong_static numkong_warnings $<$<PLATFORM_ID:Linux,FreeBSD>:m>
                               $<$<AND:$<BOOL:${OpenMP_C_FOUND}>,$<PLATFORM_ID:Linux,FreeBSD>>:OpenMP::OpenMP_C>
    )
    # The module's one export is its init function; the static library's kernels stay inside it.
    set_target_properties(numkong_python PROPERTIES C_VISIBILITY_PRESET hidden)
    if (APPLE)
        target_link_options(numkong_python PRIVATE "LINKER:-exported_symbol,_PyInit__numkong")
    elseif (NOT MSVC)
        target_link_options(numkong_python PRIVATE "LINKER:--exclude-libs,ALL")
    endif ()
    install(TARGETS numkong_python DESTINATION numkong COMPONENT numkong_python)
endif ()

# The `numkong.node` addon, which `cmake-js` builds from `package.json` and which links the C interface as an archive.
if (NUMKONG_BUILD_NODE)
    if (NOT CMAKE_JS_INC)
        message(FATAL_ERROR "The Node addon is built by cmake-js, which supplies the headers and import library")
    endif ()
    find_package(OpenMP COMPONENTS C)
    # `CMAKE_JS_SRC` is the Windows delay-load hook and empty everywhere else.
    add_library(numkong_node SHARED javascript/numkong.c src/parallel.c ${CMAKE_JS_SRC})
    set_target_properties(numkong_node PROPERTIES PREFIX "" SUFFIX ".node" OUTPUT_NAME "numkong")
    target_compile_features(numkong_node PRIVATE c_std_11)
    target_include_directories(numkong_node PRIVATE ${CMAKE_JS_INC} src)
    target_compile_definitions(numkong_node PRIVATE NAPI_VERSION=8)
    target_compile_options(numkong_node PRIVATE $<$<C_COMPILER_ID:MSVC>:/Zc:preprocessor>)
    target_link_libraries(
        numkong_node PRIVATE numkong_static numkong_warnings ${CMAKE_JS_LIB}
                             $<$<AND:$<BOOL:${OpenMP_C_FOUND}>,$<PLATFORM_ID:Linux,FreeBSD>>:OpenMP::OpenMP_C>
    )
endif ()

# Shared-library test (tests runtime dispatch path)
if (NUMKONG_BUILD_TEST AND NUMKONG_BUILD_SHARED)
    add_executable(numkong_cpu_shared_test ${numkong_test_sources_})
    target_include_directories(
        numkong_cpu_shared_test PRIVATE "${PROJECT_SOURCE_DIR}/c" "${PROJECT_SOURCE_DIR}/include"
                                        "${PROJECT_SOURCE_DIR}/cpp"
    )
    target_link_libraries(numkong_cpu_shared_test numkong_shared numkong_warnings fmt::fmt ${numkong_cxx_math_library_})
    target_compile_features(numkong_cpu_shared_test PRIVATE cxx_std_20)
    target_precompile_headers(numkong_cpu_shared_test PRIVATE ${numkong_test_precompiled_})

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

# A monoversioned build folds each dispatch table to the kernel of the best enabled capability implementing its
# operation.
if (NOT NUMKONG_MULTIVERSIONED)
    foreach (library_ ${numkong_libraries_})
        target_compile_definitions(${library_} PRIVATE NUMKONG_MULTIVERSIONED=0)
    endforeach ()
endif ()

# One unit per GPU generation, as USearch builds them, so no cubin or code object carries another generation's
# kernels and the units compile in parallel. The dispatch units see every capability some listed code runs.
if (NUMKONG_CUDA_ARCH)
    find_package(CUDAToolkit 13.0 REQUIRED)
    string(REGEX REPLACE "-(real|virtual)" "" numkong_cuda_codes_ "${numkong_cuda_resolved_architectures_}")
    numkong_select_gpu_codes(
        numkong_cuda_unit_codes_ "${numkong_cuda_resolved_architectures_}" "${numkong_cuda_codes_}"
    )
    numkong_select_gpu_codes(
        numkong_ampere_unit_codes_ "${numkong_cuda_resolved_architectures_}" "${numkong_target_ampere_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_ada_unit_codes_ "${numkong_cuda_resolved_architectures_}" "${numkong_target_ada_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_hopper_unit_codes_ "${numkong_cuda_resolved_architectures_}" "${numkong_target_hopper_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_blackwell_unit_codes_ "${numkong_cuda_resolved_architectures_}"
        "${numkong_target_blackwell_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_blackwellrtx_unit_codes_ "${numkong_cuda_resolved_architectures_}"
        "${numkong_target_blackwellrtx_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_blackwellultra_unit_codes_ "${numkong_cuda_resolved_architectures_}"
        "${numkong_target_blackwellultra_architectures}"
    )
    numkong_add_gpu_unit(
        src/capabilities/cuda.cu "${numkong_cuda_unit_codes_}" numkong_cuda_capabilities_compiled_ CUDA::cudart
    )
    numkong_add_gpu_unit(
        src/capabilities/ampere.cu "${numkong_ampere_unit_codes_}" numkong_cuda_capabilities_compiled_ CUDA::cudart
    )
    numkong_add_gpu_unit(
        src/capabilities/ada.cu "${numkong_ada_unit_codes_}" numkong_cuda_capabilities_compiled_ CUDA::cudart
    )
    numkong_add_gpu_unit(
        src/capabilities/hopper.cu "${numkong_hopper_unit_codes_}" numkong_cuda_capabilities_compiled_ CUDA::cudart
    )
    numkong_add_gpu_unit(
        src/capabilities/blackwell.cu "${numkong_blackwell_unit_codes_}" numkong_cuda_capabilities_compiled_
        CUDA::cudart
    )
    numkong_add_gpu_unit(
        src/capabilities/blackwellrtx.cu "${numkong_blackwellrtx_unit_codes_}" numkong_cuda_capabilities_compiled_
        CUDA::cudart
    )
    numkong_add_gpu_unit(
        src/capabilities/blackwellultra.cu "${numkong_blackwellultra_unit_codes_}" numkong_cuda_capabilities_compiled_
        CUDA::cudart
    )
    foreach (library_ ${numkong_libraries_})
        target_compile_definitions(${library_} PRIVATE NUMKONG_ARCH_CUDA_=1)
        target_link_libraries(${library_} PRIVATE $<BUILD_INTERFACE:numkong_cuda_capabilities_compiled_>)
    endforeach ()
endif ()

# Both GPU vendors may share one library: their units are separate files, each vendor has its own capability group,
# and each numbers its own devices.
if (NUMKONG_ROCM_ARCH)
    find_package(hip REQUIRED GLOBAL)
    numkong_select_gpu_codes(
        numkong_cdna3_unit_codes_ "${numkong_rocm_resolved_architectures_}" "${numkong_target_cdna3_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_cdna4_unit_codes_ "${numkong_rocm_resolved_architectures_}" "${numkong_target_cdna4_architectures}"
    )
    numkong_select_gpu_codes(
        numkong_cdna5_unit_codes_ "${numkong_rocm_resolved_architectures_}" "${numkong_target_cdna5_architectures}"
    )
    numkong_add_gpu_unit(
        src/capabilities/rocm.hip "${numkong_rocm_resolved_architectures_}" numkong_rocm_capabilities_compiled_
        hip::host
    )
    numkong_add_gpu_unit(
        src/capabilities/cdna3.hip "${numkong_cdna3_unit_codes_}" numkong_rocm_capabilities_compiled_ hip::host
    )
    numkong_add_gpu_unit(
        src/capabilities/cdna4.hip "${numkong_cdna4_unit_codes_}" numkong_rocm_capabilities_compiled_ hip::host
    )
    numkong_add_gpu_unit(
        src/capabilities/cdna5.hip "${numkong_cdna5_unit_codes_}" numkong_rocm_capabilities_compiled_ hip::host
    )
    foreach (library_ ${numkong_libraries_})
        target_compile_definitions(${library_} PRIVATE NUMKONG_ARCH_ROCM_=1)
        target_link_libraries(${library_} PRIVATE $<BUILD_INTERFACE:numkong_rocm_capabilities_compiled_>)
    endforeach ()
    set_target_properties(${numkong_libraries_} PROPERTIES HIP_ARCHITECTURES "${numkong_rocm_resolved_architectures_}")
    # HIP-Clang cannot read the GCC link-time-optimized objects of the CPU sources, so the host linker links.
    set_target_properties(${numkong_libraries_} PROPERTIES LINKER_LANGUAGE C)
endif ()

if (NUMKONG_METAL_ARCH)
    # Metal compiles its kernels on the device, so the C units hold only the host side of each Apple capability.
    add_library(numkong_metal_ OBJECT src/capabilities/metal.c src/capabilities/apple9.c src/capabilities/apple10.c)
    target_include_directories(
        numkong_metal_ PRIVATE "${PROJECT_SOURCE_DIR}/c" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/cpp"
                               "${PROJECT_SOURCE_DIR}/src"
    )
    set_target_properties(numkong_metal_ PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(
        numkong_metal_ PRIVATE numkong_cpu_capabilities_disabled_ numkong_metal_capabilities_compiled_ numkong_warnings
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
    target_include_directories(
        ${target_} PRIVATE "${PROJECT_SOURCE_DIR}/c" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/cpp"
    )
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
    target_include_directories(
        ${target_} PRIVATE "${PROJECT_SOURCE_DIR}/c" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/cpp"
    )
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
            set(vendor_enabled_ ${NUMKONG_CUDA_ARCH})
        elseif (vendor_ STREQUAL "rocm")
            set(vendor_sources_ test/cross_rocm.hip test/each_rocm.hip test/cast_rocm.hip)
            set(vendor_enabled_ ${NUMKONG_ROCM_ARCH})
        else ()
            set(vendor_sources_ test/cross_metal.cpp test/each_metal.cpp test/cast_metal.cpp test/shapes_metal.cpp)
            set(vendor_enabled_ ${NUMKONG_METAL_ARCH})
        endif ()
        set(leaf_ numkong_test_${vendor_})
        add_library(${leaf_} OBJECT ${vendor_sources_})
        target_include_directories(
            ${leaf_} PRIVATE "${PROJECT_SOURCE_DIR}/c" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/cpp"
        )
        target_link_libraries(${leaf_} PRIVATE fmt::fmt)
        target_link_libraries(${leaf_} PRIVATE numkong_cpu_capabilities_disabled_ numkong_warnings)
        target_compile_features(${leaf_} PRIVATE cxx_std_20)
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
    if (NUMKONG_CUDA_ARCH)
        numkong_configure_cuda_suite(numkong_test_cuda "${numkong_cuda_resolved_architectures_}")
        target_link_libraries(numkong_test_cuda PRIVATE CUDA::cudart)
    endif ()
    if (NUMKONG_ROCM_ARCH)
        numkong_configure_rocm_test(numkong_test_rocm "${numkong_rocm_resolved_architectures_}")
        target_link_libraries(numkong_test_rocm PRIVATE hip::host)
    endif ()
    if (NUMKONG_METAL_ARCH)
        target_compile_options(numkong_test_metal PRIVATE ${numkong_embed_options_})
    endif ()
endif ()

if (NUMKONG_BUILD_BENCH)
    add_library(
        numkong_bench_cuda OBJECT bench/cross_cuda.cu bench/each_cuda.cu bench/cast_cuda.cu bench/reduce_cuda.cu
    )
    add_library(numkong_bench_rocm OBJECT bench/cross_rocm.hip bench/each_rocm.hip)
    foreach (leaf_ numkong_bench_cuda numkong_bench_rocm)
        target_include_directories(
            ${leaf_} PRIVATE "${PROJECT_SOURCE_DIR}/c" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/cpp"
        )
        target_link_libraries(${leaf_} PRIVATE fmt::fmt)
        target_link_libraries(${leaf_} PRIVATE numkong_cpu_capabilities_disabled_ numkong_warnings)
        target_compile_features(${leaf_} PRIVATE cxx_std_20)
    endforeach ()
    target_sources(numkong_bench PRIVATE $<TARGET_OBJECTS:numkong_bench_cuda> $<TARGET_OBJECTS:numkong_bench_rocm>)
endif ()

# CUDA rows in the same `numkong_bench`, from bench/*_cuda.cu, with optional cuBLAS, cuDNN and cuVS baselines.
if (NUMKONG_BUILD_BENCH AND NUMKONG_CUDA_ARCH)
    find_package(CUDAToolkit 13.0 REQUIRED)
    numkong_configure_cuda_suite(numkong_bench_cuda "${numkong_cuda_resolved_architectures_}")
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
        include(cmake/numkong_cudnn.cmake)
        target_link_libraries(numkong_bench numkong_cudnn)
        target_link_libraries(numkong_bench_cuda PRIVATE numkong_cudnn)
    endif ()
    if (NUMKONG_COMPARE_TO_CUVS)
        include(cmake/numkong_cuvs.cmake)
        target_link_libraries(numkong_bench numkong_cuvs)
        target_link_libraries(numkong_bench_cuda PRIVATE numkong_cuvs)
    endif ()
endif ()

# Without CUDA its rows compile away as C++, as every `cross_*` family does off its platform.
if (NUMKONG_BUILD_BENCH AND NOT NUMKONG_CUDA_ARCH)
    set_source_files_properties(
        bench/cross_cuda.cu bench/each_cuda.cu bench/cast_cuda.cu bench/reduce_cuda.cu PROPERTIES LANGUAGE CXX
    )
endif ()

if (NUMKONG_BUILD_BENCH AND NUMKONG_ROCM_ARCH)
    target_link_libraries(numkong_bench hip::host)
    target_link_libraries(numkong_bench_rocm PRIVATE hip::host)
    target_link_libraries(numkong_bench_rocm PRIVATE numkong_rocm_capabilities_compiled_)
    set_target_properties(
        numkong_bench_rocm PROPERTIES HIP_ARCHITECTURES "${numkong_rocm_resolved_architectures_}" HIP_STANDARD 20
                                      HIP_STANDARD_REQUIRED YES HIP_EXTENSIONS NO
    )
elseif (NUMKONG_BUILD_BENCH)
    set_source_files_properties(bench/cross_rocm.hip bench/each_rocm.hip PROPERTIES LANGUAGE CXX)
endif ()

# Metal rows in the same `numkong_bench`, from bench/cross_metal.cpp, over the queue's shared memory.
if (NUMKONG_BUILD_BENCH AND NUMKONG_METAL_ARCH)
    target_link_libraries(numkong_bench numkong_metal_capabilities_compiled_)
    target_link_libraries(numkong_bench "-framework Metal" "-framework Foundation")
endif ()

include(cmake/numkong_install.cmake)
