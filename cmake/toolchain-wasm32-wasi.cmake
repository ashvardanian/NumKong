# WASI toolchain for NumKong: single-threaded, self-contained memory, so the same `.wasm` runs under Wasmtime, Wasmer
# and Node alike (standalone runtimes without a host).
# Usage: cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi.cmake -DNUMKONG_BUILD_TEST=ON
#
# The SIMD tier is a whole-module choice, so it is fixed here rather than probed: `v128` by default, or
# `-DNUMKONG_WASM_SIMD=v128relaxed`. Threads live in `toolchain-wasm32-wasi-threads.cmake`.

set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR wasm32)

# Locate the WASI SDK.
if (NOT DEFINED WASI_SDK_PATH)
    if (DEFINED ENV{WASI_SDK_PATH})
        set(WASI_SDK_PATH "$ENV{WASI_SDK_PATH}")
    elseif (EXISTS "$ENV{HOME}/wasi-sdk")
        set(WASI_SDK_PATH "$ENV{HOME}/wasi-sdk")
    else ()
        message(
            FATAL_ERROR
                "WASI_SDK_PATH not set and ~/wasi-sdk not found.\n"
                "Download from: https://github.com/WebAssembly/wasi-sdk/releases\n"
                "Then set: export WASI_SDK_PATH=~/wasi-sdk"
        )
    endif ()
endif ()

file(TO_CMAKE_PATH "${WASI_SDK_PATH}" WASI_SDK_PATH)

# Nested try-compile projects reread this file and inherit the environment, but not this cache variable.
set(ENV{WASI_SDK_PATH} "${WASI_SDK_PATH}")

# Windows SDK archives ship executable suffixes that Unix hosts do not use.
if (CMAKE_HOST_WIN32)
    set(WASI_TOOL_SUFFIX ".exe")
else ()
    set(WASI_TOOL_SUFFIX "")
endif ()

# Set compilers.
set(CMAKE_C_COMPILER "${WASI_SDK_PATH}/bin/clang${WASI_TOOL_SUFFIX}")
set(CMAKE_CXX_COMPILER "${WASI_SDK_PATH}/bin/clang++${WASI_TOOL_SUFFIX}")
set(CMAKE_AR "${WASI_SDK_PATH}/bin/llvm-ar${WASI_TOOL_SUFFIX}")
set(CMAKE_RANLIB "${WASI_SDK_PATH}/bin/llvm-ranlib${WASI_TOOL_SUFFIX}")
set(CMAKE_SYSROOT "${WASI_SDK_PATH}/share/wasi-sysroot")
set(CMAKE_FIND_ROOT_PATH "${WASI_SDK_PATH}")

# A `-shared` module needs a main module to resolve `__global_base` and kin, which no standalone runtime supplies.
set(NUMKONG_BUILD_SHARED OFF CACHE BOOL "Compile a dynamic library")

# SIMD tier: the flags define `__wasm_simd128__` / `__wasm_relaxed_simd__`, which `types.h` reads.
set(NUMKONG_WASM_SIMD "v128" CACHE STRING "WebAssembly SIMD tier of this module: v128 or v128relaxed")
if (NUMKONG_WASM_SIMD STREQUAL "v128relaxed")
    set(WASM_SIMD_FLAGS "-msimd128 -mrelaxed-simd")
elseif (NUMKONG_WASM_SIMD STREQUAL "v128")
    set(WASM_SIMD_FLAGS "-msimd128")
else ()
    message(FATAL_ERROR "NUMKONG_WASM_SIMD must be v128 or v128relaxed, not `${NUMKONG_WASM_SIMD}`")
endif ()

set(CMAKE_C_FLAGS_INIT "${WASM_SIMD_FLAGS} --target=wasm32-wasip1")
set(CMAKE_CXX_FLAGS_INIT "${WASM_SIMD_FLAGS} --target=wasm32-wasip1 -fno-exceptions")

# Optimization flags.
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Do not look for programs in build-host directories.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Verify the WASI SDK version.
if (EXISTS "${WASI_SDK_PATH}/VERSION")
    file(READ "${WASI_SDK_PATH}/VERSION" WASI_SDK_VERSION)
    string(STRIP "${WASI_SDK_VERSION}" WASI_SDK_VERSION)
    message(STATUS "NumKong WASI: WASI-SDK ${WASI_SDK_VERSION}")
else ()
    message(STATUS "NumKong WASI: WASI-SDK version unknown")
endif ()

message(STATUS "NumKong WASI: SIMD tier ${NUMKONG_WASM_SIMD}, single-threaded")
message(STATUS "NumKong WASI: Toolchain at ${WASI_SDK_PATH}")

# Runtime that CTest invokes on each cross binary, so `ctest` runs the WASI tests cross-engine without a
# wrapper script. wasmtime and wasmer execute a `.wasm` command directly; node uses the small WASI runner
# in test/wasi.mjs. Relaxed-SIMD lowers differently per engine, so testing more than one matters.
set(NUMKONG_WASM_RUNTIME "wasmtime" CACHE STRING "WASM runtime CTest uses for WASI tests (wasmtime, wasmer, or node)")
if (NUMKONG_WASM_RUNTIME STREQUAL "wasmer")
    find_program(NUMKONG_WASMER_EXE_ wasmer PATHS "$ENV{HOME}/.cargo/bin" "$ENV{HOME}/.wasmer/bin")
    set(CMAKE_CROSSCOMPILING_EMULATOR "${NUMKONG_WASMER_EXE_};run;--enable-simd;--enable-relaxed-simd")
elseif (NUMKONG_WASM_RUNTIME STREQUAL "node")
    find_program(NUMKONG_NODE_EXE_ node)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${NUMKONG_NODE_EXE_};${CMAKE_CURRENT_LIST_DIR}/../test/wasi.mjs")
else ()
    find_program(NUMKONG_WASMTIME_EXE_ wasmtime PATHS "$ENV{HOME}/.wasmtime/bin")
    set(CMAKE_CROSSCOMPILING_EMULATOR "${NUMKONG_WASMTIME_EXE_};run;-W;relaxed-simd=y")
endif ()
message(STATUS "NumKong WASI: CTest runtime = ${NUMKONG_WASM_RUNTIME}")
