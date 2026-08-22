# WASI toolchain for NumKong (standalone WASM runtimes such as Wasmer and Wasmtime).
# Usage: cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasi.cmake -DNK_BUILD_TEST=ON

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

# Cross-runtime portable mode: build single-threaded with self-contained memory so the SAME `.wasm`
# runs under node, wasmtime, AND wasmer (used by the cross-runtime kernel validation). Default OFF keeps
# the threaded, host-imported-memory build that the full nk_test / nk_bench suites rely on.
if (NK_WASI_PORTABLE)
    set(NK_WASI_TARGET_ "wasm32-wasip1")
    set(NK_WASI_THREADS_ "")
    set(NK_WASI_MEMORY_ "")
else ()
    set(NK_WASI_TARGET_ "wasm32-wasip1-threads")
    set(NK_WASI_THREADS_ "-pthread")
    set(NK_WASI_MEMORY_ "-Wl,--import-memory -Wl,--export-memory -Wl,--shared-memory -Wl,--max-memory=2147483648")
endif ()

# WASM SIMD flags (same as Emscripten for consistency).
set(WASM_SIMD_FLAGS "-msimd128 -mrelaxed-simd")
set(CMAKE_C_FLAGS_INIT "${WASM_SIMD_FLAGS} --target=${NK_WASI_TARGET_} ${NK_WASI_THREADS_}")
set(CMAKE_CXX_FLAGS_INIT "${WASM_SIMD_FLAGS} --target=${NK_WASI_TARGET_} ${NK_WASI_THREADS_} -fno-exceptions")

# Optimization flags.
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Linker flags for WASI (the imported/shared-memory group is empty in NK_WASI_PORTABLE mode).
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-Wl,--allow-undefined \
     ${NK_WASI_MEMORY_} \
     -Wl,--export=main \
     -Wl,--export=_start"
)

# `wasm-ld` resolves `__global_base`, `__data_end`, and `__heap_base` only in a main module, so a `-shared`
# WASI build cannot link. Seeded rather than forced, so `-D NK_BUILD_SHARED=ON` still overrides.
set(NK_BUILD_SHARED OFF CACHE BOOL "Compile a dynamic library")

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

message(STATUS "NumKong WASI: SIMD128 and Relaxed SIMD enabled")
message(STATUS "NumKong WASI: Toolchain at ${WASI_SDK_PATH}")

# Runtime that CTest invokes on each cross binary, so `ctest` runs the WASI tests cross-engine without a
# wrapper script. wasmtime and wasmer execute a `.wasm` command directly; node uses the small WASI runner
# in test/test-wasi.mjs. Relaxed-SIMD lowers differently per engine, so testing more than one matters.
set(NK_WASM_RUNTIME "wasmtime" CACHE STRING "WASM runtime CTest uses for WASI tests (wasmtime, wasmer, or node)")
if (NK_WASM_RUNTIME STREQUAL "wasmer")
    find_program(NK_WASMER_EXE_ wasmer PATHS "$ENV{HOME}/.cargo/bin" "$ENV{HOME}/.wasmer/bin")
    set(CMAKE_CROSSCOMPILING_EMULATOR "${NK_WASMER_EXE_};run")
elseif (NK_WASM_RUNTIME STREQUAL "node")
    find_program(NK_NODE_EXE_ node)
    set(CMAKE_CROSSCOMPILING_EMULATOR
        "${NK_NODE_EXE_};--experimental-wasi-unstable-preview1;${CMAKE_CURRENT_LIST_DIR}/../test/test-wasi.mjs"
    )
else ()
    find_program(NK_WASMTIME_EXE_ wasmtime PATHS "$ENV{HOME}/.wasmtime/bin")
    set(CMAKE_CROSSCOMPILING_EMULATOR "${NK_WASMTIME_EXE_}")
endif ()
message(STATUS "NumKong WASI: CTest runtime = ${NK_WASM_RUNTIME}")
