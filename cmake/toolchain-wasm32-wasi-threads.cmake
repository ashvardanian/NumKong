# WASI threads toolchain for NumKong: `wasm32-wasip1-threads` over an imported shared memory, for hosts that
# spawn workers (Wasmtime with `-S threads=y`, a Node host, an embedding that links the pool).
# Usage: cmake -B build-wasi-threads -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi-threads.cmake -DNUMKONG_BUILD_TEST=ON
#
# Every runtime with WASI threads also has Relaxed SIMD, so the tier defaults to `v128relaxed`; `-DNUMKONG_WASM_SIMD=v128`
# narrows it. The single-threaded module lives in `toolchain-wasm32-wasi.cmake`.

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
set(NUMKONG_WASM_SIMD "v128relaxed" CACHE STRING "WebAssembly SIMD tier of this module: v128 or v128relaxed")
if (NUMKONG_WASM_SIMD STREQUAL "v128relaxed")
    set(WASM_SIMD_FLAGS "-msimd128 -mrelaxed-simd")
elseif (NUMKONG_WASM_SIMD STREQUAL "v128")
    set(WASM_SIMD_FLAGS "-msimd128")
else ()
    message(FATAL_ERROR "NUMKONG_WASM_SIMD must be v128 or v128relaxed, not `${NUMKONG_WASM_SIMD}`")
endif ()

set(CMAKE_C_FLAGS_INIT "${WASM_SIMD_FLAGS} --target=wasm32-wasip1-threads -pthread")
set(CMAKE_CXX_FLAGS_INIT "${WASM_SIMD_FLAGS} --target=wasm32-wasip1-threads -pthread -fno-exceptions")

# Optimization flags.
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Linker flags for WASI threads: the host owns the shared memory and hands it in as an import.
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-Wl,--import-memory -Wl,--export-memory -Wl,--shared-memory -Wl,--max-memory=2147483648"
)

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

message(STATUS "NumKong WASI: SIMD tier ${NUMKONG_WASM_SIMD}, wasi-threads over shared memory")
message(STATUS "NumKong WASI: Toolchain at ${WASI_SDK_PATH}")

# Runtime that CTest invokes on each cross binary. Only hosts that spawn workers over the imported shared memory can
# run this module: wasmtime with its threads proposal and WASI threads on, or wasmer with threads on.
set(NUMKONG_WASM_RUNTIME "wasmtime" CACHE STRING "WASM runtime CTest uses for WASI tests (wasmtime or wasmer)")
if (NUMKONG_WASM_RUNTIME STREQUAL "wasmer")
    find_program(NUMKONG_WASMER_EXE_ wasmer PATHS "$ENV{HOME}/.cargo/bin" "$ENV{HOME}/.wasmer/bin")
    set(CMAKE_CROSSCOMPILING_EMULATOR "${NUMKONG_WASMER_EXE_};run;--enable-simd;--enable-relaxed-simd;--enable-threads")
else ()
    find_program(NUMKONG_WASMTIME_EXE_ wasmtime PATHS "$ENV{HOME}/.wasmtime/bin")
    set(CMAKE_CROSSCOMPILING_EMULATOR "${NUMKONG_WASMTIME_EXE_};run;-W;relaxed-simd=y,threads=y;-S;threads=y")
endif ()
message(STATUS "NumKong WASI: CTest runtime = ${NUMKONG_WASM_RUNTIME}")
