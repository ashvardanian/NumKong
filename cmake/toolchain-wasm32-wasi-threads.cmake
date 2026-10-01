# WASI threads toolchain for NumKong: `wasm32-wasip1-threads` over an imported shared memory, for hosts that
# spawn workers (Wasmtime with `-S threads=y`, a Node host, an embedding that links the pool).
# Usage: cmake --preset wasm32_wasi_threads, into `build_wasm32_wasi_threads`, or `build.rs` for `wasm32-wasip1-threads`
#
# Every runtime with WASI threads also has Relaxed SIMD, so the capability defaults to `v128relaxed`; `-DNUMKONG_TARGET_ARCH=v128`
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

# `CMakeLists.txt` turns the capability into `-msimd128` and `-mrelaxed-simd` for every unit.
set(NUMKONG_TARGET_ARCH "v128relaxed" CACHE STRING
                                            "WebAssembly SIMD capability of this module: serial, v128 or v128relaxed"
)

set(CMAKE_C_FLAGS_INIT "--target=wasm32-wasip1-threads -pthread")
set(CMAKE_CXX_FLAGS_INIT "--target=wasm32-wasip1-threads -pthread -fno-exceptions")

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

message(STATUS "NumKong WASI: SIMD capability ${NUMKONG_TARGET_ARCH}, wasi-threads over shared memory")
message(STATUS "NumKong WASI: Toolchain at ${WASI_SDK_PATH}")

# The engine CTest runs each test binary through. Only hosts that spawn workers over the imported shared memory can
# run this module: wasmtime with its threads proposal and WASI threads on, the default, or wasmer with threads on:
# `-DCMAKE_CROSSCOMPILING_EMULATOR="wasmer;run;--enable-simd;--enable-relaxed-simd;--enable-threads;--forward-host-env"`.
find_program(NUMKONG_WASMTIME_EXE_ wasmtime PATHS "$ENV{HOME}/.wasmtime/bin")
set(CMAKE_CROSSCOMPILING_EMULATOR "${NUMKONG_WASMTIME_EXE_};run;-W;relaxed-simd=y,threads=y;-S;threads=y,inherit-env=y"
    CACHE STRING "Runs the WASI tests"
)
message(STATUS "NumKong WASI: CTest runtime = ${CMAKE_CROSSCOMPILING_EMULATOR}")
