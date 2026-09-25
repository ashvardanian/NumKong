# WASM/Emscripten toolchain for NumKong: 32-bit, single-threaded, the module any page and any runtime loads.
# Usage: cmake -B build-wasm32-emscripten -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-emscripten.cmake
#
# The SIMD tier is a whole-module choice, so it is fixed here rather than probed: `v128` by default, which also
# loads where Relaxed SIMD is absent - Safari and WebKit on iOS - or `-DNUMKONG_WASM_SIMD=v128relaxed`. The flags go into
# the cache once, so use one build directory per tier.

# Verify the Emscripten SDK.
if (NOT DEFINED ENV{EMSDK})
    message(
        FATAL_ERROR
            "EMSDK environment variable not set.\n"
            "Install Emscripten: https://emscripten.org/docs/getting_started/downloads.html\n"
            "Then run: source $EMSDK/emsdk_env.sh"
    )
endif ()

# SIMD tier: the flags define `__wasm_simd128__` / `__wasm_relaxed_simd__`, which `types.h` reads.
set(NUMKONG_WASM_SIMD "v128" CACHE STRING "WebAssembly SIMD tier of this module: v128 or v128relaxed")
if (NUMKONG_WASM_SIMD STREQUAL "v128relaxed")
    set(WASM_SIMD_FLAGS "-msimd128 -mrelaxed-simd")
elseif (NUMKONG_WASM_SIMD STREQUAL "v128")
    set(WASM_SIMD_FLAGS "-msimd128")
else ()
    message(FATAL_ERROR "NUMKONG_WASM_SIMD must be v128 or v128relaxed, not `${NUMKONG_WASM_SIMD}`")
endif ()

# Emscripten's own toolchain file reads the width off these flags as it loads, so they precede the include.
set(CMAKE_C_FLAGS "${WASM_SIMD_FLAGS}" CACHE STRING "Flags used by the C compiler during all build types.")
set(CMAKE_CXX_FLAGS "${WASM_SIMD_FLAGS}" CACHE STRING "Flags used by the CXX compiler during all build types.")
set(EMSCRIPTEN_SYSTEM_PROCESSOR wasm32)
include("$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")

# Optimization.
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -sASSERTIONS=2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -sASSERTIONS=2")

# Linker flags for Node.js and browser execution.
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-sALLOW_MEMORY_GROWTH=1 \
     -sIMPORTED_MEMORY=1 \
     -sINITIAL_MEMORY=64MB \
     -sMAXIMUM_MEMORY=2GB \
     -sSTACK_SIZE=5MB \
     -sEXPORTED_FUNCTIONS=['_main'] \
     -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
)

# Report the Emscripten version.
execute_process(
    COMMAND ${CMAKE_C_COMPILER} --version OUTPUT_VARIABLE EMCC_VERSION_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" EMCC_VERSION "${EMCC_VERSION_OUTPUT}")
message(STATUS "NumKong WASM32: Emscripten ${EMCC_VERSION}")
message(STATUS "NumKong WASM32: SIMD tier ${NUMKONG_WASM_SIMD}")
