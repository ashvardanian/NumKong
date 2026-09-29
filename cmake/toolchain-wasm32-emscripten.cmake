# WASM/Emscripten toolchain for NumKong: 32-bit, single-threaded, the module any page and any runtime loads.
# Usage: cmake --preset wasm32_emscripten, which builds into `build_wasm32_emscripten`
#
# The SIMD capability is a whole-module choice, so it is fixed here rather than probed: `v128` by default, which also
# loads where Relaxed SIMD is absent - Safari and WebKit on iOS - or `-DNUMKONG_TARGET_ARCH=v128relaxed`. The module is
# named after it, so both capabilities can be built into one directory, one reconfigure apart.

# Verify the Emscripten SDK.
if (NOT DEFINED ENV{EMSDK})
    message(
        FATAL_ERROR
            "EMSDK environment variable not set.\n"
            "Install Emscripten: https://emscripten.org/docs/getting_started/downloads.html\n"
            "Then run: source $EMSDK/emsdk_env.sh"
    )
endif ()

# `CMakeLists.txt` turns the capability into `-msimd128` and `-mrelaxed-simd` for every unit.
set(NUMKONG_TARGET_ARCH "v128" CACHE STRING "WebAssembly SIMD capability of this module: serial, v128 or v128relaxed")

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
message(STATUS "NumKong WASM32: SIMD capability ${NUMKONG_TARGET_ARCH}")
