# WASM64/Emscripten Memory64 toolchain for NumKong: 64-bit addressing, the only way one module passes 4 GB.
# Usage: cmake --preset wasm64_emscripten, which builds into `build_wasm64_emscripten`
#
# Every engine with Memory64 also has Relaxed SIMD, so the capability defaults to `v128relaxed`; `-DNUMKONG_TARGET_ARCH=v128`
# narrows it. The kernels are thread-free, so no `-pthread` here. The module is named after the capability, so both can
# be built into one directory, one reconfigure apart.

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
set(NUMKONG_TARGET_ARCH "v128relaxed" CACHE STRING
                                            "WebAssembly SIMD capability of this module: serial, v128 or v128relaxed"
)

# Emscripten's own toolchain file reads the width off these flags as it loads, so they precede the include.
set(CMAKE_C_FLAGS "-sMEMORY64=1" CACHE STRING "Flags used by the C compiler during all build types.")
set(CMAKE_CXX_FLAGS "-sMEMORY64=1" CACHE STRING "Flags used by the CXX compiler during all build types.")
set(EMSCRIPTEN_SYSTEM_PROCESSOR wasm64)
include("$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")

# Optimization.
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -sASSERTIONS=2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -sASSERTIONS=2")

# Linker flags for Node.js and browser execution (larger memory limits for 64-bit).
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-sMEMORY64=1 \
     -sALLOW_MEMORY_GROWTH=1 \
     -sIMPORTED_MEMORY=1 \
     -sINITIAL_MEMORY=256MB \
     -sMAXIMUM_MEMORY=16GB \
     -sSTACK_SIZE=5MB \
     -sEXPORTED_FUNCTIONS=['_main'] \
     -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
)

# Report the Emscripten version.
execute_process(
    COMMAND ${CMAKE_C_COMPILER} --version OUTPUT_VARIABLE EMCC_VERSION_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" EMCC_VERSION "${EMCC_VERSION_OUTPUT}")
message(STATUS "NumKong WASM64: Emscripten ${EMCC_VERSION}")
message(STATUS "NumKong WASM64: SIMD capability ${NUMKONG_TARGET_ARCH}, Memory64 enabled")
