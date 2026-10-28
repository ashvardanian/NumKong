# Architecture and OS facts, and the ABI floor every C and C++ unit compiles at.

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

# Portable baseline policy (commits 43e8324d, 453ed5ff, 23f51955): each translation unit gets a per-arch
# ABI floor, auto-vectorization is forbidden so serial fallbacks stay scalar, and SIMD kernels opt in via
# per-function __attribute__((target(...))) or #pragma GCC target. NUMKONG_TARGET_ARCH opts out for a build tuned to
# one CPU: `native` for this host, or a CPU name like `graniterapids`, also when cross-compiling. Only C and C++ units
# take the flags, as nvcc would read them as its own.
if (NUMKONG_TARGET_ARCH STREQUAL "native" AND CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "NUMKONG_TARGET_ARCH=native needs the host as the target; name a CPU instead")
elseif (NUMKONG_TARGET_ARCH AND NOT numkong_arch_wasm_)
    message(
        WARNING "NUMKONG_TARGET_ARCH=${NUMKONG_TARGET_ARCH}: baking specific instructions; result may not run elsewhere"
    )
    if (MSVC AND NUMKONG_TARGET_ARCH STREQUAL "native")
        # MSVC has no `/arch:native`, so it takes its flags from the probes below.
    elseif (MSVC)
        message(
            FATAL_ERROR
                "MSVC takes NUMKONG_TARGET_ARCH=native only; pass an /arch: value such as AVX2, AVX512 or AVX10.2 "
                "through CMAKE_C_FLAGS instead of `${NUMKONG_TARGET_ARCH}`"
        )
    elseif (numkong_arch_arm64_)
        # AArch64 names CPUs through `-mcpu`, like `neoverse-v2`, as `-march` takes only architecture versions; Apple
        # Clang's `-march=native` also advertises just a subset of host features, without SME, SME2 or FP16_FML.
        set(numkong_unit_arch_flags_ -mcpu=${NUMKONG_TARGET_ARCH})
    else ()
        set(numkong_unit_arch_flags_ -march=${NUMKONG_TARGET_ARCH})
    endif ()
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:${numkong_unit_arch_flags_}>")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|IntelLLVM)$")
    # Clang and IntelLLVM accept the GCC spellings as aliases, so one pair covers all three compilers.
    add_compile_options(-fno-tree-vectorize -fno-tree-slp-vectorize)
    if (APPLE)
        # `-arch` already pins the ABI floor per slice, and a universal build passes both
        # to one clang invocation, where a per-arch `-march=` conflicts with the other slice.
        # `CMAKE_SYSTEM_PROCESSOR` names the host, so it cannot pick the flag here.
    elseif (numkong_arch_arm64_)
        set(numkong_unit_arch_flags_ -march=armv8-a)
    elseif (numkong_arch_x8664_)
        set(numkong_unit_arch_flags_ -march=x86-64)
    elseif (numkong_arch_riscv64_)
        set(numkong_unit_arch_flags_ -march=rv64gc)
    elseif (numkong_arch_ppc64_)
        set(numkong_unit_arch_flags_ -mcpu=power8)
    elseif (numkong_arch_loongarch64_)
        set(numkong_unit_arch_flags_ -march=loongarch64)
    elseif (numkong_arch_wasm_)
        # An engine validates a module whole, so every unit carries the module's one SIMD capability,
        # whatever `CMAKE_C_FLAGS` a binding passes; `types.h` reads it back from the flags. Empty is the portable floor.
        set(numkong_wasm_simd_ ${NUMKONG_TARGET_ARCH})
        if (NOT numkong_wasm_simd_)
            set(numkong_wasm_simd_ serial)
        endif ()
        if (numkong_wasm_simd_ STREQUAL "v128relaxed")
            add_compile_options(-msimd128 -mrelaxed-simd)
        elseif (numkong_wasm_simd_ STREQUAL "v128")
            add_compile_options(-msimd128)
        elseif (NOT numkong_wasm_simd_ STREQUAL "serial")
            message(
                FATAL_ERROR
                    "NUMKONG_TARGET_ARCH on WebAssembly: serial, v128 or v128relaxed, not `${NUMKONG_TARGET_ARCH}`"
            )
        endif ()
    endif ()
    add_compile_options(${numkong_unit_arch_flags_})
elseif (MSVC)
    # MSVC: no per-function target pragma; these match defaults. NVCC would read them as input files.
    if (numkong_arch_x8664_)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/arch:SSE2>)
    elseif (numkong_arch_arm64_)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/arch:armv8.0>)
    endif ()
endif ()
