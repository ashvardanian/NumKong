# cmake/numkong_probes.cmake — which CPU and GPU capabilities the build compiles
#
# One `numkong_add_cpu_capability` row per CPU capability, run here. Each probes `probes/<capability>.c` into the
# cached `numkong_target_<capability>_compiles`, caches the flags enabling the capability across a unit as
# `numkong_target_<capability>_flags`, and defines two macros on the targets below:
#
#   compiles  -> NUMKONG_ARCH_<ARCH>_<CAPABILITY>_  the toolchain builds the capability's code at the units' floor
#   compiled  -> NUMKONG_TARGET_<CAPABILITY>        it compiles, nobody set the target to 0, and the build kind keeps it
#
# A probe raises only its own ARCH flag, as a unit at the floor sees it from the verdict target.
# `-D NUMKONG_TARGET_<CAPABILITY>=0` leaves a capability's kernels out, and `numkong_target_<capability>_compiled`
# caches the verdict after it. A monoversioned build also compiles `probes/implied.c` at the floor into the cached
# `numkong_target_<capability>_implied`, and keeps only the capabilities the floor's predefined macros imply. MSVC
# enables capabilities by `_MSC_VER` instead, so its native build runs each enabled probe, caching whether this host
# runs the capability as `numkong_target_<capability>_runs`, and a monoversioned one keeps those.
#
# A compiled capability is not one the unit's own flags enable, nor one the CPU runs. `numkong_header` carries no
# macro, so its consumers enable what their flags name, and the runtime mask picks among the compiled ones.
#
# One `numkong_add_gpu_capability` row per GPU capability, run by `cmake/numkong_gpu.cmake` once the GPU architectures
# resolve.

include_guard(GLOBAL)

# Linked by the units that call CPU capability kernels and pick them at runtime: each ARCH flag is 1 where the
# toolchain compiles the capability, and each target where the build also keeps its kernels, else 0. The twin of
# `nk_capabilities_compiled_cpu`, published as `numkong::cpu_capabilities_compiled` for projects building NumKong in
# their own tree, like USearch, and never installed, as an installed header meets other toolchains.
add_library(numkong_cpu_capabilities_compiled INTERFACE)
add_library(numkong::cpu_capabilities_compiled ALIAS numkong_cpu_capabilities_compiled)
# Linked by the units held at the platform baseline and the GPU units: every CPU capability's ARCH flag is 0.
add_library(numkong_cpu_capabilities_disabled_ INTERFACE)
# Serial, then each capability row's name as its verdict enables it, for the configure summary.
set_property(GLOBAL PROPERTY numkong_capabilities_compiled serial)

# Probes one CPU capability, named `capability_name_` in its probe and cache variables, and defines
# `capability_macro_` from the verdict. `GCC_FLAGS`, or `MSVC_FLAGS` under MSVC, enable the capability across a whole
# unit. The probe compiles without them at the units' `-march`, as the library does, the capability's target pragmas
# scoping it per function; a capability this toolchain cannot scope that way stays off, and the serial kernels serve
# its calls.
function (numkong_add_cpu_capability capability_name_ capability_macro_ architecture_macro_)
    cmake_parse_arguments(PARSE_ARGV 3 arg "" "" "GCC_FLAGS;MSVC_FLAGS")
    if (MSVC)
        set(unit_flags_ ${arg_MSVC_FLAGS})
    else ()
        set(unit_flags_ ${arg_GCC_FLAGS})
    endif ()
    set(numkong_target_${capability_name_}_flags "${unit_flags_}"
        CACHE INTERNAL "Flags enabling ${capability_macro_} across a unit"
    )
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
    # Library definitions do not reach try_compile; the Windows SDK needs its ARM64 selector here too.
    if (MSVC AND numkong_arch_arm64_)
        set(probe_definitions_ -D_ARM64_=1)
    else ()
        set(probe_definitions_)
    endif ()
    if (NOT DEFINED numkong_target_${capability_name_}_compiles)
        try_compile(
            numkong_target_${capability_name_}_compiles ${CMAKE_BINARY_DIR}/numkong_probes
            ${PROJECT_SOURCE_DIR}/probes/${capability_name_}.c
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/c;${PROJECT_SOURCE_DIR}/include" C_STANDARD 99
            COMPILE_DEFINITIONS ${probe_definitions_} ${numkong_unit_arch_flags_}
        )
    endif ()
    message(
        STATUS "Performing ISA probe ${capability_macro_} - compiles: ${numkong_target_${capability_name_}_compiles}"
    )
    set(capability_compiled_ ${numkong_target_${capability_name_}_compiles})
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_compiled_ FALSE)
    endif ()
    # The host's exit code is cached per build folder, and a probe killed by a missing instruction never exits with 0.
    set(capability_runs_ FALSE)
    if (capability_compiled_ AND MSVC AND NUMKONG_MARCH STREQUAL "native")
        if (NOT DEFINED numkong_target_${capability_name_}_exit_code)
            try_run(numkong_target_${capability_name_}_exit_code numkong_target_${capability_name_}_built
                    ${CMAKE_BINARY_DIR}/numkong_probes ${PROJECT_SOURCE_DIR}/probes/${capability_name_}.c
                    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/c;${PROJECT_SOURCE_DIR}/include" C_STANDARD
                                99 COMPILE_DEFINITIONS ${probe_definitions_}
            )
        endif ()
        if (numkong_target_${capability_name_}_exit_code STREQUAL "0")
            set(capability_runs_ TRUE)
        endif ()
        set(capability_implied_ ${capability_runs_})
        message(STATUS "Performing ISA probe ${capability_macro_} - runs: ${capability_runs_}")
    elseif (capability_compiled_ AND NOT NUMKONG_MULTIVERSIONED)
        # A verdict holds for the floor it was probed at, so a build tree whose floor changed probes again.
        if (NOT "${numkong_unit_arch_flags_}" STREQUAL "${numkong_target_${capability_name_}_implied_floor}")
            unset(numkong_target_${capability_name_}_implied CACHE)
        endif ()
        if (NOT DEFINED numkong_target_${capability_name_}_implied)
            # The probe holds no `main`, as it only asks the preprocessor.
            set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
            try_compile(
                numkong_target_${capability_name_}_implied ${CMAKE_BINARY_DIR}/numkong_probes
                ${PROJECT_SOURCE_DIR}/probes/implied.c
                CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/c;${PROJECT_SOURCE_DIR}/include" C_STANDARD 99
                COMPILE_DEFINITIONS ${probe_definitions_} ${numkong_unit_arch_flags_}
                                    -DNUMKONG_PROBE_MACRO=${capability_macro_}
            )
        endif ()
        set(numkong_target_${capability_name_}_implied_floor "${numkong_unit_arch_flags_}"
            CACHE INTERNAL "The floor ${capability_macro_} was last probed at"
        )
        set(capability_implied_ ${numkong_target_${capability_name_}_implied})
        message(STATUS "Performing ISA probe ${capability_macro_} - implied: ${capability_implied_}")
    endif ()
    # `-D NUMKONG_TARGET_<CAP>=1` vouches for what no compiler macro reports, like WFET or MSVC's AVX-512 subsets.
    if (NOT NUMKONG_MULTIVERSIONED AND NOT capability_implied_ AND NOT ${capability_macro_})
        set(capability_compiled_ FALSE)
    endif ()
    set(numkong_target_${capability_name_}_compiled ${capability_compiled_}
        CACHE INTERNAL "Whether ${capability_macro_} compiles, is not switched off, and the build kind keeps it"
    )
    if (capability_compiled_)
        set_property(GLOBAL APPEND PROPERTY numkong_capabilities_compiled ${capability_name_})
    endif ()
    set(numkong_target_${capability_name_}_runs ${capability_runs_}
        CACHE INTERNAL "Whether ${capability_macro_} is enabled and this host runs it, for MSVC's native build"
    )
    target_compile_definitions(
        numkong_cpu_capabilities_compiled
        INTERFACE "${architecture_macro_}=$<BOOL:${numkong_target_${capability_name_}_compiles}>"
                  "${capability_macro_}=$<BOOL:${capability_compiled_}>"
    )
    # A target of 0 no longer hides building blocks, so only a zero flag keeps intrinsics headers from device passes.
    target_compile_definitions(numkong_cpu_capabilities_disabled_ INTERFACE ${architecture_macro_}=0)
endfunction ()

# Flags are the union of each capability's target pragmas. LASX, POWER9 and POWER10 scope per function from GCC 15 on, which
# opens `lasxintrin.h` and `altivec.h` inside their target regions; Clang never does, so its builds leave them off.
if (numkong_arch_x8664_)
    numkong_add_cpu_capability(
        haswell NUMKONG_TARGET_HASWELL NUMKONG_ARCH_X8664_HASWELL_ GCC_FLAGS -mavx2 -mf16c -mfma -mbmi -mbmi2 -mpopcnt
        MSVC_FLAGS /arch:AVX2
    )
    numkong_add_cpu_capability(
        skylake NUMKONG_TARGET_SKYLAKE NUMKONG_ARCH_X8664_SKYLAKE_ GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw
                                                                             -mavx512dq -mf16c -mfma -mbmi -mbmi2
        MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        icelake NUMKONG_TARGET_ICELAKE NUMKONG_ARCH_X8664_ICELAKE_
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512vbmi -mavx512vbmi2
                  -mavx512vpopcntdq -mf16c -mfma -mbmi -mbmi2 -mlzcnt -mpopcnt MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        genoa NUMKONG_TARGET_GENOA NUMKONG_ARCH_X8664_GENOA_
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512bf16 -mf16c -mfma -mbmi -mbmi2
        MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        sapphire NUMKONG_TARGET_SAPPHIRE NUMKONG_ARCH_X8664_SAPPHIRE_
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mf16c -mfma -mbmi -mbmi2
        MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        sapphireamx NUMKONG_TARGET_SAPPHIREAMX NUMKONG_ARCH_X8664_SAPPHIREAMX_
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mavx512vnni -mavx512vbmi -mavx512bf16
                  -mf16c -mfma -mbmi -mbmi2 -mamx-tile -mamx-bf16 -mamx-int8 MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        graniteamx NUMKONG_TARGET_GRANITEAMX NUMKONG_ARCH_X8664_GRANITEAMX_
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mavx512vbmi -mf16c -mfma -mbmi -mbmi2
                  -mamx-tile -mamx-bf16 -mamx-int8 -mamx-fp16 MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        diamond NUMKONG_TARGET_DIAMOND NUMKONG_ARCH_X8664_DIAMOND_
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mavx10.2 -mf16c -mfma -mbmi -mbmi2
        MSVC_FLAGS /arch:AVX10.2
    )
    numkong_add_cpu_capability(
        diamondamx NUMKONG_TARGET_DIAMONDAMX NUMKONG_ARCH_X8664_DIAMONDAMX_
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mavx10.2 -mf16c -mfma -mbmi -mbmi2
                  -mamx-tile -mamx-bf16 -mamx-int8 -mamx-fp8 -mamx-avx512 MSVC_FLAGS /arch:AVX10.2
    )
    numkong_add_cpu_capability(
        turin NUMKONG_TARGET_TURIN NUMKONG_ARCH_X8664_TURIN_
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512vbmi2 -mavx512bf16
                  -mavx512vp2intersect -mbmi -mbmi2 -mlzcnt -mpopcnt MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        alder NUMKONG_TARGET_ALDER NUMKONG_ARCH_X8664_ALDER_ GCC_FLAGS -mavx2 -mf16c -mfma -mbmi -mbmi2 -mavxvnni
        MSVC_FLAGS /arch:AVX2
    )
    numkong_add_cpu_capability(
        sierra NUMKONG_TARGET_SIERRA NUMKONG_ARCH_X8664_SIERRA_ GCC_FLAGS -mavx2 -mf16c -mfma -mbmi -mbmi2 -mavxvnni
                                                                          -mavxvnniint8 MSVC_FLAGS /arch:AVX2
    )
elseif (numkong_arch_arm64_)
    numkong_add_cpu_capability(neon NUMKONG_TARGET_NEON NUMKONG_ARCH_ARM64_NEON_ GCC_FLAGS -march=armv8-a+simd)
    numkong_add_cpu_capability(
        neonhalf NUMKONG_TARGET_NEONHALF NUMKONG_ARCH_ARM64_NEONHALF_ GCC_FLAGS -march=armv8.2-a+simd+fp16
        MSVC_FLAGS /arch:armv8.2
    )
    numkong_add_cpu_capability(
        neonsdot NUMKONG_TARGET_NEONSDOT NUMKONG_ARCH_ARM64_NEONSDOT_ GCC_FLAGS -march=armv8.2-a+dotprod
        MSVC_FLAGS /arch:armv8.4
    )
    numkong_add_cpu_capability(
        neonbfdot NUMKONG_TARGET_NEONBFDOT NUMKONG_ARCH_ARM64_NEONBFDOT_ GCC_FLAGS -march=armv8.2-a+simd+bf16
        MSVC_FLAGS /arch:armv8.6
    )
    numkong_add_cpu_capability(
        neonfhm NUMKONG_TARGET_NEONFHM NUMKONG_ARCH_ARM64_NEONFHM_ GCC_FLAGS -march=armv8.2-a+simd+fp16+fp16fml
        MSVC_FLAGS /arch:armv8.4
    )
    numkong_add_cpu_capability(sve NUMKONG_TARGET_SVE NUMKONG_ARCH_ARM64_SVE_ GCC_FLAGS -march=armv8.2-a+sve)
    numkong_add_cpu_capability(
        svehalf NUMKONG_TARGET_SVEHALF NUMKONG_ARCH_ARM64_SVEHALF_ GCC_FLAGS -march=armv8.2-a+sve+fp16
    )
    numkong_add_cpu_capability(
        svebfdot NUMKONG_TARGET_SVEBFDOT NUMKONG_ARCH_ARM64_SVEBFDOT_ GCC_FLAGS -march=armv8.2-a+sve+bf16
    )
    numkong_add_cpu_capability(
        svesdot NUMKONG_TARGET_SVESDOT NUMKONG_ARCH_ARM64_SVESDOT_ GCC_FLAGS -march=armv8.2-a+sve+dotprod
    )
    numkong_add_cpu_capability(sve2 NUMKONG_TARGET_SVE2 NUMKONG_ARCH_ARM64_SVE2_ GCC_FLAGS -march=armv8.2-a+sve+sve2)
    numkong_add_cpu_capability(
        neonfp8 NUMKONG_TARGET_NEONFP8 NUMKONG_ARCH_ARM64_NEONFP8_ GCC_FLAGS -march=armv8-a+simd+fp8dot4
    )
    numkong_add_cpu_capability(sme NUMKONG_TARGET_SME NUMKONG_ARCH_ARM64_SME_ GCC_FLAGS -march=armv8-a+sme)
    numkong_add_cpu_capability(
        smef64 NUMKONG_TARGET_SMEF64 NUMKONG_ARCH_ARM64_SMEF64_ GCC_FLAGS -march=armv8-a+sme+sme-f64f64
    )
    numkong_add_cpu_capability(smebi32 NUMKONG_TARGET_SMEBI32 NUMKONG_ARCH_ARM64_SMEBI32_ GCC_FLAGS -march=armv8-a+sme2)
elseif (numkong_arch_riscv64_)
    numkong_add_cpu_capability(rvv NUMKONG_TARGET_RVV NUMKONG_ARCH_RISCV64_RVV_ GCC_FLAGS -march=rv64gcv)
    numkong_add_cpu_capability(
        rvvhalf NUMKONG_TARGET_RVVHALF NUMKONG_ARCH_RISCV64_RVVHALF_ GCC_FLAGS -march=rv64gcv_zvfh
    )
    numkong_add_cpu_capability(
        rvvbf16 NUMKONG_TARGET_RVVBF16 NUMKONG_ARCH_RISCV64_RVVBF16_ GCC_FLAGS -march=rv64gcv_zvfbfwma
    )
    numkong_add_cpu_capability(rvvbb NUMKONG_TARGET_RVVBB NUMKONG_ARCH_RISCV64_RVVBB_ GCC_FLAGS -march=rv64gcv_zvbb)
elseif (numkong_arch_loongarch64_)
    numkong_add_cpu_capability(
        loongsonasx NUMKONG_TARGET_LOONGSONASX NUMKONG_ARCH_LOONGARCH64_LOONGSONASX_ GCC_FLAGS -mlasx
    )
elseif (numkong_arch_ppc64_)
    numkong_add_cpu_capability(powervsx NUMKONG_TARGET_POWERVSX NUMKONG_ARCH_PPC64_POWERVSX_ GCC_FLAGS -mcpu=power9)
    numkong_add_cpu_capability(powermma NUMKONG_TARGET_POWERMMA NUMKONG_ARCH_PPC64_POWERMMA_ GCC_FLAGS -mcpu=power10)
endif ()

# Linked by the units that call one GPU vendor's capability kernels: each macro is 1 where some architecture the build
# compiles for, without CMake's "-real" or "-virtual" or a ROCm target ID's ":xnack-", runs the capability, else 0.
# Each target also switches its vendor flag on, as a host unit calling the vendor's kernels has no compiler macro saying
# the vendor's runtime is in the build, and Metal has none even in its device code.
add_library(numkong_cuda_capabilities_compiled_ INTERFACE)
add_library(numkong::cuda_capabilities_compiled ALIAS numkong_cuda_capabilities_compiled_)
target_compile_definitions(numkong_cuda_capabilities_compiled_ INTERFACE NUMKONG_ARCH_CUDA_=1)
add_library(numkong_rocm_capabilities_compiled_ INTERFACE)
add_library(numkong::rocm_capabilities_compiled ALIAS numkong_rocm_capabilities_compiled_)
target_compile_definitions(numkong_rocm_capabilities_compiled_ INTERFACE NUMKONG_ARCH_ROCM_=1)
add_library(numkong_metal_capabilities_compiled_ INTERFACE)
add_library(numkong::metal_capabilities_compiled ALIAS numkong_metal_capabilities_compiled_)
target_compile_definitions(numkong_metal_capabilities_compiled_ INTERFACE NUMKONG_ARCH_METAL_=1)

# One GPU capability, named `capability_name_` in its cache variable: caches the `CUDA_ARCHITECTURES`,
# `ROCM_ARCHITECTURES` or `METAL_ARCHITECTURES` running it as `numkong_target_<capability>_architectures`, the twin of a
# CPU capability's `numkong_target_<capability>_flags`, so a project embedding NumKong, like USearch, reads the same lists,
# and defines `capability_macro_` on that vendor's target. `-D NUMKONG_TARGET_<CAPABILITY>=0` leaves it out, as on the
# CPU, and `numkong_target_<capability>_compiled` caches the verdict.
function (numkong_add_gpu_capability capability_name_ capability_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "" "CUDA_ARCHITECTURES;ROCM_ARCHITECTURES;METAL_ARCHITECTURES")
    if (arg_CUDA_ARCHITECTURES)
        set(capability_architectures_ ${arg_CUDA_ARCHITECTURES})
        set(compiled_architectures_ ${numkong_cuda_resolved_architectures_})
        set(capabilities_target_ numkong_cuda_capabilities_compiled_)
    elseif (arg_ROCM_ARCHITECTURES)
        set(capability_architectures_ ${arg_ROCM_ARCHITECTURES})
        set(compiled_architectures_ ${numkong_rocm_resolved_architectures_})
        set(capabilities_target_ numkong_rocm_capabilities_compiled_)
    else ()
        set(capability_architectures_ ${arg_METAL_ARCHITECTURES})
        set(compiled_architectures_ ${numkong_metal_resolved_architectures_})
        set(capabilities_target_ numkong_metal_capabilities_compiled_)
    endif ()
    set(numkong_target_${capability_name_}_architectures "${capability_architectures_}"
        CACHE INTERNAL "Architectures running ${capability_macro_}"
    )
    set(capability_compiled_ FALSE)
    foreach (architecture_ IN LISTS compiled_architectures_)
        string(REGEX REPLACE "(-(real|virtual)|:.+)$" "" architecture_ "${architecture_}")
        if (architecture_ IN_LIST capability_architectures_)
            set(capability_compiled_ TRUE)
        endif ()
    endforeach ()
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_compiled_ FALSE)
    endif ()
    set(numkong_target_${capability_name_}_compiled ${capability_compiled_}
        CACHE INTERNAL "Whether some compiled architecture runs ${capability_macro_} and it is not switched off"
    )
    if (capability_compiled_)
        set_property(GLOBAL APPEND PROPERTY numkong_capabilities_compiled ${capability_name_})
    endif ()
    target_compile_definitions(${capabilities_target_} INTERFACE "${capability_macro_}=$<BOOL:${capability_compiled_}>")
endfunction ()

if (MSVC AND NUMKONG_MARCH STREQUAL "native")
    # MSVC has no `/arch:native`, so it takes the `/arch:` of the widest capability this host runs.
    if (numkong_target_diamond_runs)
        set(numkong_native_msvc_flags_ ${numkong_target_diamond_flags})
    elseif (numkong_target_skylake_runs)
        set(numkong_native_msvc_flags_ ${numkong_target_skylake_flags})
    elseif (numkong_target_haswell_runs)
        set(numkong_native_msvc_flags_ ${numkong_target_haswell_flags})
    elseif (numkong_target_neonbfdot_runs)
        set(numkong_native_msvc_flags_ ${numkong_target_neonbfdot_flags})
    elseif (numkong_target_neonsdot_runs)
        set(numkong_native_msvc_flags_ ${numkong_target_neonsdot_flags})
    elseif (numkong_target_neonhalf_runs)
        set(numkong_native_msvc_flags_ ${numkong_target_neonhalf_flags})
    endif ()
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:${numkong_native_msvc_flags_}>")
endif ()
