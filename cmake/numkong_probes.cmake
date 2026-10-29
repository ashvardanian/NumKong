# cmake/numkong_probes.cmake — which CPU and GPU capabilities the build compiles
#
# One `numkong_add_cpu_capability` row per CPU capability, run here. Each probes `probes/<capability>.c` into the
# cached `numkong_target_<capability>_compiles`, caches the flags enabling the capability across a unit as
# `numkong_target_<capability>_flags`, and defines its `NUMKONG_TARGET_<CAPABILITY>` macro on the targets below.
# `-D NUMKONG_TARGET_<CAPABILITY>=0` leaves a capability out, and `numkong_target_<capability>_enabled` caches the
# verdict after it. A monoversioned build also compiles `probes/implied.c` at the floor into the cached
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

# Linked by the units that call CPU capability kernels and pick them at runtime: each macro is 1 where the toolchain
# compiles the capability, else 0. The twin of `nk_cpu_capabilities_compiled`, published as
# `numkong::cpu_capabilities_compiled` for projects building NumKong in their own tree, like USearch, and never
# installed, as an installed header meets other toolchains.
add_library(numkong_cpu_capabilities_compiled INTERFACE)
add_library(numkong::cpu_capabilities_compiled ALIAS numkong_cpu_capabilities_compiled)
# Linked by the units held at the platform baseline: every CPU capability's macro is 0.
add_library(numkong_cpu_capabilities_disabled_ INTERFACE)
# Serial, then each capability row's name as its verdict enables it, for the configure summary.
set_property(GLOBAL PROPERTY numkong_capabilities_enabled serial)

# Probes one CPU capability, named `capability_name_` in its probe and cache variables, and defines
# `capability_macro_` from the verdict. `GCC_FLAGS`, or `MSVC_FLAGS` under MSVC, enable the capability across a whole
# unit. The probe compiles without them at the units' `-march`, as the library does, the capability's target pragmas
# scoping it per function; a capability this toolchain cannot scope that way stays off, and the serial kernels serve
# its calls.
function (numkong_add_cpu_capability capability_name_ capability_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "" "GCC_FLAGS;MSVC_FLAGS")
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
    set(capability_enabled_ ${numkong_target_${capability_name_}_compiles})
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_enabled_ FALSE)
    endif ()
    # The host's exit code is cached per build folder, and a probe killed by a missing instruction never exits with 0.
    set(capability_runs_ FALSE)
    if (capability_enabled_ AND MSVC AND NUMKONG_MARCH STREQUAL "native")
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
    elseif (capability_enabled_ AND NOT NUMKONG_MULTIVERSIONED)
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
        set(capability_enabled_ FALSE)
    endif ()
    set(numkong_target_${capability_name_}_enabled ${capability_enabled_}
        CACHE INTERNAL "Whether ${capability_macro_} compiles, is not switched off, and the build kind keeps it"
    )
    if (capability_enabled_)
        set_property(GLOBAL APPEND PROPERTY numkong_capabilities_enabled ${capability_name_})
    endif ()
    set(numkong_target_${capability_name_}_runs ${capability_runs_}
        CACHE INTERNAL "Whether ${capability_macro_} is enabled and this host runs it, for MSVC's native build"
    )
    target_compile_definitions(
        numkong_cpu_capabilities_compiled INTERFACE "${capability_macro_}=$<BOOL:${capability_enabled_}>"
    )
    target_compile_definitions(numkong_cpu_capabilities_disabled_ INTERFACE ${capability_macro_}=0)
endfunction ()

# Flags are the union of each capability's target pragmas. LASX and POWER9 scope per function from GCC 15 on, which
# opens `lasxintrin.h` and `altivec.h` inside their target regions; Clang never does, so its builds leave both off.
if (numkong_arch_x8664_)
    numkong_add_cpu_capability(
        haswell NUMKONG_TARGET_HASWELL GCC_FLAGS -mavx2 -mf16c -mfma -mbmi -mbmi2 -mpopcnt MSVC_FLAGS /arch:AVX2
    )
    numkong_add_cpu_capability(
        skylake NUMKONG_TARGET_SKYLAKE GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mf16c -mfma -mbmi
                                                 -mbmi2 MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        icelake NUMKONG_TARGET_ICELAKE
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512vbmi -mavx512vbmi2
                  -mavx512vpopcntdq -mf16c -mfma -mbmi -mbmi2 -mlzcnt -mpopcnt MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        genoa NUMKONG_TARGET_GENOA GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512bf16
                                             -mf16c -mfma -mbmi -mbmi2 MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        sapphire NUMKONG_TARGET_SAPPHIRE GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mf16c
                                                   -mfma -mbmi -mbmi2 MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        sapphireamx NUMKONG_TARGET_SAPPHIREAMX
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mavx512vnni -mavx512vbmi -mavx512bf16
                  -mf16c -mfma -mbmi -mbmi2 -mamx-tile -mamx-bf16 -mamx-int8 MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        graniteamx NUMKONG_TARGET_GRANITEAMX
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mavx512vbmi -mf16c -mfma -mbmi -mbmi2
                  -mamx-tile -mamx-bf16 -mamx-int8 -mamx-fp16 MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        diamond NUMKONG_TARGET_DIAMOND GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16
                                                 -mavx10.2 -mf16c -mfma -mbmi -mbmi2 MSVC_FLAGS /arch:AVX10.2
    )
    numkong_add_cpu_capability(
        diamondamx NUMKONG_TARGET_DIAMONDAMX
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mavx10.2 -mf16c -mfma -mbmi -mbmi2
                  -mamx-tile -mamx-bf16 -mamx-int8 -mamx-fp8 -mamx-avx512 MSVC_FLAGS /arch:AVX10.2
    )
    numkong_add_cpu_capability(
        turin NUMKONG_TARGET_TURIN
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512vbmi2 -mavx512bf16
                  -mavx512vp2intersect -mbmi -mbmi2 -mlzcnt -mpopcnt MSVC_FLAGS /arch:AVX512
    )
    numkong_add_cpu_capability(
        alder NUMKONG_TARGET_ALDER GCC_FLAGS -mavx2 -mf16c -mfma -mbmi -mbmi2 -mavxvnni MSVC_FLAGS /arch:AVX2
    )
    numkong_add_cpu_capability(
        sierra NUMKONG_TARGET_SIERRA GCC_FLAGS -mavx2 -mf16c -mfma -mbmi -mbmi2 -mavxvnni -mavxvnniint8
        MSVC_FLAGS /arch:AVX2
    )
elseif (numkong_arch_arm64_)
    numkong_add_cpu_capability(neon NUMKONG_TARGET_NEON GCC_FLAGS -march=armv8.2-a+simd)
    numkong_add_cpu_capability(
        neonhalf NUMKONG_TARGET_NEONHALF GCC_FLAGS -march=armv8.2-a+simd+fp16 MSVC_FLAGS /arch:armv8.2
    )
    numkong_add_cpu_capability(
        neonsdot NUMKONG_TARGET_NEONSDOT GCC_FLAGS -march=armv8.2-a+dotprod MSVC_FLAGS /arch:armv8.4
    )
    numkong_add_cpu_capability(
        neonbfdot NUMKONG_TARGET_NEONBFDOT GCC_FLAGS -march=armv8.6-a+simd+bf16 MSVC_FLAGS /arch:armv8.6
    )
    numkong_add_cpu_capability(
        neonfhm NUMKONG_TARGET_NEONFHM GCC_FLAGS -march=armv8.2-a+simd+fp16+fp16fml MSVC_FLAGS /arch:armv8.4
    )
    numkong_add_cpu_capability(sve NUMKONG_TARGET_SVE GCC_FLAGS -march=armv8.2-a+sve)
    numkong_add_cpu_capability(svehalf NUMKONG_TARGET_SVEHALF GCC_FLAGS -march=armv8.2-a+sve+fp16)
    numkong_add_cpu_capability(svebfdot NUMKONG_TARGET_SVEBFDOT GCC_FLAGS -march=armv8.2-a+sve+bf16)
    numkong_add_cpu_capability(svesdot NUMKONG_TARGET_SVESDOT GCC_FLAGS -march=armv8.2-a+sve+dotprod)
    numkong_add_cpu_capability(sve2 NUMKONG_TARGET_SVE2 GCC_FLAGS -march=armv8.2-a+sve+sve2)
    numkong_add_cpu_capability(neonfp8 NUMKONG_TARGET_NEONFP8 GCC_FLAGS -march=armv8-a+simd+fp8dot4)
    numkong_add_cpu_capability(sme NUMKONG_TARGET_SME GCC_FLAGS -march=armv8-a+sme)
    numkong_add_cpu_capability(smef64 NUMKONG_TARGET_SMEF64 GCC_FLAGS -march=armv8-a+sme+sme-f64f64)
    numkong_add_cpu_capability(smebi32 NUMKONG_TARGET_SMEBI32 GCC_FLAGS -march=armv8-a+sme2)
elseif (numkong_arch_riscv64_)
    numkong_add_cpu_capability(rvv NUMKONG_TARGET_RVV GCC_FLAGS -march=rv64gcv)
    numkong_add_cpu_capability(rvvhalf NUMKONG_TARGET_RVVHALF GCC_FLAGS -march=rv64gcv_zvfh)
    numkong_add_cpu_capability(rvvbf16 NUMKONG_TARGET_RVVBF16 GCC_FLAGS -march=rv64gcv_zvfbfwma)
    numkong_add_cpu_capability(rvvbb NUMKONG_TARGET_RVVBB GCC_FLAGS -march=rv64gcv_zvbb)
elseif (numkong_arch_loongarch64_)
    numkong_add_cpu_capability(loongsonasx NUMKONG_TARGET_LOONGSONASX GCC_FLAGS -mlasx)
elseif (numkong_arch_ppc64_)
    numkong_add_cpu_capability(powervsx NUMKONG_TARGET_POWERVSX GCC_FLAGS -mcpu=power9)
endif ()

# Linked by the units that call one GPU vendor's capability kernels: each macro is 1 where some architecture the
# build compiles for, without CMake's "-real" or "-virtual", runs the capability, else 0. Metal has no compiler macro
# of its own, so its target also switches `NUMKONG_ARCH_METAL_` on.
add_library(numkong_cuda_capabilities_compiled_ INTERFACE)
add_library(numkong_rocm_capabilities_compiled_ INTERFACE)
add_library(numkong_metal_capabilities_compiled_ INTERFACE)
target_compile_definitions(numkong_metal_capabilities_compiled_ INTERFACE NUMKONG_ARCH_METAL_=1)

# One GPU capability, named `capability_name_` in its cache variable: caches the `CUDA_ARCHITECTURES`,
# `ROCM_ARCHITECTURES` or `METAL_ARCHITECTURES` running it as `numkong_target_<capability>_architectures`, the twin of a
# CPU capability's `numkong_target_<capability>_flags`, so a project embedding NumKong, like USearch, reads the same lists,
# and defines `capability_macro_` on that vendor's target. `-D NUMKONG_TARGET_<CAPABILITY>=0` leaves it out, as on the
# CPU, and `numkong_target_<capability>_enabled` caches the verdict.
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
    set(capability_enabled_ FALSE)
    foreach (architecture_ IN LISTS compiled_architectures_)
        string(REGEX REPLACE "-(real|virtual)$" "" architecture_ "${architecture_}")
        if (architecture_ IN_LIST capability_architectures_)
            set(capability_enabled_ TRUE)
        endif ()
    endforeach ()
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_enabled_ FALSE)
    endif ()
    set(numkong_target_${capability_name_}_enabled ${capability_enabled_}
        CACHE INTERNAL "Whether some compiled architecture runs ${capability_macro_} and it is not switched off"
    )
    if (capability_enabled_)
        set_property(GLOBAL APPEND PROPERTY numkong_capabilities_enabled ${capability_name_})
    endif ()
    target_compile_definitions(${capabilities_target_} INTERFACE "${capability_macro_}=$<BOOL:${capability_enabled_}>")
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
