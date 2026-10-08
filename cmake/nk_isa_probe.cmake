# cmake/nk_isa_probe.cmake — which CPU and GPU capabilities the build compiles
#
# One `nk_cpu_capability_` row per CPU capability, run here. Each probes `probes/<capability>.c` into the cached
# `nk_target_<capability>_compiles`, caches the flags enabling the capability across a unit as
# `nk_target_<capability>_flags`, and defines its `NUMKONG_TARGET_<CAPABILITY>` macro on the targets below.
# `-D NUMKONG_TARGET_<CAPABILITY>=0` leaves a capability out.
#
# A compiled capability is not one the unit's own flags enable, nor one the CPU runs. `numkong_header` carries no
# macro, so its consumers enable what their flags name, and the runtime mask picks among the compiled ones.
#
# One `nk_gpu_capability_` row per GPU capability, run by `CMakeLists.txt` once the GPU architectures resolve.

include_guard(GLOBAL)

# Linked by the units that call CPU capability kernels and pick them at runtime: each macro is 1 where the toolchain
# compiles the capability, else 0. The twin of `nk_cpu_capabilities_compiled`, published as
# `numkong::cpu_capabilities_compiled` for projects building NumKong in their own tree, like USearch, and never
# installed, as an installed header meets other toolchains.
add_library(numkong_cpu_capabilities_compiled INTERFACE)
add_library(numkong::cpu_capabilities_compiled ALIAS numkong_cpu_capabilities_compiled)
# Linked by the units held at the platform baseline: every CPU capability's macro is 0.
add_library(numkong_cpu_capabilities_disabled_ INTERFACE)

# Probes one CPU capability, named `capability_name_` in its probe and cache variables, and defines
# `capability_macro_` from the verdict. `GCC_FLAGS`, or `MSVC_FLAGS` under MSVC, enable the capability across a whole
# unit. The probe compiles without them, as the library does, the capability's target pragmas scoping it per
# function; a capability this toolchain cannot scope that way stays off, and the serial kernels serve its calls.
function (nk_cpu_capability_ capability_name_ capability_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 argument "" "" "GCC_FLAGS;MSVC_FLAGS")
    if (MSVC)
        set(unit_flags_ ${argument_MSVC_FLAGS})
    else ()
        set(unit_flags_ ${argument_GCC_FLAGS})
    endif ()
    set(nk_target_${capability_name_}_flags "${unit_flags_}" CACHE INTERNAL
                                                                   "Flags enabling ${capability_macro_} across a unit"
    )
    if (NOT DEFINED nk_target_${capability_name_}_compiles)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
        try_compile(
            nk_target_${capability_name_}_compiles ${CMAKE_BINARY_DIR}/nk_probes
            ${PROJECT_SOURCE_DIR}/probes/${capability_name_}.c
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/include" C_STANDARD 99
        )
    endif ()
    message(STATUS "Performing ISA probe ${capability_macro_} - compiles: ${nk_target_${capability_name_}_compiles}")
    set(capability_enabled_ ${nk_target_${capability_name_}_compiles})
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_enabled_ FALSE)
    endif ()
    target_compile_definitions(
        numkong_cpu_capabilities_compiled INTERFACE "${capability_macro_}=$<BOOL:${capability_enabled_}>"
    )
    target_compile_definitions(numkong_cpu_capabilities_disabled_ INTERFACE ${capability_macro_}=0)
endfunction ()

# Flags are the union of each capability's target pragmas. LASX and POWER9 scope per function from GCC 15 on, which
# opens `lasxintrin.h` and `altivec.h` inside their target regions; Clang never does, so its builds leave both off.
if (NUMKONG_ARCH_X8664_)
    nk_cpu_capability_(
        haswell NUMKONG_TARGET_HASWELL GCC_FLAGS -mavx2 -mf16c -mfma -mbmi -mbmi2 -mpopcnt MSVC_FLAGS /arch:AVX2
    )
    nk_cpu_capability_(
        skylake NUMKONG_TARGET_SKYLAKE GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mf16c -mfma -mbmi
                                                 -mbmi2 MSVC_FLAGS /arch:AVX512
    )
    nk_cpu_capability_(
        icelake NUMKONG_TARGET_ICELAKE
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512vbmi -mavx512vbmi2
                  -mavx512vpopcntdq -mf16c -mfma -mbmi -mbmi2 -mlzcnt -mpopcnt MSVC_FLAGS /arch:AVX512
    )
    nk_cpu_capability_(
        genoa NUMKONG_TARGET_GENOA GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512bf16
                                             -mf16c -mfma -mbmi -mbmi2 MSVC_FLAGS /arch:AVX512
    )
    nk_cpu_capability_(
        sapphire NUMKONG_TARGET_SAPPHIRE GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mf16c
                                                   -mfma -mbmi -mbmi2 MSVC_FLAGS /arch:AVX512
    )
    nk_cpu_capability_(
        sapphireamx NUMKONG_TARGET_SAPPHIREAMX
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mavx512vnni -mavx512vbmi -mavx512bf16
                  -mf16c -mfma -mbmi -mbmi2 -mamx-tile -mamx-bf16 -mamx-int8 MSVC_FLAGS /arch:AVX512
    )
    nk_cpu_capability_(
        graniteamx NUMKONG_TARGET_GRANITEAMX
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mavx512vbmi -mf16c -mfma -mbmi -mbmi2
                  -mamx-tile -mamx-bf16 -mamx-int8 -mamx-fp16 MSVC_FLAGS /arch:AVX512
    )
    nk_cpu_capability_(
        diamond NUMKONG_TARGET_DIAMOND GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16
                                                 -mavx10.2 -mf16c -mfma -mbmi -mbmi2 MSVC_FLAGS /arch:AVX10.2
    )
    nk_cpu_capability_(
        diamondamx NUMKONG_TARGET_DIAMONDAMX
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16 -mavx10.2 -mf16c -mfma -mbmi -mbmi2
                  -mamx-tile -mamx-bf16 -mamx-int8 -mamx-fp8 -mamx-avx512 MSVC_FLAGS /arch:AVX10.2
    )
    nk_cpu_capability_(
        turin NUMKONG_TARGET_TURIN
        GCC_FLAGS -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512vbmi2 -mavx512bf16
                  -mavx512vp2intersect -mbmi -mbmi2 -mlzcnt -mpopcnt MSVC_FLAGS /arch:AVX512
    )
    nk_cpu_capability_(
        alder NUMKONG_TARGET_ALDER GCC_FLAGS -mavx2 -mf16c -mfma -mbmi -mbmi2 -mavxvnni MSVC_FLAGS /arch:AVX2
    )
    nk_cpu_capability_(
        sierra NUMKONG_TARGET_SIERRA GCC_FLAGS -mavx2 -mf16c -mfma -mbmi -mbmi2 -mavxvnni -mavxvnniint8
        MSVC_FLAGS /arch:AVX2
    )
elseif (NUMKONG_ARCH_ARM64_)
    nk_cpu_capability_(neon NUMKONG_TARGET_NEON GCC_FLAGS -march=armv8.2-a+simd)
    nk_cpu_capability_(neonhalf NUMKONG_TARGET_NEONHALF GCC_FLAGS -march=armv8.2-a+simd+fp16 MSVC_FLAGS /arch:armv8.2)
    nk_cpu_capability_(neonsdot NUMKONG_TARGET_NEONSDOT GCC_FLAGS -march=armv8.2-a+dotprod MSVC_FLAGS /arch:armv8.4)
    nk_cpu_capability_(neonbfdot NUMKONG_TARGET_NEONBFDOT GCC_FLAGS -march=armv8.6-a+simd+bf16 MSVC_FLAGS /arch:armv8.6)
    nk_cpu_capability_(
        neonfhm NUMKONG_TARGET_NEONFHM GCC_FLAGS -march=armv8.2-a+simd+fp16+fp16fml MSVC_FLAGS /arch:armv8.4
    )
    nk_cpu_capability_(sve NUMKONG_TARGET_SVE GCC_FLAGS -march=armv8.2-a+sve)
    nk_cpu_capability_(svehalf NUMKONG_TARGET_SVEHALF GCC_FLAGS -march=armv8.2-a+sve+fp16)
    nk_cpu_capability_(svebfdot NUMKONG_TARGET_SVEBFDOT GCC_FLAGS -march=armv8.2-a+sve+bf16)
    nk_cpu_capability_(svesdot NUMKONG_TARGET_SVESDOT GCC_FLAGS -march=armv8.2-a+sve+dotprod)
    nk_cpu_capability_(sve2 NUMKONG_TARGET_SVE2 GCC_FLAGS -march=armv8.2-a+sve+sve2)
    nk_cpu_capability_(neonfp8 NUMKONG_TARGET_NEONFP8 GCC_FLAGS -march=armv8-a+simd+fp8dot4)
    nk_cpu_capability_(sme NUMKONG_TARGET_SME GCC_FLAGS -march=armv8-a+sme)
    nk_cpu_capability_(smef64 NUMKONG_TARGET_SMEF64 GCC_FLAGS -march=armv8-a+sme+sme-f64f64)
    nk_cpu_capability_(smebi32 NUMKONG_TARGET_SMEBI32 GCC_FLAGS -march=armv8-a+sme2)
elseif (NUMKONG_ARCH_RISCV64_)
    nk_cpu_capability_(rvv NUMKONG_TARGET_RVV GCC_FLAGS -march=rv64gcv)
    nk_cpu_capability_(rvvhalf NUMKONG_TARGET_RVVHALF GCC_FLAGS -march=rv64gcv_zvfh)
    nk_cpu_capability_(rvvbf16 NUMKONG_TARGET_RVVBF16 GCC_FLAGS -march=rv64gcv_zvfbfwma)
    nk_cpu_capability_(rvvbb NUMKONG_TARGET_RVVBB GCC_FLAGS -march=rv64gcv_zvbb)
elseif (NUMKONG_ARCH_LOONGARCH64_)
    nk_cpu_capability_(loongsonasx NUMKONG_TARGET_LOONGSONASX GCC_FLAGS -mlasx)
elseif (NUMKONG_ARCH_PPC64_)
    nk_cpu_capability_(powervsx NUMKONG_TARGET_POWERVSX GCC_FLAGS -mcpu=power9)
endif ()

# Linked by the units that call one GPU vendor's capability kernels: each macro is 1 where some architecture the
# build compiles for, without CMake's "-real" or "-virtual", runs the capability, else 0. Metal has no compiler macro
# of its own, so its target also switches `NUMKONG_ARCH_METAL_` on.
add_library(numkong_cuda_capabilities_compiled_ INTERFACE)
add_library(numkong_rocm_capabilities_compiled_ INTERFACE)
add_library(numkong_metal_capabilities_compiled_ INTERFACE)
target_compile_definitions(numkong_metal_capabilities_compiled_ INTERFACE NUMKONG_ARCH_METAL_=1)

# One GPU capability, named `capability_name_` in its cache variable: caches the `CUDA_ARCHITECTURES`,
# `ROCM_ARCHITECTURES` or `METAL_ARCHITECTURES` running it as `nk_target_<capability>_architectures`, the twin of a
# CPU capability's `nk_target_<capability>_flags`, so a project embedding NumKong, like USearch, reads the same lists,
# and defines `capability_macro_` on that vendor's target.
function (nk_gpu_capability_ capability_name_ capability_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 argument "" "" "CUDA_ARCHITECTURES;ROCM_ARCHITECTURES;METAL_ARCHITECTURES")
    if (argument_CUDA_ARCHITECTURES)
        set(capability_architectures_ ${argument_CUDA_ARCHITECTURES})
        set(compiled_architectures_ ${NUMKONG_CUDA_RESOLVED_ARCHITECTURES_})
        set(capabilities_target_ numkong_cuda_capabilities_compiled_)
    elseif (argument_ROCM_ARCHITECTURES)
        set(capability_architectures_ ${argument_ROCM_ARCHITECTURES})
        set(compiled_architectures_ ${NUMKONG_ROCM_ARCHITECTURES})
        set(capabilities_target_ numkong_rocm_capabilities_compiled_)
    else ()
        set(capability_architectures_ ${argument_METAL_ARCHITECTURES})
        set(compiled_architectures_ ${NUMKONG_METAL_ARCHITECTURES})
        set(capabilities_target_ numkong_metal_capabilities_compiled_)
    endif ()
    set(nk_target_${capability_name_}_architectures "${capability_architectures_}"
        CACHE INTERNAL "Architectures running ${capability_macro_}"
    )
    set(capability_enabled_ FALSE)
    foreach (architecture_ IN LISTS compiled_architectures_)
        string(REGEX REPLACE "-(real|virtual)$" "" architecture_ "${architecture_}")
        if (architecture_ IN_LIST capability_architectures_)
            set(capability_enabled_ TRUE)
        endif ()
    endforeach ()
    target_compile_definitions(${capabilities_target_} INTERFACE "${capability_macro_}=$<BOOL:${capability_enabled_}>")
endfunction ()
