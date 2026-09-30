# cmake/nk_isa_probe.cmake — which kits of the target architecture the toolchain builds
#
# Probes `probes/<kit>.c` for every kit of the architecture and folds the verdicts into
# `nk_compile_definitions_`, a list of `NUMKONG_TARGET_<KIT>=0/1`, and into
# `nk_header_only_definitions_`, the same without the file-wide kits. `-D NUMKONG_TARGET_<KIT>=0`
# drops a kit, and `=1` keeps one only where its probe compiles.

include_guard(GLOBAL)

# Whether the toolchain builds one kit's kernels: the probe calls one of them, compiled header-only
# at the baseline flags and the kit's own flags in `ARGN`, as the library compiles it. `try_compile`
# reruns even with its verdict cached, hence the guard.
function (nk_instruction_set_probe_ kit_)
    string(TOLOWER "${kit_}" kit_lowercase_)
    if (NOT DEFINED nk_target_${kit_lowercase_}_compiles)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
        try_compile(
            nk_target_${kit_lowercase_}_compiles ${CMAKE_BINARY_DIR}/nk_probes
            ${PROJECT_SOURCE_DIR}/probes/${kit_lowercase_}.c
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/include" C_STANDARD 99 COMPILE_DEFINITIONS ${ARGN}
        )
    endif ()
    message(STATUS "Performing ISA probe ${kit_} - compiles: ${nk_target_${kit_lowercase_}_compiles}")
endfunction ()

# LASX and POWER9 compile file-wide, as Clang's `lasxintrin.h` and `altivec.h` hide their contents
# without the flag; `CMakeLists.txt` gives it to the kit's unit and header-only test alone.
set(nk_kit_flags_LOONGSONASX -mlasx)
set(nk_kit_flags_POWERVSX -mcpu=power9)

if (NUMKONG_ARCH_X8664_)
    set(nk_kits_ HASWELL SKYLAKE ICELAKE GENOA SAPPHIRE SAPPHIREAMX GRANITEAMX DIAMOND DIAMONDAMX TURIN ALDER SIERRA)
elseif (NUMKONG_ARCH_ARM64_)
    set(nk_kits_ NEON NEONHALF NEONSDOT NEONBFDOT NEONFHM SVE SVEHALF SVEBFDOT SVESDOT SVE2 NEONFP8 SME SMEF64 SMEBI32)
elseif (NUMKONG_ARCH_RISCV64_)
    set(nk_kits_ RVV RVVHALF RVVBF16 RVVBB)
elseif (NUMKONG_ARCH_LOONGARCH64_)
    set(nk_kits_ LOONGSONASX)
elseif (NUMKONG_ARCH_PPC64_)
    set(nk_kits_ POWERVSX)
else ()
    set(nk_kits_)
endif ()

# Header-only units leave the file-wide kits to their own flags, which `types.h` reads.
set(nk_compile_definitions_)
set(nk_header_only_definitions_)
foreach (nk_kit_ IN LISTS nk_kits_)
    nk_instruction_set_probe_(${nk_kit_} ${nk_kit_flags_${nk_kit_}})
    string(TOLOWER "${nk_kit_}" nk_kit_lowercase_)
    set(nk_verdict_ 0)
    if (nk_target_${nk_kit_lowercase_}_compiles)
        set(nk_verdict_ 1)
    endif ()
    if (DEFINED NUMKONG_TARGET_${nk_kit_})
        message(STATUS "NUMKONG_TARGET_${nk_kit_} override in effect: ${NUMKONG_TARGET_${nk_kit_}}")
        if (NOT NUMKONG_TARGET_${nk_kit_})
            set(nk_verdict_ 0)
        elseif (NOT nk_verdict_)
            message(WARNING "NUMKONG_TARGET_${nk_kit_}=1 requested, but its probe does not compile here; ignoring")
        endif ()
    endif ()
    list(APPEND nk_compile_definitions_ "NUMKONG_TARGET_${nk_kit_}=${nk_verdict_}")
    if (NOT DEFINED nk_kit_flags_${nk_kit_})
        list(APPEND nk_header_only_definitions_ "NUMKONG_TARGET_${nk_kit_}=${nk_verdict_}")
    endif ()
endforeach ()
