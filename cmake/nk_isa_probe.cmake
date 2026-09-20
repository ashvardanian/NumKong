# cmake/nk_isa_probe.cmake — shared ISA probe infrastructure
#
# Provides the two-pass probe macro and result-list builder used by all
# per-architecture probe files (nk_x86_isa_probes.cmake, etc.).
#
# Each architecture file sets `nk_native_flags_` before including this file,
# then calls `nk_isa_probe_()` for each ISA and `nk_build_isa_defs_()` once
# to collect the results.
#
# Results are CMake check caches: a preset `-Dnk_target_<isa>_compiles=0` (or `_native`)
# skips that probe entirely, which is also how a build can force an ISA off by hand.

include(CheckSourceCompiles)
include(CheckSourceRuns)

# `check_source_runs` caches success, so results probed under one emulated CPU would
# survive a reconfigure with another (e.g. `sve-max-vq=2` → `sve=off`). Drop them.
if (NOT "$CACHE{nk_probe_emulator_}" STREQUAL "${CMAKE_CROSSCOMPILING_EMULATOR}")
    get_cmake_property(nk_cache_vars_ CACHE_VARIABLES)
    foreach (nk_cache_var_ IN LISTS nk_cache_vars_)
        if (nk_cache_var_ MATCHES "^nk_target_.*_native$")
            unset(${nk_cache_var_} CACHE)
        endif ()
    endforeach ()
    set(nk_probe_emulator_ "${CMAKE_CROSSCOMPILING_EMULATOR}" CACHE INTERNAL
                                                                    "Emulator the native ISA probes last ran under"
    )
endif ()

# Save and restore CMAKE_REQUIRED_FLAGS around probes.
# Force Release config for try_compile to avoid ASAN DLL-not-found failures.
# Call nk_isa_probes_begin_() before probe calls, nk_isa_probes_end_() after.
macro (nk_isa_probes_begin_)
    set(nk_saved_required_flags_ "${CMAKE_REQUIRED_FLAGS}")
    set(nk_saved_try_compile_config_ "${CMAKE_TRY_COMPILE_CONFIGURATION}")
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
endmacro ()

macro (nk_isa_probes_end_)
    set(CMAKE_REQUIRED_FLAGS "${nk_saved_required_flags_}")
    set(CMAKE_TRY_COMPILE_CONFIGURATION "${nk_saved_try_compile_config_}")
endmacro ()

# Two-pass probe: compile with ISA flags (Pass 1) and native flags (Pass 2).
#   Pass 1 result drives nk_shared (dynamic dispatch — compile all the compiler supports).
#   Pass 2 result drives nk_test/nk_bench (must actually run on the local CPU).
#
# The native flag (`nk_native_flags_`) is set by the caller:
#   x86/ARM/LoongArch: -march=native
#   Power:             -mcpu=native
#   RISC-V:            "" (no equivalent — falls back to Pass 1)
macro (nk_isa_probe_ var_ msvc_arch_ gcc_flags_ probe_file_)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${probe_file_}" nk_probe_source_)
    # Pass 1: can the compiler emit this ISA?
    if (MSVC)
        set(CMAKE_REQUIRED_FLAGS "${msvc_arch_}")
    else ()
        set(CMAKE_REQUIRED_FLAGS "${gcc_flags_}")
    endif ()
    check_source_compiles(C "${nk_probe_source_}" ${var_}_compiles)
    # Pass 2: does the local CPU support it?
    if (NOT CMAKE_CROSSCOMPILING AND NOT MSVC AND NOT "${nk_native_flags_}" STREQUAL "")
        set(CMAKE_REQUIRED_FLAGS "${nk_native_flags_}")
        check_source_compiles(C "${nk_probe_source_}" ${var_}_native)
    elseif (NOT CMAKE_CROSSCOMPILING AND MSVC AND ${var_}_compiles)
        # MSVC has no -march=native; instead compile with ISA-specific flags and
        # run the probe.  If the CPU lacks the ISA the process will crash and
        # check_source_runs() will report failure.
        set(CMAKE_REQUIRED_FLAGS "${msvc_arch_}")
        check_source_runs(C "${nk_probe_source_}" ${var_}_native)
    elseif (CMAKE_CROSSCOMPILING_EMULATOR AND ${var_}_compiles)
        # The emulator carries the exact CPU under test, so run the probe through it: a leg
        # like `-cpu max,sve=off` then rejects SVE kernels just as NEON-only hardware would.
        set(CMAKE_REQUIRED_FLAGS "${gcc_flags_}")
        set(nk_saved_target_type_ "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
        set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
        check_source_runs(C "${nk_probe_source_}" ${var_}_native)
        set(CMAKE_TRY_COMPILE_TARGET_TYPE "${nk_saved_target_type_}")
    else ()
        # Cross-compiling with no emulator: nothing can execute the probe, so this records what
        # the compiler can emit, not what the target runs. Such binaries need `nk_shared`.
        set(${var_}_native ${${var_}_compiles})
    endif ()
endmacro ()

# Build compile-only and native def lists from probe results.
#   arch_prefix_:  variable prefix, e.g. "x86" → nk_x86_compile_defs_
#   arch_display_: human-readable name for status messages
#   isa_list_:     semicolon-separated list of ISA names, e.g. "HASWELL;SKYLAKE;..."
function (nk_build_isa_defs_ arch_prefix_ arch_display_ isa_list_)
    set(compile_defs_ "")
    set(native_defs_ "")
    foreach (isa_ IN LISTS isa_list_)
        string(TOLOWER "${isa_}" isa_lower_)
        if (nk_target_${isa_lower_}_compiles)
            list(APPEND compile_defs_ "NK_TARGET_${isa_}=1")
        else ()
            list(APPEND compile_defs_ "NK_TARGET_${isa_}=0")
        endif ()
        if (nk_target_${isa_lower_}_native)
            list(APPEND native_defs_ "NK_TARGET_${isa_}=1")
        else ()
            list(APPEND native_defs_ "NK_TARGET_${isa_}=0")
        endif ()
    endforeach ()
    if (compile_defs_)
        message(STATUS "${arch_display_} ISA compile probes: ${compile_defs_}")
    endif ()
    if (native_defs_)
        message(STATUS "${arch_display_} ISA native probes: ${native_defs_}")
    endif ()
    set(nk_${arch_prefix_}_compile_defs_ "${compile_defs_}" PARENT_SCOPE)
    set(nk_${arch_prefix_}_native_defs_ "${native_defs_}" PARENT_SCOPE)
    set(nk_${arch_prefix_}_isa_defs_ "${native_defs_}" PARENT_SCOPE)
endfunction ()
