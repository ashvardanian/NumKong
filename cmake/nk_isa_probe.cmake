# cmake/nk_isa_probe.cmake — shared ISA probe infrastructure
#
# `nk_instruction_set_probe_()` compiles one probe source per kit twice: with the kit's own flags
# for `nk_target_<kit>_compiles`, and as this machine would execute it for `nk_target_<kit>_runs`.
# `nk_build_instruction_set_definitions_()` folds those verdicts into `NUMKONG_TARGET_<KIT>=0/1` entries
# appended per architecture to the cached `nk_compile_definitions_` and `nk_run_definitions_` lists.
#
# Each architecture file sets `nk_native_flags_` before including this file, then calls the two.
# Both verdicts are CMake check caches, so a preset `-D nk_target_<kit>_compiles=0` or `_runs=0`
# skips that probe and every reader follows it.

include_guard(GLOBAL)

include(CheckSourceCompiles)
include(CheckSourceRuns)

set(nk_compile_definitions_ "" CACHE INTERNAL "")
set(nk_run_definitions_ "" CACHE INTERNAL "")

# `check_source_runs` caches success, so results probed under one emulated CPU would
# survive a reconfigure with another (e.g. `sve-max-vq=2` → `sve=off`). Drop them.
if (NOT "$CACHE{nk_probe_emulator_}" STREQUAL "${CMAKE_CROSSCOMPILING_EMULATOR}")
    get_cmake_property(nk_cache_vars_ CACHE_VARIABLES)
    foreach (nk_cache_var_ IN LISTS nk_cache_vars_)
        if (nk_cache_var_ MATCHES "^nk_target_.*_runs$")
            unset(${nk_cache_var_} CACHE)
        endif ()
    endforeach ()
    set(nk_probe_emulator_ "${CMAKE_CROSSCOMPILING_EMULATOR}" CACHE INTERNAL
                                                                    "Emulator the ISA run probes last ran under"
    )
endif ()

# Two-pass probe: the kit's own flags say whether the toolchain builds it, then the native flags,
# an MSVC run, or an emulator run say whether this machine executes it. Function scope keeps the
# `CMAKE_REQUIRED_FLAGS` and try_compile settings local; Release keeps sanitizer runtimes out.
function (nk_instruction_set_probe_ variable_ msvc_flags_ gnu_flags_ probe_file_)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${probe_file_}" probe_source_)
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
    if (MSVC)
        set(CMAKE_REQUIRED_FLAGS "${msvc_flags_}")
    else ()
        set(CMAKE_REQUIRED_FLAGS "${gnu_flags_}")
    endif ()
    check_source_compiles(C "${probe_source_}" ${variable_}_compiles)
    # The flags the verdict was reached with, for a consumer composing one unit out of several kits.
    set(${variable_}_flags "${CMAKE_REQUIRED_FLAGS}" CACHE INTERNAL "")
    if (NOT CMAKE_CROSSCOMPILING AND NOT MSVC AND NOT "${nk_native_flags_}" STREQUAL "")
        set(CMAKE_REQUIRED_FLAGS "${nk_native_flags_}")
        check_source_compiles(C "${probe_source_}" ${variable_}_runs)
    elseif (NOT CMAKE_CROSSCOMPILING AND MSVC AND ${variable_}_compiles)
        # MSVC has no `-march=native`, so run the probe: a CPU lacking the kit crashes it.
        set(CMAKE_REQUIRED_FLAGS "${msvc_flags_}")
        check_source_runs(C "${probe_source_}" ${variable_}_runs)
    elseif (CMAKE_CROSSCOMPILING_EMULATOR AND ${variable_}_compiles)
        # The emulator carries the exact CPU under test, so run the probe through it: a leg
        # like `-cpu max,sve=off` then rejects SVE kernels just as NEON-only hardware would.
        set(CMAKE_REQUIRED_FLAGS "${gnu_flags_}")
        set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
        check_source_runs(C "${probe_source_}" ${variable_}_runs)
    else ()
        # Cross-compiling with no emulator: nothing can execute the probe, so this records what
        # the compiler can emit, not what the target runs. Such binaries need `numkong_shared`.
        set(${variable_}_runs "${${variable_}_compiles}" CACHE INTERNAL "")
    endif ()
endfunction ()

# Appends `NUMKONG_TARGET_<KIT>=0/1` per kit to the cached `nk_compile_definitions_` (what the toolchain
# builds) and `nk_run_definitions_` (what this machine runs).
function (nk_build_instruction_set_definitions_ architecture_name_ instruction_set_names_)
    set(compile_definitions_ "${nk_compile_definitions_}")
    set(run_definitions_ "${nk_run_definitions_}")
    foreach (instruction_set_ IN LISTS instruction_set_names_)
        string(TOLOWER "${instruction_set_}" instruction_set_lowercase_)
        if (nk_target_${instruction_set_lowercase_}_compiles)
            list(APPEND compile_definitions_ "NUMKONG_TARGET_${instruction_set_}=1")
        else ()
            list(APPEND compile_definitions_ "NUMKONG_TARGET_${instruction_set_}=0")
        endif ()
        if (nk_target_${instruction_set_lowercase_}_runs)
            list(APPEND run_definitions_ "NUMKONG_TARGET_${instruction_set_}=1")
        else ()
            list(APPEND run_definitions_ "NUMKONG_TARGET_${instruction_set_}=0")
        endif ()
    endforeach ()
    list(JOIN compile_definitions_ " " compile_summary_)
    list(JOIN run_definitions_ " " run_summary_)
    message(STATUS "${architecture_name_} compile verdicts: ${compile_summary_}")
    message(STATUS "${architecture_name_} run verdicts: ${run_summary_}")
    set(nk_compile_definitions_ "${compile_definitions_}" CACHE INTERNAL "")
    set(nk_run_definitions_ "${run_definitions_}" CACHE INTERNAL "")
endfunction ()
