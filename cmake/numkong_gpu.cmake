# GPU languages and architectures, the GPU capability rows, and the helpers adding one GPU unit.

# CUDA and HIP come on before any target: `numkong_static` carries their units, so every target linking it, the
# bindings, tests and benchmarks included, needs the architecture list. One cubin per device family plus a PTX tail,
# needing CUDA 13 or newer and a driver from 580 on; Turing, CUDA 13's floor, runs the CUDA baseline only. Every
# code lands in one fat binary, and each kernel checks its own architecture in device code.
if (NUMKONG_BUILD_CUDA)
    enable_language(CUDA)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13)
        message(FATAL_ERROR "NumKong needs CUDA 13 or newer, found ${CMAKE_CUDA_COMPILER_VERSION}")
    endif ()
    set(NUMKONG_CUDA_ARCHITECTURES "75-real;80-real;89-real;90a-real;100f-real;120f-real;80-virtual"
        CACHE STRING "GPU architectures baked into the CUDA targets"
    )
    # "native" maps each visible GPU to the code its capability rows need, family suffix included, where CMake's own
    # resolution would drop it; "all" and "all-major" stand for the lists CMake resolved them to.
    set(numkong_cuda_native_codes_
        75-real=75-real 80-real=80-real 86-real=86-real 89-real=89-real 90-real=90a-real 100-real=100f-real
        103-real=103f-real 110-real=110f-real 120-real=120f-real 121-real=121f-real
    )
    set(numkong_cuda_resolved_architectures_)
    foreach (entry_ IN LISTS NUMKONG_CUDA_ARCHITECTURES)
        if (entry_ STREQUAL "native")
            foreach (visible_ IN LISTS CMAKE_CUDA_ARCHITECTURES_NATIVE)
                set(native_code_)
                foreach (pair_ IN LISTS numkong_cuda_native_codes_)
                    if (pair_ MATCHES "^${visible_}=(.+)$")
                        set(native_code_ ${CMAKE_MATCH_1})
                    endif ()
                endforeach ()
                if (NOT native_code_)
                    message(
                        FATAL_ERROR "NUMKONG_CUDA_ARCHITECTURES=native has no code for the visible ${visible_} GPU; "
                                    "name its code instead"
                    )
                endif ()
                list(APPEND numkong_cuda_resolved_architectures_ ${native_code_})
            endforeach ()
        elseif (entry_ STREQUAL "all")
            list(APPEND numkong_cuda_resolved_architectures_ ${CMAKE_CUDA_ARCHITECTURES_ALL})
        elseif (entry_ STREQUAL "all-major")
            list(APPEND numkong_cuda_resolved_architectures_ ${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR})
        else ()
            list(APPEND numkong_cuda_resolved_architectures_ ${entry_})
        endif ()
    endforeach ()
    if (CMAKE_VERSION VERSION_LESS 3.31 AND numkong_cuda_resolved_architectures_ MATCHES "1[0-9][0-9][af]")
        message(FATAL_ERROR "Three-digit \"a\" and \"f\" CUDA codes need CMake 3.31, found ${CMAKE_VERSION}")
    endif ()
    set(CMAKE_CUDA_ARCHITECTURES "${numkong_cuda_resolved_architectures_}")
    # NVCC's default leaves Turing and Ampere cubins uncompressed, ten times their compressed size.
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=size>)
endif ()
if (NUMKONG_BUILD_ROCM)
    enable_language(HIP)
    set(NUMKONG_ROCM_ARCHITECTURES "gfx942;gfx950;gfx1250;gfx1251"
        CACHE STRING "AMD GPU architectures baked into the ROCm targets"
    )
    set(CMAKE_HIP_ARCHITECTURES "${NUMKONG_ROCM_ARCHITECTURES}")
endif ()

if (NUMKONG_BUILD_METAL AND NOT APPLE)
    message(FATAL_ERROR "NUMKONG_BUILD_METAL needs an Apple platform")
elseif (NUMKONG_BUILD_METAL)
    # Metal compiles on the device, so the families only choose the capabilities embedded: the Metal baseline always,
    # Apple9 from M3, Apple10 from M5.
    set(NUMKONG_METAL_ARCHITECTURES "apple7;apple9;apple10"
        CACHE STRING "Apple GPU families whose capabilities the Metal targets embed"
    )
endif ()

# One `numkong_add_gpu_capability` row per GPU capability. Past Ampere, the tensor-core capabilities need family- or
# architecture-specific code, like "120f" rather than "120".
# `mma.sync`: A100, A30, A40, RTX 30, Jetson Orin, L40, RTX 40, and every GPU since.
numkong_add_gpu_capability(
    ampere NUMKONG_TARGET_AMPERE BIT 49 CUDA_ARCHITECTURES 80 86 87 88 89 90 90a 100 100a 100f 103 103a 103f 110 110a
                                                           110f 120 120a 120f 121 121a 121f
)
# FP8 `mma.sync`: L4, L40S, RTX 40, and every GPU since.
numkong_add_gpu_capability(
    ada NUMKONG_TARGET_ADA BIT 50 CUDA_ARCHITECTURES 89 90 90a 100 100a 100f 103 103a 103f 110 110a 110f 120 120a 120f
                                                     121 121a 121f
)
# `wgmma`: H100, H200, GH200.
numkong_add_gpu_capability(hopper NUMKONG_TARGET_HOPPER BIT 51 CUDA_ARCHITECTURES 90a)
# `tcgen05`: B200, GB200, B300, GB300.
numkong_add_gpu_capability(blackwell NUMKONG_TARGET_BLACKWELL BIT 52 CUDA_ARCHITECTURES 100a 100f 103a 103f)
# FP6/FP4 `mma.sync`: RTX 50, RTX PRO 6000, DGX Spark.
numkong_add_gpu_capability(blackwellrtx NUMKONG_TARGET_BLACKWELLRTX BIT 53 CUDA_ARCHITECTURES 120a 120f 121a 121f)
# `tcgen05.ld.red`: B300, GB300.
numkong_add_gpu_capability(blackwellultra NUMKONG_TARGET_BLACKWELLULTRA BIT 54 CUDA_ARCHITECTURES 103a 103f)
# MFMA over BF16, F16 and I8: MI300X, MI300A, MI325X.
numkong_add_gpu_capability(cdna3 NUMKONG_TARGET_CDNA3 BIT 57 ROCM_ARCHITECTURES gfx942)
# MFMA over FP8, FP6 and FP4: MI350X, MI355X.
numkong_add_gpu_capability(cdna4 NUMKONG_TARGET_CDNA4 BIT 58 ROCM_ARCHITECTURES gfx950)
# WMMA over FP8, FP6 and FP4: MI450X and MI455X on gfx1250, MI430X on gfx1251.
numkong_add_gpu_capability(cdna5 NUMKONG_TARGET_CDNA5 BIT 59 ROCM_ARCHITECTURES gfx1250 gfx1251)
# Apple9 from M3, Apple10 from M5.
numkong_add_gpu_capability(apple9 NUMKONG_TARGET_APPLE9 BIT 61 METAL_ARCHITECTURES apple9 apple10)
numkong_add_gpu_capability(apple10 NUMKONG_TARGET_APPLE10 BIT 62 METAL_ARCHITECTURES apple10)

# Sets `output_` to the entries of `codes_` that `capability_codes_` names, keeping CMake's "-real" and "-virtual"
# suffixes.
function (numkong_select_gpu_codes output_ codes_ capability_codes_)
    set(kept_)
    foreach (code_ IN LISTS codes_)
        string(REGEX REPLACE "-(real|virtual)$" "" architecture_ "${code_}")
        if (architecture_ IN_LIST capability_codes_)
            list(APPEND kept_ "${code_}")
        endif ()
    endforeach ()
    set(${output_} "${kept_}" PARENT_SCOPE)
endfunction ()

# Adds one GPU generation's unit, like c/target/hopper.cu, to both libraries: compiled for `codes_` alone, in the
# language its extension names, without the CPU capabilities and with its vendor's `capabilities_` target. A unit
# without codes to compile is skipped. The rest are the runtime libraries its calls need.
function (numkong_add_gpu_unit source_ codes_ capabilities_)
    if (NOT codes_)
        return()
    endif ()
    get_filename_component(unit_ "${source_}" NAME_WE)
    add_library(numkong_${unit_}_ OBJECT ${source_})
    target_include_directories(numkong_${unit_}_ PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/c")
    # C++20 makes `NUMKONG_CONSTEXPR` constexpr, which kernels reach through relaxed constexpr.
    set_target_properties(
        numkong_${unit_}_
        PROPERTIES POSITION_INDEPENDENT_CODE ON
                   CUDA_STANDARD 20
                   CUDA_STANDARD_REQUIRED YES
                   CUDA_ARCHITECTURES "${codes_}"
                   HIP_STANDARD 20
                   HIP_STANDARD_REQUIRED YES
                   HIP_ARCHITECTURES "${codes_}"
    )
    # Every CPU capability off, as NVCC's front end cannot parse GCC's `arm_neon.h` and its kin, and every GPU
    # capability the listed architectures run on, as the dispatch units see them; the unit turns off the ones its
    # headers include.
    target_link_libraries(
        numkong_${unit_}_ PRIVATE numkong_cpu_capabilities_disabled_ numkong_warnings ${capabilities_}
    )
    if (MSVC AND numkong_arch_arm64_)
        target_compile_definitions(numkong_${unit_}_ PRIVATE _ARM64_)
    endif ()
    # NVCC compiles a unit's codes one after another unless told to spread them over threads.
    target_compile_options(
        numkong_${unit_}_ PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--threads=0>"
                                  "$<$<COMPILE_LANGUAGE:HIP>:-fvisibility=hidden>"
    )
    if (NOT MSVC)
        target_compile_options(numkong_${unit_}_ PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>")
    endif ()
    # A machine profile links the host side as bitcode too, but only bitcode the C units' compiler can read back.
    set(host_compiler_ "${CMAKE_CUDA_HOST_COMPILER_ID} ${CMAKE_CUDA_HOST_COMPILER_VERSION}")
    if (NUMKONG_PROFILE_CAPABILITIES AND numkong_ipo_supported_
        AND host_compiler_ STREQUAL "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}"
    )
        if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
            target_compile_options(numkong_${unit_}_ PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-flto=thin>")
        elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
            target_compile_options(numkong_${unit_}_ PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-flto>")
        endif ()
    endif ()
    target_link_libraries(numkong_${unit_}_ PRIVATE ${ARGN})
    foreach (library_ ${numkong_libraries_})
        target_sources(${library_} PRIVATE $<TARGET_OBJECTS:numkong_${unit_}_>)
        target_link_libraries(${library_} PUBLIC ${ARGN})
    endforeach ()
endfunction ()
