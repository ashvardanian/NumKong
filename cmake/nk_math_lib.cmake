# cmake/nk_math_lib.cmake
#
# Detect whether C++ targets need explicit linkage against `libm`.
# Some Unix toolchains require `-lm` for cmath symbols, while MSVC and
# WASM/WASI toolchains do not provide a separate math archive.
#
# Contract:
#   Input:  MSVC, NUMKONG_OS_WASI_, CMAKE_SYSTEM_NAME
#   Output: caller-provided output variable with either "" or "m"

include(CheckCXXSourceCompiles)

function (nk_detect_cxx_math_lib_ output_var_)
    set(${output_var_} "" PARENT_SCOPE)

    if (MSVC OR NUMKONG_OS_WASI_ OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        return()
    endif ()

    set(nk_cxx_math_probe_src_
        "
        #include <cmath>
        int main() {
            volatile double x = std::sqrt(2.0);
            volatile double y = std::sin(x);
            volatile double z = std::fma(x, y, 1.0);
            return z == 0.0;
        }
        "
    )

    set(CMAKE_REQUIRED_QUIET ON)
    set(CMAKE_REQUIRED_LIBRARIES "")
    check_cxx_source_compiles("${nk_cxx_math_probe_src_}" nk_cxx_links_math_without_libm_)

    if (NOT nk_cxx_links_math_without_libm_)
        set(CMAKE_REQUIRED_LIBRARIES m)
        check_cxx_source_compiles("${nk_cxx_math_probe_src_}" nk_cxx_links_math_with_libm_)
        if (nk_cxx_links_math_with_libm_)
            set(${output_var_} "m" PARENT_SCOPE)
        endif ()
    endif ()

    unset(CMAKE_REQUIRED_LIBRARIES)
    unset(CMAKE_REQUIRED_QUIET)
endfunction ()
