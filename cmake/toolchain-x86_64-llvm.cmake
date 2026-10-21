# x86_64 LLVM toolchain for NumKong, driving Clang.
#
# Native x86_64 hosts use this as a pinned LLVM toolchain.
# Apple Silicon hosts use it to target x86_64 macOS via `-arch x86_64`.

if (NOT DEFINED LLVM_ROOT)
    if (EXISTS "/opt/homebrew/opt/llvm/bin/clang")
        set(LLVM_ROOT "/opt/homebrew/opt/llvm")
    elseif (EXISTS "/usr/local/opt/llvm/bin/clang")
        set(LLVM_ROOT "/usr/local/opt/llvm")
    endif ()
endif ()

if (DEFINED LLVM_ROOT)
    set(numkong_llvm_bin_ "${LLVM_ROOT}/bin")
    set(CMAKE_C_COMPILER "${numkong_llvm_bin_}/clang")
    set(CMAKE_CXX_COMPILER "${numkong_llvm_bin_}/clang++")
    set(CMAKE_AR "${numkong_llvm_bin_}/llvm-ar")
    set(CMAKE_RANLIB "${numkong_llvm_bin_}/llvm-ranlib")
    if (EXISTS "${numkong_llvm_bin_}/ld.lld")
        set(CMAKE_LINKER "${numkong_llvm_bin_}/ld.lld")
    endif ()
else ()
    find_program(CMAKE_C_COMPILER clang REQUIRED)
    find_program(CMAKE_CXX_COMPILER clang++ REQUIRED)
    find_program(CMAKE_AR llvm-ar REQUIRED)
    find_program(CMAKE_RANLIB llvm-ranlib REQUIRED)
    find_program(CMAKE_LINKER ld.lld)
endif ()

set(CMAKE_SYSTEM_PROCESSOR x86_64)

if (APPLE)
    set(CMAKE_SYSTEM_NAME Darwin)
    set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Target macOS architecture")
    set(CMAKE_C_FLAGS_INIT "-arch x86_64")
    set(CMAKE_CXX_FLAGS_INIT "-arch x86_64")

    if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        find_program(numkong_arch_program_ arch)
        if (numkong_arch_program_)
            set(CMAKE_CROSSCOMPILING_EMULATOR "${numkong_arch_program_};-x86_64")
        endif ()
    endif ()
else ()
    set(CMAKE_SYSTEM_NAME "${CMAKE_HOST_SYSTEM_NAME}")
endif ()
