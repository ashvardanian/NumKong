# x86_64 LLVM Clang toolchain for NumKong.
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
    set(_NUMKONG_LLVM_BIN "${LLVM_ROOT}/bin")
    set(CMAKE_C_COMPILER "${_NUMKONG_LLVM_BIN}/clang")
    set(CMAKE_CXX_COMPILER "${_NUMKONG_LLVM_BIN}/clang++")
    set(CMAKE_AR "${_NUMKONG_LLVM_BIN}/llvm-ar")
    set(CMAKE_RANLIB "${_NUMKONG_LLVM_BIN}/llvm-ranlib")
    if (EXISTS "${_NUMKONG_LLVM_BIN}/ld.lld")
        set(CMAKE_LINKER "${_NUMKONG_LLVM_BIN}/ld.lld")
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
    set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Target macOS architecture" FORCE)
    set(CMAKE_C_FLAGS_INIT "-arch x86_64")
    set(CMAKE_CXX_FLAGS_INIT "-arch x86_64")

    if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        find_program(_NUMKONG_ARCH_PROGRAM arch)
        if (_NUMKONG_ARCH_PROGRAM)
            set(CMAKE_CROSSCOMPILING_EMULATOR "${_NUMKONG_ARCH_PROGRAM};-x86_64")
        endif ()
    endif ()
else ()
    set(CMAKE_SYSTEM_NAME "${CMAKE_HOST_SYSTEM_NAME}")
endif ()
