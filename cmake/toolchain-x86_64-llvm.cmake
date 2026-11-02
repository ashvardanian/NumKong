# x86_64 LLVM toolchain for NumKong, driving Clang.
#
# Native x86_64 hosts use this as a pinned LLVM toolchain.
# Apple Silicon hosts use it to target x86_64 macOS via `-arch x86_64`.
# Other Linux hosts cross-compile and run tests under QEMU:
#   cmake --preset linux_x86_64_llvm -D LLVM_ROOT=/usr/lib/llvm-23 -D X86_QEMU_CPU=Haswell-noTSX-IBRS
#
# Optional inputs, on a foreign Linux host:
#   -D LLVM_ROOT=/path/to/llvm
#   -D X86_QEMU_LD_PREFIX=/usr/x86_64-linux-gnu
#   -D X86_QEMU_CPU=max

if (NOT DEFINED LLVM_ROOT)
    if (DEFINED ENV{LLVM_ROOT})
        set(LLVM_ROOT "$ENV{LLVM_ROOT}")
    elseif (EXISTS "/opt/homebrew/opt/llvm/bin/clang")
        set(LLVM_ROOT "/opt/homebrew/opt/llvm")
    elseif (EXISTS "/usr/local/opt/llvm/bin/clang")
        set(LLVM_ROOT "/usr/local/opt/llvm")
    endif ()
endif ()

if (DEFINED LLVM_ROOT)
    set(ENV{LLVM_ROOT} "${LLVM_ROOT}")
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
elseif (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_SYSTEM_NAME "${CMAKE_HOST_SYSTEM_NAME}")
else ()
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_C_COMPILER_TARGET x86_64-linux-gnu)
    set(CMAKE_CXX_COMPILER_TARGET x86_64-linux-gnu)

    # `max` enables every feature QEMU implements for this target.
    if (DEFINED ENV{X86_QEMU_CPU})
        set(X86_QEMU_CPU "$ENV{X86_QEMU_CPU}" CACHE STRING "CPU model for `qemu-x86_64 -cpu`")
    else ()
        set(X86_QEMU_CPU "max" CACHE STRING "CPU model for `qemu-x86_64 -cpu`")
    endif ()
    set(ENV{X86_QEMU_CPU} "${X86_QEMU_CPU}")
    set(X86_QEMU_LD_PREFIX "/usr/x86_64-linux-gnu" CACHE PATH "Guest loader prefix for `qemu-x86_64 -L`")

    # `check_ipo_supported()` ignores `CMAKE_TRY_COMPILE_PLATFORM_VARIABLES`, hence the `set(ENV{...})` above.
    set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES LLVM_ROOT X86_QEMU_CPU X86_QEMU_LD_PREFIX)

    find_program(numkong_qemu_x86_64_ qemu-x86_64)
    if (numkong_qemu_x86_64_)
        set(CMAKE_CROSSCOMPILING_EMULATOR "${numkong_qemu_x86_64_};-L;${X86_QEMU_LD_PREFIX};-cpu;${X86_QEMU_CPU}")
    endif ()
endif ()
