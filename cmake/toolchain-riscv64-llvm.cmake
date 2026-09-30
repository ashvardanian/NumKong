# RISC-V 64 LLVM toolchain for NumKong, driving Clang.
# The RVV kernels need Clang 21 or newer; with an older compiler those kits probe as 0.
#
# Usage:
#   sudo apt install clang lld gcc-riscv64-linux-gnu g++-riscv64-linux-gnu libc6-dev-riscv64-cross qemu-user
#   cmake -B build_riscv_llvm -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-llvm.cmake
#
# Optional inputs:
#   -D LLVM_ROOT=/path/to/llvm
#   -D RISCV_SYSROOT=/path/to/riscv64/sysroot
#   -D RISCV_QEMU_LD_PREFIX=/usr/riscv64-linux-gnu
#   -D RISCV_QEMU_CPU=max
#   -D RISCV_TARGET=riscv64-linux-gnu
#   -D RISCV_MABI=lp64d
#
# Testing with QEMU:
#   Tests will automatically run under QEMU via CMAKE_CROSSCOMPILING_EMULATOR, on `max` by default,
#   which runs every kit, or on the RV64GC baseline with `-D RISCV_QEMU_CPU=rv64`.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Only a self-contained rootfs is a sysroot. Distribution cross packages install into the build
# root, so the sysroot is `/` — Clang's default — and `--target` alone finds the multiarch paths.
if (NOT DEFINED RISCV_SYSROOT AND DEFINED ENV{RISCV_SYSROOT})
    set(RISCV_SYSROOT "$ENV{RISCV_SYSROOT}")
endif ()
if (DEFINED RISCV_SYSROOT)
    set(ENV{RISCV_SYSROOT} "${RISCV_SYSROOT}")
    set(RISCV_SYSROOT "${RISCV_SYSROOT}" CACHE PATH "RISC-V sysroot for LLVM cross-compilation")
endif ()

# `qemu-user`'s `-L` is the guest *interpreter* prefix, not a sysroot: it is where
# `ld-linux-riscv64-lp64d.so.1` and the guest shared libraries live. On Debian that is
# `/usr/<triple>` even though the compiler sysroot is `/`, so it is its own input.
if (NOT DEFINED RISCV_QEMU_LD_PREFIX)
    if (DEFINED ENV{QEMU_LD_PREFIX})
        set(RISCV_QEMU_LD_PREFIX "$ENV{QEMU_LD_PREFIX}")
    else ()
        set(RISCV_QEMU_LD_PREFIX "/usr/riscv64-linux-gnu")
    endif ()
endif ()
set(RISCV_QEMU_LD_PREFIX "${RISCV_QEMU_LD_PREFIX}" CACHE PATH "Guest loader prefix for `qemu-riscv64 -L`")

# `max` enables every extension QEMU implements for this target.
if (DEFINED ENV{RISCV_QEMU_CPU})
    set(RISCV_QEMU_CPU "$ENV{RISCV_QEMU_CPU}" CACHE STRING "CPU model for `qemu-riscv64 -cpu`")
else ()
    set(RISCV_QEMU_CPU "max" CACHE STRING "CPU model for `qemu-riscv64 -cpu`")
endif ()
set(ENV{RISCV_QEMU_CPU} "${RISCV_QEMU_CPU}")

if (NOT DEFINED RISCV_TARGET)
    if (DEFINED ENV{RISCV_TARGET})
        set(RISCV_TARGET "$ENV{RISCV_TARGET}")
    else ()
        set(RISCV_TARGET "riscv64-linux-gnu")
    endif ()
endif ()
set(RISCV_TARGET "${RISCV_TARGET}" CACHE STRING "RISC-V target triple for LLVM cross-compilation")
set(ENV{RISCV_TARGET} "${RISCV_TARGET}")

if (NOT DEFINED RISCV_MABI)
    if (DEFINED ENV{RISCV_MABI})
        set(RISCV_MABI "$ENV{RISCV_MABI}")
    else ()
        set(RISCV_MABI "lp64d")
    endif ()
endif ()
set(RISCV_MABI "${RISCV_MABI}" CACHE STRING "RISC-V ABI for LLVM cross-compilation")
set(ENV{RISCV_MABI} "${RISCV_MABI}")

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
    set(LLVM_ROOT "${LLVM_ROOT}" CACHE PATH "LLVM toolchain root for cross-compilation")
    set(ENV{LLVM_ROOT} "${LLVM_ROOT}")
endif ()

# `check_ipo_supported()` uses try_compile's whole-project signature, which ignores
# `CMAKE_TRY_COMPILE_PLATFORM_VARIABLES` — hence the duplicate `set(ENV{...})` above.
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES LLVM_ROOT RISCV_SYSROOT RISCV_QEMU_LD_PREFIX RISCV_QEMU_CPU RISCV_TARGET
                                         RISCV_MABI
)

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

if (DEFINED RISCV_SYSROOT)
    set(CMAKE_SYSROOT "${RISCV_SYSROOT}")
endif ()
set(CMAKE_C_COMPILER_TARGET "${RISCV_TARGET}")
set(CMAKE_CXX_COMPILER_TARGET "${RISCV_TARGET}")

# No `-march` here: `CMakeLists.txt` pins the dispatch floor with
# `add_compile_options(-march=rv64gc)`, which lands after `CMAKE_C_FLAGS` and wins. `-mabi` is the
# one flag nothing else sets.
set(CMAKE_C_FLAGS_INIT "-mabi=${RISCV_MABI}")
set(CMAKE_CXX_FLAGS_INIT "-mabi=${RISCV_MABI}")

find_program(_NUMKONG_QEMU_RISCV64 qemu-riscv64)
if (_NUMKONG_QEMU_RISCV64)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${_NUMKONG_QEMU_RISCV64};-L;${RISCV_QEMU_LD_PREFIX};-cpu;${RISCV_QEMU_CPU}")
endif ()

if (DEFINED RISCV_SYSROOT)
    set(CMAKE_FIND_ROOT_PATH "${RISCV_SYSROOT}")
endif ()
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
