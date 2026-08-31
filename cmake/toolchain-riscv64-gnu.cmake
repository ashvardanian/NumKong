# RISC-V 64 GNU toolchain for NumKong.
#
# Two toolchain layouts are supported, selected by whether `RISCV_TOOLCHAIN_PATH` is given.
#
#   Distribution cross packages, the default and what CI uses:
#     sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu libc6-dev-riscv64-cross qemu-user
#     cmake -B build_riscv -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-gnu.cmake
#
#   A self-contained toolchain carrying its own rootfs:
#     cmake -B build_riscv -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-gnu.cmake \
#           -D RISCV_TOOLCHAIN_PATH=/opt/riscv
#
# Optional inputs:
#   -D RISCV_TOOLCHAIN_PATH=/opt/riscv # selects the self-contained layout
#   -D RISCV_TRIPLE=riscv64-linux-gnu  # binary prefix and multiarch directory
#   -D RISCV_COMPILER_SUFFIX=-14       # selects `riscv64-linux-gnu-gcc-14`
#   -D RISCV_SYSROOT=/opt/riscv/sysroot
#   -D RISCV_QEMU_LD_PREFIX=/usr/riscv64-linux-gnu
#   -D RISCV_QEMU_CPU=max
#   -D RISCV_MABI=lp64d
#
# Needs GCC 16 or newer. The RVV kernels gate on `#pragma GCC target("arch=+v")`, which GCC
# implements for RISC-V only from 16; 14 and 15 ignore the pragma and then fail on the
# intrinsics it was meant to enable. Use `toolchain-riscv64-llvm.cmake` for older toolchains.
#
# Testing with QEMU:
#   Tests will automatically run under QEMU via CMAKE_CROSSCOMPILING_EMULATOR

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if (NOT DEFINED RISCV_TOOLCHAIN_PATH AND DEFINED ENV{RISCV_TOOLCHAIN_PATH})
    set(RISCV_TOOLCHAIN_PATH "$ENV{RISCV_TOOLCHAIN_PATH}")
endif ()
if (DEFINED RISCV_TOOLCHAIN_PATH)
    set(RISCV_TOOLCHAIN_PATH "${RISCV_TOOLCHAIN_PATH}" CACHE PATH "Self-contained RISC-V GNU toolchain root")
    set(ENV{RISCV_TOOLCHAIN_PATH} "${RISCV_TOOLCHAIN_PATH}")
endif ()

if (NOT DEFINED RISCV_TRIPLE)
    if (DEFINED ENV{RISCV_TRIPLE})
        set(RISCV_TRIPLE "$ENV{RISCV_TRIPLE}")
    else ()
        set(RISCV_TRIPLE "riscv64-linux-gnu")
    endif ()
endif ()
set(RISCV_TRIPLE "${RISCV_TRIPLE}" CACHE STRING "RISC-V target triple, used as the compiler prefix")
set(ENV{RISCV_TRIPLE} "${RISCV_TRIPLE}")

# Debian versions its cross compilers as `<triple>-gcc-14` and only provides an unsuffixed
# `<triple>-gcc` when the unversioned metapackage is installed, which CI does not install.
if (NOT DEFINED RISCV_COMPILER_SUFFIX)
    if (DEFINED ENV{RISCV_COMPILER_SUFFIX})
        set(RISCV_COMPILER_SUFFIX "$ENV{RISCV_COMPILER_SUFFIX}")
    else ()
        set(RISCV_COMPILER_SUFFIX "")
    endif ()
endif ()
set(RISCV_COMPILER_SUFFIX "${RISCV_COMPILER_SUFFIX}" CACHE STRING "Version suffix on the cross compiler, e.g. `-14`")
set(ENV{RISCV_COMPILER_SUFFIX} "${RISCV_COMPILER_SUFFIX}")

# Only a self-contained rootfs is a sysroot. Distribution cross packages install into the build
# root, so the sysroot is `/` — the compiler's default — and setting it buys nothing.
if (NOT DEFINED RISCV_SYSROOT)
    if (DEFINED ENV{RISCV_SYSROOT})
        set(RISCV_SYSROOT "$ENV{RISCV_SYSROOT}")
    elseif (DEFINED RISCV_TOOLCHAIN_PATH)
        set(RISCV_SYSROOT "${RISCV_TOOLCHAIN_PATH}/sysroot")
    endif ()
endif ()
if (DEFINED RISCV_SYSROOT)
    set(RISCV_SYSROOT "${RISCV_SYSROOT}" CACHE PATH "Self-contained riscv64 rootfs")
    set(ENV{RISCV_SYSROOT} "${RISCV_SYSROOT}")
endif ()

# `qemu-user`'s `-L` is the guest *interpreter* prefix, not a sysroot: it is where `ld-linux-riscv64-lp64d.so.1` and
# the guest shared libraries live. With distribution packages that is `/usr/<triple>` even though
# the compiler sysroot is `/`, and `qemu-riscv64 -L /` fails to open the loader.
if (NOT DEFINED RISCV_QEMU_LD_PREFIX)
    if (DEFINED ENV{QEMU_LD_PREFIX})
        set(RISCV_QEMU_LD_PREFIX "$ENV{QEMU_LD_PREFIX}")
    elseif (DEFINED RISCV_SYSROOT)
        set(RISCV_QEMU_LD_PREFIX "${RISCV_SYSROOT}")
    else ()
        set(RISCV_QEMU_LD_PREFIX "/usr/${RISCV_TRIPLE}")
    endif ()
endif ()
set(RISCV_QEMU_LD_PREFIX "${RISCV_QEMU_LD_PREFIX}" CACHE PATH "Guest loader prefix for `qemu-riscv64 -L`")
set(ENV{RISCV_QEMU_LD_PREFIX} "${RISCV_QEMU_LD_PREFIX}")

# `max` enables every extension QEMU implements for this target.
if (NOT DEFINED RISCV_QEMU_CPU)
    if (DEFINED ENV{RISCV_QEMU_CPU})
        set(RISCV_QEMU_CPU "$ENV{RISCV_QEMU_CPU}")
    else ()
        set(RISCV_QEMU_CPU "max")
    endif ()
endif ()
set(RISCV_QEMU_CPU "${RISCV_QEMU_CPU}" CACHE STRING "CPU model for `qemu-riscv64 -cpu`")
set(ENV{RISCV_QEMU_CPU} "${RISCV_QEMU_CPU}")

# Forward settings to nested try_compile() invocations.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RISCV_TOOLCHAIN_PATH RISCV_TRIPLE RISCV_COMPILER_SUFFIX
     RISCV_SYSROOT RISCV_QEMU_LD_PREFIX RISCV_QEMU_CPU RISCV_MABI)

# Toolchain validation only needs to prove the project compiles.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if (DEFINED RISCV_TOOLCHAIN_PATH)
    set(_NK_RISCV_PREFIX "${RISCV_TOOLCHAIN_PATH}/bin/${RISCV_TRIPLE}-")
else ()
    set(_NK_RISCV_PREFIX "${RISCV_TRIPLE}-")
endif ()
set(CMAKE_C_COMPILER "${_NK_RISCV_PREFIX}gcc${RISCV_COMPILER_SUFFIX}")
set(CMAKE_CXX_COMPILER "${_NK_RISCV_PREFIX}g++${RISCV_COMPILER_SUFFIX}")

if (NOT DEFINED RISCV_MABI)
    if (DEFINED ENV{RISCV_MABI})
        set(RISCV_MABI "$ENV{RISCV_MABI}")
    else ()
        set(RISCV_MABI "lp64d")
    endif ()
endif ()
set(RISCV_MABI "${RISCV_MABI}" CACHE STRING "RISC-V ABI for GNU cross-compilation")
set(ENV{RISCV_MABI} "${RISCV_MABI}")

# No `-march` here: `CMakeLists.txt` pins the dispatch floor with `add_compile_options(-march=rv64gc)`,
# which lands after `CMAKE_C_FLAGS` and wins. `-mabi` is the one flag nothing else sets.
set(CMAKE_C_FLAGS_INIT "-mabi=${RISCV_MABI}")
set(CMAKE_CXX_FLAGS_INIT "-mabi=${RISCV_MABI}")

if (DEFINED RISCV_SYSROOT)
    set(CMAKE_SYSROOT "${RISCV_SYSROOT}")
    set(CMAKE_FIND_ROOT_PATH "${RISCV_SYSROOT}")
endif ()

find_program(_NK_QEMU_RISCV64 qemu-riscv64)
if (_NK_QEMU_RISCV64)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${_NK_QEMU_RISCV64};-L;${RISCV_QEMU_LD_PREFIX};-cpu;${RISCV_QEMU_CPU}")
endif ()

# Search paths for libraries and headers (target system only).
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
