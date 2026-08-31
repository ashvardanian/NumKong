# LoongArch 64 GNU toolchain for NumKong.
#
# Two toolchain layouts are supported, selected by whether `LOONGARCH_TOOLCHAIN_PATH` is given.
#
#   Distribution cross packages, the default and what CI uses:
#     sudo apt install gcc-loongarch64-linux-gnu g++-loongarch64-linux-gnu libc6-dev-loong64-cross qemu-user
#     cmake -B build_loongarch64 -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-loongarch64-gnu.cmake
#
#   A self-contained toolchain carrying its own rootfs:
#     cmake -B build_loongarch64 -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-loongarch64-gnu.cmake \
#           -D LOONGARCH_TOOLCHAIN_PATH=/opt/loongarch64
#
# Optional inputs:
#   -D LOONGARCH_TOOLCHAIN_PATH=/opt/loongarch64 # selects the self-contained layout
#   -D LOONGARCH_TRIPLE=loongarch64-linux-gnu    # binary prefix and multiarch directory
#   -D LOONGARCH_COMPILER_SUFFIX=-14             # selects `loongarch64-linux-gnu-gcc-14`
#   -D LOONGARCH_SYSROOT=/opt/loongarch64/sysroot
#   -D LOONGARCH_QEMU_LD_PREFIX=/usr/loongarch64-linux-gnu
#   -D LOONGARCH_QEMU_CPU=max
#
# Testing with QEMU:
#   Tests will automatically run under QEMU via CMAKE_CROSSCOMPILING_EMULATOR

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR loongarch64)

if (NOT DEFINED LOONGARCH_TOOLCHAIN_PATH AND DEFINED ENV{LOONGARCH_TOOLCHAIN_PATH})
    set(LOONGARCH_TOOLCHAIN_PATH "$ENV{LOONGARCH_TOOLCHAIN_PATH}")
endif ()
if (DEFINED LOONGARCH_TOOLCHAIN_PATH)
    set(LOONGARCH_TOOLCHAIN_PATH "${LOONGARCH_TOOLCHAIN_PATH}" CACHE PATH "Self-contained LoongArch 64 toolchain root")
    set(ENV{LOONGARCH_TOOLCHAIN_PATH} "${LOONGARCH_TOOLCHAIN_PATH}")
endif ()

if (NOT DEFINED LOONGARCH_TRIPLE)
    if (DEFINED ENV{LOONGARCH_TRIPLE})
        set(LOONGARCH_TRIPLE "$ENV{LOONGARCH_TRIPLE}")
    else ()
        set(LOONGARCH_TRIPLE "loongarch64-linux-gnu")
    endif ()
endif ()
set(LOONGARCH_TRIPLE "${LOONGARCH_TRIPLE}" CACHE STRING "LoongArch 64 target triple, used as the compiler prefix")
set(ENV{LOONGARCH_TRIPLE} "${LOONGARCH_TRIPLE}")

# Debian versions its cross compilers as `<triple>-gcc-14` and only provides an unsuffixed
# `<triple>-gcc` when the unversioned metapackage is installed, which CI does not install.
if (NOT DEFINED LOONGARCH_COMPILER_SUFFIX)
    if (DEFINED ENV{LOONGARCH_COMPILER_SUFFIX})
        set(LOONGARCH_COMPILER_SUFFIX "$ENV{LOONGARCH_COMPILER_SUFFIX}")
    else ()
        set(LOONGARCH_COMPILER_SUFFIX "")
    endif ()
endif ()
set(LOONGARCH_COMPILER_SUFFIX "${LOONGARCH_COMPILER_SUFFIX}" CACHE STRING
    "Version suffix on the cross compiler, e.g. `-14`"
)
set(ENV{LOONGARCH_COMPILER_SUFFIX} "${LOONGARCH_COMPILER_SUFFIX}")

# Only a self-contained rootfs is a sysroot. Distribution cross packages install into the build
# root, so the sysroot is `/` — the compiler's default — and setting it buys nothing.
if (NOT DEFINED LOONGARCH_SYSROOT)
    if (DEFINED ENV{LOONGARCH_SYSROOT})
        set(LOONGARCH_SYSROOT "$ENV{LOONGARCH_SYSROOT}")
    elseif (DEFINED LOONGARCH_TOOLCHAIN_PATH)
        set(LOONGARCH_SYSROOT "${LOONGARCH_TOOLCHAIN_PATH}/sysroot")
    endif ()
endif ()
if (DEFINED LOONGARCH_SYSROOT)
    set(LOONGARCH_SYSROOT "${LOONGARCH_SYSROOT}" CACHE PATH "Self-contained loongarch64 rootfs")
    set(ENV{LOONGARCH_SYSROOT} "${LOONGARCH_SYSROOT}")
endif ()

# `qemu-user`'s `-L` is the guest *interpreter* prefix, not a sysroot: it is where `ld.so.1` and
# the guest shared libraries live. With distribution packages that is `/usr/<triple>` even though
# the compiler sysroot is `/`, and `qemu-loongarch64 -L /` fails to open the loader.
if (NOT DEFINED LOONGARCH_QEMU_LD_PREFIX)
    if (DEFINED ENV{QEMU_LD_PREFIX})
        set(LOONGARCH_QEMU_LD_PREFIX "$ENV{QEMU_LD_PREFIX}")
    elseif (DEFINED LOONGARCH_SYSROOT)
        set(LOONGARCH_QEMU_LD_PREFIX "${LOONGARCH_SYSROOT}")
    else ()
        set(LOONGARCH_QEMU_LD_PREFIX "/usr/${LOONGARCH_TRIPLE}")
    endif ()
endif ()
set(LOONGARCH_QEMU_LD_PREFIX "${LOONGARCH_QEMU_LD_PREFIX}" CACHE PATH "Guest loader prefix for `qemu-loongarch64 -L`")
set(ENV{LOONGARCH_QEMU_LD_PREFIX} "${LOONGARCH_QEMU_LD_PREFIX}")

# `max` enables every extension QEMU implements for this target.
if (NOT DEFINED LOONGARCH_QEMU_CPU)
    if (DEFINED ENV{LOONGARCH_QEMU_CPU})
        set(LOONGARCH_QEMU_CPU "$ENV{LOONGARCH_QEMU_CPU}")
    else ()
        set(LOONGARCH_QEMU_CPU "max")
    endif ()
endif ()
set(LOONGARCH_QEMU_CPU "${LOONGARCH_QEMU_CPU}" CACHE STRING "CPU model for `qemu-loongarch64 -cpu`")
set(ENV{LOONGARCH_QEMU_CPU} "${LOONGARCH_QEMU_CPU}")

# Forward settings to nested try_compile() invocations.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES LOONGARCH_TOOLCHAIN_PATH LOONGARCH_TRIPLE LOONGARCH_COMPILER_SUFFIX
     LOONGARCH_SYSROOT LOONGARCH_QEMU_LD_PREFIX LOONGARCH_QEMU_CPU)

# Toolchain validation only needs to prove the project compiles.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if (DEFINED LOONGARCH_TOOLCHAIN_PATH)
    set(_NK_LOONGARCH_PREFIX "${LOONGARCH_TOOLCHAIN_PATH}/bin/${LOONGARCH_TRIPLE}-")
else ()
    set(_NK_LOONGARCH_PREFIX "${LOONGARCH_TRIPLE}-")
endif ()
set(CMAKE_C_COMPILER "${_NK_LOONGARCH_PREFIX}gcc${LOONGARCH_COMPILER_SUFFIX}")
set(CMAKE_CXX_COMPILER "${_NK_LOONGARCH_PREFIX}g++${LOONGARCH_COMPILER_SUFFIX}")

# No `-march`/`-mlasx` here: `CMakeLists.txt` sets them with
# `add_compile_options(-march=loongarch64 -mlasx)`, which lands after `CMAKE_C_FLAGS` and wins.

if (DEFINED LOONGARCH_SYSROOT)
    set(CMAKE_SYSROOT "${LOONGARCH_SYSROOT}")
    set(CMAKE_FIND_ROOT_PATH "${LOONGARCH_SYSROOT}")
endif ()

find_program(_NK_QEMU_LOONGARCH qemu-loongarch64)
if (_NK_QEMU_LOONGARCH)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${_NK_QEMU_LOONGARCH};-L;${LOONGARCH_QEMU_LD_PREFIX};-cpu;${LOONGARCH_QEMU_CPU}")
endif ()

# Search paths for libraries and headers (target system only).
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
