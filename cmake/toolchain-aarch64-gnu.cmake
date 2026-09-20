# AArch64 GNU toolchain for NumKong.
#
# Two toolchain layouts are supported, selected by whether `AARCH64_TOOLCHAIN_PATH` is given.
#
#   Distribution cross packages, the default and what CI uses:
#     sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross qemu-user
#     cmake -B build_aarch64 -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-gnu.cmake
#
#   A self-contained toolchain carrying its own rootfs, e.g. the Arm GNU Toolchain, whose triple
#   and sysroot layout differ from the distribution's and so are named explicitly:
#     cmake -B build_aarch64 -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-gnu.cmake \
#           -D AARCH64_TOOLCHAIN_PATH=/opt/arm-gnu \
#           -D AARCH64_TRIPLE=aarch64-none-linux-gnu \
#           -D AARCH64_SYSROOT=/opt/arm-gnu/aarch64-none-linux-gnu/libc
#
# Optional inputs:
#   -D AARCH64_TOOLCHAIN_PATH=/opt/aarch64 # selects the self-contained layout
#   -D AARCH64_TRIPLE=aarch64-linux-gnu    # binary prefix and multiarch directory
#   -D AARCH64_COMPILER_SUFFIX=-14         # selects `aarch64-linux-gnu-gcc-14`
#   -D AARCH64_SYSROOT=/opt/aarch64/sysroot
#   -D AARCH64_QEMU_LD_PREFIX=/usr/aarch64-linux-gnu
#   -D AARCH64_QEMU_CPU=max
#
# Testing with QEMU:
#   Tests will automatically run under QEMU via CMAKE_CROSSCOMPILING_EMULATOR

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if (NOT DEFINED AARCH64_TOOLCHAIN_PATH AND DEFINED ENV{AARCH64_TOOLCHAIN_PATH})
    set(AARCH64_TOOLCHAIN_PATH "$ENV{AARCH64_TOOLCHAIN_PATH}")
endif ()
if (DEFINED AARCH64_TOOLCHAIN_PATH)
    set(AARCH64_TOOLCHAIN_PATH "${AARCH64_TOOLCHAIN_PATH}" CACHE PATH "Self-contained AArch64 toolchain root")
    set(ENV{AARCH64_TOOLCHAIN_PATH} "${AARCH64_TOOLCHAIN_PATH}")
endif ()

if (NOT DEFINED AARCH64_TRIPLE)
    if (DEFINED ENV{AARCH64_TRIPLE})
        set(AARCH64_TRIPLE "$ENV{AARCH64_TRIPLE}")
    else ()
        set(AARCH64_TRIPLE "aarch64-linux-gnu")
    endif ()
endif ()
set(AARCH64_TRIPLE "${AARCH64_TRIPLE}" CACHE STRING "AArch64 target triple, used as the compiler prefix")
set(ENV{AARCH64_TRIPLE} "${AARCH64_TRIPLE}")

# Debian versions its cross compilers as `<triple>-gcc-14` and only provides an unsuffixed
# `<triple>-gcc` when the unversioned metapackage is installed, which CI does not install.
if (NOT DEFINED AARCH64_COMPILER_SUFFIX)
    if (DEFINED ENV{AARCH64_COMPILER_SUFFIX})
        set(AARCH64_COMPILER_SUFFIX "$ENV{AARCH64_COMPILER_SUFFIX}")
    else ()
        set(AARCH64_COMPILER_SUFFIX "")
    endif ()
endif ()
set(AARCH64_COMPILER_SUFFIX "${AARCH64_COMPILER_SUFFIX}" CACHE STRING
    "Version suffix on the cross compiler, e.g. `-14`"
)
set(ENV{AARCH64_COMPILER_SUFFIX} "${AARCH64_COMPILER_SUFFIX}")

# Only a self-contained rootfs is a sysroot. Distribution cross packages install into the build
# root, so the sysroot is `/` — the compiler's default — and setting it buys nothing.
if (NOT DEFINED AARCH64_SYSROOT)
    if (DEFINED ENV{AARCH64_SYSROOT})
        set(AARCH64_SYSROOT "$ENV{AARCH64_SYSROOT}")
    elseif (DEFINED AARCH64_TOOLCHAIN_PATH)
        set(AARCH64_SYSROOT "${AARCH64_TOOLCHAIN_PATH}/sysroot")
    endif ()
endif ()
if (DEFINED AARCH64_SYSROOT)
    set(AARCH64_SYSROOT "${AARCH64_SYSROOT}" CACHE PATH "Self-contained aarch64 rootfs")
    set(ENV{AARCH64_SYSROOT} "${AARCH64_SYSROOT}")
endif ()

# `qemu-user`'s `-L` is the guest *interpreter* prefix, not a sysroot: it is where `ld-linux-aarch64.so.1` and
# the guest shared libraries live. With distribution packages that is `/usr/<triple>` even though
# the compiler sysroot is `/`, and `qemu-aarch64 -L /` fails to open the loader.
if (NOT DEFINED AARCH64_QEMU_LD_PREFIX)
    if (DEFINED ENV{QEMU_LD_PREFIX})
        set(AARCH64_QEMU_LD_PREFIX "$ENV{QEMU_LD_PREFIX}")
    elseif (DEFINED AARCH64_SYSROOT)
        set(AARCH64_QEMU_LD_PREFIX "${AARCH64_SYSROOT}")
    else ()
        set(AARCH64_QEMU_LD_PREFIX "/usr/${AARCH64_TRIPLE}")
    endif ()
endif ()
set(AARCH64_QEMU_LD_PREFIX "${AARCH64_QEMU_LD_PREFIX}" CACHE PATH "Guest loader prefix for `qemu-aarch64 -L`")
set(ENV{AARCH64_QEMU_LD_PREFIX} "${AARCH64_QEMU_LD_PREFIX}")

# `max` enables every extension QEMU implements for this target.
if (NOT DEFINED AARCH64_QEMU_CPU)
    if (DEFINED ENV{AARCH64_QEMU_CPU})
        set(AARCH64_QEMU_CPU "$ENV{AARCH64_QEMU_CPU}")
    else ()
        set(AARCH64_QEMU_CPU "max")
    endif ()
endif ()
set(AARCH64_QEMU_CPU "${AARCH64_QEMU_CPU}" CACHE STRING "CPU model for `qemu-aarch64 -cpu`")
set(ENV{AARCH64_QEMU_CPU} "${AARCH64_QEMU_CPU}")

# Forward settings to nested try_compile() invocations.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES AARCH64_TOOLCHAIN_PATH AARCH64_TRIPLE AARCH64_COMPILER_SUFFIX
     AARCH64_SYSROOT AARCH64_QEMU_LD_PREFIX AARCH64_QEMU_CPU)

# Toolchain validation only needs to prove the project compiles.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if (DEFINED AARCH64_TOOLCHAIN_PATH)
    set(_NK_AARCH64_PREFIX "${AARCH64_TOOLCHAIN_PATH}/bin/${AARCH64_TRIPLE}-")
else ()
    set(_NK_AARCH64_PREFIX "${AARCH64_TRIPLE}-")
endif ()
set(CMAKE_C_COMPILER "${_NK_AARCH64_PREFIX}gcc${AARCH64_COMPILER_SUFFIX}")
set(CMAKE_CXX_COMPILER "${_NK_AARCH64_PREFIX}g++${AARCH64_COMPILER_SUFFIX}")

# No `-march` here: `CMakeLists.txt` pins the dispatch floor with
# `add_compile_options(-march=armv8-a)`, which lands after `CMAKE_C_FLAGS` and wins, and the
# per-ISA kernels are gated by `nk_arm_isa_probes.cmake` per translation unit.

if (DEFINED AARCH64_SYSROOT)
    set(CMAKE_SYSROOT "${AARCH64_SYSROOT}")
    set(CMAKE_FIND_ROOT_PATH "${AARCH64_SYSROOT}")
endif ()

find_program(_NK_QEMU_AARCH64 qemu-aarch64)
if (_NK_QEMU_AARCH64)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${_NK_QEMU_AARCH64};-L;${AARCH64_QEMU_LD_PREFIX};-cpu;${AARCH64_QEMU_CPU}")
endif ()

# Search paths for libraries and headers (target system only).
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
