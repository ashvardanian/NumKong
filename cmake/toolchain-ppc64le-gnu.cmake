# Power ppc64le GNU toolchain for NumKong, driving GCC 15.
#
# GCC 15 is the floor: its `#pragma GCC target("power9-vector")` opens the POWER9 half of `altivec.h` inside the POWER9
# kernels' region, so they compile beside the `power8` floor like every other capability. Clang never opens it
# without `-mcpu=power9`, so it leaves POWER9 off.
#
# Two toolchain layouts are supported, selected by whether `PPC_TOOLCHAIN_PATH` is given.
#
#   Distribution cross packages, the default and what CI uses:
#     sudo apt install gcc-15-powerpc64le-linux-gnu g++-15-powerpc64le-linux-gnu libc6-dev-ppc64el-cross qemu-user
#     cmake -B build_ppc -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-ppc64le-gnu.cmake
#
#   A self-contained toolchain carrying its own rootfs:
#     cmake -B build_ppc -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-ppc64le-gnu.cmake \
#           -D PPC_TOOLCHAIN_PATH=/opt/powerpc64le
#
# Optional inputs:
#   -D PPC_TOOLCHAIN_PATH=/opt/powerpc64le             # selects the self-contained layout
#   -D PPC_TRIPLE=powerpc64le-linux-gnu                # binary prefix and multiarch directory
#   -D PPC_COMPILER_SUFFIX=-15                         # selects `powerpc64le-linux-gnu-gcc-15`, the default
#   -D PPC_SYSROOT=/opt/powerpc64le/sysroot
#   -D PPC_QEMU_LD_PREFIX=/usr/powerpc64le-linux-gnu
#   -D PPC_QEMU_CPU=power10
#
# Testing with QEMU:
#   Tests will automatically run under QEMU via CMAKE_CROSSCOMPILING_EMULATOR, on `power10` by
#   default, which runs the POWER9 kernels, or on the POWER8 baseline with `-D PPC_QEMU_CPU=power8`.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)

if (NOT DEFINED PPC_TOOLCHAIN_PATH AND DEFINED ENV{PPC_TOOLCHAIN_PATH})
    set(PPC_TOOLCHAIN_PATH "$ENV{PPC_TOOLCHAIN_PATH}")
endif ()
if (DEFINED PPC_TOOLCHAIN_PATH)
    set(PPC_TOOLCHAIN_PATH "${PPC_TOOLCHAIN_PATH}" CACHE PATH "Self-contained Power GNU toolchain root")
    set(ENV{PPC_TOOLCHAIN_PATH} "${PPC_TOOLCHAIN_PATH}")
endif ()

if (NOT DEFINED PPC_TRIPLE)
    if (DEFINED ENV{PPC_TRIPLE})
        set(PPC_TRIPLE "$ENV{PPC_TRIPLE}")
    else ()
        set(PPC_TRIPLE "powerpc64le-linux-gnu")
    endif ()
endif ()
set(PPC_TRIPLE "${PPC_TRIPLE}" CACHE STRING "Power target triple, used as the compiler prefix")
set(ENV{PPC_TRIPLE} "${PPC_TRIPLE}")

# Ubuntu and Debian version their cross compilers as `<triple>-gcc-15`, and an unsuffixed `<triple>-gcc` is whatever
# the unversioned metapackage points at, so the suffix pins GCC 15; pass an empty one for an unsuffixed GCC 15.
if (DEFINED ENV{PPC_COMPILER_SUFFIX})
    set(PPC_COMPILER_SUFFIX_DEFAULT_ "$ENV{PPC_COMPILER_SUFFIX}")
else ()
    set(PPC_COMPILER_SUFFIX_DEFAULT_ "-15")
endif ()
set(PPC_COMPILER_SUFFIX "${PPC_COMPILER_SUFFIX_DEFAULT_}" CACHE STRING "Version suffix on the cross compiler")
set(ENV{PPC_COMPILER_SUFFIX} "${PPC_COMPILER_SUFFIX}")

# Only a self-contained rootfs is a sysroot. Distribution cross packages install into the build
# root, so the sysroot is `/` — the compiler's default — and setting it buys nothing.
if (NOT DEFINED PPC_SYSROOT)
    if (DEFINED ENV{PPC_SYSROOT})
        set(PPC_SYSROOT "$ENV{PPC_SYSROOT}")
    elseif (DEFINED PPC_TOOLCHAIN_PATH)
        set(PPC_SYSROOT "${PPC_TOOLCHAIN_PATH}/sysroot")
    endif ()
endif ()
if (DEFINED PPC_SYSROOT)
    set(PPC_SYSROOT "${PPC_SYSROOT}" CACHE PATH "Self-contained ppc64le rootfs")
    set(ENV{PPC_SYSROOT} "${PPC_SYSROOT}")
endif ()

# `qemu-user`'s `-L` is the guest *interpreter* prefix, not a sysroot: it is where `ld64.so.2` and
# the guest shared libraries live. With distribution packages that is `/usr/<triple>` even though
# the compiler sysroot is `/`, and `qemu-ppc64le -L /` fails to open the loader.
if (NOT DEFINED PPC_QEMU_LD_PREFIX)
    if (DEFINED ENV{QEMU_LD_PREFIX})
        set(PPC_QEMU_LD_PREFIX "$ENV{QEMU_LD_PREFIX}")
    elseif (DEFINED PPC_SYSROOT)
        set(PPC_QEMU_LD_PREFIX "${PPC_SYSROOT}")
    else ()
        set(PPC_QEMU_LD_PREFIX "/usr/${PPC_TRIPLE}")
    endif ()
endif ()
set(PPC_QEMU_LD_PREFIX "${PPC_QEMU_LD_PREFIX}" CACHE PATH "Guest loader prefix for `qemu-ppc64le -L`")
set(ENV{PPC_QEMU_LD_PREFIX} "${PPC_QEMU_LD_PREFIX}")

# Older QEMU lacks `-cpu max` for ppc64le, so the emulated core is named. `power10` is a superset
# of the POWER9 floor the `powervsx` kernels are probed against, so it exercises strictly more.
if (NOT DEFINED PPC_QEMU_CPU)
    if (DEFINED ENV{PPC_QEMU_CPU})
        set(PPC_QEMU_CPU "$ENV{PPC_QEMU_CPU}")
    else ()
        set(PPC_QEMU_CPU "power10")
    endif ()
endif ()
set(PPC_QEMU_CPU "${PPC_QEMU_CPU}" CACHE STRING "CPU model for `qemu-ppc64le -cpu`")
set(ENV{PPC_QEMU_CPU} "${PPC_QEMU_CPU}")

# Forward settings to nested try_compile() invocations.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES PPC_TOOLCHAIN_PATH PPC_TRIPLE PPC_COMPILER_SUFFIX PPC_SYSROOT
     PPC_QEMU_LD_PREFIX PPC_QEMU_CPU
)

# Toolchain validation only needs to prove the project compiles.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if (DEFINED PPC_TOOLCHAIN_PATH)
    set(_NUMKONG_PPC_PREFIX "${PPC_TOOLCHAIN_PATH}/bin/${PPC_TRIPLE}-")
else ()
    set(_NUMKONG_PPC_PREFIX "${PPC_TRIPLE}-")
endif ()
set(CMAKE_C_COMPILER "${_NUMKONG_PPC_PREFIX}gcc${PPC_COMPILER_SUFFIX}")
set(CMAKE_CXX_COMPILER "${_NUMKONG_PPC_PREFIX}g++${PPC_COMPILER_SUFFIX}")

# No `-mcpu` here: `CMakeLists.txt` pins the dispatch floor with `add_compile_options(-mcpu=power8)`,
# which lands after `CMAKE_C_FLAGS` and wins.
# Raising the floor here would SIGILL on POWER8.

if (DEFINED PPC_SYSROOT)
    set(CMAKE_SYSROOT "${PPC_SYSROOT}")
    set(CMAKE_FIND_ROOT_PATH "${PPC_SYSROOT}")
endif ()

find_program(_NUMKONG_QEMU_PPC64LE qemu-ppc64le)
if (_NUMKONG_QEMU_PPC64LE)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${_NUMKONG_QEMU_PPC64LE};-L;${PPC_QEMU_LD_PREFIX};-cpu;${PPC_QEMU_CPU}")
endif ()

# Search paths for libraries and headers (target system only).
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
