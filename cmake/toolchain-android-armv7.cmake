# Android ARMv7 toolchain for NumKong.
#
# Usage:
#   export ANDROID_NDK_ROOT=/path/to/android-ndk-r29
#   cmake -B build_android_armv7 -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-android-armv7.cmake \
#         -D NUMKONG_BUILD_SHARED=ON
#   cmake --build build_android_armv7
#
# Requires ANDROID_NDK_ROOT or ANDROID_NDK to point to the NDK.

# Locate the NDK.
if (DEFINED ENV{ANDROID_NDK_ROOT})
    set(_ndk_root "$ENV{ANDROID_NDK_ROOT}")
elseif (DEFINED ENV{ANDROID_NDK})
    set(_ndk_root "$ENV{ANDROID_NDK}")
else ()
    message(FATAL_ERROR "Set ANDROID_NDK_ROOT or ANDROID_NDK to the NDK installation path")
endif ()

# Defaults for 32-bit ARM Android with NEON.
set(ANDROID_ABI "armeabi-v7a" CACHE STRING "Android ABI")
set(ANDROID_PLATFORM "android-28" CACHE STRING "Minimum Android API level")

# Delegate to the NDK's own toolchain.
include("${_ndk_root}/build/cmake/android.toolchain.cmake")
