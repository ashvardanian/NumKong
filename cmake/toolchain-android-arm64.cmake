# Android ARM64 toolchain for NumKong.
#
# Usage:
#   export ANDROID_NDK_ROOT=/path/to/android-ndk-r29
#   cmake -B build_android -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-android-arm64.cmake \
#         -D NUMKONG_BUILD_SHARED=ON
#   cmake --build build_android
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

# Defaults for ARM64 Android.
set(ANDROID_ABI "arm64-v8a" CACHE STRING "Android ABI")
set(ANDROID_PLATFORM "android-28" CACHE STRING "Minimum Android API level")

# Delegate to the NDK's own toolchain.
include("${_ndk_root}/build/cmake/android.toolchain.cmake")
