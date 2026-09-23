# cmake/nk_cudnn.cmake — cuDNN from a tarball or package root
#
# Defines the `nk_cudnn` interface target from `NK_CUDNN_ROOT`, holding include/cudnn.h and lib/libcudnn.so*.

include_guard(GLOBAL)

find_path(NK_CUDNN_INCLUDE_DIR cudnn.h PATHS ${NK_CUDNN_ROOT} PATH_SUFFIXES include REQUIRED)
find_library(NK_CUDNN_LIBRARY NAMES cudnn libcudnn.so.9 PATHS ${NK_CUDNN_ROOT} PATH_SUFFIXES lib lib64 REQUIRED)

add_library(nk_cudnn INTERFACE)
target_include_directories(nk_cudnn SYSTEM INTERFACE "${NK_CUDNN_INCLUDE_DIR}")
target_link_libraries(nk_cudnn INTERFACE "${NK_CUDNN_LIBRARY}")
