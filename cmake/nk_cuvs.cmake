# cmake/nk_cuvs.cmake — cuVS C API with the RAPIDS libraries it links
#
# Defines the `nk_cuvs` interface target from the `NK_CUVS_ROOTS` prefixes, holding libcuvs_c, librmm,
# librapids_logger and the dlpack headers.

include_guard(GLOBAL)

find_path(NK_CUVS_INCLUDE_DIR cuvs/distance/pairwise_distance.h PATHS ${NK_CUVS_ROOTS} PATH_SUFFIXES include REQUIRED)
find_path(NK_DLPACK_INCLUDE_DIR dlpack/dlpack.h PATHS ${NK_CUVS_ROOTS} PATH_SUFFIXES include REQUIRED)
find_library(NK_CUVS_LIBRARY cuvs_c PATHS ${NK_CUVS_ROOTS} PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NK_RMM_LIBRARY rmm PATHS ${NK_CUVS_ROOTS} PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NK_RAPIDS_LOGGER_LIBRARY rapids_logger PATHS ${NK_CUVS_ROOTS} PATH_SUFFIXES lib lib64 REQUIRED)

add_library(nk_cuvs INTERFACE)
target_include_directories(nk_cuvs SYSTEM INTERFACE "${NK_CUVS_INCLUDE_DIR}" "${NK_DLPACK_INCLUDE_DIR}")
target_link_libraries(nk_cuvs INTERFACE "${NK_CUVS_LIBRARY}" "${NK_RMM_LIBRARY}" "${NK_RAPIDS_LOGGER_LIBRARY}")
