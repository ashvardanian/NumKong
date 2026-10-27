# cmake/numkong_cuvs.cmake — cuVS C API with the RAPIDS libraries it links
#
# Defines the `numkong_cuvs` interface target from the `NUMKONG_CUVS_ROOTS` prefixes, holding libcuvs_c, librmm,
# librapids_logger and the dlpack headers.

include_guard(GLOBAL)

find_path(NUMKONG_CUVS_INCLUDE_DIR cuvs/distance/pairwise_distance.h PATHS ${NUMKONG_CUVS_ROOTS} PATH_SUFFIXES include
                                                                                                               REQUIRED
)
find_path(NUMKONG_DLPACK_INCLUDE_DIR dlpack/dlpack.h PATHS ${NUMKONG_CUVS_ROOTS} PATH_SUFFIXES include REQUIRED)
find_library(NUMKONG_CUVS_LIBRARY cuvs_c PATHS ${NUMKONG_CUVS_ROOTS} PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NUMKONG_RMM_LIBRARY rmm PATHS ${NUMKONG_CUVS_ROOTS} PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NUMKONG_RAPIDS_LOGGER_LIBRARY rapids_logger PATHS ${NUMKONG_CUVS_ROOTS} PATH_SUFFIXES lib lib64 REQUIRED)

add_library(numkong_cuvs INTERFACE)
target_include_directories(numkong_cuvs SYSTEM INTERFACE "${NUMKONG_CUVS_INCLUDE_DIR}" "${NUMKONG_DLPACK_INCLUDE_DIR}")
target_link_libraries(
    numkong_cuvs INTERFACE "${NUMKONG_CUVS_LIBRARY}" "${NUMKONG_RMM_LIBRARY}" "${NUMKONG_RAPIDS_LOGGER_LIBRARY}"
)
