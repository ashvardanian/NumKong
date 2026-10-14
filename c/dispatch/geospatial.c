/**
 *  @file c/dispatch/geospatial.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Dispatch points of the geospatial kernels: 3D squares, Haversine, Vincenty and terrain.
 */
#include "dispatch.h"

static nk_capability_kernels_t const *nk_sqeuclideans3d_i32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclideans3d_i32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sqeuclideans3d_i32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclideans3d_i32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_sqeuclideans3d_i32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclideans3d_i32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclideans3d_i32_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_sqeuclideans3d_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclideans3d_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sqeuclideans3d_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclideans3d_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_sqeuclideans3d_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclideans3d_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclideans3d_f64_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_haversines_packed_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_haversines_packed_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_haversines_packed_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_haversines_packed_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_haversines_packed_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_haversines_packed_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_haversines_packed_f64_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_vincentys_packed_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_vincentys_packed_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_vincentys_packed_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_vincentys_packed_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_vincentys_packed_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_vincentys_packed_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_vincentys_packed_f64_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_haversines_pack_size_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_haversines_pack_size_f64_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_haversines_pack_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_haversines_pack_f64_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_vincentys_pack_size_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_vincentys_pack_size_f64_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_vincentys_pack_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_vincentys_pack_f64_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_terrain_pack_size_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_terrain_pack_size_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_terrain_pack_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_terrain_pack_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_terrain_heights_packed_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_terrain_heights_packed_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_clearances_packed_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_clearances_packed_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_best(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                   nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                   nk_capability_t capabilities, nk_stream_t stream) {
    nk_sqeuclideans3d_punned_t const kernel = (nk_sqeuclideans3d_punned_t)nk_kernel_pick_(
        capabilities, nk_sqeuclideans3d_i32_capabilities());
    return kernel ? kernel(a, rows, b, columns, results, results_stride, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_best(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                   nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                   nk_capability_t capabilities, nk_stream_t stream) {
    nk_sqeuclideans3d_punned_t const kernel = (nk_sqeuclideans3d_punned_t)nk_kernel_pick_(
        capabilities, nk_sqeuclideans3d_f64_capabilities());
    return kernel ? kernel(a, rows, b, columns, results, results_stride, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_haversines_pack_size_f64_best(nk_size_t points, nk_capability_t capabilities,
                                                         nk_size_t *bytes) {
    nk_geospatial_pack_size_punned_t const kernel = (nk_geospatial_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_haversines_pack_size_f64_capabilities());
    return kernel ? kernel(points, bytes) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_vincentys_pack_size_f64_best(nk_size_t points, nk_capability_t capabilities,
                                                        nk_size_t *bytes) {
    nk_geospatial_pack_size_punned_t const kernel = (nk_geospatial_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_vincentys_pack_size_f64_capabilities());
    return kernel ? kernel(points, bytes) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_haversines_pack_f64_best(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                    nk_size_t count, void *packed, nk_capability_t capabilities,
                                                    nk_stream_t stream) {
    nk_geospatial_pack_punned_t const kernel = (nk_geospatial_pack_punned_t)nk_kernel_pick_(
        capabilities, nk_haversines_pack_f64_capabilities());
    return kernel ? kernel(latitudes, longitudes, count, packed, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_vincentys_pack_f64_best(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                   nk_size_t count, void *packed, nk_capability_t capabilities,
                                                   nk_stream_t stream) {
    nk_geospatial_pack_punned_t const kernel = (nk_geospatial_pack_punned_t)nk_kernel_pick_(
        capabilities, nk_vincentys_pack_f64_capabilities());
    return kernel ? kernel(latitudes, longitudes, count, packed, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_haversines_packed_f64_best(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_capability_t capabilities,
                                                      nk_stream_t stream) {
    nk_geospatial_packed_punned_t const kernel = (nk_geospatial_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_haversines_packed_f64_capabilities());
    return kernel ? kernel(a_packed, b_packed, results, results_stride, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_vincentys_packed_f64_best(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                     nk_size_t results_stride, nk_capability_t capabilities,
                                                     nk_stream_t stream) {
    nk_geospatial_packed_punned_t const kernel = (nk_geospatial_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_vincentys_packed_f64_capabilities());
    return kernel ? kernel(a_packed, b_packed, results, results_stride, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_terrain_pack_size_f32_best(nk_size_t rows, nk_size_t columns, nk_capability_t capabilities,
                                                      nk_size_t *bytes) {
    nk_terrain_pack_size_punned_t const kernel = (nk_terrain_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_terrain_pack_size_f32_capabilities());
    return kernel ? kernel(rows, columns, bytes) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_terrain_pack_f32_best(nk_f32_t const *heights, nk_size_t rows, nk_size_t columns,
                                                 nk_size_t heights_stride, nk_f64_t row_metres, nk_f64_t column_metres,
                                                 void *packed, nk_capability_t capabilities, nk_stream_t stream) {
    nk_terrain_pack_punned_t const kernel = (nk_terrain_pack_punned_t)nk_kernel_pick_(
        capabilities, nk_terrain_pack_f32_capabilities());
    return kernel ? kernel(heights, rows, columns, heights_stride, row_metres, column_metres, packed, stream)
                  : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_terrain_heights_packed_f32_best(nk_f32_t const *heights, void const *packed,
                                                           nk_f32_t const *points, nk_size_t count, nk_f32_t *results,
                                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_terrain_heights_packed_punned_t const kernel = (nk_terrain_heights_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_terrain_heights_packed_f32_capabilities());
    return kernel ? kernel(heights, packed, points, count, results, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_clearances_packed_f32_best(nk_f32_t const *heights, void const *packed,
                                                      nk_f32_t const *viewpoints, nk_f32_t const *views, nk_size_t rows,
                                                      nk_f32_t const *targets, nk_size_t columns, nk_f64_t refraction,
                                                      nk_f32_t *results, nk_size_t results_stride,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_clearances_packed_punned_t const kernel = (nk_clearances_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_clearances_packed_f32_capabilities());
    return kernel ? kernel(heights, packed, viewpoints, views, rows, targets, columns, refraction, results,
                           results_stride, stream)
                  : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_geospatial_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                                  nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_i32_k:
        if (kind == nk_kernel_sqeuclideans3d_k) lists = nk_sqeuclideans3d_i32_capabilities();
        break;
    case nk_f64_k:
        switch (kind) {
        case nk_kernel_sqeuclideans3d_k: lists = nk_sqeuclideans3d_f64_capabilities(); break;
        case nk_kernel_haversines_pack_size_k: lists = nk_haversines_pack_size_f64_capabilities(); break;
        case nk_kernel_haversines_pack_k: lists = nk_haversines_pack_f64_capabilities(); break;
        case nk_kernel_haversines_packed_k: lists = nk_haversines_packed_f64_capabilities(); break;
        case nk_kernel_vincentys_pack_size_k: lists = nk_vincentys_pack_size_f64_capabilities(); break;
        case nk_kernel_vincentys_pack_k: lists = nk_vincentys_pack_f64_capabilities(); break;
        case nk_kernel_vincentys_packed_k: lists = nk_vincentys_packed_f64_capabilities(); break;
        default: break;
        }
        break;
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_terrain_pack_size_k: lists = nk_terrain_pack_size_f32_capabilities(); break;
        case nk_kernel_terrain_pack_k: lists = nk_terrain_pack_f32_capabilities(); break;
        case nk_kernel_terrain_heights_packed_k: lists = nk_terrain_heights_packed_f32_capabilities(); break;
        case nk_kernel_clearances_packed_k: lists = nk_clearances_packed_f32_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
