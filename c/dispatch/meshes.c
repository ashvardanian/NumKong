/**
 *  @file c/dispatch/meshes.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief The shape capability lists, @c _best dispatch points and @c nk_shapes_find_kernel.
 */
#include "dispatch.h"
#include "numkong/meshes.h"

static nk_capability_kernels_t const *nk_shapes_pack_size_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_pack_size_f16_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_pack_size_f16_best(nk_size_t points, nk_capability_t capabilities, nk_size_t *bytes) {
    nk_shapes_pack_size_punned_t const kernel = (nk_shapes_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_pack_size_f16_capabilities());
    return kernel ? kernel(points, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_pack_size_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_pack_size_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_pack_size_f32_best(nk_size_t points, nk_capability_t capabilities, nk_size_t *bytes) {
    nk_shapes_pack_size_punned_t const kernel = (nk_shapes_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_pack_size_f32_capabilities());
    return kernel ? kernel(points, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_pack_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_pack_f16_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_pack_f16_best(nk_f16_t const *points, nk_size_t count, void *packed,
                                                nk_capability_t capabilities, nk_stream_t stream) {
    nk_shapes_pack_punned_t const kernel = (nk_shapes_pack_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_shapes_pack_f16_capabilities());
    return kernel ? kernel(points, count, packed, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_pack_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_pack_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_pack_f32_best(nk_f32_t const *points, nk_size_t count, void *packed,
                                                nk_capability_t capabilities, nk_stream_t stream) {
    nk_shapes_pack_punned_t const kernel = (nk_shapes_pack_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_shapes_pack_f32_capabilities());
    return kernel ? kernel(points, count, packed, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_pack_normalized_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_pack_normalized_f16_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f16_best(nk_f16_t const *points, nk_size_t count, void *packed,
                                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_shapes_pack_punned_t const kernel = (nk_shapes_pack_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_pack_normalized_f16_capabilities());
    return kernel ? kernel(points, count, packed, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_pack_normalized_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_pack_normalized_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f32_best(nk_f32_t const *points, nk_size_t count, void *packed,
                                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_shapes_pack_punned_t const kernel = (nk_shapes_pack_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_pack_normalized_f32_capabilities());
    return kernel ? kernel(points, count, packed, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_chamfers_packed_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f16_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f16_neonfhm,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_chamfers_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                           void const *const *b_packed, nk_size_t columns,
                                                           nk_f32_t *results, nk_size_t results_stride,
                                                           nk_f32_t *rotations, nk_capability_t capabilities,
                                                           nk_stream_t stream) {
    nk_shapes_packed_punned_t const kernel = (nk_shapes_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_chamfers_packed_f16_capabilities());
    return kernel ? kernel(a_packed, rows, b_packed, columns, results, results_stride, rotations, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_chamfers_packed_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_shapes_chamfers_packed_f32_v128relaxed,
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

NUMKONG_API nk_status_t nk_shapes_chamfers_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                           void const *const *b_packed, nk_size_t columns,
                                                           nk_f64_t *results, nk_size_t results_stride,
                                                           nk_f32_t *rotations, nk_capability_t capabilities,
                                                           nk_stream_t stream) {
    nk_shapes_packed_punned_t const kernel = (nk_shapes_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_chamfers_packed_f32_capabilities());
    return kernel ? kernel(a_packed, rows, b_packed, columns, results, results_stride, rotations, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_hausdorffs_packed_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f16_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f16_neonfhm,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_hausdorffs_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                             void const *const *b_packed, nk_size_t columns,
                                                             nk_f32_t *results, nk_size_t results_stride,
                                                             nk_f32_t *rotations, nk_capability_t capabilities,
                                                             nk_stream_t stream) {
    nk_shapes_packed_punned_t const kernel = (nk_shapes_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_hausdorffs_packed_f16_capabilities());
    return kernel ? kernel(a_packed, rows, b_packed, columns, results, results_stride, rotations, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_hausdorffs_packed_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_shapes_hausdorffs_packed_f32_v128relaxed,
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

NUMKONG_API nk_status_t nk_shapes_hausdorffs_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                             void const *const *b_packed, nk_size_t columns,
                                                             nk_f64_t *results, nk_size_t results_stride,
                                                             nk_f32_t *rotations, nk_capability_t capabilities,
                                                             nk_stream_t stream) {
    nk_shapes_packed_punned_t const kernel = (nk_shapes_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_hausdorffs_packed_f32_capabilities());
    return kernel ? kernel(a_packed, rows, b_packed, columns, results, results_stride, rotations, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_wassersteinprocrustes_workspace_size_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_workspace_size_f16_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_wassersteinprocrustes_workspace_size_f16_best(nk_size_t row_points,
                                                                                nk_size_t column_points,
                                                                                nk_capability_t capabilities,
                                                                                nk_size_t *bytes) {
    nk_shapes_workspace_size_punned_t const kernel = (nk_shapes_workspace_size_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_wassersteinprocrustes_workspace_size_f16_capabilities());
    return kernel ? kernel(row_points, column_points, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_wassersteinprocrustes_workspace_size_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_workspace_size_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_wassersteinprocrustes_workspace_size_f32_best(nk_size_t row_points,
                                                                                nk_size_t column_points,
                                                                                nk_capability_t capabilities,
                                                                                nk_size_t *bytes) {
    nk_shapes_workspace_size_punned_t const kernel = (nk_shapes_workspace_size_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_wassersteinprocrustes_workspace_size_f32_capabilities());
    return kernel ? kernel(row_points, column_points, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_wassersteinprocrustes_packed_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f16_v128relaxed,
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

NUMKONG_API nk_status_t nk_shapes_wassersteinprocrustes_packed_f16_best(
    void const *const *a_packed, nk_size_t rows, void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
    nk_size_t results_stride, nk_f32_t *rotations, void *workspace, nk_size_t workspace_bytes,
    nk_capability_t capabilities, nk_stream_t stream) {
    nk_shapes_packed_workspace_punned_t const kernel = (nk_shapes_packed_workspace_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_wassersteinprocrustes_packed_f16_capabilities());
    return kernel ? kernel(a_packed, rows, b_packed, columns, results, results_stride, rotations, workspace,
                           workspace_bytes, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_wassersteinprocrustes_packed_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_shapes_wassersteinprocrustes_packed_f32_v128relaxed,
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

NUMKONG_API nk_status_t nk_shapes_wassersteinprocrustes_packed_f32_best(
    void const *const *a_packed, nk_size_t rows, void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
    nk_size_t results_stride, nk_f32_t *rotations, void *workspace, nk_size_t workspace_bytes,
    nk_capability_t capabilities, nk_stream_t stream) {
    nk_shapes_packed_workspace_punned_t const kernel = (nk_shapes_packed_workspace_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_wassersteinprocrustes_packed_f32_capabilities());
    return kernel ? kernel(a_packed, rows, b_packed, columns, results, results_stride, rotations, workspace,
                           workspace_bytes, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_gromovwassersteins_workspace_size_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_workspace_size_f16_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_gromovwassersteins_workspace_size_f16_best(nk_size_t row_points,
                                                                             nk_size_t column_points,
                                                                             nk_capability_t capabilities,
                                                                             nk_size_t *bytes) {
    nk_shapes_workspace_size_punned_t const kernel = (nk_shapes_workspace_size_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_gromovwassersteins_workspace_size_f16_capabilities());
    return kernel ? kernel(row_points, column_points, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_gromovwassersteins_workspace_size_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_workspace_size_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_shapes_gromovwassersteins_workspace_size_f32_best(nk_size_t row_points,
                                                                             nk_size_t column_points,
                                                                             nk_capability_t capabilities,
                                                                             nk_size_t *bytes) {
    nk_shapes_workspace_size_punned_t const kernel = (nk_shapes_workspace_size_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_gromovwassersteins_workspace_size_f32_capabilities());
    return kernel ? kernel(row_points, column_points, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_gromovwassersteins_packed_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f16_v128relaxed,
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

NUMKONG_API nk_status_t nk_shapes_gromovwassersteins_packed_f16_best(void const *const *a_packed, nk_size_t rows,
                                                                     void const *const *b_packed, nk_size_t columns,
                                                                     nk_f32_t *results, nk_size_t results_stride,
                                                                     nk_f32_t *rotations, void *workspace,
                                                                     nk_size_t workspace_bytes,
                                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_shapes_packed_workspace_punned_t const kernel = (nk_shapes_packed_workspace_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_gromovwassersteins_packed_f16_capabilities());
    return kernel ? kernel(a_packed, rows, b_packed, columns, results, results_stride, rotations, workspace,
                           workspace_bytes, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_shapes_gromovwassersteins_packed_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_shapes_gromovwassersteins_packed_f32_v128relaxed,
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

NUMKONG_API nk_status_t nk_shapes_gromovwassersteins_packed_f32_best(void const *const *a_packed, nk_size_t rows,
                                                                     void const *const *b_packed, nk_size_t columns,
                                                                     nk_f64_t *results, nk_size_t results_stride,
                                                                     nk_f32_t *rotations, void *workspace,
                                                                     nk_size_t workspace_bytes,
                                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_shapes_packed_workspace_punned_t const kernel = (nk_shapes_packed_workspace_punned_t)nk_kernel_pick_(
        capabilities, nk_shapes_gromovwassersteins_packed_f32_capabilities());
    return kernel ? kernel(a_packed, rows, b_packed, columns, results, results_stride, rotations, workspace,
                           workspace_bytes, stream)
                  : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_shapes_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f16_k:
        switch (kind) {
        case nk_kernel_shapes_pack_size_k: lists = nk_shapes_pack_size_f16_capabilities(); break;
        case nk_kernel_shapes_pack_k: lists = nk_shapes_pack_f16_capabilities(); break;
        case nk_kernel_shapes_pack_normalized_k: lists = nk_shapes_pack_normalized_f16_capabilities(); break;
        case nk_kernel_shapes_chamfers_packed_k: lists = nk_shapes_chamfers_packed_f16_capabilities(); break;
        case nk_kernel_shapes_hausdorffs_packed_k: lists = nk_shapes_hausdorffs_packed_f16_capabilities(); break;
        case nk_kernel_shapes_wassersteinprocrustes_workspace_size_k:
            lists = nk_shapes_wassersteinprocrustes_workspace_size_f16_capabilities();
            break;
        case nk_kernel_shapes_wassersteinprocrustes_packed_k:
            lists = nk_shapes_wassersteinprocrustes_packed_f16_capabilities();
            break;
        case nk_kernel_shapes_gromovwassersteins_workspace_size_k:
            lists = nk_shapes_gromovwassersteins_workspace_size_f16_capabilities();
            break;
        case nk_kernel_shapes_gromovwassersteins_packed_k:
            lists = nk_shapes_gromovwassersteins_packed_f16_capabilities();
            break;
        default: break;
        }
        break;
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_shapes_pack_size_k: lists = nk_shapes_pack_size_f32_capabilities(); break;
        case nk_kernel_shapes_pack_k: lists = nk_shapes_pack_f32_capabilities(); break;
        case nk_kernel_shapes_pack_normalized_k: lists = nk_shapes_pack_normalized_f32_capabilities(); break;
        case nk_kernel_shapes_chamfers_packed_k: lists = nk_shapes_chamfers_packed_f32_capabilities(); break;
        case nk_kernel_shapes_hausdorffs_packed_k: lists = nk_shapes_hausdorffs_packed_f32_capabilities(); break;
        case nk_kernel_shapes_wassersteinprocrustes_workspace_size_k:
            lists = nk_shapes_wassersteinprocrustes_workspace_size_f32_capabilities();
            break;
        case nk_kernel_shapes_wassersteinprocrustes_packed_k:
            lists = nk_shapes_wassersteinprocrustes_packed_f32_capabilities();
            break;
        case nk_kernel_shapes_gromovwassersteins_workspace_size_k:
            lists = nk_shapes_gromovwassersteins_workspace_size_f32_capabilities();
            break;
        case nk_kernel_shapes_gromovwassersteins_packed_k:
            lists = nk_shapes_gromovwassersteins_packed_f32_capabilities();
            break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
