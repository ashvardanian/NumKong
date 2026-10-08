/**
 *  @file c/dispatch/mesh.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Dispatch points of the point-cloud alignments: RMSD, Kabsch and Umeyama.
 */
#include "dispatch.h"

static nk_capability_kernels_t const *nk_rmsd_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_rmsd_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_rmsd_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_rmsd_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_rmsd_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_rmsd_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_rmsd_f64_v128relaxed,
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

NUMKONG_API nk_status_t nk_rmsd_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                         nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                         nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_rmsd_f64_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_rmsd_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_rmsd_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_rmsd_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_rmsd_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_rmsd_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_rmsd_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_rmsd_f32_v128relaxed,
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

NUMKONG_API nk_status_t nk_rmsd_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                         nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_rmsd_f32_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_rmsd_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_rmsd_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_rmsd_f16_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_rmsd_f16_neonfhm,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_rmsd_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_rmsd_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_rmsd_f16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_rmsd_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                         nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_rmsd_f16_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_rmsd_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_rmsd_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_rmsd_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_rmsd_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_rmsd_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_rmsd_bf16_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_rmsd_bf16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_rmsd_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                          nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                          nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_rmsd_bf16_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_kabsch_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_kabsch_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_kabsch_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_kabsch_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_kabsch_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_kabsch_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_kabsch_f64_v128relaxed,
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

NUMKONG_API nk_status_t nk_kabsch_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                           nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_kabsch_f64_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_kabsch_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_kabsch_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_kabsch_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_kabsch_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_kabsch_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_kabsch_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_kabsch_f32_v128relaxed,
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

NUMKONG_API nk_status_t nk_kabsch_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_kabsch_f32_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_kabsch_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_kabsch_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_kabsch_f16_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_kabsch_f16_neonfhm,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_kabsch_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_kabsch_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_kabsch_f16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_kabsch_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                           nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_kabsch_f16_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_kabsch_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_kabsch_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_kabsch_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_kabsch_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_kabsch_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_kabsch_bf16_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_kabsch_bf16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_kabsch_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_kabsch_bf16_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_umeyama_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_umeyama_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_umeyama_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_umeyama_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_umeyama_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_umeyama_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_umeyama_f64_v128relaxed,
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

NUMKONG_API nk_status_t nk_umeyama_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                            nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_umeyama_f64_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_umeyama_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_umeyama_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_umeyama_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_umeyama_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_umeyama_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_umeyama_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_umeyama_f32_v128relaxed,
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

NUMKONG_API nk_status_t nk_umeyama_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_umeyama_f32_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_umeyama_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_umeyama_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_umeyama_f16_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_umeyama_f16_neonfhm,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_umeyama_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_umeyama_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_umeyama_f16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_umeyama_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                            nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_umeyama_f16_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_umeyama_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_umeyama_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_umeyama_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_umeyama_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_umeyama_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_umeyama_bf16_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_umeyama_bf16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_umeyama_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                             nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                             nk_f32_t *result, nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_mesh_punned_t const kernel = (nk_metric_mesh_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_umeyama_bf16_capabilities());
    return kernel ? kernel(a, b, n, a_centroid, b_centroid, rotation, scale, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_mesh_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                            nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f64_k:
        switch (kind) {
        case nk_kernel_rmsd_k: lists = nk_rmsd_f64_capabilities(); break;
        case nk_kernel_kabsch_k: lists = nk_kabsch_f64_capabilities(); break;
        case nk_kernel_umeyama_k: lists = nk_umeyama_f64_capabilities(); break;
        default: break;
        }
        break;
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_rmsd_k: lists = nk_rmsd_f32_capabilities(); break;
        case nk_kernel_kabsch_k: lists = nk_kabsch_f32_capabilities(); break;
        case nk_kernel_umeyama_k: lists = nk_umeyama_f32_capabilities(); break;
        default: break;
        }
        break;
    case nk_bf16_k:
        switch (kind) {
        case nk_kernel_rmsd_k: lists = nk_rmsd_bf16_capabilities(); break;
        case nk_kernel_kabsch_k: lists = nk_kabsch_bf16_capabilities(); break;
        case nk_kernel_umeyama_k: lists = nk_umeyama_bf16_capabilities(); break;
        default: break;
        }
        break;
    case nk_f16_k:
        switch (kind) {
        case nk_kernel_rmsd_k: lists = nk_rmsd_f16_capabilities(); break;
        case nk_kernel_kabsch_k: lists = nk_kabsch_f16_capabilities(); break;
        case nk_kernel_umeyama_k: lists = nk_umeyama_f16_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
