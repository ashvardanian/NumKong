/**
 *  @file c/dispatch/curved.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Dispatch points of the curved-space metrics: bilinear forms and Mahalanobis distances.
 */
#include "dispatch.h"

static nk_capability_kernels_t const *nk_bilinear_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_bilinear_f64_serial,
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_bilinear_f64_smef64,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_bilinear_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_bilinear_f64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_bilinear_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                             nk_f64_t *result, nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(capabilities,
                                                                                        nk_bilinear_f64_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_bilinear_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_bilinear_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_bilinear_f32_neon,
#endif
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_bilinear_f32_smef64,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_bilinear_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_bilinear_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_bilinear_f32_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_bilinear_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                             nk_f64_t *result, nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(capabilities,
                                                                                        nk_bilinear_f32_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_bilinear_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_bilinear_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_bilinear_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_bilinear_f16_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_bilinear_f16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_bilinear_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                             nk_f32_t *result, nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(capabilities,
                                                                                        nk_bilinear_f16_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_bilinear_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_bilinear_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_bilinear_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_bilinear_bf16_haswell,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_bilinear_bf16_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_bilinear_bf16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_bilinear_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                              nk_f32_t *result, nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(
        capabilities, nk_bilinear_bf16_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_mahalanobis_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_mahalanobis_f64_serial,
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_mahalanobis_f64_smef64,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_mahalanobis_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_mahalanobis_f64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_mahalanobis_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                                nk_f64_t *result, nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(
        capabilities, nk_mahalanobis_f64_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_mahalanobis_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_mahalanobis_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_mahalanobis_f32_neon,
#endif
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_mahalanobis_f32_smef64,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_mahalanobis_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_mahalanobis_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_mahalanobis_f32_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_mahalanobis_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                                nk_f64_t *result, nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(
        capabilities, nk_mahalanobis_f32_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_mahalanobis_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_mahalanobis_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_mahalanobis_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_mahalanobis_f16_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_mahalanobis_f16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_mahalanobis_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                                nk_f32_t *result, nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(
        capabilities, nk_mahalanobis_f16_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_mahalanobis_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_mahalanobis_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_mahalanobis_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_mahalanobis_bf16_haswell,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_mahalanobis_bf16_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_mahalanobis_bf16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_mahalanobis_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                 nk_size_t n, nk_f32_t *result, nk_capability_t capabilities,
                                                 nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(
        capabilities, nk_mahalanobis_bf16_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_bilinear_f64c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_bilinear_f64c_serial,
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_bilinear_f64c_smef64,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_bilinear_f64c_skylake,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_bilinear_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                              nk_f64c_t *result, nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(
        capabilities, nk_bilinear_f64c_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_bilinear_f32c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_bilinear_f32c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_bilinear_f32c_neon,
#endif
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_bilinear_f32c_smef64,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_bilinear_f32c_skylake,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_bilinear_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                              nk_f64c_t *result, nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(
        capabilities, nk_bilinear_f32c_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_bilinear_f16c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_bilinear_f16c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_bilinear_f16c_neon,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_bilinear_f16c_best(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                              nk_f32c_t *result, nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(
        capabilities, nk_bilinear_f16c_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_bilinear_bf16c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_bilinear_bf16c_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_bilinear_bf16c_neonbfdot,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_bilinear_bf16c_genoa,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_genoa_k * NUMKONG_TARGET_GENOA, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_bilinear_bf16c_best(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c,
                                               nk_size_t n, nk_f32c_t *result, nk_capability_t capabilities,
                                               nk_stream_t stream) {
    nk_metric_curved_punned_t const kernel = (nk_metric_curved_punned_t)nk_kernel_pick_(
        capabilities, nk_bilinear_bf16c_capabilities());
    return kernel ? kernel(a, b, c, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_curved_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f64c_k:
        switch (kind) {
        case nk_kernel_bilinear_k: lists = nk_bilinear_f64c_capabilities(); break;
        default: break;
        }
        break;
    case nk_f32c_k:
        switch (kind) {
        case nk_kernel_bilinear_k: lists = nk_bilinear_f32c_capabilities(); break;
        default: break;
        }
        break;
    case nk_bf16c_k:
        switch (kind) {
        case nk_kernel_bilinear_k: lists = nk_bilinear_bf16c_capabilities(); break;
        default: break;
        }
        break;
    case nk_f16c_k:
        switch (kind) {
        case nk_kernel_bilinear_k: lists = nk_bilinear_f16c_capabilities(); break;
        default: break;
        }
        break;
    case nk_f64_k:
        switch (kind) {
        case nk_kernel_bilinear_k: lists = nk_bilinear_f64_capabilities(); break;
        case nk_kernel_mahalanobis_k: lists = nk_mahalanobis_f64_capabilities(); break;
        default: break;
        }
        break;
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_bilinear_k: lists = nk_bilinear_f32_capabilities(); break;
        case nk_kernel_mahalanobis_k: lists = nk_mahalanobis_f32_capabilities(); break;
        default: break;
        }
        break;
    case nk_bf16_k:
        switch (kind) {
        case nk_kernel_bilinear_k: lists = nk_bilinear_bf16_capabilities(); break;
        case nk_kernel_mahalanobis_k: lists = nk_mahalanobis_bf16_capabilities(); break;
        default: break;
        }
        break;
    case nk_f16_k:
        switch (kind) {
        case nk_kernel_bilinear_k: lists = nk_bilinear_f16_capabilities(); break;
        case nk_kernel_mahalanobis_k: lists = nk_mahalanobis_f16_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
