/**
 *  @file c/dispatch/probability.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Dispatch points of the probability divergences: Kullback-Leibler and Jensen-Shannon.
 */
#include "dispatch.h"

static nk_capability_kernels_t const *nk_kld_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_kld_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_kld_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_kld_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_kld_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_kld_f16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_kld_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_kld_f16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_kld_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_kld_bf16_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_kld_bf16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_kld_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                         nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_kld_bf16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_kld_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_kld_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_kld_f32_neon,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_kld_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_kld_f32_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_kld_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_kld_f32_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_kld_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_kld_f64_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_kld_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_kld_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_kld_f64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_kld_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_kld_f64_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_jsd_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_jsd_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_jsd_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_jsd_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_jsd_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_jsd_f16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_jsd_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_jsd_f16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_jsd_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_jsd_bf16_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_jsd_bf16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_jsd_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                         nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_jsd_bf16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_jsd_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_jsd_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_jsd_f32_neon,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_jsd_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_jsd_f32_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_jsd_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_jsd_f32_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_jsd_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_jsd_f64_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_jsd_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_jsd_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_jsd_f64_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_jsd_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_jsd_f64_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_probability_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype,
                                                   nk_capability_t capabilities, nk_kernel_punned_t *kernel,
                                                   nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f64_k:
        switch (kind) {
        case nk_kernel_kld_k: lists = nk_kld_f64_capabilities(); break;
        case nk_kernel_jsd_k: lists = nk_jsd_f64_capabilities(); break;
        default: break;
        }
        break;
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_kld_k: lists = nk_kld_f32_capabilities(); break;
        case nk_kernel_jsd_k: lists = nk_jsd_f32_capabilities(); break;
        default: break;
        }
        break;
    case nk_bf16_k:
        switch (kind) {
        case nk_kernel_kld_k: lists = nk_kld_bf16_capabilities(); break;
        case nk_kernel_jsd_k: lists = nk_jsd_bf16_capabilities(); break;
        default: break;
        }
        break;
    case nk_f16_k:
        switch (kind) {
        case nk_kernel_kld_k: lists = nk_kld_f16_capabilities(); break;
        case nk_kernel_jsd_k: lists = nk_jsd_f16_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
