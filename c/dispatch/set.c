/**
 *  @file c/dispatch/set.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief The set distance capability lists, @c _best dispatch points and @c nk_set_find_kernel.
 */
#include "dispatch.h"
#include "numkong/set.h"

static nk_capability_kernels_t const *nk_hamming_u1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_hamming_u1_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_hamming_u1_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_hamming_u1_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_hamming_u1_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_hamming_u1_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_hamming_u1_rvv,
#endif
#if NUMKONG_TARGET_RVVBB
        (nk_kernel_punned_t)&nk_hamming_u1_rvvbb,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_hamming_u1_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_hamming_u1_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_hamming_u1_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_rvvbb_k * NUMKONG_TARGET_RVVBB |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_hamming_u1_best(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                           nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_hamming_u1_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_jaccard_u1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_jaccard_u1_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_jaccard_u1_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_jaccard_u1_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_jaccard_u1_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_jaccard_u1_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_jaccard_u1_rvv,
#endif
#if NUMKONG_TARGET_RVVBB
        (nk_kernel_punned_t)&nk_jaccard_u1_rvvbb,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_jaccard_u1_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_jaccard_u1_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_jaccard_u1_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_rvvbb_k * NUMKONG_TARGET_RVVBB |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_jaccard_u1_best(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_jaccard_u1_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_jaccard_u32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_jaccard_u32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_jaccard_u32_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_jaccard_u32_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_jaccard_u32_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_jaccard_u32_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_jaccard_u32_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_jaccard_u32_v128,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_jaccard_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_jaccard_u32_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_hamming_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_hamming_u8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_hamming_u8_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_hamming_u8_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_hamming_u8_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_hamming_u8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_hamming_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_hamming_u8_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_hamming_u8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_hamming_u8_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_hamming_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                           nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_hamming_u8_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_jaccard_u16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_jaccard_u16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_jaccard_u16_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_jaccard_u16_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_jaccard_u16_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_jaccard_u16_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_jaccard_u16_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_jaccard_u16_v128,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_jaccard_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_jaccard_u16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_set_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                           nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_u32_k:
        switch (kind) {
        case nk_kernel_jaccard_k: lists = nk_jaccard_u32_capabilities(); break;
        default: break;
        }
        break;
    case nk_u16_k:
        switch (kind) {
        case nk_kernel_jaccard_k: lists = nk_jaccard_u16_capabilities(); break;
        default: break;
        }
        break;
    case nk_u8_k:
        switch (kind) {
        case nk_kernel_hamming_k: lists = nk_hamming_u8_capabilities(); break;
        default: break;
        }
        break;
    case nk_u1_k:
        switch (kind) {
        case nk_kernel_hamming_k: lists = nk_hamming_u1_capabilities(); break;
        case nk_kernel_jaccard_k: lists = nk_jaccard_u1_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
