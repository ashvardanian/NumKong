/**
 *  @file c/dispatch/sparse.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief The sparse capability lists, @c _best dispatch points and @c nk_sparse_find_kernel.
 */
#include "dispatch.h"
#include "numkong/sparse.h"

static nk_capability_kernels_t const *nk_sparse_intersect_u16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_sparse_intersect_u16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sparse_intersect_u16_neon,
#endif
#if NUMKONG_TARGET_SVE2
        (nk_kernel_punned_t)&nk_sparse_intersect_u16_sve2,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_sparse_intersect_u16_icelake,
#endif
#if NUMKONG_TARGET_TURIN
        (nk_kernel_punned_t)&nk_sparse_intersect_u16_turin,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve2_k * NUMKONG_TARGET_SVE2 |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_turin_k * NUMKONG_TARGET_TURIN,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sparse_intersect_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u16_t *result, nk_size_t *count,
                                                     nk_capability_t capabilities, void *stream) {
    nk_sparse_intersect_punned_t const kernel = (nk_sparse_intersect_punned_t)nk_kernel_pick_(
        capabilities, nk_sparse_intersect_u16_capabilities());
    return kernel ? kernel(a, b, a_length, b_length, result, count, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sparse_intersect_u32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_sparse_intersect_u32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sparse_intersect_u32_neon,
#endif
#if NUMKONG_TARGET_SVE2
        (nk_kernel_punned_t)&nk_sparse_intersect_u32_sve2,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_sparse_intersect_u32_icelake,
#endif
#if NUMKONG_TARGET_TURIN
        (nk_kernel_punned_t)&nk_sparse_intersect_u32_turin,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve2_k * NUMKONG_TARGET_SVE2 |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_turin_k * NUMKONG_TARGET_TURIN,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sparse_intersect_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u32_t *result, nk_size_t *count,
                                                     nk_capability_t capabilities, void *stream) {
    nk_sparse_intersect_punned_t const kernel = (nk_sparse_intersect_punned_t)nk_kernel_pick_(
        capabilities, nk_sparse_intersect_u32_capabilities());
    return kernel ? kernel(a, b, a_length, b_length, result, count, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sparse_intersect_u64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_sparse_intersect_u64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sparse_intersect_u64_neon,
#endif
#if NUMKONG_TARGET_SVE2
        (nk_kernel_punned_t)&nk_sparse_intersect_u64_sve2,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_sparse_intersect_u64_icelake,
#endif
#if NUMKONG_TARGET_TURIN
        (nk_kernel_punned_t)&nk_sparse_intersect_u64_turin,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve2_k * NUMKONG_TARGET_SVE2 |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_turin_k * NUMKONG_TARGET_TURIN,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sparse_intersect_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u64_t *result, nk_size_t *count,
                                                     nk_capability_t capabilities, void *stream) {
    nk_sparse_intersect_punned_t const kernel = (nk_sparse_intersect_punned_t)nk_kernel_pick_(
        capabilities, nk_sparse_intersect_u64_capabilities());
    return kernel ? kernel(a, b, a_length, b_length, result, count, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sparse_dot_u16bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_sparse_dot_u16bf16_serial,
#if NUMKONG_TARGET_SVE2
        (nk_kernel_punned_t)&nk_sparse_dot_u16bf16_sve2,
#endif
#if NUMKONG_TARGET_TURIN
        (nk_kernel_punned_t)&nk_sparse_dot_u16bf16_turin,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_sve2_k * NUMKONG_TARGET_SVE2 | nk_cap_turin_k * NUMKONG_TARGET_TURIN, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sparse_dot_u16bf16_best(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                                   nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                   nk_f32_t *product, nk_capability_t capabilities, void *stream) {
    nk_sparse_dot_punned_t const kernel = (nk_sparse_dot_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_sparse_dot_u16bf16_capabilities());
    return kernel ? kernel(a, b, a_weights, b_weights, a_length, b_length, product, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sparse_dot_u32f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_sparse_dot_u32f32_serial,
#if NUMKONG_TARGET_SVE2
        (nk_kernel_punned_t)&nk_sparse_dot_u32f32_sve2,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sparse_dot_u32f32_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_sparse_dot_u32f32_icelake,
#endif
#if NUMKONG_TARGET_TURIN
        (nk_kernel_punned_t)&nk_sparse_dot_u32f32_turin,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_sve2_k * NUMKONG_TARGET_SVE2 | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_turin_k * NUMKONG_TARGET_TURIN,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sparse_dot_u32f32_best(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                                  nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                  nk_f64_t *product, nk_capability_t capabilities, void *stream) {
    nk_sparse_dot_punned_t const kernel = (nk_sparse_dot_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_sparse_dot_u32f32_capabilities());
    return kernel ? kernel(a, b, a_weights, b_weights, a_length, b_length, product, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_sparse_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_sparse_dot_k: lists = nk_sparse_dot_u32f32_capabilities(); break;
        default: break;
        }
        break;
    case nk_bf16_k:
        switch (kind) {
        case nk_kernel_sparse_dot_k: lists = nk_sparse_dot_u16bf16_capabilities(); break;
        default: break;
        }
        break;
    case nk_u64_k:
        switch (kind) {
        case nk_kernel_sparse_intersect_k: lists = nk_sparse_intersect_u64_capabilities(); break;
        default: break;
        }
        break;
    case nk_u32_k:
        switch (kind) {
        case nk_kernel_sparse_intersect_k: lists = nk_sparse_intersect_u32_capabilities(); break;
        default: break;
        }
        break;
    case nk_u16_k:
        switch (kind) {
        case nk_kernel_sparse_intersect_k: lists = nk_sparse_intersect_u16_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
