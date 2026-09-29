/**
 *  @file c/dispatch/sets.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Batched set distances: capability lists, @c _best points and @c nk_sets_find_kernel.
 */
#include "dispatch.h"
#include "numkong/sets.h"

static nk_capability_kernels_t const *nk_hammings_packed_u1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_hammings_packed_u1_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_hammings_packed_u1_neon,
#endif
#if NUMKONG_TARGET_SMEBI32
        (nk_kernel_punned_t)&nk_hammings_packed_u1_smebi32,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_hammings_packed_u1_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_hammings_packed_u1_icelake,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_hammings_packed_u1_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_hammings_packed_u1_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_hammings_packed_u1_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smebi32_k * NUMKONG_TARGET_SMEBI32 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_hammings_packed_u1_best(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c,
                                                   nk_size_t height, nk_size_t width, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                   void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_hammings_packed_u1_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_hammings_symmetric_u1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_hammings_symmetric_u1_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_hammings_symmetric_u1_neon,
#endif
#if NUMKONG_TARGET_SMEBI32
        (nk_kernel_punned_t)&nk_hammings_symmetric_u1_smebi32,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_hammings_symmetric_u1_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_hammings_symmetric_u1_icelake,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_hammings_symmetric_u1_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_hammings_symmetric_u1_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_hammings_symmetric_u1_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smebi32_k * NUMKONG_TARGET_SMEBI32 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_hammings_symmetric_u1_best(nk_u1x8_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_u32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_hammings_symmetric_u1_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_jaccards_packed_u1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_jaccards_packed_u1_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_jaccards_packed_u1_neon,
#endif
#if NUMKONG_TARGET_SMEBI32
        (nk_kernel_punned_t)&nk_jaccards_packed_u1_smebi32,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_jaccards_packed_u1_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_jaccards_packed_u1_icelake,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_jaccards_packed_u1_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_jaccards_packed_u1_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_jaccards_packed_u1_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smebi32_k * NUMKONG_TARGET_SMEBI32 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_jaccards_packed_u1_best(nk_u1x8_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t height, nk_size_t width, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                   void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_jaccards_packed_u1_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_jaccards_symmetric_u1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_jaccards_symmetric_u1_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_jaccards_symmetric_u1_neon,
#endif
#if NUMKONG_TARGET_SMEBI32
        (nk_kernel_punned_t)&nk_jaccards_symmetric_u1_smebi32,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_jaccards_symmetric_u1_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_jaccards_symmetric_u1_icelake,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_jaccards_symmetric_u1_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_jaccards_symmetric_u1_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_jaccards_symmetric_u1_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smebi32_k * NUMKONG_TARGET_SMEBI32 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_jaccards_symmetric_u1_best(nk_u1x8_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_jaccards_symmetric_u1_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_sets_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                            nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_u1_k:
        switch (kind) {
        case nk_kernel_hammings_packed_k: lists = nk_hammings_packed_u1_capabilities(); break;
        case nk_kernel_hammings_symmetric_k: lists = nk_hammings_symmetric_u1_capabilities(); break;
        case nk_kernel_jaccards_packed_k: lists = nk_jaccards_packed_u1_capabilities(); break;
        case nk_kernel_jaccards_symmetric_k: lists = nk_jaccards_symmetric_u1_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
