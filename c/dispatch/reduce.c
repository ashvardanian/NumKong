/**
 *  @file c/dispatch/reduce.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Dispatch points of the reductions: moments, min/max with indices, and RMS normalization.
 */
#include "dispatch.h"

static nk_capability_kernels_t const *nk_reduce_moments_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_f64_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_reduce_moments_f64_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_f64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_f64_best(nk_f64_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_f64_t *sum, nk_f64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_f64_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_f64_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_f64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_f64_best(nk_f64_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_f64_t *min_value, nk_size_t *min_index, nk_f64_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_f64_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_moments_f32_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_f32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_f32_best(nk_f32_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_f64_t *sum, nk_f64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_f32_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_f32_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_f32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_f32_best(nk_f32_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_f32_t *min_value, nk_size_t *min_index, nk_f32_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_f32_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_i8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_i8_neon,
#endif
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_reduce_moments_i8_neonsdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_i8_haswell,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_reduce_moments_i8_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_i8_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_reduce_moments_i8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_i8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_reduce_moments_i8_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_i8_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_sierra_k * NUMKONG_TARGET_SIERRA |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_i8_best(nk_i8_t const *data, nk_size_t count, nk_size_t stride, nk_i64_t *sum,
                                                  nk_u64_t *sumsq, nk_capability_t capabilities, void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_i8_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_i8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_i8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_i8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_i8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_i8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_i8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_i8_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_i8_best(nk_i8_t const *data, nk_size_t count, nk_size_t stride,
                                                 nk_i8_t *min_value, nk_size_t *min_index, nk_i8_t *max_value,
                                                 nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_i8_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_u8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_u8_neon,
#endif
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_reduce_moments_u8_neonsdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_u8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_reduce_moments_u8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_reduce_moments_u8_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_u8_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_reduce_moments_u8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_reduce_moments_u8_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_u8_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_u8_best(nk_u8_t const *data, nk_size_t count, nk_size_t stride, nk_u64_t *sum,
                                                  nk_u64_t *sumsq, nk_capability_t capabilities, void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_u8_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_u8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_u8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_u8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_u8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_u8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_u8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_u8_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u8_best(nk_u8_t const *data, nk_size_t count, nk_size_t stride,
                                                 nk_u8_t *min_value, nk_size_t *min_index, nk_u8_t *max_value,
                                                 nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_u8_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_i16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_i16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_i16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_i16_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_reduce_moments_i16_alder,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_i16_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_reduce_moments_i16_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_i16_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_reduce_moments_i16_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_i16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_i16_best(nk_i16_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_i64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_i16_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_i16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_i16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_i16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_i16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_i16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_i16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_i16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_i16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_i16_best(nk_i16_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_i16_t *min_value, nk_size_t *min_index, nk_i16_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_i16_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_u16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_u16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_u16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_u16_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_reduce_moments_u16_alder,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_u16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_u16_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_reduce_moments_u16_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_u16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_u16_best(nk_u16_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_u64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_u16_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_u16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_u16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_u16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_u16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_u16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_u16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_u16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_u16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u16_best(nk_u16_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_u16_t *min_value, nk_size_t *min_index, nk_u16_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_u16_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_i32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_i32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_i32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_i32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_i32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_i32_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_reduce_moments_i32_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_i32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_i32_best(nk_i32_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_i64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_i32_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_i32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_i32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_i32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_i32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_i32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_i32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_i32_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_i32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_i32_best(nk_i32_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_i32_t *min_value, nk_size_t *min_index, nk_i32_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_i32_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_u32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_u32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_u32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_u32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_u32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_u32_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_reduce_moments_u32_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_u32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_u32_best(nk_u32_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_u64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_u32_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_u32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_u32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_u32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_u32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_u32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_u32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_u32_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_u32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u32_best(nk_u32_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_u32_t *min_value, nk_size_t *min_index, nk_u32_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_u32_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_i64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_i64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_i64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_i64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_i64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_i64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_moments_i64_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_i64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_i64_best(nk_i64_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_i64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_i64_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_i64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_i64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_i64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_i64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_i64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_i64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_i64_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_i64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_i64_best(nk_i64_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_i64_t *min_value, nk_size_t *min_index, nk_i64_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_i64_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_u64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_u64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_u64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_u64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_u64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_u64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_moments_u64_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_u64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_u64_best(nk_u64_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_u64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_u64_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_u64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_u64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_u64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_u64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_u64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_u64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_u64_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_u64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u64_best(nk_u64_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_u64_t *min_value, nk_size_t *min_index, nk_u64_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_u64_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_moments_f16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_f16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_f16_best(nk_f16_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                   void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_f16_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_f16_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_f16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_f16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_f16_best(nk_f16_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_f16_t *min_value, nk_size_t *min_index, nk_f16_t *max_value,
                                                  nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_f16_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_reduce_moments_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_reduce_moments_bf16_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_reduce_moments_bf16_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_bf16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_bf16_best(nk_bf16_t const *data, nk_size_t count, nk_size_t stride,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_bf16_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_bf16_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_bf16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_bf16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_bf16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_bf16_best(nk_bf16_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_bf16_t *min_value, nk_size_t *min_index, nk_bf16_t *max_value,
                                                   nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_bf16_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_e4m3_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_reduce_moments_e4m3_neonfhm,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_e4m3_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_reduce_moments_e4m3_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_moments_e4m3_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_e4m3_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_e4m3_best(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_e4m3_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_e4m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_e4m3_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_e4m3_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_e4m3_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_e4m3_best(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_e4m3_t *min_value, nk_size_t *min_index, nk_e4m3_t *max_value,
                                                   nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_e4m3_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_e5m2_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_reduce_moments_e5m2_neonfhm,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_e5m2_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_reduce_moments_e5m2_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_e5m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_moments_e5m2_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_e5m2_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_e5m2_best(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_e5m2_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_e5m2_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_e5m2_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_e5m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_e5m2_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_e5m2_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_e5m2_best(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_e5m2_t *min_value, nk_size_t *min_index, nk_e5m2_t *max_value,
                                                   nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_e5m2_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_e2m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_e2m3_neon,
#endif
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_reduce_moments_e2m3_neonsdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_e2m3_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_reduce_moments_e2m3_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_reduce_moments_e2m3_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_e2m3_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_reduce_moments_e2m3_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_e2m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_moments_e2m3_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_e2m3_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_e2m3_best(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_e2m3_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_e2m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_e2m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_e2m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_e2m3_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_e2m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_e2m3_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_e2m3_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_e2m3_best(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_e2m3_t *min_value, nk_size_t *min_index, nk_e2m3_t *max_value,
                                                   nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_e2m3_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_e3m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_e3m2_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_e3m2_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_reduce_moments_e3m2_alder,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_e3m2_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_reduce_moments_e3m2_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_moments_e3m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_moments_e3m2_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_e3m2_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_e3m2_best(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_e3m2_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_e3m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_minmax_e3m2_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_minmax_e3m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_minmax_e3m2_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_reduce_minmax_e3m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_reduce_minmax_e3m2_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_e3m2_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_e3m2_best(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride,
                                                   nk_e3m2_t *min_value, nk_size_t *min_index, nk_e3m2_t *max_value,
                                                   nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_e3m2_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_e2m1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_e2m1_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_e2m1_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_e2m1_best(nk_e2m1x2_t const *data, nk_size_t count, nk_size_t stride,
                                                    nk_f32_t *sum, nk_f32_t *sumsq, nk_capability_t capabilities,
                                                    void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_e2m1_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_i4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_i4_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_i4_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_i4_skylake,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_i4_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_i4_best(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_i64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                  void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_i4_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_i4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_i4_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_i4_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_i4_best(nk_i4x2_t const *data, nk_size_t count, nk_size_t stride,
                                                 nk_i8_t *min_value, nk_size_t *min_index, nk_i8_t *max_value,
                                                 nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_i4_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_u4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_u4_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_u4_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_u4_skylake,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_u4_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_u4_best(nk_u4x2_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_u64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                  void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_u4_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_u4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_u4_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_u4_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u4_best(nk_u4x2_t const *data, nk_size_t count, nk_size_t stride,
                                                 nk_u8_t *min_value, nk_size_t *min_index, nk_u8_t *max_value,
                                                 nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_u4_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_moments_u1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_moments_u1_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_reduce_moments_u1_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_reduce_moments_u1_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_reduce_moments_u1_skylake,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_moments_u1_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_moments_u1_best(nk_u1x8_t const *data, nk_size_t count, nk_size_t stride,
                                                  nk_u64_t *sum, nk_u64_t *sumsq, nk_capability_t capabilities,
                                                  void *stream) {
    nk_reduce_moments_punned_t const kernel = (nk_reduce_moments_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_moments_u1_capabilities());
    return kernel ? kernel(data, count, stride, sum, sumsq, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_reduce_minmax_u1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_reduce_minmax_u1_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_reduce_minmax_u1_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_reduce_minmax_u1_best(nk_u1x8_t const *data, nk_size_t count, nk_size_t stride,
                                                 nk_u8_t *min_value, nk_size_t *min_index, nk_u8_t *max_value,
                                                 nk_size_t *max_index, nk_capability_t capabilities, void *stream) {
    nk_reduce_minmax_punned_t const kernel = (nk_reduce_minmax_punned_t)nk_kernel_pick_(
        capabilities, nk_reduce_minmax_u1_capabilities());
    return kernel ? kernel(data, count, stride, min_value, min_index, max_value, max_index, stream)
                  : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_reduce_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f64_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_f64_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_f64_capabilities(); break;
        default: break;
        }
        break;
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_f32_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_f32_capabilities(); break;
        default: break;
        }
        break;
    case nk_bf16_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_bf16_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_bf16_capabilities(); break;
        default: break;
        }
        break;
    case nk_f16_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_f16_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_f16_capabilities(); break;
        default: break;
        }
        break;
    case nk_e5m2_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_e5m2_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_e5m2_capabilities(); break;
        default: break;
        }
        break;
    case nk_e4m3_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_e4m3_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_e4m3_capabilities(); break;
        default: break;
        }
        break;
    case nk_e3m2_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_e3m2_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_e3m2_capabilities(); break;
        default: break;
        }
        break;
    case nk_e2m3_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_e2m3_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_e2m3_capabilities(); break;
        default: break;
        }
        break;
    case nk_e2m1_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_e2m1_capabilities(); break;
        default: break;
        }
        break;
    case nk_i64_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_i64_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_i64_capabilities(); break;
        default: break;
        }
        break;
    case nk_i32_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_i32_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_i32_capabilities(); break;
        default: break;
        }
        break;
    case nk_i16_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_i16_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_i16_capabilities(); break;
        default: break;
        }
        break;
    case nk_i8_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_i8_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_i8_capabilities(); break;
        default: break;
        }
        break;
    case nk_i4_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_i4_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_i4_capabilities(); break;
        default: break;
        }
        break;
    case nk_u64_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_u64_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_u64_capabilities(); break;
        default: break;
        }
        break;
    case nk_u32_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_u32_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_u32_capabilities(); break;
        default: break;
        }
        break;
    case nk_u16_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_u16_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_u16_capabilities(); break;
        default: break;
        }
        break;
    case nk_u8_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_u8_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_u8_capabilities(); break;
        default: break;
        }
        break;
    case nk_u4_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_u4_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_u4_capabilities(); break;
        default: break;
        }
        break;
    case nk_u1_k:
        switch (kind) {
        case nk_kernel_reduce_moments_k: lists = nk_reduce_moments_u1_capabilities(); break;
        case nk_kernel_reduce_minmax_k: lists = nk_reduce_minmax_u1_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
