/**
 *  @file c/dispatch/each.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Dispatch points of the element-wise operations: scale, sum, blend, FMA and SwiGLU.
 */
#include "dispatch.h"

static nk_capability_kernels_t const *nk_each_scale_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_f64_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_f64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_f64_best(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                               nk_f64_t const *beta, nk_f64_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_f64_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_scale_f32_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_f32_cuda,
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

NUMKONG_API nk_status_t nk_each_scale_f32_best(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                               nk_f32_t const *beta, nk_f32_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_f32_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_scale_f16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_f16_cuda,
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

NUMKONG_API nk_status_t nk_each_scale_f16_best(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                               nk_f32_t const *beta, nk_f16_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_f16_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_each_scale_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_bf16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_scale_bf16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_bf16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_bf16_best(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                nk_f32_t const *beta, nk_bf16_t *result, nk_capability_t capabilities,
                                                void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_bf16_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_i8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_i8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_i8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_i8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_i8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_scale_i8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_i8_cuda,
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

NUMKONG_API nk_status_t nk_each_scale_i8_best(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                              nk_f32_t const *beta, nk_i8_t *result, nk_capability_t capabilities,
                                              void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_i8_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_u8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_u8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_u8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_u8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_u8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_scale_u8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_u8_cuda,
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

NUMKONG_API nk_status_t nk_each_scale_u8_best(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                              nk_f32_t const *beta, nk_u8_t *result, nk_capability_t capabilities,
                                              void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_u8_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_i16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_i16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_i16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_i16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_i16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_i16_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_i16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_i16_best(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                               nk_f32_t const *beta, nk_i16_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_i16_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_u16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_u16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_u16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_u16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_u16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_u16_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_u16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_u16_best(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                               nk_f32_t const *beta, nk_u16_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_u16_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_i32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_i32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_i32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_i32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_i32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_i32_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_i32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_i32_best(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                               nk_f64_t const *beta, nk_i32_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_i32_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_u32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_u32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_u32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_u32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_u32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_u32_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_u32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_u32_best(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                               nk_f64_t const *beta, nk_u32_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_u32_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_i64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_i64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_i64_neon,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_i64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_i64_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_i64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_i64_best(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                               nk_f64_t const *beta, nk_i64_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_i64_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_u64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_u64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_u64_neon,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_u64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_u64_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_u64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_u64_best(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha,
                                               nk_f64_t const *beta, nk_u64_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_u64_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_e4m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_e4m3_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_e4m3_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_e4m3_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_e4m3_best(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                nk_f32_t const *beta, nk_e4m3_t *result, nk_capability_t capabilities,
                                                void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_e4m3_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_e5m2_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_e5m2_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_e5m2_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_e5m2_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_e5m2_best(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                nk_f32_t const *beta, nk_e5m2_t *result, nk_capability_t capabilities,
                                                void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_e5m2_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_e2m3_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_e2m3_cuda,
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

NUMKONG_API nk_status_t nk_each_scale_e2m3_best(nk_e2m3_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                nk_f32_t const *beta, nk_e2m3_t *result, nk_capability_t capabilities,
                                                void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_e2m3_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_e3m2_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_e3m2_cuda,
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

NUMKONG_API nk_status_t nk_each_scale_e3m2_best(nk_e3m2_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                nk_f32_t const *beta, nk_e3m2_t *result, nk_capability_t capabilities,
                                                void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_e3m2_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_f32c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_f32c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_f32c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_f32c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_f32c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_f32c_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_f32c_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_f32c_best(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                                nk_f32c_t const *beta, nk_f32c_t *result, nk_capability_t capabilities,
                                                void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_f32c_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_scale_f64c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_scale_f64c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_scale_f64c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_scale_f64c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_scale_f64c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_scale_f64c_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_scale_f64c_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_scale_f64c_best(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                                nk_f64c_t const *beta, nk_f64c_t *result, nk_capability_t capabilities,
                                                void *stream) {
    nk_each_scale_punned_t const kernel = (nk_each_scale_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_scale_f64c_capabilities());
    return kernel ? kernel(a, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_sum_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_f64_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_f64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_f64_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_sum_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_f32_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_each_sum_f32_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_f32_cuda,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_each_sum_f32_rocm,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM, rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_f32_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_f16_serial,
#if NUMKONG_TARGET_NEONHALF
        (nk_kernel_punned_t)&nk_each_sum_f16_neonhalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_f16_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_each_sum_f16_sapphire,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_sum_f16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_f16_cuda,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_each_sum_f16_rocm,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM, rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_f16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_each_sum_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_sum_bf16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_each_sum_bf16_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_bf16_cuda,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_each_sum_bf16_rocm,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM, rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_bf16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_i8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_i8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_i8_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_i8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_i8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_each_sum_i8_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_i8_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_i8_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_u8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_u8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_u8_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_u8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_each_sum_u8_v128,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_u8_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_u8_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_i16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_i16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_i16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_i16_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_i16_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_i16_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_i16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_i16_best(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_i16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_u16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_u16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_u16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_u16_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_u16_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_u16_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_u16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_u16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_i32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_i32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_i32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_i32_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_i32_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_i32_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_i32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_i32_best(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_i32_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_u32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_u32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_u32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_u32_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_u32_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_u32_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_u32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_u32_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_i64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_i64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_i64_neon,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_i64_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_i64_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_i64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_i64_best(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_i64_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_u64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_u64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_u64_neon,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_each_sum_u64_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_u64_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_u64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_u64_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_e4m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_sum_e4m3_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_each_sum_e4m3_sapphire,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_e4m3_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_e4m3_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_e4m3_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_e4m3_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_sum_e5m2_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_sum_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_sum_e5m2_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_sum_e5m2_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_e5m2_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_sum_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_e5m2_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_e5m2_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_e2m3_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_e2m3_cuda,
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

NUMKONG_API nk_status_t nk_each_sum_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_e2m3_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_e2m3_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_e3m2_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_e3m2_cuda,
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

NUMKONG_API nk_status_t nk_each_sum_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_e3m2_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_e3m2_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_f32c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_f32c_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_f32c_cuda,
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

NUMKONG_API nk_status_t nk_each_sum_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_f32c_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_sum_f64c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_sum_f64c_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_sum_f64c_cuda,
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

NUMKONG_API nk_status_t nk_each_sum_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_sum_punned_t const kernel = (nk_each_sum_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_sum_f64c_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_f64_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_f64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_blend_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                               nk_f64_t const *beta, nk_f64_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_f64_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_blend_f32_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_f32_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                               nk_f32_t const *beta, nk_f32_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_f32_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_blend_f16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_f16_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                               nk_f32_t const *beta, nk_f16_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_f16_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_each_blend_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_bf16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_blend_bf16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_bf16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_blend_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result,
                                                nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_bf16_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_i8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_i8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_i8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_i8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_i8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_blend_i8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_i8_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                              nk_f32_t const *beta, nk_i8_t *result, nk_capability_t capabilities,
                                              void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_i8_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_u8_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_u8_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_u8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_u8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_u8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_blend_u8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_u8_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                              nk_f32_t const *beta, nk_u8_t *result, nk_capability_t capabilities,
                                              void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_u8_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_i16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_i16_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_i16_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_i16_best(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                               nk_f32_t const *beta, nk_i16_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_i16_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_u16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_u16_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_u16_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                               nk_f32_t const *beta, nk_u16_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_u16_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_i32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_i32_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_i32_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_i32_best(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                               nk_f64_t const *beta, nk_i32_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_i32_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_u32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_u32_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_u32_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                               nk_f64_t const *beta, nk_u32_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_u32_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_i64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_i64_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_i64_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_i64_best(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                               nk_f64_t const *beta, nk_i64_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_i64_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_u64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_u64_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_u64_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                               nk_f64_t const *beta, nk_u64_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_u64_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_e4m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_e4m3_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_e4m3_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_e4m3_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_blend_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                nk_f32_t const *alpha, nk_f32_t const *beta, nk_e4m3_t *result,
                                                nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_e4m3_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_e5m2_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_e5m2_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_e5m2_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_e5m2_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_blend_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                nk_f32_t const *alpha, nk_f32_t const *beta, nk_e5m2_t *result,
                                                nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_e5m2_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_e2m3_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_e2m3_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                nk_f32_t const *alpha, nk_f32_t const *beta, nk_e2m3_t *result,
                                                nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_e2m3_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_e3m2_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_e3m2_cuda,
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

NUMKONG_API nk_status_t nk_each_blend_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                nk_f32_t const *alpha, nk_f32_t const *beta, nk_e3m2_t *result,
                                                nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_e3m2_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_f32c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_f32c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_f32c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_f32c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_f32c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_f32c_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_f32c_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_blend_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                nk_f32c_t const *alpha, nk_f32c_t const *beta, nk_f32c_t *result,
                                                nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_f32c_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_blend_f64c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_blend_f64c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_blend_f64c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_blend_f64c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_blend_f64c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_blend_f64c_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_blend_f64c_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_blend_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                nk_f64c_t const *alpha, nk_f64c_t const *beta, nk_f64c_t *result,
                                                nk_capability_t capabilities, void *stream) {
    nk_each_blend_punned_t const kernel = (nk_each_blend_punned_t)nk_kernel_pick_(capabilities,
                                                                                  nk_each_blend_f64c_capabilities());
    return kernel ? kernel(a, b, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_f64_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_f64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                             nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_f64_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_fma_f32_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_f32_cuda,
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

NUMKONG_API nk_status_t nk_each_fma_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                             nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_f32_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_f16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_fma_f16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_f16_cuda,
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

NUMKONG_API nk_status_t nk_each_fma_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                             nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_f16_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_each_fma_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_bf16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_fma_bf16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_bf16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                              nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_bf16_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_i8_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_i8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_i8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_i8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_fma_i8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_i8_cuda,
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

NUMKONG_API nk_status_t nk_each_fma_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n,
                                            nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_i8_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_u8_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_u8_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_u8_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_u8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_each_fma_u8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_u8_cuda,
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

NUMKONG_API nk_status_t nk_each_fma_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n,
                                            nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_u8_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_i16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_i16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_i16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_i16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_i16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_i16_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_i16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_i16_best(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n,
                                             nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_i16_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_u16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_u16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_u16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_u16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_u16_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_u16_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_u16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n,
                                             nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_u16_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_i32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_i32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_i32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_i32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_i32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_i32_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_i32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_i32_best(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n,
                                             nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_i32_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_u32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_u32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_u32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_u32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_u32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_u32_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_u32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n,
                                             nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_u32_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_i64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_i64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_i64_neon,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_i64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_i64_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_i64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_i64_best(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, nk_size_t n,
                                             nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_i64_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_u64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_u64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_u64_neon,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_u64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_u64_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_u64_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, nk_size_t n,
                                             nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result,
                                             nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_u64_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_e4m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_e4m3_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_e4m3_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_e4m3_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c, nk_size_t n,
                                              nk_f32_t const *alpha, nk_f32_t const *beta, nk_e4m3_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_e4m3_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_e5m2_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_e5m2_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_e5m2_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_e5m2_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c, nk_size_t n,
                                              nk_f32_t const *alpha, nk_f32_t const *beta, nk_e5m2_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_e5m2_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_e2m3_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_e2m3_cuda,
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

NUMKONG_API nk_status_t nk_each_fma_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_e2m3_t const *c, nk_size_t n,
                                              nk_f32_t const *alpha, nk_f32_t const *beta, nk_e2m3_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_e2m3_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_e3m2_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_e3m2_cuda,
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

NUMKONG_API nk_status_t nk_each_fma_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_e3m2_t const *c, nk_size_t n,
                                              nk_f32_t const *alpha, nk_f32_t const *beta, nk_e3m2_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_e3m2_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_f32c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_f32c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_f32c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_f32c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_f32c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_f32c_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_f32c_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                              nk_f32c_t const *alpha, nk_f32c_t const *beta, nk_f32c_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_f32c_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_fma_f64c_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_fma_f64c_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_fma_f64c_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_fma_f64c_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_fma_f64c_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_each_fma_f64c_rvv,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_fma_f64c_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_fma_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                              nk_f64c_t const *alpha, nk_f64c_t const *beta, nk_f64c_t *result,
                                              nk_capability_t capabilities, void *stream) {
    nk_each_fma_punned_t const kernel = (nk_each_fma_punned_t)nk_kernel_pick_(capabilities,
                                                                              nk_each_fma_f64c_capabilities());
    return kernel ? kernel(a, b, c, n, alpha, beta, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_swiglu_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_swiglu_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_swiglu_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_swiglu_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_swiglu_f32_skylake,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_swiglu_f32_cuda,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_each_swiglu_f32_rocm,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM, rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_swiglu_f32_best(nk_f32_t const *gate, nk_f32_t const *up, nk_f32_t *y, nk_size_t rows,
                                                nk_size_t columns, nk_size_t gate_stride, nk_size_t up_stride,
                                                nk_size_t y_stride, nk_f32_t gate_scale, nk_f32_t output_scale,
                                                nk_capability_t capabilities, void *stream) {
    nk_each_swiglu_punned_t const kernel = (nk_each_swiglu_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_each_swiglu_f32_capabilities());
    return kernel
               ? kernel(gate, up, y, rows, columns, gate_stride, up_stride, y_stride, gate_scale, output_scale, stream)
               : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_swiglu_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_swiglu_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_swiglu_f16_neon,
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

NUMKONG_API nk_status_t nk_each_swiglu_f16_best(nk_f16_t const *gate, nk_f16_t const *up, nk_f16_t *y, nk_size_t rows,
                                                nk_size_t columns, nk_size_t gate_stride, nk_size_t up_stride,
                                                nk_size_t y_stride, nk_f32_t gate_scale, nk_f32_t output_scale,
                                                nk_capability_t capabilities, void *stream) {
    nk_each_swiglu_punned_t const kernel = (nk_each_swiglu_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_each_swiglu_f16_capabilities());
    return kernel
               ? kernel(gate, up, y, rows, columns, gate_stride, up_stride, y_stride, gate_scale, output_scale, stream)
               : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_swiglu_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_swiglu_bf16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_swiglu_bf16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_swiglu_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_swiglu_bf16_skylake,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_swiglu_bf16_cuda,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_each_swiglu_bf16_rocm,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM, rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_swiglu_bf16_best(nk_bf16_t const *gate, nk_bf16_t const *up, nk_bf16_t *y,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t gate_stride,
                                                 nk_size_t up_stride, nk_size_t y_stride, nk_f32_t gate_scale,
                                                 nk_f32_t output_scale, nk_capability_t capabilities, void *stream) {
    nk_each_swiglu_punned_t const kernel = (nk_each_swiglu_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_each_swiglu_bf16_capabilities());
    return kernel
               ? kernel(gate, up, y, rows, columns, gate_stride, up_stride, y_stride, gate_scale, output_scale, stream)
               : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_swiglu_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_swiglu_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_swiglu_e4m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_swiglu_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_swiglu_e4m3_skylake,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_swiglu_e4m3_cuda,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_each_swiglu_e4m3_rocm,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM, rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_swiglu_e4m3_best(nk_e4m3_t const *gate, nk_e4m3_t const *up, nk_e4m3_t *y,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t gate_stride,
                                                 nk_size_t up_stride, nk_size_t y_stride, nk_f32_t gate_scale,
                                                 nk_f32_t output_scale, nk_capability_t capabilities, void *stream) {
    nk_each_swiglu_punned_t const kernel = (nk_each_swiglu_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_each_swiglu_e4m3_capabilities());
    return kernel
               ? kernel(gate, up, y, rows, columns, gate_stride, up_stride, y_stride, gate_scale, output_scale, stream)
               : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_rmsnorm_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_rmsnorm_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_rmsnorm_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_rmsnorm_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_rmsnorm_f32_skylake,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_rmsnorm_f32_cuda,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_each_rmsnorm_f32_rocm,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM, rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_rmsnorm_f32_best(nk_f32_t const *x, nk_f32_t const *gamma, nk_f32_t *y, nk_size_t rows,
                                                 nk_size_t groups, nk_size_t columns, nk_size_t x_stride,
                                                 nk_size_t y_stride, nk_f32_t epsilon, nk_capability_t capabilities,
                                                 void *stream) {
    nk_each_rmsnorm_punned_t const kernel = (nk_each_rmsnorm_punned_t)nk_kernel_pick_(
        capabilities, nk_each_rmsnorm_f32_capabilities());
    return kernel ? kernel(x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_rmsnorm_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_rmsnorm_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_rmsnorm_f16_neon,
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

NUMKONG_API nk_status_t nk_each_rmsnorm_f16_best(nk_f16_t const *x, nk_f32_t const *gamma, nk_f16_t *y, nk_size_t rows,
                                                 nk_size_t groups, nk_size_t columns, nk_size_t x_stride,
                                                 nk_size_t y_stride, nk_f32_t epsilon, nk_capability_t capabilities,
                                                 void *stream) {
    nk_each_rmsnorm_punned_t const kernel = (nk_each_rmsnorm_punned_t)nk_kernel_pick_(
        capabilities, nk_each_rmsnorm_f16_capabilities());
    return kernel ? kernel(x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_rmsnorm_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_rmsnorm_bf16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_rmsnorm_bf16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_rmsnorm_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_rmsnorm_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_each_rmsnorm_bf16_genoa,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_rmsnorm_bf16_cuda,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_each_rmsnorm_bf16_rocm,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM, rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_rmsnorm_bf16_best(nk_bf16_t const *x, nk_f32_t const *gamma, nk_bf16_t *y,
                                                  nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                  nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                  nk_capability_t capabilities, void *stream) {
    nk_each_rmsnorm_punned_t const kernel = (nk_each_rmsnorm_punned_t)nk_kernel_pick_(
        capabilities, nk_each_rmsnorm_bf16_capabilities());
    return kernel ? kernel(x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_each_rmsnorm_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_each_rmsnorm_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_each_rmsnorm_e4m3_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_each_rmsnorm_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_each_rmsnorm_e4m3_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_each_rmsnorm_e4m3_genoa,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_each_rmsnorm_e4m3_cuda,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_each_rmsnorm_e4m3_rocm,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM, rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_each_rmsnorm_e4m3_best(nk_e4m3_t const *x, nk_f32_t const *gamma, nk_e4m3_t *y,
                                                  nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                  nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                  nk_capability_t capabilities, void *stream) {
    nk_each_rmsnorm_punned_t const kernel = (nk_each_rmsnorm_punned_t)nk_kernel_pick_(
        capabilities, nk_each_rmsnorm_e4m3_capabilities());
    return kernel ? kernel(x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream)
                  : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_each_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                            nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f64c_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_f64c_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_f64c_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_f64c_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_f64c_capabilities(); break;
        default: break;
        }
        break;
    case nk_f32c_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_f32c_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_f32c_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_f32c_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_f32c_capabilities(); break;
        default: break;
        }
        break;
    case nk_f64_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_f64_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_f64_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_f64_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_f64_capabilities(); break;
        default: break;
        }
        break;
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_f32_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_f32_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_f32_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_f32_capabilities(); break;
        case nk_kernel_each_swiglu_k: lists = nk_each_swiglu_f32_capabilities(); break;
        case nk_kernel_each_rmsnorm_k: lists = nk_each_rmsnorm_f32_capabilities(); break;
        default: break;
        }
        break;
    case nk_bf16_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_bf16_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_bf16_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_bf16_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_bf16_capabilities(); break;
        case nk_kernel_each_swiglu_k: lists = nk_each_swiglu_bf16_capabilities(); break;
        case nk_kernel_each_rmsnorm_k: lists = nk_each_rmsnorm_bf16_capabilities(); break;
        default: break;
        }
        break;
    case nk_f16_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_f16_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_f16_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_f16_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_f16_capabilities(); break;
        case nk_kernel_each_rmsnorm_k: lists = nk_each_rmsnorm_f16_capabilities(); break;
        case nk_kernel_each_swiglu_k: lists = nk_each_swiglu_f16_capabilities(); break;
        default: break;
        }
        break;
    case nk_e5m2_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_e5m2_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_e5m2_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_e5m2_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_e5m2_capabilities(); break;
        default: break;
        }
        break;
    case nk_e4m3_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_e4m3_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_e4m3_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_e4m3_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_e4m3_capabilities(); break;
        case nk_kernel_each_swiglu_k: lists = nk_each_swiglu_e4m3_capabilities(); break;
        case nk_kernel_each_rmsnorm_k: lists = nk_each_rmsnorm_e4m3_capabilities(); break;
        default: break;
        }
        break;
    case nk_e3m2_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_e3m2_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_e3m2_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_e3m2_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_e3m2_capabilities(); break;
        default: break;
        }
        break;
    case nk_e2m3_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_e2m3_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_e2m3_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_e2m3_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_e2m3_capabilities(); break;
        default: break;
        }
        break;
    case nk_i64_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_i64_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_i64_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_i64_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_i64_capabilities(); break;
        default: break;
        }
        break;
    case nk_i32_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_i32_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_i32_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_i32_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_i32_capabilities(); break;
        default: break;
        }
        break;
    case nk_i16_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_i16_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_i16_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_i16_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_i16_capabilities(); break;
        default: break;
        }
        break;
    case nk_i8_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_i8_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_i8_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_i8_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_i8_capabilities(); break;
        default: break;
        }
        break;
    case nk_u64_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_u64_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_u64_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_u64_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_u64_capabilities(); break;
        default: break;
        }
        break;
    case nk_u32_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_u32_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_u32_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_u32_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_u32_capabilities(); break;
        default: break;
        }
        break;
    case nk_u16_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_u16_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_u16_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_u16_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_u16_capabilities(); break;
        default: break;
        }
        break;
    case nk_u8_k:
        switch (kind) {
        case nk_kernel_each_scale_k: lists = nk_each_scale_u8_capabilities(); break;
        case nk_kernel_each_sum_k: lists = nk_each_sum_u8_capabilities(); break;
        case nk_kernel_each_blend_k: lists = nk_each_blend_u8_capabilities(); break;
        case nk_kernel_each_fma_k: lists = nk_each_fma_u8_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
