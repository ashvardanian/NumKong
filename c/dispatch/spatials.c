/**
 *  @file c/dispatch/spatials.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Batched spatial distances: capability lists, @c _best points, @c nk_spatials_find_kernel.
 */
#include "dispatch.h"
#include "numkong/spatials.h"

static nk_capability_kernels_t const *nk_angulars_packed_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angulars_packed_f32_neon,
#endif
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_angulars_packed_f32_smef64,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_packed_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_packed_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_f32_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_packed_f32_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_packed_f32_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_f32_cuda,
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_f32_rocm,
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_f32_best(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                    nk_size_t height, nk_size_t width, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride,
                                                    nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_f32_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angulars_symmetric_f32_neon,
#endif
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_angulars_symmetric_f32_smef64,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_symmetric_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_f32_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_symmetric_f32_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_symmetric_f32_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_f32_cuda,
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_f32_rocm,
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f32_best(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, nk_capability_t capabilities,
                                                       void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_f32_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclideans_packed_f32_neon,
#endif
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_euclideans_packed_f32_smef64,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_packed_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_f32_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_packed_f32_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_packed_f32_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_f32_cuda,
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_f32_rocm,
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_f32_best(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                      nk_size_t height, nk_size_t width, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_f32_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f32_neon,
#endif
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f32_smef64,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f32_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f32_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f32_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f32_cuda,
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f32_rocm,
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_best(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_f32_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angulars_packed_f64_neon,
#endif
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_angulars_packed_f64_smef64,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_packed_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_packed_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_f64_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_packed_f64_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_packed_f64_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_f64_cuda,
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_f64_rocm,
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_f64_best(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                    nk_size_t height, nk_size_t width, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride,
                                                    nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_f64_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angulars_symmetric_f64_neon,
#endif
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_angulars_symmetric_f64_smef64,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_symmetric_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_f64_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_symmetric_f64_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_symmetric_f64_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_f64_cuda,
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_f64_rocm,
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f64_best(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, nk_capability_t capabilities,
                                                       void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_f64_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclideans_packed_f64_neon,
#endif
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_euclideans_packed_f64_smef64,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_packed_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_f64_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_packed_f64_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_packed_f64_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_f64_cuda,
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_f64_rocm,
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_f64_best(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                      nk_size_t height, nk_size_t width, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_f64_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f64_neon,
#endif
#if NUMKONG_TARGET_SMEF64
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f64_smef64,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f64_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f64_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f64_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f64_cuda,
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f64_rocm,
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_smef64_k * NUMKONG_TARGET_SMEF64 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_best(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_f64_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angulars_packed_f16_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_angulars_packed_f16_neonfhm,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_packed_f16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_packed_f16_skylake,
#endif
#if NUMKONG_TARGET_GRANITEAMX
        (nk_kernel_punned_t)&nk_angulars_packed_f16_graniteamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_packed_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_f16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_packed_f16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_packed_f16_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_f16_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_packed_f16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_packed_f16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_packed_f16_blackwell,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_f16_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_packed_f16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_packed_f16_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_graniteamx_k * NUMKONG_TARGET_GRANITEAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_f16_best(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t height, nk_size_t width, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride,
                                                    nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_f16_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_neonfhm,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_skylake,
#endif
#if NUMKONG_TARGET_GRANITEAMX
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_graniteamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_blackwell,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_symmetric_f16_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_graniteamx_k * NUMKONG_TARGET_GRANITEAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f16_best(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, nk_capability_t capabilities,
                                                       void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_f16_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_neonfhm,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_skylake,
#endif
#if NUMKONG_TARGET_GRANITEAMX
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_graniteamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_blackwell,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_packed_f16_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_graniteamx_k * NUMKONG_TARGET_GRANITEAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_f16_best(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t height, nk_size_t width, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_f16_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_neon,
#endif
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_neonfhm,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_skylake,
#endif
#if NUMKONG_TARGET_GRANITEAMX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_graniteamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_blackwell,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_symmetric_f16_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_graniteamx_k * NUMKONG_TARGET_GRANITEAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_best(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_f16_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_neon,
#endif
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_blackwell,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_packed_bf16_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_bf16_best(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t height, nk_size_t width, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_bf16_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_neon,
#endif
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_blackwell,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_symmetric_bf16_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_best(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_bf16_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_neon,
#endif
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_blackwell,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_packed_bf16_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_bf16_best(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t height, nk_size_t width, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_bf16_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_neon,
#endif
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_blackwell,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_symmetric_bf16_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_best(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_bf16_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_neonfhm,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_genoa,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_diamond,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_packed_e4m3_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_e4m3_best(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t height, nk_size_t width, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_e4m3_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_neonfhm,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_genoa,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_diamond,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_symmetric_e4m3_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_best(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_e4m3_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_neonfhm,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_genoa,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_diamond,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_packed_e4m3_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_best(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t height, nk_size_t width, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_e4m3_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_neonfhm,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_genoa,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_diamond,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e4m3_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_best(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_e4m3_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_neonfhm,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_genoa,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_diamond,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_sapphireamx,
#endif
#if NUMKONG_TARGET_GRANITEAMX
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_graniteamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_packed_e5m2_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_graniteamx_k * NUMKONG_TARGET_GRANITEAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_e5m2_best(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t height, nk_size_t width, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_e5m2_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_neonfhm,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_genoa,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_diamond,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_sapphireamx,
#endif
#if NUMKONG_TARGET_GRANITEAMX
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_graniteamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_symmetric_e5m2_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_graniteamx_k * NUMKONG_TARGET_GRANITEAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_best(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_e5m2_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_neonfhm,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_genoa,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_diamond,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_sapphireamx,
#endif
#if NUMKONG_TARGET_GRANITEAMX
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_graniteamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_packed_e5m2_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_graniteamx_k * NUMKONG_TARGET_GRANITEAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_best(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t height, nk_size_t width, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_e5m2_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_neonfhm,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_genoa,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_diamond,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_sapphireamx,
#endif
#if NUMKONG_TARGET_GRANITEAMX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_graniteamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e5m2_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_genoa_k * NUMKONG_TARGET_GENOA |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_graniteamx_k * NUMKONG_TARGET_GRANITEAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_best(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_e5m2_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_packed_e2m3_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_sierra_k * NUMKONG_TARGET_SIERRA |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_e2m3_best(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t height, nk_size_t width, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_e2m3_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_e2m1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_packed_e2m1_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_sierra_k * NUMKONG_TARGET_SIERRA |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_e2m1_best(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t height, nk_size_t width, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_e2m1_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m3_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_sierra_k * NUMKONG_TARGET_SIERRA |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_best(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_e2m3_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_e2m1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_symmetric_e2m1_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_sierra_k * NUMKONG_TARGET_SIERRA |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_best(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_e2m1_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m3_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_sierra_k * NUMKONG_TARGET_SIERRA |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_best(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t height, nk_size_t width, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_e2m3_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_e2m1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_packed_e2m1_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_sierra_k * NUMKONG_TARGET_SIERRA |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_best(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t height, nk_size_t width, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_e2m1_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m3_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_sierra_k * NUMKONG_TARGET_SIERRA |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_best(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_e2m3_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_e2m1_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e2m1_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_alder_k * NUMKONG_TARGET_ALDER | nk_cap_sierra_k * NUMKONG_TARGET_SIERRA |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_best(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_e2m1_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_packed_e3m2_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_e3m2_best(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t height, nk_size_t width, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_e3m2_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_symmetric_e3m2_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_best(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_e3m2_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_packed_e3m2_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_best(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t height, nk_size_t width, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_e3m2_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_neonsdot,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_neonfp8,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_blackwellrtx,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_symmetric_e3m2_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_sme_k * NUMKONG_TARGET_SME | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX,
         nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_best(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_e3m2_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_packed_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_packed_i8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_i8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angulars_packed_i8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angulars_packed_i8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angulars_packed_i8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_packed_i8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_packed_i8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_angulars_packed_i8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_i8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_packed_i8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_packed_i8_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_i8_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_packed_i8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_packed_i8_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_i8_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_packed_i8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_packed_i8_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_i8_best(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t height, nk_size_t width, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                   void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_i8_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_symmetric_i8_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_i8_best(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_i8_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_packed_i8_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_i8_best(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t height, nk_size_t width, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_i8_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i8_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_best(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_i8_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_u8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_packed_u8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_packed_u8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_u8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angulars_packed_u8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angulars_packed_u8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angulars_packed_u8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_packed_u8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_packed_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_angulars_packed_u8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_u8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_packed_u8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_packed_u8_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_u8_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_packed_u8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_packed_u8_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_u8_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_packed_u8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_packed_u8_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_u8_best(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t height, nk_size_t width, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                   void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_u8_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_symmetric_u8_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_u8_best(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_u8_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_packed_u8_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_u8_best(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t height, nk_size_t width, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_u8_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_loongsonasx,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u8_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128_k * NUMKONG_TARGET_V128 | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_best(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_u8_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_i4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_i4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_packed_i4_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_packed_i4_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_i4_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angulars_packed_i4_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_i4_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_i4_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_packed_i4_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_packed_i4_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_i4_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_packed_i4_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_packed_i4_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_i4_best(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t height, nk_size_t width, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                   void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_i4_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_i4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_symmetric_i4_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_i4_best(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_i4_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_i4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_packed_i4_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_i4_best(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t height, nk_size_t width, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_i4_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_i4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_symmetric_i4_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_best(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_i4_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_packed_u4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_u4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_packed_u4_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_packed_u4_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_packed_u4_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angulars_packed_u4_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_packed_u4_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_u4_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_packed_u4_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_packed_u4_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_packed_u4_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_packed_u4_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_packed_u4_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_packed_u4_best(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t height, nk_size_t width, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                   void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_packed_u4_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angulars_symmetric_u4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_angulars_symmetric_u4_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_u4_best(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_angulars_symmetric_u4_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_packed_u4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_packed_u4_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_packed_u4_best(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t height, nk_size_t width, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_dots_packed_punned_t const kernel = (nk_dots_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_packed_u4_capabilities());
    return kernel ? kernel(a, b_packed, c, height, width, depth, a_stride, c_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclideans_symmetric_u4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_v128relaxed,
#endif
    };
#if NUMKONG_ARCH_CUDA_
    static nk_kernel_punned_t const nvidia[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_cuda,
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_hopper,
#endif
    };
#endif
#if NUMKONG_ARCH_ROCM_
    static nk_kernel_punned_t const amd[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_rocm,
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_euclideans_symmetric_u4_cdna5,
#endif
    };
#endif
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
#if NUMKONG_ARCH_CUDA_
        {nk_cap_cuda_k | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE | nk_cap_hopper_k * NUMKONG_TARGET_HOPPER, nvidia},
#else
        {0, nk_no_kernels_},
#endif
#if NUMKONG_ARCH_ROCM_
        {nk_cap_rocm_k | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5, amd},
#else
        {0, nk_no_kernels_},
#endif
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_best(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_dots_symmetric_punned_t const kernel = (nk_dots_symmetric_punned_t)nk_kernel_pick_(
        capabilities, nk_euclideans_symmetric_u4_capabilities());
    return kernel ? kernel(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count, stream)
                  : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_spatials_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                                nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f64_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_f64_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_f64_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_f64_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_f64_capabilities(); break;
        default: break;
        }
        break;
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_f32_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_f32_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_f32_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_f32_capabilities(); break;
        default: break;
        }
        break;
    case nk_bf16_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_bf16_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_bf16_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_bf16_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_bf16_capabilities(); break;
        default: break;
        }
        break;
    case nk_f16_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_f16_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_f16_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_f16_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_f16_capabilities(); break;
        default: break;
        }
        break;
    case nk_e5m2_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_e5m2_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_e5m2_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_e5m2_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_e5m2_capabilities(); break;
        default: break;
        }
        break;
    case nk_e4m3_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_e4m3_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_e4m3_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_e4m3_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_e4m3_capabilities(); break;
        default: break;
        }
        break;
    case nk_e3m2_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_e3m2_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_e3m2_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_e3m2_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_e3m2_capabilities(); break;
        default: break;
        }
        break;
    case nk_e2m3_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_e2m3_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_e2m3_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_e2m3_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_e2m3_capabilities(); break;
        default: break;
        }
        break;
    case nk_e2m1_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_e2m1_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_e2m1_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_e2m1_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_e2m1_capabilities(); break;
        default: break;
        }
        break;
    case nk_i8_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_i8_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_i8_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_i8_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_i8_capabilities(); break;
        default: break;
        }
        break;
    case nk_i4_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_i4_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_i4_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_i4_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_i4_capabilities(); break;
        default: break;
        }
        break;
    case nk_u8_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_u8_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_u8_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_u8_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_u8_capabilities(); break;
        default: break;
        }
        break;
    case nk_u4_k:
        switch (kind) {
        case nk_kernel_angulars_packed_k: lists = nk_angulars_packed_u4_capabilities(); break;
        case nk_kernel_angulars_symmetric_k: lists = nk_angulars_symmetric_u4_capabilities(); break;
        case nk_kernel_euclideans_packed_k: lists = nk_euclideans_packed_u4_capabilities(); break;
        case nk_kernel_euclideans_symmetric_k: lists = nk_euclideans_symmetric_u4_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
