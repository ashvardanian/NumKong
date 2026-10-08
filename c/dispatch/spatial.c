/**
 *  @file c/dispatch/spatial.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Spatial distances: capability lists, @c _best dispatch points, @c nk_spatial_find_kernel.
 */
#include "dispatch.h"
#include "numkong/spatial.h"

static nk_capability_kernels_t const *nk_euclidean_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclidean_f64_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_euclidean_f64_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclidean_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclidean_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclidean_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclidean_f64_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclidean_f64_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclidean_f64_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_f64_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclidean_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclidean_f32_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_euclidean_f32_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclidean_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclidean_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclidean_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclidean_f32_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclidean_f32_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclidean_f32_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_f32_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclidean_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclidean_f16_neon,
#endif
#if NUMKONG_TARGET_SVEHALF
        (nk_kernel_punned_t)&nk_euclidean_f16_svehalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclidean_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclidean_f16_skylake,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_euclidean_f16_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclidean_f16_rvv,
#endif
#if NUMKONG_TARGET_RVVHALF
        (nk_kernel_punned_t)&nk_euclidean_f16_rvvhalf,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclidean_f16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclidean_f16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclidean_f16_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_svehalf_k * NUMKONG_TARGET_SVEHALF |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_rvvhalf_k * NUMKONG_TARGET_RVVHALF | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_f16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclidean_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_bf16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclidean_bf16_neon,
#endif
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_euclidean_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SVEBFDOT
        (nk_kernel_punned_t)&nk_euclidean_bf16_svebfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclidean_bf16_haswell,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_euclidean_bf16_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclidean_bf16_rvv,
#endif
#if NUMKONG_TARGET_RVVBF16
        (nk_kernel_punned_t)&nk_euclidean_bf16_rvvbf16,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_euclidean_bf16_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclidean_bf16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclidean_bf16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclidean_bf16_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT |
             nk_cap_svebfdot_k * NUMKONG_TARGET_SVEBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_rvvbf16_k * NUMKONG_TARGET_RVVBF16 | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_bf16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclidean_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclidean_e4m3_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclidean_e4m3_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclidean_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclidean_e4m3_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclidean_e4m3_icelake,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_euclidean_e4m3_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclidean_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclidean_e4m3_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_e4m3_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclidean_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclidean_e5m2_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclidean_e5m2_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclidean_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclidean_e5m2_skylake,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_euclidean_e5m2_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclidean_e5m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclidean_e5m2_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_e5m2_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclidean_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_e2m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclidean_e2m3_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclidean_e2m3_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclidean_e2m3_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclidean_e2m3_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclidean_e2m3_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclidean_e2m3_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclidean_e2m3_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclidean_e2m3_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_e2m3_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclidean_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_e3m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_euclidean_e3m2_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_euclidean_e3m2_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclidean_e3m2_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclidean_e3m2_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclidean_e3m2_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_euclidean_e3m2_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclidean_e3m2_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclidean_e3m2_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_e3m2_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclidean_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclidean_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SVESDOT
        (nk_kernel_punned_t)&nk_euclidean_i8_svesdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclidean_i8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclidean_i8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclidean_i8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclidean_i8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclidean_i8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_euclidean_i8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclidean_i8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclidean_i8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclidean_i8_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_svesdot_k * NUMKONG_TARGET_SVESDOT |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_i8_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclidean_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_u8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclidean_u8_neonsdot,
#endif
#if NUMKONG_TARGET_SVESDOT
        (nk_kernel_punned_t)&nk_euclidean_u8_svesdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_euclidean_u8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_euclidean_u8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_euclidean_u8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclidean_u8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclidean_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_euclidean_u8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_euclidean_u8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_euclidean_u8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_euclidean_u8_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_svesdot_k * NUMKONG_TARGET_SVESDOT |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_u8_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclidean_i4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_i4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclidean_i4_neonsdot,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclidean_i4_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclidean_i4_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_i4_best(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_i4_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_euclidean_u4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_euclidean_u4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_euclidean_u4_neonsdot,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_euclidean_u4_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_euclidean_u4_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_euclidean_u4_best(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_euclidean_u4_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sqeuclidean_f64_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_sqeuclidean_f64_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclidean_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclidean_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclidean_f64_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_sqeuclidean_f64_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_sqeuclidean_f64_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(
        capabilities, nk_sqeuclidean_f64_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sqeuclidean_f32_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_sqeuclidean_f32_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclidean_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclidean_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclidean_f32_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_sqeuclidean_f32_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_sqeuclidean_f32_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(
        capabilities, nk_sqeuclidean_f32_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sqeuclidean_f16_neon,
#endif
#if NUMKONG_TARGET_SVEHALF
        (nk_kernel_punned_t)&nk_sqeuclidean_f16_svehalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclidean_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_f16_skylake,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_sqeuclidean_f16_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclidean_f16_rvv,
#endif
#if NUMKONG_TARGET_RVVHALF
        (nk_kernel_punned_t)&nk_sqeuclidean_f16_rvvhalf,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclidean_f16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_sqeuclidean_f16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_sqeuclidean_f16_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_svehalf_k * NUMKONG_TARGET_SVEHALF |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_rvvhalf_k * NUMKONG_TARGET_RVVHALF | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(
        capabilities, nk_sqeuclidean_f16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_neon,
#endif
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SVEBFDOT
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_svebfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_haswell,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_rvv,
#endif
#if NUMKONG_TARGET_RVVBF16
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_rvvbf16,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_sqeuclidean_bf16_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT |
             nk_cap_svebfdot_k * NUMKONG_TARGET_SVEBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_rvvbf16_k * NUMKONG_TARGET_RVVBF16 | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(
        capabilities, nk_sqeuclidean_bf16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sqeuclidean_e4m3_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_sqeuclidean_e4m3_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclidean_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_e4m3_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_e4m3_icelake,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_sqeuclidean_e4m3_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclidean_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclidean_e4m3_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(
        capabilities, nk_sqeuclidean_e4m3_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sqeuclidean_e5m2_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_sqeuclidean_e5m2_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclidean_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_e5m2_skylake,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_sqeuclidean_e5m2_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclidean_e5m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclidean_e5m2_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(
        capabilities, nk_sqeuclidean_e5m2_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_e2m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sqeuclidean_e2m3_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_sqeuclidean_e2m3_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclidean_e2m3_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_sqeuclidean_e2m3_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_sqeuclidean_e2m3_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_e2m3_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_e2m3_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclidean_e2m3_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(
        capabilities, nk_sqeuclidean_e2m3_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_e3m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_sqeuclidean_e3m2_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_sqeuclidean_e3m2_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclidean_e3m2_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_sqeuclidean_e3m2_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_sqeuclidean_e3m2_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_e3m2_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_e3m2_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclidean_e3m2_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(
        capabilities, nk_sqeuclidean_e3m2_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SVESDOT
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_svesdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_sqeuclidean_i8_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_svesdot_k * NUMKONG_TARGET_SVESDOT |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_sqeuclidean_i8_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_neonsdot,
#endif
#if NUMKONG_TARGET_SVESDOT
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_svesdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_sqeuclidean_u8_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_svesdot_k * NUMKONG_TARGET_SVESDOT |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_sqeuclidean_u8_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_i4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_i4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_sqeuclidean_i4_neonsdot,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_i4_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclidean_i4_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_i4_best(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_sqeuclidean_i4_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_sqeuclidean_u4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_sqeuclidean_u4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_sqeuclidean_u4_neonsdot,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_sqeuclidean_u4_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_sqeuclidean_u4_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_sqeuclidean_u4_best(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_sqeuclidean_u4_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angular_f64_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_angular_f64_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angular_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angular_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angular_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angular_f64_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angular_f64_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angular_f64_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_f64_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angular_f32_neon,
#endif
#if NUMKONG_TARGET_SVE
        (nk_kernel_punned_t)&nk_angular_f32_sve,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angular_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angular_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angular_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angular_f32_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angular_f32_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angular_f32_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_sve_k * NUMKONG_TARGET_SVE |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_f32_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angular_f16_neon,
#endif
#if NUMKONG_TARGET_SVEHALF
        (nk_kernel_punned_t)&nk_angular_f16_svehalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angular_f16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angular_f16_skylake,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_angular_f16_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angular_f16_rvv,
#endif
#if NUMKONG_TARGET_RVVHALF
        (nk_kernel_punned_t)&nk_angular_f16_rvvhalf,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angular_f16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angular_f16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angular_f16_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_svehalf_k * NUMKONG_TARGET_SVEHALF |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_rvvhalf_k * NUMKONG_TARGET_RVVHALF | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_f16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_bf16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angular_bf16_neon,
#endif
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_angular_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SVEBFDOT
        (nk_kernel_punned_t)&nk_angular_bf16_svebfdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angular_bf16_haswell,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_angular_bf16_genoa,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angular_bf16_rvv,
#endif
#if NUMKONG_TARGET_RVVBF16
        (nk_kernel_punned_t)&nk_angular_bf16_rvvbf16,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_angular_bf16_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angular_bf16_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angular_bf16_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angular_bf16_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT |
             nk_cap_svebfdot_k * NUMKONG_TARGET_SVEBFDOT | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_rvvbf16_k * NUMKONG_TARGET_RVVBF16 | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_bf16_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_e4m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angular_e4m3_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angular_e4m3_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angular_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angular_e4m3_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angular_e4m3_icelake,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_angular_e4m3_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angular_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angular_e4m3_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_e4m3_best(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_e4m3_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_e5m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angular_e5m2_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angular_e5m2_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angular_e5m2_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angular_e5m2_skylake,
#endif
#if NUMKONG_TARGET_DIAMOND
        (nk_kernel_punned_t)&nk_angular_e5m2_diamond,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angular_e5m2_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angular_e5m2_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_e5m2_best(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_e5m2_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_e2m3_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angular_e2m3_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angular_e2m3_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angular_e2m3_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angular_e2m3_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angular_e2m3_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angular_e2m3_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angular_e2m3_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angular_e2m3_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_e2m3_best(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_e2m3_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_e3m2_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_angular_e3m2_neon,
#endif
#if NUMKONG_TARGET_NEONFP8
        (nk_kernel_punned_t)&nk_angular_e3m2_neonfp8,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angular_e3m2_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angular_e3m2_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angular_e3m2_sierra,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_angular_e3m2_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angular_e3m2_icelake,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angular_e3m2_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8 |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_e3m2_best(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_e3m2_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angular_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SVESDOT
        (nk_kernel_punned_t)&nk_angular_i8_svesdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angular_i8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angular_i8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angular_i8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angular_i8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angular_i8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_angular_i8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angular_i8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angular_i8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angular_i8_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_svesdot_k * NUMKONG_TARGET_SVESDOT |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_i8_best(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_i8_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_u8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_u8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angular_u8_neonsdot,
#endif
#if NUMKONG_TARGET_SVESDOT
        (nk_kernel_punned_t)&nk_angular_u8_svesdot,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_angular_u8_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_angular_u8_alder,
#endif
#if NUMKONG_TARGET_SIERRA
        (nk_kernel_punned_t)&nk_angular_u8_sierra,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angular_u8_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angular_u8_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_angular_u8_v128,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_angular_u8_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_angular_u8_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_angular_u8_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_svesdot_k * NUMKONG_TARGET_SVESDOT |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_sierra_k * NUMKONG_TARGET_SIERRA | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX |
             nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_u8_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_i4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_i4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angular_i4_neonsdot,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angular_i4_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angular_i4_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_i4_best(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_i4_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_angular_u4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_angular_u4_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_angular_u4_neonsdot,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_angular_u4_icelake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_angular_u4_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_angular_u4_best(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_metric_dense_punned_t const kernel = (nk_metric_dense_punned_t)nk_kernel_pick_(capabilities,
                                                                                      nk_angular_u4_capabilities());
    return kernel ? kernel(a, b, n, result, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_spatial_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                               nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f64_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_f64_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_f64_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_f64_capabilities(); break;
        default: break;
        }
        break;
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_f32_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_f32_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_f32_capabilities(); break;
        default: break;
        }
        break;
    case nk_bf16_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_bf16_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_bf16_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_bf16_capabilities(); break;
        default: break;
        }
        break;
    case nk_f16_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_f16_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_f16_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_f16_capabilities(); break;
        default: break;
        }
        break;
    case nk_e5m2_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_e5m2_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_e5m2_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_e5m2_capabilities(); break;
        default: break;
        }
        break;
    case nk_e4m3_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_e4m3_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_e4m3_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_e4m3_capabilities(); break;
        default: break;
        }
        break;
    case nk_e3m2_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_e3m2_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_e3m2_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_e3m2_capabilities(); break;
        default: break;
        }
        break;
    case nk_e2m3_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_e2m3_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_e2m3_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_e2m3_capabilities(); break;
        default: break;
        }
        break;
    case nk_i8_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_i8_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_i8_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_i8_capabilities(); break;
        default: break;
        }
        break;
    case nk_i4_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_i4_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_i4_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_i4_capabilities(); break;
        default: break;
        }
        break;
    case nk_u8_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_u8_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_u8_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_u8_capabilities(); break;
        default: break;
        }
        break;
    case nk_u4_k:
        switch (kind) {
        case nk_kernel_angular_k: lists = nk_angular_u4_capabilities(); break;
        case nk_kernel_euclidean_k: lists = nk_euclidean_u4_capabilities(); break;
        case nk_kernel_sqeuclidean_k: lists = nk_sqeuclidean_u4_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
