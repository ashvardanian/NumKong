/**
 *  @file c/dispatch/trigonometry.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Dispatch points of the trigonometric functions and rotary embeddings.
 */
#include "dispatch.h"

static nk_capability_kernels_t const *nk_trig_sin_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_trig_sin_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_trig_sin_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_trig_sin_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_trig_sin_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_trig_sin_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_trig_sin_f64_v128relaxed,
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

NUMKONG_API nk_status_t nk_trig_sin_f64_best(nk_f64_t const *inputs, nk_size_t n, nk_f64_t *outputs,
                                             nk_capability_t capabilities, void *stream) {
    nk_kernel_trig_punned_t const kernel = (nk_kernel_trig_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_trig_sin_f64_capabilities());
    return kernel ? kernel(inputs, n, outputs, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_trig_cos_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_trig_cos_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_trig_cos_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_trig_cos_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_trig_cos_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_trig_cos_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_trig_cos_f64_v128relaxed,
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

NUMKONG_API nk_status_t nk_trig_cos_f64_best(nk_f64_t const *inputs, nk_size_t n, nk_f64_t *outputs,
                                             nk_capability_t capabilities, void *stream) {
    nk_kernel_trig_punned_t const kernel = (nk_kernel_trig_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_trig_cos_f64_capabilities());
    return kernel ? kernel(inputs, n, outputs, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_trig_atan_f64_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_trig_atan_f64_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_trig_atan_f64_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_trig_atan_f64_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_trig_atan_f64_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_trig_atan_f64_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_trig_atan_f64_v128relaxed,
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

NUMKONG_API nk_status_t nk_trig_atan_f64_best(nk_f64_t const *inputs, nk_size_t n, nk_f64_t *outputs,
                                              nk_capability_t capabilities, void *stream) {
    nk_kernel_trig_punned_t const kernel = (nk_kernel_trig_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_trig_atan_f64_capabilities());
    return kernel ? kernel(inputs, n, outputs, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_trig_sin_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_trig_sin_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_trig_sin_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_trig_sin_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_trig_sin_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_trig_sin_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_trig_sin_f32_v128relaxed,
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

NUMKONG_API nk_status_t nk_trig_sin_f32_best(nk_f32_t const *inputs, nk_size_t n, nk_f32_t *outputs,
                                             nk_capability_t capabilities, void *stream) {
    nk_kernel_trig_punned_t const kernel = (nk_kernel_trig_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_trig_sin_f32_capabilities());
    return kernel ? kernel(inputs, n, outputs, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_trig_cos_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_trig_cos_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_trig_cos_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_trig_cos_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_trig_cos_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_trig_cos_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_trig_cos_f32_v128relaxed,
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

NUMKONG_API nk_status_t nk_trig_cos_f32_best(nk_f32_t const *inputs, nk_size_t n, nk_f32_t *outputs,
                                             nk_capability_t capabilities, void *stream) {
    nk_kernel_trig_punned_t const kernel = (nk_kernel_trig_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_trig_cos_f32_capabilities());
    return kernel ? kernel(inputs, n, outputs, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_trig_atan_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_trig_atan_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_trig_atan_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_trig_atan_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_trig_atan_f32_skylake,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_trig_atan_f32_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_trig_atan_f32_v128relaxed,
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

NUMKONG_API nk_status_t nk_trig_atan_f32_best(nk_f32_t const *inputs, nk_size_t n, nk_f32_t *outputs,
                                              nk_capability_t capabilities, void *stream) {
    nk_kernel_trig_punned_t const kernel = (nk_kernel_trig_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_trig_atan_f32_capabilities());
    return kernel ? kernel(inputs, n, outputs, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_trig_sin_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_trig_sin_f16_serial,
#if NUMKONG_TARGET_NEONHALF
        (nk_kernel_punned_t)&nk_trig_sin_f16_neonhalf,
#endif
#if NUMKONG_TARGET_SVEHALF
        (nk_kernel_punned_t)&nk_trig_sin_f16_svehalf,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_trig_sin_f16_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_trig_sin_f16_sapphire,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_trig_sin_f16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF | nk_cap_svehalf_k * NUMKONG_TARGET_SVEHALF |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_trig_sin_f16_best(nk_f16_t const *inputs, nk_size_t n, nk_f16_t *outputs,
                                             nk_capability_t capabilities, void *stream) {
    nk_kernel_trig_punned_t const kernel = (nk_kernel_trig_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_trig_sin_f16_capabilities());
    return kernel ? kernel(inputs, n, outputs, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_trig_cos_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_trig_cos_f16_serial,
#if NUMKONG_TARGET_NEONHALF
        (nk_kernel_punned_t)&nk_trig_cos_f16_neonhalf,
#endif
#if NUMKONG_TARGET_SVEHALF
        (nk_kernel_punned_t)&nk_trig_cos_f16_svehalf,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_trig_cos_f16_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_trig_cos_f16_sapphire,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_trig_cos_f16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF | nk_cap_svehalf_k * NUMKONG_TARGET_SVEHALF |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_trig_cos_f16_best(nk_f16_t const *inputs, nk_size_t n, nk_f16_t *outputs,
                                             nk_capability_t capabilities, void *stream) {
    nk_kernel_trig_punned_t const kernel = (nk_kernel_trig_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_trig_cos_f16_capabilities());
    return kernel ? kernel(inputs, n, outputs, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_trig_atan_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_trig_atan_f16_serial,
#if NUMKONG_TARGET_NEONHALF
        (nk_kernel_punned_t)&nk_trig_atan_f16_neonhalf,
#endif
#if NUMKONG_TARGET_SVEHALF
        (nk_kernel_punned_t)&nk_trig_atan_f16_svehalf,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_trig_atan_f16_skylake,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_trig_atan_f16_sapphire,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_trig_atan_f16_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF | nk_cap_svehalf_k * NUMKONG_TARGET_SVEHALF |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_trig_atan_f16_best(nk_f16_t const *inputs, nk_size_t n, nk_f16_t *outputs,
                                              nk_capability_t capabilities, void *stream) {
    nk_kernel_trig_punned_t const kernel = (nk_kernel_trig_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_trig_atan_f16_capabilities());
    return kernel ? kernel(inputs, n, outputs, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_trigonometry_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype,
                                                    nk_capability_t capabilities, nk_kernel_punned_t *kernel,
                                                    nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f64_k:
        switch (kind) {
        case nk_kernel_trig_sin_k: lists = nk_trig_sin_f64_capabilities(); break;
        case nk_kernel_trig_cos_k: lists = nk_trig_cos_f64_capabilities(); break;
        case nk_kernel_trig_atan_k: lists = nk_trig_atan_f64_capabilities(); break;
        default: break;
        }
        break;
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_trig_sin_k: lists = nk_trig_sin_f32_capabilities(); break;
        case nk_kernel_trig_cos_k: lists = nk_trig_cos_f32_capabilities(); break;
        case nk_kernel_trig_atan_k: lists = nk_trig_atan_f32_capabilities(); break;
        default: break;
        }
        break;
    case nk_f16_k:
        switch (kind) {
        case nk_kernel_trig_sin_k: lists = nk_trig_sin_f16_capabilities(); break;
        case nk_kernel_trig_cos_k: lists = nk_trig_cos_f16_capabilities(); break;
        case nk_kernel_trig_atan_k: lists = nk_trig_atan_f16_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
