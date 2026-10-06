/**
 *  @file c/dispatch/cast.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Dispatch points of the type conversions: bulk, block-scaled and scalar casts.
 */
#include "dispatch.h"

static nk_capability_kernels_t const *nk_cast_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_cast_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_cast_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_cast_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_cast_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_cast_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_cast_sapphire,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_cast_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_cast_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_cast_powervsx,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_cast_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_cast_ampere,
#endif
#if NUMKONG_TARGET_ADA
        (nk_kernel_punned_t)&nk_cast_ada,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_ada_k * NUMKONG_TARGET_ADA,
         cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_f16_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f16_to_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f16_to_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f16_to_f32_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f16_to_f32_sapphire,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f16_to_f32_powervsx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f16_to_f32_best(nk_f16_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_f16_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                             nk_f16_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_bf16_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_bf16_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_bf16_to_f32_best(nk_bf16_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_bf16_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                              nk_bf16_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_e4m3_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e4m3_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_e4m3_to_f32_best(nk_e4m3_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_u8_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_e4m3_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_e5m2_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e5m2_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_e5m2_to_f32_best(nk_e5m2_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_u8_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_e5m2_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_e2m3_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e2m3_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_e2m3_to_f32_best(nk_e2m3_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_u8_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_e2m3_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_e3m2_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e3m2_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_e3m2_to_f32_best(nk_e3m2_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_u8_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_e3m2_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f32_to_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f32_to_f16_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f32_to_f16_sapphire,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f32_to_f16_powervsx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_f16_best(nk_f32_t const *source, nk_f16_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_f16_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                             nk_f32_to_f16_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_bf16_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_bf16_best(nk_f32_t const *source, nk_bf16_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_bf16_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                              nk_f32_to_bf16_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_e4m3_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_e4m3_best(nk_f32_t const *source, nk_e4m3_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_u8_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_f32_to_e4m3_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_e5m2_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_e5m2_best(nk_f32_t const *source, nk_e5m2_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_u8_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_f32_to_e5m2_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_e2m3_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_e2m3_best(nk_f32_t const *source, nk_e2m3_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_u8_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_f32_to_e2m3_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_e3m2_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_e3m2_best(nk_f32_t const *source, nk_e3m2_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_u8_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_f32_to_e3m2_capabilities()))(source, destination);
}

NUMKONG_API nk_status_t nk_cast_best(void const *from, nk_dtype_t from_dtype, void *to, nk_dtype_t to_dtype,
                                     nk_size_t count, nk_capability_t capabilities, void *stream) {
    nk_kernel_cast_punned_t const kernel = (nk_kernel_cast_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_cast_capabilities());
    return kernel ? kernel(from, from_dtype, to, to_dtype, count, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_cast_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                            nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_dtype_unknown_k:
        switch (kind) {
        case nk_kernel_cast_k: lists = nk_cast_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
