/**
 *  @file c/dispatch/scalar.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Dispatch points of the scalar math: square roots, FMA, saturating arithmetic and ordering.
 */
#include "dispatch.h"

static nk_capability_kernels_t const *nk_f32_sqrt_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_sqrt_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f32_sqrt_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f32_sqrt_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f32_sqrt_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_f32_sqrt_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f32_sqrt_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_f32_sqrt_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_f32_t nk_f32_sqrt_best(nk_f32_t x, nk_capability_t capabilities) {
    return ((nk_f32_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f32_sqrt_capabilities()))(x);
}

static nk_capability_kernels_t const *nk_f64_sqrt_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f64_sqrt_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f64_sqrt_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f64_sqrt_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f64_sqrt_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_f64_sqrt_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f64_sqrt_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_f64_sqrt_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_f64_t nk_f64_sqrt_best(nk_f64_t x, nk_capability_t capabilities) {
    return ((nk_f64_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f64_sqrt_capabilities()))(x);
}

static nk_capability_kernels_t const *nk_f32_rsqrt_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_rsqrt_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f32_rsqrt_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f32_rsqrt_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f32_rsqrt_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_f32_rsqrt_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f32_rsqrt_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_f32_rsqrt_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_f32_t nk_f32_rsqrt_best(nk_f32_t x, nk_capability_t capabilities) {
    return ((nk_f32_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f32_rsqrt_capabilities()))(x);
}

static nk_capability_kernels_t const *nk_f64_rsqrt_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f64_rsqrt_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f64_rsqrt_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f64_rsqrt_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f64_rsqrt_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_f64_rsqrt_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f64_rsqrt_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_f64_rsqrt_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_f64_t nk_f64_rsqrt_best(nk_f64_t x, nk_capability_t capabilities) {
    return ((nk_f64_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f64_rsqrt_capabilities()))(x);
}

static nk_capability_kernels_t const *nk_f32_fma_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_fma_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f32_fma_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f32_fma_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f32_fma_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_f32_fma_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f32_fma_powervsx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_f32_t nk_f32_fma_best(nk_f32_t a, nk_f32_t b, nk_f32_t c, nk_capability_t capabilities) {
    return ((nk_f32_ternary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                     nk_f32_fma_capabilities()))(a, b, c);
}

static nk_capability_kernels_t const *nk_f64_fma_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f64_fma_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f64_fma_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f64_fma_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f64_fma_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_f64_fma_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f64_fma_powervsx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_f64_t nk_f64_fma_best(nk_f64_t a, nk_f64_t b, nk_f64_t c, nk_capability_t capabilities) {
    return ((nk_f64_ternary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                     nk_f64_fma_capabilities()))(a, b, c);
}

static nk_capability_kernels_t const *nk_f16_sqrt_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f16_sqrt_serial,
#if NUMKONG_TARGET_NEONHALF
        (nk_kernel_punned_t)&nk_f16_sqrt_neonhalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f16_sqrt_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f16_sqrt_sapphire,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_f16_t nk_f16_sqrt_best(nk_f16_t x, nk_capability_t capabilities) {
    return ((nk_f16_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f16_sqrt_capabilities()))(x);
}

static nk_capability_kernels_t const *nk_f16_rsqrt_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f16_rsqrt_serial,
#if NUMKONG_TARGET_NEONHALF
        (nk_kernel_punned_t)&nk_f16_rsqrt_neonhalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f16_rsqrt_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f16_rsqrt_sapphire,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_f16_t nk_f16_rsqrt_best(nk_f16_t x, nk_capability_t capabilities) {
    return ((nk_f16_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f16_rsqrt_capabilities()))(x);
}

static nk_capability_kernels_t const *nk_f16_fma_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f16_fma_serial,
#if NUMKONG_TARGET_NEONHALF
        (nk_kernel_punned_t)&nk_f16_fma_neonhalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f16_fma_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f16_fma_sapphire,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_f16_t nk_f16_fma_best(nk_f16_t a, nk_f16_t b, nk_f16_t c, nk_capability_t capabilities) {
    return ((nk_f16_ternary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                     nk_f16_fma_capabilities()))(a, b, c);
}

static nk_capability_kernels_t const *nk_u8_saturating_add_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_u8_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_u8_saturating_add_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_u8_saturating_add_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u8_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_u8_t nk_u8_saturating_add_best(nk_u8_t a, nk_u8_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_u8_saturating_add_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_i8_saturating_add_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_i8_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_i8_saturating_add_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_i8_saturating_add_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i8_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_i8_t nk_i8_saturating_add_best(nk_i8_t a, nk_i8_t b, nk_capability_t capabilities) {
    return ((nk_i8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_i8_saturating_add_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_u16_saturating_add_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_u16_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_u16_saturating_add_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_u16_saturating_add_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u16_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_u16_t nk_u16_saturating_add_best(nk_u16_t a, nk_u16_t b, nk_capability_t capabilities) {
    return ((nk_u16_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u16_saturating_add_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_i16_saturating_add_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_i16_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_i16_saturating_add_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_i16_saturating_add_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i16_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_i16_t nk_i16_saturating_add_best(nk_i16_t a, nk_i16_t b, nk_capability_t capabilities) {
    return ((nk_i16_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i16_saturating_add_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_u32_saturating_add_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_u32_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_u32_saturating_add_neon,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u32_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_u32_t nk_u32_saturating_add_best(nk_u32_t a, nk_u32_t b, nk_capability_t capabilities) {
    return ((nk_u32_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u32_saturating_add_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_i32_saturating_add_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_i32_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_i32_saturating_add_neon,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i32_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_i32_t nk_i32_saturating_add_best(nk_i32_t a, nk_i32_t b, nk_capability_t capabilities) {
    return ((nk_i32_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i32_saturating_add_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_u64_saturating_add_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_u64_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_u64_saturating_add_neon,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u64_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_u64_t nk_u64_saturating_add_best(nk_u64_t a, nk_u64_t b, nk_capability_t capabilities) {
    return ((nk_u64_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u64_saturating_add_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_i64_saturating_add_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_i64_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_i64_saturating_add_neon,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i64_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_i64_t nk_i64_saturating_add_best(nk_i64_t a, nk_i64_t b, nk_capability_t capabilities) {
    return ((nk_i64_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i64_saturating_add_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_i4x2_saturating_add_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_i4x2_saturating_add_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_i4x2_t nk_i4x2_saturating_add_best(nk_i4x2_t a, nk_i4x2_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_i4x2_saturating_add_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_u4x2_saturating_add_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_u4x2_saturating_add_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_u4x2_t nk_u4x2_saturating_add_best(nk_u4x2_t a, nk_u4x2_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_u4x2_saturating_add_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_u8_saturating_mul_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_u8_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u8_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_u8_t nk_u8_saturating_mul_best(nk_u8_t a, nk_u8_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_u8_saturating_mul_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_i8_saturating_mul_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_i8_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i8_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_i8_t nk_i8_saturating_mul_best(nk_i8_t a, nk_i8_t b, nk_capability_t capabilities) {
    return ((nk_i8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_i8_saturating_mul_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_u16_saturating_mul_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_u16_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u16_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_u16_t nk_u16_saturating_mul_best(nk_u16_t a, nk_u16_t b, nk_capability_t capabilities) {
    return ((nk_u16_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u16_saturating_mul_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_i16_saturating_mul_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_i16_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i16_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_i16_t nk_i16_saturating_mul_best(nk_i16_t a, nk_i16_t b, nk_capability_t capabilities) {
    return ((nk_i16_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i16_saturating_mul_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_u32_saturating_mul_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_u32_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u32_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_u32_t nk_u32_saturating_mul_best(nk_u32_t a, nk_u32_t b, nk_capability_t capabilities) {
    return ((nk_u32_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u32_saturating_mul_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_i32_saturating_mul_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_i32_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i32_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_i32_t nk_i32_saturating_mul_best(nk_i32_t a, nk_i32_t b, nk_capability_t capabilities) {
    return ((nk_i32_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i32_saturating_mul_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_u64_saturating_mul_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_u64_saturating_mul_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_u64_saturating_mul_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_u64_saturating_mul_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u64_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_u64_t nk_u64_saturating_mul_best(nk_u64_t a, nk_u64_t b, nk_capability_t capabilities) {
    return ((nk_u64_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u64_saturating_mul_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_i64_saturating_mul_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_i64_saturating_mul_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_i64_saturating_mul_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_i64_saturating_mul_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i64_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_i64_t nk_i64_saturating_mul_best(nk_i64_t a, nk_i64_t b, nk_capability_t capabilities) {
    return ((nk_i64_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i64_saturating_mul_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_i4x2_saturating_mul_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_i4x2_saturating_mul_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_i4x2_t nk_i4x2_saturating_mul_best(nk_i4x2_t a, nk_i4x2_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_i4x2_saturating_mul_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_u4x2_saturating_mul_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_u4x2_saturating_mul_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_u4x2_t nk_u4x2_saturating_mul_best(nk_u4x2_t a, nk_u4x2_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_u4x2_saturating_mul_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_f16_order_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f16_order_serial,
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f16_order_sapphire,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API int nk_f16_order_best(nk_f16_t a, nk_f16_t b, nk_capability_t capabilities) {
    return ((nk_f16_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                     nk_f16_order_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_bf16_order_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_bf16_order_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API int nk_bf16_order_best(nk_bf16_t a, nk_bf16_t b, nk_capability_t capabilities) {
    return ((nk_bf16_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                      nk_bf16_order_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_e4m3_order_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e4m3_order_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API int nk_e4m3_order_best(nk_e4m3_t a, nk_e4m3_t b, nk_capability_t capabilities) {
    return ((nk_u8_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_e4m3_order_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_e5m2_order_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e5m2_order_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API int nk_e5m2_order_best(nk_e5m2_t a, nk_e5m2_t b, nk_capability_t capabilities) {
    return ((nk_u8_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_e5m2_order_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_e2m3_order_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e2m3_order_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API int nk_e2m3_order_best(nk_e2m3_t a, nk_e2m3_t b, nk_capability_t capabilities) {
    return ((nk_u8_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_e2m3_order_capabilities()))(a, b);
}

static nk_capability_kernels_t const *nk_e3m2_order_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e3m2_order_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API int nk_e3m2_order_best(nk_e3m2_t a, nk_e3m2_t b, nk_capability_t capabilities) {
    return ((nk_u8_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_e3m2_order_capabilities()))(a, b);
}

NUMKONG_API nk_status_t nk_scalar_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = (nk_kernel_punned_t)NUMKONG_NULL, *capability = 0;
    return nk_missing_kernel_k;
}
