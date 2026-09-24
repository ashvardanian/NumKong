/**
 *  @brief SIMD-accelerated Scalar Math Helpers for NEON FP16 (FEAT_FP16).
 *  @file include/numkong/scalar/neonhalf.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  ARMv8.2-A FEAT_FP16 provides native scalar f16 sqrt, rsqrt estimate, and fma.
 *  `vrsqrte_f16` gives ~4-bit estimate; 2 Newton-Raphson steps refine to ~16 bits,
 *  exceeding f16's 10-bit mantissa precision.
 */
#ifndef NK_SCALAR_NEONHALF_H
#define NK_SCALAR_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

NK_PUBLIC nk_f16_t nk_f16_sqrt_neonhalf(nk_f16_t x) {
    float16x4_t x_f16x4 = vld1_dup_f16((nk_f16_for_arm_simd_t const *)&x);
    x_f16x4 = vsqrt_f16(x_f16x4);
    nk_f16_t result;
    vst1_lane_f16((nk_f16_for_arm_simd_t *)&result, x_f16x4, 0);
    return result;
}
NK_PUBLIC nk_f16_t nk_f16_rsqrt_neonhalf(nk_f16_t x) {
    float16x4_t x_f16x4 = vld1_dup_f16((nk_f16_for_arm_simd_t const *)&x);
    float16x4_t estimate_f16x4 = vrsqrte_f16(x_f16x4);
    estimate_f16x4 = vmul_f16(estimate_f16x4, vrsqrts_f16(vmul_f16(x_f16x4, estimate_f16x4), estimate_f16x4));
    estimate_f16x4 = vmul_f16(estimate_f16x4, vrsqrts_f16(vmul_f16(x_f16x4, estimate_f16x4), estimate_f16x4));
    nk_f16_t result;
    vst1_lane_f16((nk_f16_for_arm_simd_t *)&result, estimate_f16x4, 0);
    return result;
}
NK_PUBLIC nk_f16_t nk_f16_fma_neonhalf(nk_f16_t a, nk_f16_t b, nk_f16_t c) {
    float16x4_t a_f16x4 = vld1_dup_f16((nk_f16_for_arm_simd_t const *)&a);
    float16x4_t b_f16x4 = vld1_dup_f16((nk_f16_for_arm_simd_t const *)&b);
    float16x4_t c_f16x4 = vld1_dup_f16((nk_f16_for_arm_simd_t const *)&c);
    c_f16x4 = vfma_f16(c_f16x4, a_f16x4, b_f16x4);
    nk_f16_t result;
    vst1_lane_f16((nk_f16_for_arm_simd_t *)&result, c_f16x4, 0);
    return result;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_
#endif // NK_SCALAR_NEONHALF_H
