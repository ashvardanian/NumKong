/**
 *  @brief SIMD-accelerated Elementwise Arithmetic for NEON FHM.
 *  @file include/numkong/each/neonfhm.h
 *  @author Ash Vardanian
 *  @date October 31, 2026
 *
 *  @sa include/numkong/each.h
 *
 *  @section elementwise_neonfhm_instructions ARM NEON FP16 FHM Instructions (ARMv8.4-FHM)
 *
 *      Intrinsic         Instruction                A76       M5
 *      vfmlalq_low_f16   FMLAL (V.4S, V.8H, V.8H)   4cy @ 2p  4cy @ 4p
 *      vfmlalq_high_f16  FMLAL2 (V.4S, V.8H, V.8H)  4cy @ 2p  4cy @ 4p
 *      vcvt_f32_f16      FCVTL (V.4S, V.4H)         3cy @ 2p  3cy @ 4p
 *      vcvt_f16_f32      FCVTN (V.4H, V.4S)         3cy @ 2p  3cy @ 4p
 *      vmulq_f32         FMUL (V.4S, V.4S, V.4S)    3cy @ 2p  3cy @ 4p
 *
 *  FMLAL multiplies two F16 values into F32 lanes, where their product is always exact, so it takes
 *  the place of two FCVTLs and an FMUL wherever both factors are F16. That covers `a * b` in fma. It
 *  cannot take an F32 weight, which is why scale and blend stay in `each/neon.h`. Every kernel here
 *  returns the same bits as its NEON sibling.
 */
#ifndef NK_EACH_NEONFHM_H
#define NK_EACH_NEONFHM_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONFHM

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b16x8_serial_`, `nk_partial_store_b16x8_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

/** Computes `a * b * alpha + c * beta` over 8 F16 values in F32, unfused, as the serial kernel does. */
NK_INTERNAL float16x8_t nk_fma_f16x8_neonfhm_(float16x8_t a_f16x8, float16x8_t b_f16x8, float16x8_t c_f16x8,
                                              float32x4_t alpha_f32x4, float32x4_t beta_f32x4) {
    // Adding an exact product to −0 keeps the sign of a zero, as FMUL does
    float32x4_t const negative_zero_f32x4 = vdupq_n_f32(-0.0f);
    float32x4_t ab_low_f32x4 = vfmlalq_low_f16(negative_zero_f32x4, a_f16x8, b_f16x8);
    float32x4_t ab_high_f32x4 = vfmlalq_high_f16(negative_zero_f32x4, a_f16x8, b_f16x8);
    float32x4_t c_scaled_low_f32x4 = vmulq_f32(vcvt_f32_f16(vget_low_f16(c_f16x8)), beta_f32x4);
    float32x4_t c_scaled_high_f32x4 = vmulq_f32(vcvt_high_f32_f16(c_f16x8), beta_f32x4);
    float32x4_t result_low_f32x4 = vaddq_f32(vmulq_f32(ab_low_f32x4, alpha_f32x4), c_scaled_low_f32x4);
    float32x4_t result_high_f32x4 = vaddq_f32(vmulq_f32(ab_high_f32x4, alpha_f32x4), c_scaled_high_f32x4);
    return vcvt_high_f16_f32(vcvt_f16_f32(result_low_f32x4), result_high_f32x4);
}

NK_PUBLIC void nk_each_fma_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                       nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {
    float32x4_t alpha_f32x4 = vdupq_n_f32(*alpha);
    float32x4_t beta_f32x4 = vdupq_n_f32(*beta);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)a + i));
        float16x8_t b_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)b + i));
        float16x8_t c_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)c + i));
        vst1q_u16((nk_u16_t *)result + i,
                  vreinterpretq_u16_f16(nk_fma_f16x8_neonfhm_(a_f16x8, b_f16x8, c_f16x8, alpha_f32x4, beta_f32x4)));
    }

    // The tail:
    if (i < n) {
        nk_b128_vec_t a_vec, b_vec, c_vec, result_vec;
        nk_partial_load_b16x8_serial_(a + i, &a_vec, n - i);
        nk_partial_load_b16x8_serial_(b + i, &b_vec, n - i);
        nk_partial_load_b16x8_serial_(c + i, &c_vec, n - i);
        float16x8_t a_f16x8 = vreinterpretq_f16_u16(a_vec.u16x8), b_f16x8 = vreinterpretq_f16_u16(b_vec.u16x8);
        float16x8_t c_f16x8 = vreinterpretq_f16_u16(c_vec.u16x8);
        result_vec.u16x8 = vreinterpretq_u16_f16(
            nk_fma_f16x8_neonfhm_(a_f16x8, b_f16x8, c_f16x8, alpha_f32x4, beta_f32x4));
        nk_partial_store_b16x8_serial_(&result_vec, result + i, n - i);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM64_
#endif // NK_EACH_NEONFHM_H
