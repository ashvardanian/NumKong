/**
 *  @file include/numkong/each/neonfhm.h
 *  @author Ash Vardanian
 *  @date October 31, 2026
 *  @brief SIMD-accelerated elementwise arithmetic for NEON FHM.
 *
 *  @sa include/numkong/each.h
 *
 *  @section elementwise_neonfhm_instructions ARM NEON FP16 FHM Instructions (ARMv8.4-FHM)
 *
 *  @verbatim
 *  Intrinsic         Instruction                A76       M5
 *  vfmlalq_low_f16   FMLAL (V.4S, V.8H, V.8H)   4cy @ 2p  4cy @ 4p
 *  vfmlalq_high_f16  FMLAL2 (V.4S, V.8H, V.8H)  4cy @ 2p  4cy @ 4p
 *  vcvt_f32_f16      FCVTL (V.4S, V.4H)         3cy @ 2p  3cy @ 4p
 *  vcvt_f16_f32      FCVTN (V.4H, V.4S)         3cy @ 2p  3cy @ 4p
 *  vmulq_f32         FMUL (V.4S, V.4S, V.4S)    3cy @ 2p  3cy @ 4p
 *  @endverbatim
 *
 *  FMLAL multiplies two F16 values into F32 lanes, where their product is always exact, so it takes
 *  the place of two FCVTLs and an FMUL wherever both factors are F16, as `a * b` in fma. FMLAL by 1
 *  folds an FCVTL and an FADD into one rounding, as in `shapes/neonfhm.h`. It cannot take an F32
 *  weight, which is why scale and blend stay in `each/neon.h`. Every kernel here returns the same
 *  bits as its NEON sibling.
 */
#ifndef NUMKONG_EACH_NEONFHM_H
#define NUMKONG_EACH_NEONFHM_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONFHM

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b16x8_serial_`, `nk_partial_store_b16x8_serial_`
#include "numkong/cast/neon.h"   // `nk_e4m3x8_to_f16x8_neon_`
#include "numkong/each/neon.h"   // `nk_f32x4_pair_to_e4m3x8_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

/** Computes `a * b * alpha + c * beta` over 8 F16 values in F32. */
NUMKONG_INLINE float16x8_t nk_fma_f16x8_neonfhm_(float16x8_t a_f16x8, float16x8_t b_f16x8, float16x8_t c_f16x8,
                                                 float32x4_t alpha_f32x4, float32x4_t beta_f32x4) {
    // Adding an exact product to −0 keeps the sign of a zero, as FMUL does
    float32x4_t const negative_zero_f32x4 = vdupq_n_f32(-0.0f);
    float32x4_t ab_low_f32x4 = vfmlalq_low_f16(negative_zero_f32x4, a_f16x8, b_f16x8);
    float32x4_t ab_high_f32x4 = vfmlalq_high_f16(negative_zero_f32x4, a_f16x8, b_f16x8);
    float32x4_t result_low_f32x4 = vfmaq_f32(vmulq_f32(ab_low_f32x4, alpha_f32x4), vcvt_f32_f16(vget_low_f16(c_f16x8)),
                                             beta_f32x4);
    float32x4_t result_high_f32x4 = vfmaq_f32(vmulq_f32(ab_high_f32x4, alpha_f32x4), vcvt_high_f32_f16(c_f16x8),
                                              beta_f32x4);
    return vcvt_high_f16_f32(vcvt_f16_f32(result_low_f32x4), result_high_f32x4);
}

/** Adds 8 E4M3 values in F32, with one FCVTL and FADD pair folded into FMLAL by 1. */
NUMKONG_INLINE uint8x8_t nk_sum_e4m3x8_neonfhm_(uint8x8_t a_e4m3x8, uint8x8_t b_e4m3x8) {
    float16x8_t const ones_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neon_(a_e4m3x8), b_f16x8 = nk_e4m3x8_to_f16x8_neon_(b_e4m3x8);
    float32x4_t result_low_f32x4 = vfmlalq_low_f16(vcvt_f32_f16(vget_low_f16(a_f16x8)), b_f16x8, ones_f16x8);
    float32x4_t result_high_f32x4 = vfmlalq_high_f16(vcvt_high_f32_f16(a_f16x8), b_f16x8, ones_f16x8);
    return nk_f32x4_pair_to_e4m3x8_neon_(result_low_f32x4, result_high_f32x4);
}

/** Computes `a * b * alpha + c * beta` over 8 E4M3 values in F32. */
NUMKONG_INLINE uint8x8_t nk_fma_e4m3x8_neonfhm_(uint8x8_t a_e4m3x8, uint8x8_t b_e4m3x8, uint8x8_t c_e4m3x8,
                                                float32x4_t alpha_f32x4, float32x4_t beta_f32x4) {
    float32x4_t const negative_zero_f32x4 = vdupq_n_f32(-0.0f);
    float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neon_(a_e4m3x8), b_f16x8 = nk_e4m3x8_to_f16x8_neon_(b_e4m3x8);
    float16x8_t c_f16x8 = nk_e4m3x8_to_f16x8_neon_(c_e4m3x8);
    float32x4_t ab_low_f32x4 = vfmlalq_low_f16(negative_zero_f32x4, a_f16x8, b_f16x8);
    float32x4_t ab_high_f32x4 = vfmlalq_high_f16(negative_zero_f32x4, a_f16x8, b_f16x8);
    return nk_f32x4_pair_to_e4m3x8_neon_(
        vfmaq_f32(vmulq_f32(ab_low_f32x4, alpha_f32x4), vcvt_f32_f16(vget_low_f16(c_f16x8)), beta_f32x4),
        vfmaq_f32(vmulq_f32(ab_high_f32x4, alpha_f32x4), vcvt_high_f32_f16(c_f16x8), beta_f32x4));
}

/** Adds 8 E5M2 values in F32, with one FCVTL and FADD pair folded into FMLAL by 1. */
NUMKONG_INLINE uint8x8_t nk_sum_e5m2x8_neonfhm_(uint8x8_t a_e5m2x8, uint8x8_t b_e5m2x8) {
    float16x8_t const ones_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neon_(a_e5m2x8), b_f16x8 = nk_e5m2x8_to_f16x8_neon_(b_e5m2x8);
    float32x4_t result_low_f32x4 = vfmlalq_low_f16(vcvt_f32_f16(vget_low_f16(a_f16x8)), b_f16x8, ones_f16x8);
    float32x4_t result_high_f32x4 = vfmlalq_high_f16(vcvt_high_f32_f16(a_f16x8), b_f16x8, ones_f16x8);
    return nk_f32x4_pair_to_e5m2x8_neon_(result_low_f32x4, result_high_f32x4);
}

/** Computes `a * b * alpha + c * beta` over 8 E5M2 values in F32. */
NUMKONG_INLINE uint8x8_t nk_fma_e5m2x8_neonfhm_(uint8x8_t a_e5m2x8, uint8x8_t b_e5m2x8, uint8x8_t c_e5m2x8,
                                                float32x4_t alpha_f32x4, float32x4_t beta_f32x4) {
    float32x4_t const negative_zero_f32x4 = vdupq_n_f32(-0.0f);
    float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neon_(a_e5m2x8), b_f16x8 = nk_e5m2x8_to_f16x8_neon_(b_e5m2x8);
    float16x8_t c_f16x8 = nk_e5m2x8_to_f16x8_neon_(c_e5m2x8);
    float32x4_t ab_low_f32x4 = vfmlalq_low_f16(negative_zero_f32x4, a_f16x8, b_f16x8);
    float32x4_t ab_high_f32x4 = vfmlalq_high_f16(negative_zero_f32x4, a_f16x8, b_f16x8);
    return nk_f32x4_pair_to_e5m2x8_neon_(
        vfmaq_f32(vmulq_f32(ab_low_f32x4, alpha_f32x4), vcvt_f32_f16(vget_low_f16(c_f16x8)), beta_f32x4),
        vfmaq_f32(vmulq_f32(ab_high_f32x4, alpha_f32x4), vcvt_high_f32_f16(c_f16x8), beta_f32x4));
}

NUMKONG_API nk_status_t nk_each_fma_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                                nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result,
                                                nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float32x4_t alpha_f32x4 = vdupq_n_f32(*alpha);
    float32x4_t beta_f32x4 = vdupq_n_f32(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)a + i));
        float16x8_t b_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)b + i));
        float16x8_t c_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)c + i));
        vst1q_u16((nk_u16_t *)result + i,
                  vreinterpretq_u16_f16(nk_fma_f16x8_neonfhm_(a_f16x8, b_f16x8, c_f16x8, alpha_f32x4, beta_f32x4)));
    }
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
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_sum_e4m3_neonfhm(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_e4m3_t *result,
                                                 nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) vst1_u8(result + i, nk_sum_e4m3x8_neonfhm_(vld1_u8(a + i), vld1_u8(b + i)));
    if (i < n) {
        nk_b64_vec_t a_vec, b_vec, result_vec;
        nk_partial_load_b8x8_serial_(a + i, &a_vec, n - i);
        nk_partial_load_b8x8_serial_(b + i, &b_vec, n - i);
        result_vec.u8x8 = nk_sum_e4m3x8_neonfhm_(a_vec.u8x8, b_vec.u8x8);
        nk_partial_store_b8x8_serial_(&result_vec, result + i, n - i);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_fma_e4m3_neonfhm(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c,
                                                 nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                 nk_e4m3_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float32x4_t alpha_f32x4 = vdupq_n_f32(*alpha);
    float32x4_t beta_f32x4 = vdupq_n_f32(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8)
        vst1_u8(result + i,
                nk_fma_e4m3x8_neonfhm_(vld1_u8(a + i), vld1_u8(b + i), vld1_u8(c + i), alpha_f32x4, beta_f32x4));
    if (i < n) {
        nk_b64_vec_t a_vec, b_vec, c_vec, result_vec;
        nk_partial_load_b8x8_serial_(a + i, &a_vec, n - i);
        nk_partial_load_b8x8_serial_(b + i, &b_vec, n - i);
        nk_partial_load_b8x8_serial_(c + i, &c_vec, n - i);
        result_vec.u8x8 = nk_fma_e4m3x8_neonfhm_(a_vec.u8x8, b_vec.u8x8, c_vec.u8x8, alpha_f32x4, beta_f32x4);
        nk_partial_store_b8x8_serial_(&result_vec, result + i, n - i);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_sum_e5m2_neonfhm(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_e5m2_t *result,
                                                 nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) vst1_u8(result + i, nk_sum_e5m2x8_neonfhm_(vld1_u8(a + i), vld1_u8(b + i)));
    if (i < n) {
        nk_b64_vec_t a_vec, b_vec, result_vec;
        nk_partial_load_b8x8_serial_(a + i, &a_vec, n - i);
        nk_partial_load_b8x8_serial_(b + i, &b_vec, n - i);
        result_vec.u8x8 = nk_sum_e5m2x8_neonfhm_(a_vec.u8x8, b_vec.u8x8);
        nk_partial_store_b8x8_serial_(&result_vec, result + i, n - i);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_fma_e5m2_neonfhm(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c,
                                                 nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                                 nk_e5m2_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float32x4_t alpha_f32x4 = vdupq_n_f32(*alpha);
    float32x4_t beta_f32x4 = vdupq_n_f32(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8)
        vst1_u8(result + i,
                nk_fma_e5m2x8_neonfhm_(vld1_u8(a + i), vld1_u8(b + i), vld1_u8(c + i), alpha_f32x4, beta_f32x4));
    if (i < n) {
        nk_b64_vec_t a_vec, b_vec, c_vec, result_vec;
        nk_partial_load_b8x8_serial_(a + i, &a_vec, n - i);
        nk_partial_load_b8x8_serial_(b + i, &b_vec, n - i);
        nk_partial_load_b8x8_serial_(c + i, &c_vec, n - i);
        result_vec.u8x8 = nk_fma_e5m2x8_neonfhm_(a_vec.u8x8, b_vec.u8x8, c_vec.u8x8, alpha_f32x4, beta_f32x4);
        nk_partial_store_b8x8_serial_(&result_vec, result + i, n - i);
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_NEONFHM
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_EACH_NEONFHM_H
