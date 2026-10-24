/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for NEON BF16.
 *  @file include/numkong/spatial/neonbfdot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_neonbfdot_instructions ARM NEON BF16 Instructions (ARMv8.6-BF16)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vbfdotq_f32                 BFDOT (V.4S, V.8H, V.8H)        3cy         2/cy        4/cy
 *      vcvt_f32_bf16               BFCVTN (V.4H, V.4S)             3cy         2/cy        4/cy
 *      vld1q_bf16                  LD1 (V.8H)                      4cy         2/cy        3/cy
 *      vsubq_f32                   FSUB (V.4S, V.4S, V.4S)         2cy         2/cy        4/cy
 *      vfmaq_f64                   FMLA (V.2D, V.2D, V.2D)         4cy         2/cy        4/cy
 *      vaddvq_f32                  FADDP+FADDP (V.4S)              4cy         1/cy        2/cy
 *      vaddvq_f64                  FADDP (V.2D)                    3cy         1/cy        2/cy
 *
 *  The ARMv8.6-BF16 extension provides BFDOT for accelerated dot products on BF16 data, useful for
 *  angular distance (cosine similarity) computations. BF16's larger exponent range (matching FP32)
 *  prevents overflow during norm accumulation compared to FP16.
 *
 *  For L2 distance, inputs are converted to F32 for subtraction, then accumulated in F64 for
 *  numerical stability. Angular distance leverages BFDOT directly since it only requires dot
 *  products, not element-wise differences.
 */
#ifndef NK_SPATIAL_NEONBFDOT_H
#define NK_SPATIAL_NEONBFDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT

#include "numkong/types.h"
#include "numkong/cast/serial.h"  // `nk_partial_load_b16x4_serial_`
#include "numkong/reduce/neon.h"  // `nk_partial_load_b16x8_serial_`
#include "numkong/spatial/neon.h" // `nk_angular_normalize_f32_neon_`, `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#endif

NK_PUBLIC void nk_angular_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {

    // Similar to `nk_angular_i8_neon`, we can use the `BFMMLA` instruction through
    // the `vbfmmlaq_f32` intrinsic to compute matrix products and later drop 1/4 of values.
    // The only difference is that `zip` isn't provided for `bf16` and we need to reinterpret back
    // and forth before zipping. Same as with integers, on modern Arm CPUs, this "smart"
    // approach is actually slower by around 25%.
    //
    //   float32x4_t products_low_vec = vdupq_n_f32(0.0f);
    //   float32x4_t products_high_vec = vdupq_n_f32(0.0f);
    //   for (; i + 8 <= n; i += 8) {
    //       bfloat16x8_t a_vec = vld1q_bf16((nk_bf16_for_arm_simd_t const*)a + i);
    //       bfloat16x8_t b_vec = vld1q_bf16((nk_bf16_for_arm_simd_t const*)b + i);
    //       int16x8_t a_vec_s16 = vreinterpretq_s16_bf16(a_vec);
    //       int16x8_t b_vec_s16 = vreinterpretq_s16_bf16(b_vec);
    //       int16x8x2_t y_w_vecs_s16 = vzipq_s16(a_vec_s16, b_vec_s16);
    //       bfloat16x8_t y_vec = vreinterpretq_bf16_s16(y_w_vecs_s16.val[0]);
    //       bfloat16x8_t w_vec = vreinterpretq_bf16_s16(y_w_vecs_s16.val[1]);
    //       bfloat16x4_t a_low = vget_low_bf16(a_vec);
    //       bfloat16x4_t b_low = vget_low_bf16(b_vec);
    //       bfloat16x4_t a_high = vget_high_bf16(a_vec);
    //       bfloat16x4_t b_high = vget_high_bf16(b_vec);
    //       bfloat16x8_t x_vec = vcombine_bf16(a_low, b_low);
    //       bfloat16x8_t v_vec = vcombine_bf16(a_high, b_high);
    //       products_low_vec = vbfmmlaq_f32(products_low_vec, x_vec, y_vec);
    //       products_high_vec = vbfmmlaq_f32(products_high_vec, v_vec, w_vec);
    //   }
    //   float32x4_t products_vec = vaddq_f32(products_high_vec, products_low_vec);
    //   nk_f32_t a2 = products_vec[0], ab = products_vec[1], b2 = products_vec[3];
    //
    // Another way of accomplishing the same thing is to process the odd and even elements separately,
    // using special `vbfmlaltq_f32` and `vbfmlalbq_f32` intrinsics:
    //
    //      ab_high_vec = vbfmlaltq_f32(ab_high_vec, a_vec, b_vec);
    //      ab_low_vec = vbfmlalbq_f32(ab_low_vec, a_vec, b_vec);
    //      a2_high_vec = vbfmlaltq_f32(a2_high_vec, a_vec, a_vec);
    //      a2_low_vec = vbfmlalbq_f32(a2_low_vec, a_vec, a_vec);
    //      b2_high_vec = vbfmlaltq_f32(b2_high_vec, b_vec, b_vec);
    //      b2_low_vec = vbfmlalbq_f32(b2_low_vec, b_vec, b_vec);
    //

    float32x4_t dot_product_f32x4 = vdupq_n_f32(0);
    float32x4_t a_norm_sq_f32x4 = vdupq_n_f32(0);
    float32x4_t b_norm_sq_f32x4 = vdupq_n_f32(0);
    bfloat16x8_t a_bf16x8, b_bf16x8;

nk_angular_bf16_neonbfdot_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a, &a_vec, n);
        nk_partial_load_b16x8_serial_(b, &b_vec, n);
        a_bf16x8 = vreinterpretq_bf16_u16(a_vec.u16x8);
        b_bf16x8 = vreinterpretq_bf16_u16(b_vec.u16x8);
        n = 0;
    }
    else {
        a_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)a);
        b_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)b);
        n -= 8, a += 8, b += 8;
    }
    dot_product_f32x4 = vbfdotq_f32(dot_product_f32x4, a_bf16x8, b_bf16x8);
    a_norm_sq_f32x4 = vbfdotq_f32(a_norm_sq_f32x4, a_bf16x8, a_bf16x8);
    b_norm_sq_f32x4 = vbfdotq_f32(b_norm_sq_f32x4, b_bf16x8, b_bf16x8);
    if (n) goto nk_angular_bf16_neonbfdot_cycle;

    nk_f32_t dot_product_f32 = vaddvq_f32(dot_product_f32x4);
    nk_f32_t a_norm_sq_f32 = vaddvq_f32(a_norm_sq_f32x4);
    nk_f32_t b_norm_sq_f32 = vaddvq_f32(b_norm_sq_f32x4);
    *result = nk_angular_normalize_f32_neon_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_sqeuclidean_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    bfloat16x4_t a_bf16x4, b_bf16x4;

nk_sqeuclidean_bf16_neonbfdot_cycle:
    if (n < 4) {
        nk_b64_vec_t a_tail, b_tail;
        nk_partial_load_b16x4_serial_(a, &a_tail, n);
        nk_partial_load_b16x4_serial_(b, &b_tail, n);
        a_bf16x4 = vreinterpret_bf16_u16(a_tail.u16x4);
        b_bf16x4 = vreinterpret_bf16_u16(b_tail.u16x4);
        n = 0;
    }
    else {
        a_bf16x4 = vld1_bf16((nk_bf16_for_arm_simd_t const *)a);
        b_bf16x4 = vld1_bf16((nk_bf16_for_arm_simd_t const *)b);
        n -= 4, a += 4, b += 4;
    }
    float32x4_t a_f32x4 = vcvt_f32_bf16(a_bf16x4);
    float32x4_t b_f32x4 = vcvt_f32_bf16(b_bf16x4);
    float32x4_t diff_f32x4 = vsubq_f32(a_f32x4, b_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_f32x4, diff_f32x4);
    if (n) goto nk_sqeuclidean_bf16_neonbfdot_cycle;

    *result = vaddvq_f32(sum_f32x4);
}
NK_PUBLIC void nk_euclidean_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_bf16_neonbfdot(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM_
#endif // NK_SPATIAL_NEONBFDOT_H
