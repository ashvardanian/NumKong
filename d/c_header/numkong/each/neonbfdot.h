/**
 *  @brief SIMD-accelerated Elementwise Arithmetic for NEON BF16.
 *  @file include/numkong/each/neonbfdot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/each.h
 *
 *  @section elementwise_neonbfdot_instructions ARM NEON BF16 Instructions (ARMv8.6-BF16)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vld1_bf16                   LD1 (V.4H)                      4cy         2/cy        3/cy
 *      vst1_bf16                   ST1 (V.4H)                      2cy         2/cy        3/cy
 *      vcvt_f32_bf16               BFCVTN (V.4H, V.4S)             3cy         2/cy        4/cy
 *      vcvt_bf16_f32               BFCVT (V.4H, V.4S)              3cy         2/cy        4/cy
 *      vaddq_f32                   FADD (V.4S, V.4S, V.4S)         2cy         2/cy        4/cy
 *      vmulq_f32                   FMUL (V.4S, V.4S, V.4S)         3cy         2/cy        4/cy
 *      vmulq_n_f32                 FMUL (V.4S, V.4S, scalar)       3cy         2/cy        4/cy
 *      vfmaq_f32                   FMLA (V.4S, V.4S, V.4S)         4cy         2/cy        4/cy
 *      vfmaq_n_f32                 FMLA (V.4S, V.4S, scalar)       4cy         2/cy        4/cy
 *      vdupq_n_f32                 DUP (V.4S, scalar)              2cy         2/cy        4/cy
 *
 *  The ARMv8.6-BF16 extension provides element-wise operations on BF16 data by converting to F32
 *  for arithmetic, then back to BF16 for storage. This preserves the dynamic range benefits of BF16
 *  (matching F32 exponent) while using F32 precision for intermediate calculations.
 *
 *  Operations process 4 BF16 elements at a time, widening to F32 for computation. While this gives
 *  lower throughput than native F16 operations, it prevents overflow issues common with FP16's
 *  limited exponent range in ML training workloads.
 */
#ifndef NK_EACH_NEONBFDOT_H
#define NK_EACH_NEONBFDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT

#include "numkong/types.h"
#include "numkong/cast/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#endif

NK_PUBLIC void nk_each_sum_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        bfloat16x4_t a_bf16x4 = vld1_bf16((bfloat16_t const *)a + i);
        bfloat16x4_t b_bf16x4 = vld1_bf16((bfloat16_t const *)b + i);
        float32x4_t a_f32x4 = vcvt_f32_bf16(a_bf16x4);
        float32x4_t b_f32x4 = vcvt_f32_bf16(b_bf16x4);
        float32x4_t result_f32x4 = vaddq_f32(a_f32x4, b_f32x4);
        bfloat16x4_t result_bf16x4 = vcvt_bf16_f32(result_f32x4);
        vst1_bf16((bfloat16_t *)result + i, result_bf16x4);
    }
    if (i < n) {
        nk_b64_vec_t a_tail, b_tail;
        nk_partial_load_b16x4_serial_(a + i, &a_tail, n - i);
        nk_partial_load_b16x4_serial_(b + i, &b_tail, n - i);
        bfloat16x4_t a_bf16x4 = vreinterpret_bf16_u16(a_tail.u16x4);
        bfloat16x4_t b_bf16x4 = vreinterpret_bf16_u16(b_tail.u16x4);
        float32x4_t a_f32x4 = vcvt_f32_bf16(a_bf16x4);
        float32x4_t b_f32x4 = vcvt_f32_bf16(b_bf16x4);
        float32x4_t result_f32x4 = vaddq_f32(a_f32x4, b_f32x4);
        bfloat16x4_t result_bf16x4 = vcvt_bf16_f32(result_f32x4);
        nk_b64_vec_t result_vec;
        result_vec.u16x4 = vreinterpret_u16_bf16(result_bf16x4);
        nk_partial_store_b16x4_serial_(result + i, &result_vec, n - i);
    }
}

NK_PUBLIC void nk_each_scale_bf16_neonbfdot(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                            nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    float32x4_t alpha_f32x4 = vdupq_n_f32(alpha_val);
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        bfloat16x4_t a_bf16x4 = vld1_bf16((bfloat16_t const *)a + i);
        float32x4_t a_f32x4 = vcvt_f32_bf16(a_bf16x4);
        float32x4_t result_f32x4 = vfmaq_f32(beta_f32x4, a_f32x4, alpha_f32x4);
        bfloat16x4_t result_bf16x4 = vcvt_bf16_f32(result_f32x4);
        vst1_bf16((bfloat16_t *)result + i, result_bf16x4);
    }
    if (i < n) {
        nk_b64_vec_t a_tail;
        nk_partial_load_b16x4_serial_(a + i, &a_tail, n - i);
        bfloat16x4_t a_bf16x4 = vreinterpret_bf16_u16(a_tail.u16x4);
        float32x4_t a_f32x4 = vcvt_f32_bf16(a_bf16x4);
        float32x4_t result_f32x4 = vfmaq_f32(beta_f32x4, a_f32x4, alpha_f32x4);
        bfloat16x4_t result_bf16x4 = vcvt_bf16_f32(result_f32x4);
        nk_b64_vec_t result_vec;
        result_vec.u16x4 = vreinterpret_u16_bf16(result_bf16x4);
        nk_partial_store_b16x4_serial_(result + i, &result_vec, n - i);
    }
}

NK_PUBLIC void nk_each_blend_bf16_neonbfdot(             //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_bf16_neonbfdot(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_bf16_neonbfdot(a, n, alpha, &zero, result); }
        else { nk_each_scale_bf16_neonbfdot(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        bfloat16x4_t a_bf16x4 = vld1_bf16((bfloat16_t const *)a + i);
        bfloat16x4_t b_bf16x4 = vld1_bf16((bfloat16_t const *)b + i);
        float32x4_t a_f32x4 = vcvt_f32_bf16(a_bf16x4);
        float32x4_t b_f32x4 = vcvt_f32_bf16(b_bf16x4);
        float32x4_t a_scaled_f32x4 = vmulq_n_f32(a_f32x4, alpha_val);
        float32x4_t b_scaled_f32x4 = vmulq_n_f32(b_f32x4, beta_val);
        float32x4_t result_f32x4 = vaddq_f32(a_scaled_f32x4, b_scaled_f32x4);
        bfloat16x4_t result_bf16x4 = vcvt_bf16_f32(result_f32x4);
        vst1_bf16((bfloat16_t *)result + i, result_bf16x4);
    }
    if (i < n) {
        nk_b64_vec_t a_tail, b_tail;
        nk_partial_load_b16x4_serial_(a + i, &a_tail, n - i);
        nk_partial_load_b16x4_serial_(b + i, &b_tail, n - i);
        bfloat16x4_t a_bf16x4 = vreinterpret_bf16_u16(a_tail.u16x4);
        bfloat16x4_t b_bf16x4 = vreinterpret_bf16_u16(b_tail.u16x4);
        float32x4_t a_f32x4 = vcvt_f32_bf16(a_bf16x4);
        float32x4_t b_f32x4 = vcvt_f32_bf16(b_bf16x4);
        float32x4_t a_scaled_f32x4 = vmulq_n_f32(a_f32x4, alpha_val);
        float32x4_t b_scaled_f32x4 = vmulq_n_f32(b_f32x4, beta_val);
        float32x4_t result_f32x4 = vaddq_f32(a_scaled_f32x4, b_scaled_f32x4);
        bfloat16x4_t result_bf16x4 = vcvt_bf16_f32(result_f32x4);
        nk_b64_vec_t result_vec;
        result_vec.u16x4 = vreinterpret_u16_bf16(result_bf16x4);
        nk_partial_store_b16x4_serial_(result + i, &result_vec, n - i);
    }
}

NK_PUBLIC void nk_each_fma_bf16_neonbfdot(                      //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        bfloat16x4_t a_bf16x4 = vld1_bf16((bfloat16_t const *)a + i);
        bfloat16x4_t b_bf16x4 = vld1_bf16((bfloat16_t const *)b + i);
        bfloat16x4_t c_bf16x4 = vld1_bf16((bfloat16_t const *)c + i);
        float32x4_t a_f32x4 = vcvt_f32_bf16(a_bf16x4);
        float32x4_t b_f32x4 = vcvt_f32_bf16(b_bf16x4);
        float32x4_t c_f32x4 = vcvt_f32_bf16(c_bf16x4);
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_val);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_val);
        bfloat16x4_t result_bf16x4 = vcvt_bf16_f32(result_f32x4);
        vst1_bf16((bfloat16_t *)result + i, result_bf16x4);
    }
    if (i < n) {
        nk_b64_vec_t a_tail, b_tail, c_tail;
        nk_partial_load_b16x4_serial_(a + i, &a_tail, n - i);
        nk_partial_load_b16x4_serial_(b + i, &b_tail, n - i);
        nk_partial_load_b16x4_serial_(c + i, &c_tail, n - i);
        bfloat16x4_t a_bf16x4 = vreinterpret_bf16_u16(a_tail.u16x4);
        bfloat16x4_t b_bf16x4 = vreinterpret_bf16_u16(b_tail.u16x4);
        bfloat16x4_t c_bf16x4 = vreinterpret_bf16_u16(c_tail.u16x4);
        float32x4_t a_f32x4 = vcvt_f32_bf16(a_bf16x4);
        float32x4_t b_f32x4 = vcvt_f32_bf16(b_bf16x4);
        float32x4_t c_f32x4 = vcvt_f32_bf16(c_bf16x4);
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_val);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_val);
        bfloat16x4_t result_bf16x4 = vcvt_bf16_f32(result_f32x4);
        nk_b64_vec_t result_vec;
        result_vec.u16x4 = vreinterpret_u16_bf16(result_bf16x4);
        nk_partial_store_b16x4_serial_(result + i, &result_vec, n - i);
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

#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM_
#endif // NK_EACH_NEONBFDOT_H
