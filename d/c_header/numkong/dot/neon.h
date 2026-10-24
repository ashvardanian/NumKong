/**
 *  @brief SIMD-accelerated Dot Products for NEON.
 *  @file include/numkong/dot/neon.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_neon_instructions NEON Dot Product Instructions
 *
 *  Key NEON instructions for dot products:
 *
 *      Intrinsic         Instruction                   Latency     Throughput
 *                                                                  A76     M4+/V1+/Oryon
 *      vfmaq_f32         FMLA (V.4S, V.4S, V.4S)       4cy         2/cy    4/cy
 *      vfmaq_f64         FMLA (V.2D, V.2D, V.2D)       4cy         2/cy    4/cy
 *      vmulq_f32         FMUL (V.4S, V.4S, V.4S)       3cy         2/cy    4/cy
 *      vaddvq_f32        FADDP+FADDP (reduce)          5cy         1/cy    1/cy
 *      vaddvq_f64        FADDP (V.2D to scalar)        3cy         1/cy    1/cy
 *      vcvt_f64_f32      FCVTL (V.2D, V.2S)            3cy         2/cy    2/cy
 *      vld2_f32          LD2 ({Vt.2S, Vt2.2S}, [Xn])   4cy         1/cy    1/cy
 *
 *  FMA throughput doubles on cores with 4 SIMD pipes (Apple M4+, Graviton3+, Oryon), but
 *  horizontal reductions remain at 1/cy on all cores and become the main bottleneck.
 *
 *  For f32 dot products, we upcast to f64 for accumulation to preserve precision and
 *  avoid catastrophic cancellation in large-magnitude sums.
 *
 *  @section dot_neon_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_f32x2 state for f32 inputs with double-precision accumulation,
 *  - nk_dot_f64x2 state with Dot2 stable dot-products for f64 inputs.
 *
 *  @code{c}
 *  nk_dot_f32x2_state_neon_t state_first, state_second, state_third, state_fourth;
 *  float32x2_t query_f32x2, target_first_f32x2, target_second_f32x2, target_third_f32x2, target_fourth_f32x2;
 *  nk_dot_f32x2_init_neon(&state_first);
 *  nk_dot_f32x2_init_neon(&state_second);
 *  nk_dot_f32x2_init_neon(&state_third);
 *  nk_dot_f32x2_init_neon(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 2 <= depth; idx += 2) {
 *      query_f32x2 = vld1_f32(query_ptr + idx);
 *      target_first_f32x2 = vld1_f32(target_first_ptr + idx);
 *      target_second_f32x2 = vld1_f32(target_second_ptr + idx);
 *      target_third_f32x2 = vld1_f32(target_third_ptr + idx);
 *      target_fourth_f32x2 = vld1_f32(target_fourth_ptr + idx);
 *      nk_dot_f32x2_update_neon(&state_first, query_f32x2, target_first_f32x2, idx, 2);
 *      nk_dot_f32x2_update_neon(&state_second, query_f32x2, target_second_f32x2, idx, 2);
 *      nk_dot_f32x2_update_neon(&state_third, query_f32x2, target_third_f32x2, idx, 2);
 *      nk_dot_f32x2_update_neon(&state_fourth, query_f32x2, target_fourth_f32x2, idx, 2);
 *  }
 *  float32x4_t results_f32x4;
 *  nk_dot_f32x2_finalize_neon(&state_first, &state_second, &state_third, &state_fourth, depth, &results_f32x4);
 *  @endcode
 *
 *  For f64 inputs, Dot2 compensated summation provides numerical stability:
 *
 *  @code{c}
 *  nk_dot_f64x2_state_neon_t state_first, state_second, state_third, state_fourth;
 *  float64x2_t query_f64x2, target_first_f64x2, target_second_f64x2, target_third_f64x2, target_fourth_f64x2;
 *  nk_dot_f64x2_init_neon(&state_first);
 *  nk_dot_f64x2_init_neon(&state_second);
 *  nk_dot_f64x2_init_neon(&state_third);
 *  nk_dot_f64x2_init_neon(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 2 <= depth; idx += 2) {
 *      query_f64x2 = vld1q_f64(query_ptr + idx);
 *      target_first_f64x2 = vld1q_f64(target_first_ptr + idx);
 *      target_second_f64x2 = vld1q_f64(target_second_ptr + idx);
 *      target_third_f64x2 = vld1q_f64(target_third_ptr + idx);
 *      target_fourth_f64x2 = vld1q_f64(target_fourth_ptr + idx);
 *      nk_dot_f64x2_update_neon(&state_first, query_f64x2, target_first_f64x2, idx, 2);
 *      nk_dot_f64x2_update_neon(&state_second, query_f64x2, target_second_f64x2, idx, 2);
 *      nk_dot_f64x2_update_neon(&state_third, query_f64x2, target_third_f64x2, idx, 2);
 *      nk_dot_f64x2_update_neon(&state_fourth, query_f64x2, target_fourth_f64x2, idx, 2);
 *  }
 *  float64x4_t results_f64x4;
 *  nk_dot_f64x2_finalize_neon(&state_first, &state_second, &state_third, &state_fourth, depth, &results_f64x4);
 *  @endcode
 */
#ifndef NK_DOT_NEON_H
#define NK_DOT_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/cast/neon.h" // `nk_e4m3x8_to_f16x8_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/** @brief Compensated horizontal sum of 2 f64 lanes via TwoSum. */
NK_INTERNAL nk_f64_t nk_dot_stable_sum_f64x2_neon_(float64x2_t sum_f64x2, float64x2_t compensation_f64x2) {
    // TwoSum merge of sum + compensation (2-wide)
    float64x2_t tentative_sum_f64x2 = vaddq_f64(sum_f64x2, compensation_f64x2);
    float64x2_t virtual_addend_f64x2 = vsubq_f64(tentative_sum_f64x2, sum_f64x2);
    float64x2_t rounding_error_f64x2 = vaddq_f64(
        vsubq_f64(sum_f64x2, vsubq_f64(tentative_sum_f64x2, virtual_addend_f64x2)),
        vsubq_f64(compensation_f64x2, virtual_addend_f64x2));
    // Scalar TwoSum 2→1
    nk_f64_t lower_sum = vgetq_lane_f64(tentative_sum_f64x2, 0);
    nk_f64_t upper_sum = vgetq_lane_f64(tentative_sum_f64x2, 1);
    nk_f64_t lower_error = vgetq_lane_f64(rounding_error_f64x2, 0);
    nk_f64_t upper_error = vgetq_lane_f64(rounding_error_f64x2, 1);
    nk_f64_t tentative_sum = lower_sum + upper_sum;
    nk_f64_t virtual_addend = tentative_sum - lower_sum;
    nk_f64_t rounding_error = (lower_sum - (tentative_sum - virtual_addend)) + (upper_sum - virtual_addend);
    return tentative_sum + (lower_error + upper_error + rounding_error);
}

#pragma region - Traditional Floats

NK_PUBLIC void nk_dot_f32_neon(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                               nk_f64_t *result) {
    // Upcast f32 to f64 for accumulation (2 f32s per iteration, avoids slow vget_low/high)
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 2 <= count_scalars; idx_scalars += 2) {
        float32x2_t a_f32x2 = vld1_f32(a_scalars + idx_scalars);
        float32x2_t b_f32x2 = vld1_f32(b_scalars + idx_scalars);
        float64x2_t a_f64x2 = vcvt_f64_f32(a_f32x2);
        float64x2_t b_f64x2 = vcvt_f64_f32(b_f32x2);
        sum_f64x2 = vfmaq_f64(sum_f64x2, a_f64x2, b_f64x2);
    }
    nk_f64_t sum_f64 = vaddvq_f64(sum_f64x2);
    for (; idx_scalars < count_scalars; ++idx_scalars)
        sum_f64 += (nk_f64_t)a_scalars[idx_scalars] * (nk_f64_t)b_scalars[idx_scalars];
    *result = sum_f64;
}

NK_PUBLIC void nk_dot_f32c_neon(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f64c_t *result) {
    // Upcast f32 to f64 for accumulation (2 complex pairs per iteration, avoids slow vget_low/high)
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_pairs = 0;
    // ARMv8.3-A FCMLA (`vcmlaq_rot0/rot90_f32`) was benchmarked as an alternative to the
    // deinterleave+4FMA pattern below. FCMLA processes only 2 complex pairs per iteration
    // (interleaved 128-bit operands, 2x `vcmlaq`), while `vld2_f32` deinterleaves 2 pairs
    // with 4 independent FMA instructions that fully utilize M4's 4 SIMD pipes. Result on
    // Apple M4 at n=4096: manual f32 39.7 GiB/s, FCMLA 17.1 GiB/s (2.3x slower).
    // The f64 upcast here trades throughput for precision — FCMLA offers neither advantage.
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        // Unpack 2 complex pairs into real and imaginary parts:
        float32x2x2_t a_f32x2x2 = vld2_f32((nk_f32_t const *)(a_pairs + idx_pairs));
        float32x2x2_t b_f32x2x2 = vld2_f32((nk_f32_t const *)(b_pairs + idx_pairs));
        // Upcast to f64
        float64x2_t a_real_f64x2 = vcvt_f64_f32(a_f32x2x2.val[0]);
        float64x2_t a_imag_f64x2 = vcvt_f64_f32(a_f32x2x2.val[1]);
        float64x2_t b_real_f64x2 = vcvt_f64_f32(b_f32x2x2.val[0]);
        float64x2_t b_imag_f64x2 = vcvt_f64_f32(b_f32x2x2.val[1]);
        // Compute the dot product: real = aᵣ × bᵣ - aᵢ × bᵢ, imag = aᵣ × bᵢ + aᵢ × bᵣ
        sum_real_f64x2 = vfmaq_f64(sum_real_f64x2, a_real_f64x2, b_real_f64x2);
        sum_real_f64x2 = vfmsq_f64(sum_real_f64x2, a_imag_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmaq_f64(sum_imag_f64x2, a_real_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmaq_f64(sum_imag_f64x2, a_imag_f64x2, b_real_f64x2);
    }
    // Reduce horizontal sums:
    nk_f64_t sum_real_f64 = vaddvq_f64(sum_real_f64x2);
    nk_f64_t sum_imag_f64 = vaddvq_f64(sum_imag_f64x2);
    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        nk_f64_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real_f64 += ar * br - ai * bi;
        sum_imag_f64 += ar * bi + ai * br;
    }
    result->real = sum_real_f64;
    result->imag = sum_imag_f64;
}

NK_PUBLIC void nk_vdot_f32c_neon(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                 nk_f64c_t *result) {
    // Upcast f32 to f64 for accumulation (2 complex pairs per iteration, avoids slow vget_low/high)
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 2 <= count_pairs; idx_pairs += 2) {
        // Unpack 2 complex pairs into real and imaginary parts:
        float32x2x2_t a_f32x2x2 = vld2_f32((nk_f32_t const *)(a_pairs + idx_pairs));
        float32x2x2_t b_f32x2x2 = vld2_f32((nk_f32_t const *)(b_pairs + idx_pairs));
        // Upcast to f64
        float64x2_t a_real_f64x2 = vcvt_f64_f32(a_f32x2x2.val[0]);
        float64x2_t a_imag_f64x2 = vcvt_f64_f32(a_f32x2x2.val[1]);
        float64x2_t b_real_f64x2 = vcvt_f64_f32(b_f32x2x2.val[0]);
        float64x2_t b_imag_f64x2 = vcvt_f64_f32(b_f32x2x2.val[1]);
        // Compute conjugate dot product: real = aᵣ × bᵣ + aᵢ × bᵢ, imag = aᵣ × bᵢ - aᵢ × bᵣ
        sum_real_f64x2 = vfmaq_f64(sum_real_f64x2, a_real_f64x2, b_real_f64x2);
        sum_real_f64x2 = vfmaq_f64(sum_real_f64x2, a_imag_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmaq_f64(sum_imag_f64x2, a_real_f64x2, b_imag_f64x2);
        sum_imag_f64x2 = vfmsq_f64(sum_imag_f64x2, a_imag_f64x2, b_real_f64x2);
    }
    // Reduce horizontal sums:
    nk_f64_t sum_real_f64 = vaddvq_f64(sum_real_f64x2);
    nk_f64_t sum_imag_f64 = vaddvq_f64(sum_imag_f64x2);
    // Handle the tail:
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        nk_f64_t ar = a_pair.real, ai = a_pair.imag, br = b_pair.real, bi = b_pair.imag;
        sum_real_f64 += ar * br + ai * bi;
        sum_imag_f64 += ar * bi - ai * br;
    }
    result->real = sum_real_f64;
    result->imag = sum_imag_f64;
}

/**
 *  @brief Running state for 64-bit dot accumulation over f32 scalars on NEON.
 *
 *  Processes 2 f32 values at a time, upcasting to f64 for accumulation to avoid
 *  catastrophic cancellation in long reductions.
 */
typedef struct nk_dot_f32x2_state_neon_t {
    float64x2_t sum_f64x2;
} nk_dot_f32x2_state_neon_t;

NK_INTERNAL void nk_dot_f32x2_init_neon(nk_dot_f32x2_state_neon_t *state) { state->sum_f64x2 = vdupq_n_f64(0); }

NK_INTERNAL void nk_dot_f32x2_update_neon(nk_dot_f32x2_state_neon_t *state, nk_b64_vec_t a, nk_b64_vec_t b,
                                          nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Upcast 2 f32s to f64s for high-precision accumulation
    float32x2_t a_f32x2 = vreinterpret_f32_u32(a.u32x2);
    float32x2_t b_f32x2 = vreinterpret_f32_u32(b.u32x2);
    float64x2_t a_f64x2 = vcvt_f64_f32(a_f32x2);
    float64x2_t b_f64x2 = vcvt_f64_f32(b_f32x2);
    state->sum_f64x2 = vfmaq_f64(state->sum_f64x2, a_f64x2, b_f64x2);
}

NK_INTERNAL void nk_dot_f32x2_finalize_neon(                                            //
    nk_dot_f32x2_state_neon_t const *state_a, nk_dot_f32x2_state_neon_t const *state_b, //
    nk_dot_f32x2_state_neon_t const *state_c, nk_dot_f32x2_state_neon_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f64s[0] = vaddvq_f64(state_a->sum_f64x2);
    result->f64s[1] = vaddvq_f64(state_b->sum_f64x2);
    result->f64s[2] = vaddvq_f64(state_c->sum_f64x2);
    result->f64s[3] = vaddvq_f64(state_d->sum_f64x2);
}

NK_PUBLIC void nk_dot_f64_neon(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                               nk_f64_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_f64x2 = vdupq_n_f64(0);
    float64x2_t a_f64x2, b_f64x2;

nk_dot_f64_neon_cycle:
    if (count_scalars < 2) {
        nk_b128_vec_t a_tail, b_tail;
        nk_partial_load_b64x2_serial_(a_scalars, &a_tail, count_scalars);
        nk_partial_load_b64x2_serial_(b_scalars, &b_tail, count_scalars);
        a_f64x2 = a_tail.f64x2;
        b_f64x2 = b_tail.f64x2;
        count_scalars = 0;
    }
    else {
        a_f64x2 = vld1q_f64(a_scalars);
        b_f64x2 = vld1q_f64(b_scalars);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }

    // TwoProd: h = a × b, r = fma(a, b, -h) captures the rounding error
    float64x2_t product_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
    float64x2_t product_error_f64x2 = vnegq_f64(vfmsq_f64(product_f64x2, a_f64x2, b_f64x2));
    // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
    float64x2_t tentative_sum_f64x2 = vaddq_f64(sum_f64x2, product_f64x2);
    float64x2_t virtual_addend_f64x2 = vsubq_f64(tentative_sum_f64x2, sum_f64x2);
    float64x2_t sum_error_f64x2 = vaddq_f64(vsubq_f64(sum_f64x2, vsubq_f64(tentative_sum_f64x2, virtual_addend_f64x2)),
                                            vsubq_f64(product_f64x2, virtual_addend_f64x2));
    // Update: sum = t, compensation += q + r
    sum_f64x2 = tentative_sum_f64x2;
    compensation_f64x2 = vaddq_f64(compensation_f64x2, vaddq_f64(sum_error_f64x2, product_error_f64x2));

    if (count_scalars) goto nk_dot_f64_neon_cycle;
    // Compensated horizontal reduction preserving Dot2 error tracking
    *result = nk_dot_stable_sum_f64x2_neon_(sum_f64x2, compensation_f64x2);
}

NK_PUBLIC void nk_dot_f64c_neon(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f64c_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated complex dot product
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_real_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_imag_f64x2 = vdupq_n_f64(0);
    float64x2_t a_real_f64x2, a_imag_f64x2, b_real_f64x2, b_imag_f64x2;

nk_dot_f64c_neon_cycle:
    if (count_pairs < 2) {
        nk_b128_vec_t a_tail, b_tail;
        nk_partial_load_b64x2_serial_(a_pairs, &a_tail, count_pairs * 2);
        nk_partial_load_b64x2_serial_(b_pairs, &b_tail, count_pairs * 2);
        float64x2_t zeros = vdupq_n_f64(0);
        a_real_f64x2 = vzip1q_f64(a_tail.f64x2, zeros);
        a_imag_f64x2 = vzip2q_f64(a_tail.f64x2, zeros);
        b_real_f64x2 = vzip1q_f64(b_tail.f64x2, zeros);
        b_imag_f64x2 = vzip2q_f64(b_tail.f64x2, zeros);
        count_pairs = 0;
    }
    else {
        float64x2x2_t a_f64x2x2 = vld2q_f64((nk_f64_t const *)a_pairs);
        float64x2x2_t b_f64x2x2 = vld2q_f64((nk_f64_t const *)b_pairs);
        a_real_f64x2 = a_f64x2x2.val[0];
        a_imag_f64x2 = a_f64x2x2.val[1];
        b_real_f64x2 = b_f64x2x2.val[0];
        b_imag_f64x2 = b_f64x2x2.val[1];
        a_pairs += 2, b_pairs += 2, count_pairs -= 2;
    }

    // Real part: aᵣ × bᵣ - aᵢ × bᵢ (using TwoProd and TwoSum)
    // First term: +aᵣ × bᵣ
    float64x2_t product_rr_f64x2 = vmulq_f64(a_real_f64x2, b_real_f64x2);
    float64x2_t error_rr_f64x2 = vnegq_f64(vfmsq_f64(product_rr_f64x2, a_real_f64x2, b_real_f64x2));
    float64x2_t tentative_sum_real_f64x2 = vaddq_f64(sum_real_f64x2, product_rr_f64x2);
    float64x2_t virtual_addend_real_f64x2 = vsubq_f64(tentative_sum_real_f64x2, sum_real_f64x2);
    float64x2_t error_sum_real_f64x2 = vaddq_f64(
        vsubq_f64(sum_real_f64x2, vsubq_f64(tentative_sum_real_f64x2, virtual_addend_real_f64x2)),
        vsubq_f64(product_rr_f64x2, virtual_addend_real_f64x2));
    sum_real_f64x2 = tentative_sum_real_f64x2;
    compensation_real_f64x2 = vaddq_f64(compensation_real_f64x2, vaddq_f64(error_sum_real_f64x2, error_rr_f64x2));
    // Second term: -aᵢ × bᵢ (negate product and error, then standard TwoSum)
    float64x2_t product_ii_f64x2 = vmulq_f64(a_imag_f64x2, b_imag_f64x2);
    float64x2_t error_ii_f64x2 = vnegq_f64(vfmsq_f64(product_ii_f64x2, a_imag_f64x2, b_imag_f64x2));
    float64x2_t neg_product_ii_f64x2 = vnegq_f64(product_ii_f64x2);
    float64x2_t neg_error_ii_f64x2 = vnegq_f64(error_ii_f64x2);
    tentative_sum_real_f64x2 = vaddq_f64(sum_real_f64x2, neg_product_ii_f64x2);
    virtual_addend_real_f64x2 = vsubq_f64(tentative_sum_real_f64x2, sum_real_f64x2);
    error_sum_real_f64x2 = vaddq_f64(
        vsubq_f64(sum_real_f64x2, vsubq_f64(tentative_sum_real_f64x2, virtual_addend_real_f64x2)),
        vsubq_f64(neg_product_ii_f64x2, virtual_addend_real_f64x2));
    sum_real_f64x2 = tentative_sum_real_f64x2;
    compensation_real_f64x2 = vaddq_f64(compensation_real_f64x2, vaddq_f64(error_sum_real_f64x2, neg_error_ii_f64x2));

    // Imag part: aᵣ × bᵢ + aᵢ × bᵣ (using TwoProd and TwoSum)
    // First term: +aᵣ × bᵢ
    float64x2_t product_ri_f64x2 = vmulq_f64(a_real_f64x2, b_imag_f64x2);
    float64x2_t error_ri_f64x2 = vnegq_f64(vfmsq_f64(product_ri_f64x2, a_real_f64x2, b_imag_f64x2));
    float64x2_t tentative_sum_imag_f64x2 = vaddq_f64(sum_imag_f64x2, product_ri_f64x2);
    float64x2_t virtual_addend_imag_f64x2 = vsubq_f64(tentative_sum_imag_f64x2, sum_imag_f64x2);
    float64x2_t error_sum_imag_f64x2 = vaddq_f64(
        vsubq_f64(sum_imag_f64x2, vsubq_f64(tentative_sum_imag_f64x2, virtual_addend_imag_f64x2)),
        vsubq_f64(product_ri_f64x2, virtual_addend_imag_f64x2));
    sum_imag_f64x2 = tentative_sum_imag_f64x2;
    compensation_imag_f64x2 = vaddq_f64(compensation_imag_f64x2, vaddq_f64(error_sum_imag_f64x2, error_ri_f64x2));
    // Second term: +aᵢ × bᵣ
    float64x2_t product_ir_f64x2 = vmulq_f64(a_imag_f64x2, b_real_f64x2);
    float64x2_t error_ir_f64x2 = vnegq_f64(vfmsq_f64(product_ir_f64x2, a_imag_f64x2, b_real_f64x2));
    tentative_sum_imag_f64x2 = vaddq_f64(sum_imag_f64x2, product_ir_f64x2);
    virtual_addend_imag_f64x2 = vsubq_f64(tentative_sum_imag_f64x2, sum_imag_f64x2);
    error_sum_imag_f64x2 = vaddq_f64(
        vsubq_f64(sum_imag_f64x2, vsubq_f64(tentative_sum_imag_f64x2, virtual_addend_imag_f64x2)),
        vsubq_f64(product_ir_f64x2, virtual_addend_imag_f64x2));
    sum_imag_f64x2 = tentative_sum_imag_f64x2;
    compensation_imag_f64x2 = vaddq_f64(compensation_imag_f64x2, vaddq_f64(error_sum_imag_f64x2, error_ir_f64x2));

    if (count_pairs) goto nk_dot_f64c_neon_cycle;
    // Compensated horizontal reduction preserving Dot2 error tracking
    result->real = nk_dot_stable_sum_f64x2_neon_(sum_real_f64x2, compensation_real_f64x2);
    result->imag = nk_dot_stable_sum_f64x2_neon_(sum_imag_f64x2, compensation_imag_f64x2);
}

NK_PUBLIC void nk_vdot_f64c_neon(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                 nk_f64c_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated conjugate dot product
    float64x2_t sum_real_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_imag_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_real_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_imag_f64x2 = vdupq_n_f64(0);
    float64x2_t a_real_f64x2, a_imag_f64x2, b_real_f64x2, b_imag_f64x2;

nk_vdot_f64c_neon_cycle:
    if (count_pairs < 2) {
        nk_b128_vec_t a_tail, b_tail;
        nk_partial_load_b64x2_serial_(a_pairs, &a_tail, count_pairs * 2);
        nk_partial_load_b64x2_serial_(b_pairs, &b_tail, count_pairs * 2);
        float64x2_t zeros = vdupq_n_f64(0);
        a_real_f64x2 = vzip1q_f64(a_tail.f64x2, zeros);
        a_imag_f64x2 = vzip2q_f64(a_tail.f64x2, zeros);
        b_real_f64x2 = vzip1q_f64(b_tail.f64x2, zeros);
        b_imag_f64x2 = vzip2q_f64(b_tail.f64x2, zeros);
        count_pairs = 0;
    }
    else {
        float64x2x2_t a_f64x2x2 = vld2q_f64((nk_f64_t const *)a_pairs);
        float64x2x2_t b_f64x2x2 = vld2q_f64((nk_f64_t const *)b_pairs);
        a_real_f64x2 = a_f64x2x2.val[0];
        a_imag_f64x2 = a_f64x2x2.val[1];
        b_real_f64x2 = b_f64x2x2.val[0];
        b_imag_f64x2 = b_f64x2x2.val[1];
        a_pairs += 2, b_pairs += 2, count_pairs -= 2;
    }

    // Real part: aᵣ × bᵣ + aᵢ × bᵢ (using TwoProd and TwoSum)
    // First term: +aᵣ × bᵣ
    float64x2_t product_rr_f64x2 = vmulq_f64(a_real_f64x2, b_real_f64x2);
    float64x2_t error_rr_f64x2 = vnegq_f64(vfmsq_f64(product_rr_f64x2, a_real_f64x2, b_real_f64x2));
    float64x2_t tentative_sum_real_f64x2 = vaddq_f64(sum_real_f64x2, product_rr_f64x2);
    float64x2_t virtual_addend_real_f64x2 = vsubq_f64(tentative_sum_real_f64x2, sum_real_f64x2);
    float64x2_t error_sum_real_f64x2 = vaddq_f64(
        vsubq_f64(sum_real_f64x2, vsubq_f64(tentative_sum_real_f64x2, virtual_addend_real_f64x2)),
        vsubq_f64(product_rr_f64x2, virtual_addend_real_f64x2));
    sum_real_f64x2 = tentative_sum_real_f64x2;
    compensation_real_f64x2 = vaddq_f64(compensation_real_f64x2, vaddq_f64(error_sum_real_f64x2, error_rr_f64x2));
    // Second term: +aᵢ × bᵢ (conjugate: add instead of subtract)
    float64x2_t product_ii_f64x2 = vmulq_f64(a_imag_f64x2, b_imag_f64x2);
    float64x2_t error_ii_f64x2 = vnegq_f64(vfmsq_f64(product_ii_f64x2, a_imag_f64x2, b_imag_f64x2));
    tentative_sum_real_f64x2 = vaddq_f64(sum_real_f64x2, product_ii_f64x2);
    virtual_addend_real_f64x2 = vsubq_f64(tentative_sum_real_f64x2, sum_real_f64x2);
    error_sum_real_f64x2 = vaddq_f64(
        vsubq_f64(sum_real_f64x2, vsubq_f64(tentative_sum_real_f64x2, virtual_addend_real_f64x2)),
        vsubq_f64(product_ii_f64x2, virtual_addend_real_f64x2));
    sum_real_f64x2 = tentative_sum_real_f64x2;
    compensation_real_f64x2 = vaddq_f64(compensation_real_f64x2, vaddq_f64(error_sum_real_f64x2, error_ii_f64x2));

    // Imag part: aᵣ × bᵢ - aᵢ × bᵣ (using TwoProd and TwoSum)
    // First term: +aᵣ × bᵢ
    float64x2_t product_ri_f64x2 = vmulq_f64(a_real_f64x2, b_imag_f64x2);
    float64x2_t error_ri_f64x2 = vnegq_f64(vfmsq_f64(product_ri_f64x2, a_real_f64x2, b_imag_f64x2));
    float64x2_t tentative_sum_imag_f64x2 = vaddq_f64(sum_imag_f64x2, product_ri_f64x2);
    float64x2_t virtual_addend_imag_f64x2 = vsubq_f64(tentative_sum_imag_f64x2, sum_imag_f64x2);
    float64x2_t error_sum_imag_f64x2 = vaddq_f64(
        vsubq_f64(sum_imag_f64x2, vsubq_f64(tentative_sum_imag_f64x2, virtual_addend_imag_f64x2)),
        vsubq_f64(product_ri_f64x2, virtual_addend_imag_f64x2));
    sum_imag_f64x2 = tentative_sum_imag_f64x2;
    compensation_imag_f64x2 = vaddq_f64(compensation_imag_f64x2, vaddq_f64(error_sum_imag_f64x2, error_ri_f64x2));
    // Second term: -aᵢ × bᵣ (conjugate: negate product and error, then standard TwoSum)
    float64x2_t product_ir_f64x2 = vmulq_f64(a_imag_f64x2, b_real_f64x2);
    float64x2_t error_ir_f64x2 = vnegq_f64(vfmsq_f64(product_ir_f64x2, a_imag_f64x2, b_real_f64x2));
    float64x2_t neg_product_ir_f64x2 = vnegq_f64(product_ir_f64x2);
    float64x2_t neg_error_ir_f64x2 = vnegq_f64(error_ir_f64x2);
    tentative_sum_imag_f64x2 = vaddq_f64(sum_imag_f64x2, neg_product_ir_f64x2);
    virtual_addend_imag_f64x2 = vsubq_f64(tentative_sum_imag_f64x2, sum_imag_f64x2);
    error_sum_imag_f64x2 = vaddq_f64(
        vsubq_f64(sum_imag_f64x2, vsubq_f64(tentative_sum_imag_f64x2, virtual_addend_imag_f64x2)),
        vsubq_f64(neg_product_ir_f64x2, virtual_addend_imag_f64x2));
    sum_imag_f64x2 = tentative_sum_imag_f64x2;
    compensation_imag_f64x2 = vaddq_f64(compensation_imag_f64x2, vaddq_f64(error_sum_imag_f64x2, neg_error_ir_f64x2));

    if (count_pairs) goto nk_vdot_f64c_neon_cycle;
    // Compensated horizontal reduction preserving Dot2 error tracking
    result->real = nk_dot_stable_sum_f64x2_neon_(sum_real_f64x2, compensation_real_f64x2);
    result->imag = nk_dot_stable_sum_f64x2_neon_(sum_imag_f64x2, compensation_imag_f64x2);
}

/**
 *  @brief Running state for 128-bit dot accumulation over f64 scalars on NEON.
 *
 *  Uses the Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product.
 */
typedef struct nk_dot_f64x2_state_neon_t {
    float64x2_t sum_f64x2;
    float64x2_t compensation_f64x2;
} nk_dot_f64x2_state_neon_t;

NK_INTERNAL void nk_dot_f64x2_init_neon(nk_dot_f64x2_state_neon_t *state) {
    state->sum_f64x2 = vdupq_n_f64(0);
    state->compensation_f64x2 = vdupq_n_f64(0);
}

NK_INTERNAL void nk_dot_f64x2_update_neon(nk_dot_f64x2_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                          nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    float64x2_t sum_f64x2 = state->sum_f64x2;
    float64x2_t compensation_f64x2 = state->compensation_f64x2;
    float64x2_t a_f64x2 = vreinterpretq_f64_u64(a.u64x2);
    float64x2_t b_f64x2 = vreinterpretq_f64_u64(b.u64x2);

    // TwoProd: h = a × b, r = fma(a, b, -h) captures the rounding error
    float64x2_t product_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
    float64x2_t product_error_f64x2 = vnegq_f64(vfmsq_f64(product_f64x2, a_f64x2, b_f64x2));

    // TwoSum: (t, q) = TwoSum(sum, h) where t = sum + h rounded, q = error
    float64x2_t tentative_sum_f64x2 = vaddq_f64(sum_f64x2, product_f64x2);
    float64x2_t virtual_addend_f64x2 = vsubq_f64(tentative_sum_f64x2, sum_f64x2);
    float64x2_t sum_error_f64x2 = vaddq_f64(vsubq_f64(sum_f64x2, vsubq_f64(tentative_sum_f64x2, virtual_addend_f64x2)),
                                            vsubq_f64(product_f64x2, virtual_addend_f64x2));

    // Update: sum = t, compensation += q + r
    state->sum_f64x2 = tentative_sum_f64x2;
    state->compensation_f64x2 = vaddq_f64(compensation_f64x2, vaddq_f64(sum_error_f64x2, product_error_f64x2));
}

NK_INTERNAL void nk_dot_f64x2_finalize_neon(                                            //
    nk_dot_f64x2_state_neon_t const *state_a, nk_dot_f64x2_state_neon_t const *state_b, //
    nk_dot_f64x2_state_neon_t const *state_c, nk_dot_f64x2_state_neon_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    // Compensated horizontal reduction preserving Dot2 error tracking per state
    result->f64s[0] = nk_dot_stable_sum_f64x2_neon_(state_a->sum_f64x2, state_a->compensation_f64x2);
    result->f64s[1] = nk_dot_stable_sum_f64x2_neon_(state_b->sum_f64x2, state_b->compensation_f64x2);
    result->f64s[2] = nk_dot_stable_sum_f64x2_neon_(state_c->sum_f64x2, state_c->compensation_f64x2);
    result->f64s[3] = nk_dot_stable_sum_f64x2_neon_(state_d->sum_f64x2, state_d->compensation_f64x2);
}

#pragma endregion - Traditional Floats

#pragma region - Smaller Floats

NK_PUBLIC void nk_dot_bf16_neon(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                nk_f32_t *result) {
    uint16x8_t a_u16x8, b_u16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_bf16_neon_cycle:
    if (count_scalars < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b16x8_serial_(b_scalars, &b_vec, count_scalars);
        a_u16x8 = a_vec.u16x8;
        b_u16x8 = b_vec.u16x8;
        count_scalars = 0;
    }
    else {
        a_u16x8 = vld1q_u16((nk_u16_t const *)a_scalars);
        b_u16x8 = vld1q_u16((nk_u16_t const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    float32x4_t a_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a_u16x8), 16));
    float32x4_t a_high_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(a_u16x8), 16));
    float32x4_t b_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b_u16x8), 16));
    float32x4_t b_high_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b_u16x8), 16));
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_low_f32x4, b_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_high_f32x4, b_high_f32x4);
    if (count_scalars) goto nk_dot_bf16_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over bf16 scalars on plain NEON.
 *
 *  Processes 8 bf16 values at a time (128 bits), converting to f32 via USHLL shift-16
 *  for accumulation without requiring the ARMv8.6-BF16 extension.
 */
typedef struct nk_dot_bf16x8_state_neon_t {
    float32x4_t sum_f32x4;
} nk_dot_bf16x8_state_neon_t;

NK_INTERNAL void nk_dot_bf16x8_init_neon(nk_dot_bf16x8_state_neon_t *state) { state->sum_f32x4 = vdupq_n_f32(0); }

NK_INTERNAL void nk_dot_bf16x8_update_neon(nk_dot_bf16x8_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                           nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Convert bf16 to f32 via USHLL shift-16 (low and high halves)
    float32x4_t a_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a.u16x8), 16));
    float32x4_t a_high_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(a.u16x8), 16));
    float32x4_t b_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b.u16x8), 16));
    float32x4_t b_high_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b.u16x8), 16));
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, a_low_f32x4, b_low_f32x4);
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, a_high_f32x4, b_high_f32x4);
}

NK_INTERNAL void nk_dot_bf16x8_finalize_neon(                                             //
    nk_dot_bf16x8_state_neon_t const *state_a, nk_dot_bf16x8_state_neon_t const *state_b, //
    nk_dot_bf16x8_state_neon_t const *state_c, nk_dot_bf16x8_state_neon_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = vaddvq_f32(state_a->sum_f32x4);
    result->f32s[1] = vaddvq_f32(state_b->sum_f32x4);
    result->f32s[2] = vaddvq_f32(state_c->sum_f32x4);
    result->f32s[3] = vaddvq_f32(state_d->sum_f32x4);
}

NK_PUBLIC void nk_dot_f16_neon(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    uint16x8_t a_u16x8, b_u16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_f16_neon_cycle:
    if (count_scalars < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b16x8_serial_(b_scalars, &b_vec, count_scalars);
        a_u16x8 = a_vec.u16x8;
        b_u16x8 = b_vec.u16x8;
        count_scalars = 0;
    }
    else {
        a_u16x8 = vld1q_u16((nk_u16_t const *)a_scalars);
        b_u16x8 = vld1q_u16((nk_u16_t const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    float32x4_t a_low_f32x4 = nk_f16x4_to_f32x4_neon_(vget_low_u16(a_u16x8));
    float32x4_t a_high_f32x4 = nk_f16x4_to_f32x4_neon_(vget_high_u16(a_u16x8));
    float32x4_t b_low_f32x4 = nk_f16x4_to_f32x4_neon_(vget_low_u16(b_u16x8));
    float32x4_t b_high_f32x4 = nk_f16x4_to_f32x4_neon_(vget_high_u16(b_u16x8));
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_low_f32x4, b_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_high_f32x4, b_high_f32x4);
    if (count_scalars) goto nk_dot_f16_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over f16 scalars on plain NEON.
 *
 *  Processes 8 f16 values at a time (128 bits), converting to f32 via integer bit
 *  manipulation for accumulation without requiring the ARMv8.2-A FP16 extension.
 */
typedef struct nk_dot_f16x8_state_neon_t {
    float32x4_t sum_f32x4;
} nk_dot_f16x8_state_neon_t;

NK_INTERNAL void nk_dot_f16x8_init_neon(nk_dot_f16x8_state_neon_t *state) { state->sum_f32x4 = vdupq_n_f32(0); }

NK_INTERNAL void nk_dot_f16x8_update_neon(nk_dot_f16x8_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                          nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Convert f16 to f32 via integer bit manipulation (low and high halves)
    float32x4_t a_low_f32x4 = nk_f16x4_to_f32x4_neon_(vget_low_u16(a.u16x8));
    float32x4_t a_high_f32x4 = nk_f16x4_to_f32x4_neon_(vget_high_u16(a.u16x8));
    float32x4_t b_low_f32x4 = nk_f16x4_to_f32x4_neon_(vget_low_u16(b.u16x8));
    float32x4_t b_high_f32x4 = nk_f16x4_to_f32x4_neon_(vget_high_u16(b.u16x8));
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, a_low_f32x4, b_low_f32x4);
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, a_high_f32x4, b_high_f32x4);
}

NK_INTERNAL void nk_dot_f16x8_finalize_neon(                                            //
    nk_dot_f16x8_state_neon_t const *state_a, nk_dot_f16x8_state_neon_t const *state_b, //
    nk_dot_f16x8_state_neon_t const *state_c, nk_dot_f16x8_state_neon_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = vaddvq_f32(state_a->sum_f32x4);
    result->f32s[1] = vaddvq_f32(state_b->sum_f32x4);
    result->f32s[2] = vaddvq_f32(state_c->sum_f32x4);
    result->f32s[3] = vaddvq_f32(state_d->sum_f32x4);
}

NK_PUBLIC void nk_dot_e4m3_neon(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e4m3_neon_cycle:
    if (count_scalars < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x8_serial_(b_scalars, &b_vec, count_scalars);
        a_f16x8 = nk_e4m3x8_to_f16x8_neon_(a_vec.u8x8);
        b_f16x8 = nk_e4m3x8_to_f16x8_neon_(b_vec.u8x8);
        count_scalars = 0;
    }
    else {
        a_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(a_scalars));
        b_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(b_scalars));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_low_f32x4, b_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_high_f32x4, b_high_f32x4);
    if (count_scalars) goto nk_dot_e4m3_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_dot_e5m2_neon(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e5m2_neon_cycle:
    if (count_scalars < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x8_serial_(b_scalars, &b_vec, count_scalars);
        a_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(a_vec.u8x8, 8));
        b_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(b_vec.u8x8, 8));
        count_scalars = 0;
    }
    else {
        a_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(vld1_u8(a_scalars), 8));
        b_f16x8 = vreinterpretq_f16_u16(vshll_n_u8(vld1_u8(b_scalars), 8));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_low_f32x4, b_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_high_f32x4, b_high_f32x4);
    if (count_scalars) goto nk_dot_e5m2_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_dot_e2m3_neon(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                nk_f32_t *result) {
    float16x8_t a_low_f16x8, a_high_f16x8, b_low_f16x8, b_high_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    // x16 TBL path: process 16 elements per iteration via lookup table upcast
nk_dot_e2m3_neon_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        nk_e2m3x16_to_f16x8x2_neon_(a_vec.u8x16, &a_low_f16x8, &a_high_f16x8);
        nk_e2m3x16_to_f16x8x2_neon_(b_vec.u8x16, &b_low_f16x8, &b_high_f16x8);
        count_scalars = 0;
    }
    else {
        nk_e2m3x16_to_f16x8x2_neon_(vld1q_u8(a_scalars), &a_low_f16x8, &a_high_f16x8);
        nk_e2m3x16_to_f16x8x2_neon_(vld1q_u8(b_scalars), &b_low_f16x8, &b_high_f16x8);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_low_f16(a_low_f16x8)), vcvt_f32_f16(vget_low_f16(b_low_f16x8)));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_high_f16(a_low_f16x8)),
                          vcvt_f32_f16(vget_high_f16(b_low_f16x8)));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_low_f16(a_high_f16x8)),
                          vcvt_f32_f16(vget_low_f16(b_high_f16x8)));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_high_f16(a_high_f16x8)),
                          vcvt_f32_f16(vget_high_f16(b_high_f16x8)));
    if (count_scalars) goto nk_dot_e2m3_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_dot_e3m2_neon(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                                nk_f32_t *result) {
    float16x8_t a_low_f16x8, a_high_f16x8, b_low_f16x8, b_high_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    // x16 TBL path: process 16 elements per iteration via lookup table upcast
nk_dot_e3m2_neon_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        nk_e3m2x16_to_f16x8x2_neon_(a_vec.u8x16, &a_low_f16x8, &a_high_f16x8);
        nk_e3m2x16_to_f16x8x2_neon_(b_vec.u8x16, &b_low_f16x8, &b_high_f16x8);
        count_scalars = 0;
    }
    else {
        nk_e3m2x16_to_f16x8x2_neon_(vld1q_u8(a_scalars), &a_low_f16x8, &a_high_f16x8);
        nk_e3m2x16_to_f16x8x2_neon_(vld1q_u8(b_scalars), &b_low_f16x8, &b_high_f16x8);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_low_f16(a_low_f16x8)), vcvt_f32_f16(vget_low_f16(b_low_f16x8)));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_high_f16(a_low_f16x8)),
                          vcvt_f32_f16(vget_high_f16(b_low_f16x8)));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_low_f16(a_high_f16x8)),
                          vcvt_f32_f16(vget_low_f16(b_high_f16x8)));
    sum_f32x4 = vfmaq_f32(sum_f32x4, vcvt_f32_f16(vget_high_f16(a_high_f16x8)),
                          vcvt_f32_f16(vget_high_f16(b_high_f16x8)));
    if (count_scalars) goto nk_dot_e3m2_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

#pragma endregion - Smaller Floats

#pragma region - Binary

NK_PUBLIC void nk_dot_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n_bits, NK_BITS_PER_BYTE);
    nk_u32_t dot = 0;
    nk_size_t i = 0;
    while (i + 16 <= n_bytes) {
        uint8x16_t popcount_u8x16 = vdupq_n_u8(0);
        for (nk_size_t cycle = 0; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            uint8x16_t a_u8x16 = vld1q_u8(a + i);
            uint8x16_t b_u8x16 = vld1q_u8(b + i);
            popcount_u8x16 = vaddq_u8(popcount_u8x16, vcntq_u8(vandq_u8(a_u8x16, b_u8x16)));
        }
        dot += (nk_u32_t)vaddlvq_u8(popcount_u8x16);
    }
    for (; i != n_bytes; ++i) dot += nk_u1x8_popcount_(a[i] & b[i]);
    *result = dot;
}

typedef struct nk_dot_u1x128_state_neon_t {
    uint32x4_t dot_count_u32x4;
} nk_dot_u1x128_state_neon_t;

NK_INTERNAL void nk_dot_u1x128_init_neon(nk_dot_u1x128_state_neon_t *state) { state->dot_count_u32x4 = vdupq_n_u32(0); }

NK_INTERNAL void nk_dot_u1x128_update_neon(nk_dot_u1x128_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                           nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    uint8x16_t and_u8x16 = vandq_u8(a.u8x16, b.u8x16);
    uint8x16_t popcount_u8x16 = vcntq_u8(and_u8x16);
    uint16x8_t popcount_u16x8 = vpaddlq_u8(popcount_u8x16);
    uint32x4_t popcount_u32x4 = vpaddlq_u16(popcount_u16x8);
    state->dot_count_u32x4 = vaddq_u32(state->dot_count_u32x4, popcount_u32x4);
}

NK_INTERNAL void nk_dot_u1x128_finalize_neon( //
    nk_dot_u1x128_state_neon_t const *state_a, nk_dot_u1x128_state_neon_t const *state_b,
    nk_dot_u1x128_state_neon_t const *state_c, nk_dot_u1x128_state_neon_t const *state_d, nk_size_t total_dimensions,
    nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    uint32x4_t ab_sum_u32x4 = vpaddq_u32(state_a->dot_count_u32x4, state_b->dot_count_u32x4);
    uint32x4_t cd_sum_u32x4 = vpaddq_u32(state_c->dot_count_u32x4, state_d->dot_count_u32x4);
    result->u32x4 = vpaddq_u32(ab_sum_u32x4, cd_sum_u32x4);
}

#pragma endregion - Binary

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_DOT_NEON_H
