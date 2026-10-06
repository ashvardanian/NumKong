/**
 *  @brief SIMD-accelerated Dot Products for NEON BF16.
 *  @file include/numkong/dot/neonbfdot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_neonbfdot_instructions ARM NEON BF16 Instructions (ARMv8.6-BF16)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vbfdotq_f32                 BFDOT (V.4S, V.8H, V.8H)        3cy         2/cy        4/cy
 *      vcvt_f32_bf16               BFCVTN (V.4H, V.4S)             3cy         2/cy        4/cy
 *      vld1q_bf16                  LD1 (V.8H)                      4cy         2/cy        3/cy
 *      vaddvq_f32                  FADDP+FADDP (V.4S)              4cy         1/cy        2/cy
 *      vfmaq_f32                   FMLA (V.4S, V.4S, V.4S)         4cy         2/cy        4/cy
 *      vfmsq_f32                   FMLS (V.4S, V.4S, V.4S)         4cy         2/cy        4/cy
 *
 *  The ARMv8.6-BF16 extension provides the BFDOT instruction for accelerated BF16 dot products,
 *  targeting machine learning inference workloads. BF16 trades mantissa precision (7 bits vs 10 in
 *  FP16) for a larger exponent range matching FP32, eliminating overflow concerns during training.
 *
 *  BFDOT computes two BF16 dot products per lane, accumulating directly into FP32 without explicit
 *  conversion. This provides higher throughput than FP16 convert-then-FMA sequences for ML inference
 *  where the reduced precision is acceptable.
 *
 *  @section dot_neonbfdot_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines following structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_bf16x8 state with native BFDOT bf16 dot-products.
 *
 *  @code{c}
 *  nk_dot_bf16x8_state_neonbfdot_t state_first, state_second, state_third, state_fourth;
 *  bfloat16x8_t query_bf16x8, target_first_bf16x8, target_second_bf16x8, target_third_bf16x8, target_fourth_bf16x8;
 *  nk_dot_bf16x8_init_neonbfdot(&state_first);
 *  nk_dot_bf16x8_init_neonbfdot(&state_second);
 *  nk_dot_bf16x8_init_neonbfdot(&state_third);
 *  nk_dot_bf16x8_init_neonbfdot(&state_fourth);
 *  for (nk_size_t idx = 0; idx + 8 <= depth; idx += 8) {
 *      query_bf16x8 = vld1q_bf16(query_ptr + idx);
 *      target_first_bf16x8 = vld1q_bf16(target_first_ptr + idx);
 *      target_second_bf16x8 = vld1q_bf16(target_second_ptr + idx);
 *      target_third_bf16x8 = vld1q_bf16(target_third_ptr + idx);
 *      target_fourth_bf16x8 = vld1q_bf16(target_fourth_ptr + idx);
 *      nk_dot_bf16x8_update_neonbfdot(&state_first, query_bf16x8, target_first_bf16x8, idx, 8);
 *      nk_dot_bf16x8_update_neonbfdot(&state_second, query_bf16x8, target_second_bf16x8, idx, 8);
 *      nk_dot_bf16x8_update_neonbfdot(&state_third, query_bf16x8, target_third_bf16x8, idx, 8);
 *      nk_dot_bf16x8_update_neonbfdot(&state_fourth, query_bf16x8, target_fourth_bf16x8, idx, 8);
 *  }
 *  float32x4_t results_f32x4;
 *  nk_dot_bf16x8_finalize_neonbfdot(&state_first, &state_second, &state_third, &state_fourth, depth, &results_f32x4);
 *  @endcode
 */
#ifndef NK_DOT_NEONBFDOT_H
#define NK_DOT_NEONBFDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b8x8_serial_`
#include "numkong/cast/neon.h"   // `nk_e4m3x8_to_bf16x8_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#endif

NK_PUBLIC void nk_dot_bf16_neonbfdot(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    bfloat16x8_t a_bf16x8, b_bf16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_bf16_neonbfdot_cycle:
    if (count_scalars < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b16x8_serial_(b_scalars, &b_vec, count_scalars);
        a_bf16x8 = vreinterpretq_bf16_u16(a_vec.u16x8);
        b_bf16x8 = vreinterpretq_bf16_u16(b_vec.u16x8);
        count_scalars = 0;
    }
    else {
        a_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)a_scalars);
        b_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x4 = vbfdotq_f32(sum_f32x4, a_bf16x8, b_bf16x8);
    if (count_scalars) goto nk_dot_bf16_neonbfdot_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_dot_bf16c_neonbfdot(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                                      nk_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_bf16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_bf16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short const *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmsq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    nk_f32c_t tail_result;
    nk_dot_bf16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

NK_PUBLIC void nk_vdot_bf16c_neonbfdot(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                                       nk_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_bf16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_bf16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short const *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmsq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    nk_f32c_t tail_result;
    nk_vdot_bf16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

NK_PUBLIC void nk_dot_e4m3_neonbfdot(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    bfloat16x8_t a_bf16x8, b_bf16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e4m3_neonbfdot_cycle:
    if (count_scalars < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x8_serial_(b_scalars, &b_vec, count_scalars);
        a_bf16x8 = vreinterpretq_bf16_u16(nk_e4m3x8_to_bf16x8_neon_(a_vec.u8x8));
        b_bf16x8 = vreinterpretq_bf16_u16(nk_e4m3x8_to_bf16x8_neon_(b_vec.u8x8));
        count_scalars = 0;
    }
    else {
        a_bf16x8 = vreinterpretq_bf16_u16(nk_e4m3x8_to_bf16x8_neon_(vld1_u8(a_scalars)));
        b_bf16x8 = vreinterpretq_bf16_u16(nk_e4m3x8_to_bf16x8_neon_(vld1_u8(b_scalars)));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x4 = vbfdotq_f32(sum_f32x4, a_bf16x8, b_bf16x8);
    if (count_scalars) goto nk_dot_e4m3_neonbfdot_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_dot_e5m2_neonbfdot(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    bfloat16x8_t a_bf16x8, b_bf16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e5m2_neonbfdot_cycle:
    if (count_scalars < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x8_serial_(b_scalars, &b_vec, count_scalars);
        a_bf16x8 = vreinterpretq_bf16_u16(nk_e5m2x8_to_bf16x8_neon_(a_vec.u8x8));
        b_bf16x8 = vreinterpretq_bf16_u16(nk_e5m2x8_to_bf16x8_neon_(b_vec.u8x8));
        count_scalars = 0;
    }
    else {
        a_bf16x8 = vreinterpretq_bf16_u16(nk_e5m2x8_to_bf16x8_neon_(vld1_u8(a_scalars)));
        b_bf16x8 = vreinterpretq_bf16_u16(nk_e5m2x8_to_bf16x8_neon_(vld1_u8(b_scalars)));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x4 = vbfdotq_f32(sum_f32x4, a_bf16x8, b_bf16x8);
    if (count_scalars) goto nk_dot_e5m2_neonbfdot_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over bf16 scalars on NEON.
 */
typedef struct nk_dot_bf16x8_state_neonbfdot_t {
    float32x4_t sum_f32x4;
} nk_dot_bf16x8_state_neonbfdot_t;

NK_INTERNAL void nk_dot_bf16x8_init_neonbfdot(nk_dot_bf16x8_state_neonbfdot_t *state) {
    state->sum_f32x4 = vdupq_n_f32(0);
}

NK_INTERNAL void nk_dot_bf16x8_update_neonbfdot(nk_dot_bf16x8_state_neonbfdot_t *state, nk_b128_vec_t a,
                                                nk_b128_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    bfloat16x8_t a_bf16x8 = vreinterpretq_bf16_u16(a.u16x8);
    bfloat16x8_t b_bf16x8 = vreinterpretq_bf16_u16(b.u16x8);
    state->sum_f32x4 = vbfdotq_f32(state->sum_f32x4, a_bf16x8, b_bf16x8);
}

NK_INTERNAL void nk_dot_bf16x8_finalize_neonbfdot(                                                  //
    nk_dot_bf16x8_state_neonbfdot_t const *state_a, nk_dot_bf16x8_state_neonbfdot_t const *state_b, //
    nk_dot_bf16x8_state_neonbfdot_t const *state_c, nk_dot_bf16x8_state_neonbfdot_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = vaddvq_f32(state_a->sum_f32x4);
    result->f32s[1] = vaddvq_f32(state_b->sum_f32x4);
    result->f32s[2] = vaddvq_f32(state_c->sum_f32x4);
    result->f32s[3] = vaddvq_f32(state_d->sum_f32x4);
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
#endif // NK_DOT_NEONBFDOT_H
